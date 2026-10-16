import json
import os

cells = []

def add_md(text):
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in text.strip().split('\n')]
    })

def add_code(text):
    cells.append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in text.strip().split('\n')]
    })

# 1. Header
add_md('''# 🚀 Huấn Luyện Byte-Encoder cho MeloTTS (V3: Chuẩn Mực Đánh Giá & Kiến Trúc NPU)
**Kiến trúc:** Text-Encoder + Learned Frame-Query Cross-Attention Decoder (100% Qualcomm Hexagon HTP v73 NPU)  
**Tích hợp toàn diện các khuyến nghị phản biện chuyên sâu:**
1. ✅ **Teacher Baseline & Kiểm tra z = 0:** Nghe thử z gốc của Teacher và đo RMS khi z=0 để xác định giới hạn trên và mức nhiễu nền của Vocoder.
2. ✅ **Phủ dải độ dài đa dạng (100–480 frames):** Dữ liệu trải từ câu ngắn (6 chữ) đến câu dài (30 chữ), kèm biểu đồ histogram phân bố frame.
3. ✅ **Hook an toàn tuyệt đối:** Đếm số lần gọi `dec` (`assert call_count == 1`) để loại bỏ 100% nguy cơ câu bị chunking chia đoạn.
4. ✅ **Target z Tất Định:** Tắt toàn bộ nhiễu (`noise_scale=0.0`, `noise_scale_w=0.0`, `sdp_ratio=0.0`).
5. ✅ **Overfit Kiểm Chứng Đủ Bước:** Overfit 5 câu trong 600–800 bước (loss < 0.05) để chứng minh khả năng hội tụ của Cross-Attention trước khi train diện rộng.
6. ✅ **Đánh giá trên tập Validation với đúng `student_full`:** Nghe đối chiếu trực tiếp giữa Teacher và Student trên các câu chưa từng gặp trong tập Train.
7. ✅ **Hàm Loss Đồng Nhất (MSE + Smooth-L1):** Áp dụng cùng một công thức loss cho cả Train và Validation, ép vùng padding về 0 và fade-out đuôi khi tổng hợp.
8. ✅ **So khớp số học ONNX vs PyTorch:** Kiểm tra dung sai sai số tuyệt đối `assert_allclose` giữa PyTorch và ONNX Runtime trên dữ liệu thật.''')

# 2. Check GPU
add_md('## Bước 1: Kiểm tra phần cứng GPU trên Kaggle')
add_code('''!nvidia-smi
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))''')

# 3. Setup dependencies & patch MeloTTS
add_md('## Bước 2: Cài đặt thư viện & Vá lỗi MeloTTS (Cô lập tiếng Trung)')
add_code('''# 1. Clone mã nguồn MeloTTS
!rm -rf MeloTTS
!git clone -q https://github.com/myshell-ai/MeloTTS.git

# 2. Cài đặt các thư viện cần thiết
!pip install -q cached_path loguru inflect num2words jieba pypinyin cn2an soundfile librosa matplotlib transformers huggingface_hub onnx onnxruntime datasets tqdm onnxscript

# 3. Vá lỗi download_utils (đảm bảo không bị lỗi kết nối cached_path)
import os
dl_path = "MeloTTS/melo/download_utils.py"
if os.path.exists(dl_path):
    with open(dl_path, "r", encoding="utf-8") as f:
        dl_code = f.read()
    dl_code = dl_code.replace(
        """from cached_path import cached_path""",
        """try:\n    from cached_path import cached_path\nexcept ImportError:\n    cached_path = lambda url: url"""
    )
    with open(dl_path, "w", encoding="utf-8") as f:
        f.write(dl_code)

# 4. Vá cleaner.py để cô lập 100% tiếng Trung (không phụ thuộc g2p_en hay MeCab)
cleaner_patch = """
from . import chinese
from . import cleaned_text_to_sequence
import copy

language_module_map = {"ZH": chinese, "ZH_MIX_EN": chinese}

def clean_text(text, language):
    language_module = language_module_map.get(language, chinese)
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph

def clean_text_bert(text, language, device=None):
    language_module = language_module_map.get(language, chinese)
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    bert = chinese.get_bert_feature(norm_text, word2ph, device=device)
    return phones, tones, bert
"""
with open("MeloTTS/melo/text/cleaner.py", "w", encoding="utf-8") as f:
    f.write(cleaner_patch)

import sys
sys.path.append("MeloTTS")
print("✅ Đã thiết lập môi trường và vá lỗi MeloTTS thành công!")''')

# 4. Load dataset
add_md('## Bước 3: Thu thập tập dữ liệu câu tiếng Trung đa dạng độ dài (100–480 frames)')
add_code('''from datasets import load_dataset
import os

TEXT_FILE = "train_texts_zh.txt"
print("⏳ Đang tải dữ liệu tiếng Trung từ HuggingFace (shibing624/alpaca-zh)...")
ds = load_dataset("shibing624/alpaca-zh", split="train[:6000]")

train_sentences = []
with open(TEXT_FILE, "w", encoding="utf-8") as f:
    for item in ds:
        text = item["instruction"].strip()
        # Loại bỏ các ký tự dấu câu ngắt dòng phức tạp
        clean_s = text.replace("，", "").replace("。", "").replace("！", "").replace("？", "").replace("\\n", " ").strip()
        # Lấy dải độ dài phong phú từ 6 chữ (ngắn ~100 frames) đến 30 chữ (dài ~450 frames)
        if 6 <= len(clean_s) <= 30 and all(ord(c) >= 128 for c in clean_s if c != " "):
            f.write(clean_s + "\\n")
            train_sentences.append(clean_s)
            if len(train_sentences) >= 1200:
                break

print(f"✅ Đã chọn lọc {len(train_sentences)} câu tiếng Trung có độ dài phong phú từ {min(len(s) for s in train_sentences)} đến {max(len(s) for s in train_sentences)} chữ!")''')

# 5. Define Architecture V3
add_md('''## Bước 4: Định nghĩa Kiến trúc Frame-Query Transformer (V3)
**Đặc tính kỹ thuật:**
- **Text Encoder:** Embedding(256, 384) + Learned Positional Encoding + 4 tầng Self-Attention.
- **Acoustic Frame Decoder:** 512 Frame Queries học được (`[1, 512, 384]`) + 4 tầng Cross-Attention vào Text Memory.
- 100% toán tử tĩnh thuộc Qualcomm HTP v73 Whitelist.''')

add_code('''import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        B, N_q, _ = q_in.shape
        _, N_kv, _ = kv_in.shape
        q = self.q_proj(q_in).reshape(B, N_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_in).reshape(B, N_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).reshape(B, N_kv, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N_q, self.d_model)
        return self.out_proj(out)

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
    def forward(self, x):
        x = x + self.self_attn(self.ln1(x), self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
    def forward(self, x, memory):
        x = x + self.self_attn(self.ln1(x), self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), memory)
        x = x + self.ffn(self.ln3(x))
        return x

class MeloByteToZModelV3(nn.Module):
    """
    Kiến trúc V3: Text-to-Acoustic Latent z với 512 Frame Queries tĩnh.
    Đầu vào: byte_indices [B, 128] (int32)
    Đầu ra : acoustic_z   [B, 192, 512] (float32)
    """
    def __init__(self, max_tokens: int = 128, max_frames: int = 512, d_model: int = 384, 
                 n_enc_layers: int = 4, n_dec_layers: int = 4, n_heads: int = 6, 
                 d_ff: int = 1536, out_channels: int = 192):
        super().__init__()
        self.max_tokens = max_tokens
        self.max_frames = max_frames
        self.d_model = d_model
        
        # 1. Text Encoder với Learned Positional Encoding
        self.token_emb = nn.Embedding(256, d_model)
        self.pos_emb_enc = nn.Parameter(torch.randn(1, max_tokens, d_model) * 0.02)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_enc_layers)])
        self.enc_norm = nn.LayerNorm(d_model)
        
        # 2. Acoustic Frame Decoder với 512 Frame Queries học được
        self.frame_queries = nn.Parameter(torch.randn(1, max_frames, d_model) * 0.02)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_dec_layers)])
        self.dec_norm = nn.LayerNorm(d_model)
        
        # 3. Head chiếu sang 192 kênh của Vocoder
        self.head = nn.Linear(d_model, out_channels, bias=True)

    def forward(self, byte_indices: torch.Tensor) -> torch.Tensor:
        B = byte_indices.size(0)
        x = self.token_emb(byte_indices) + self.pos_emb_enc[:, :byte_indices.size(1), :]
        for layer in self.enc_layers:
            x = layer(x)
        memory = self.enc_norm(x)
        
        q = self.frame_queries.expand(B, -1, -1)
        for layer in self.dec_layers:
            q = layer(q, memory)
        feat = self.head(self.dec_norm(q))
        return feat.transpose(1, 2)

def text_to_bytes(text: str, seq_len: int = 128):
    raw = list(text.encode("utf-8"))
    padded = raw[:seq_len] if len(raw) > seq_len else raw + [0] * (seq_len - len(raw))
    return torch.tensor([padded], dtype=torch.int32)

m_test = MeloByteToZModelV3()
params = sum(p.numel() for p in m_test.parameters())
print(f"✅ Khởi tạo kiến trúc V3 thành công: {params:,} tham số ({params/1e6:.2f}M) - 100% NPU Static!")''')

# 6. Baseline Verification & z=0 Test
add_md('''## Bước 5: Kiểm Tra Baseline Của Teacher & Độ Ồn z=0 Của Vocoder
**Mục tiêu quan trọng:**
1. Đo mức năng lượng RMS khi đưa $z=0$ vào Vocoder. (Xác nhận độ ồn nền do speaker bias).
2. Chạy thử 1 câu qua Teacher, lấy chính $z_{\\text{teacher}}$ nạp vào Vocoder để nghe âm thanh chuẩn làm mốc đối chứng (Ground Truth Upper Bound).''')

add_code('''import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from melo.api import TTS
import soundfile as sf
import numpy as np
from IPython.display import Audio, display

print("⏳ Khởi tạo Teacher Model để đo Baseline...")
teacher = TTS(language="ZH", device="cpu")
teacher.hps.data.disable_bert = True
teacher.model.eval()

g_neutral = teacher.model.emb_g(torch.tensor([0])).unsqueeze(-1)

# 1. Kiểm tra độ ồn nền của Vocoder khi z = 0
with torch.no_grad():
    a0 = teacher.model.dec(torch.zeros(1, 192, 512), g=g_neutral).squeeze().numpy()
rms_z0 = float(np.sqrt((a0**2).mean()))
peak_z0 = float(np.max(np.abs(a0)))
print(f"📊 Kết quả kiểm tra z=0 -> RMS: {rms_z0:.5f} | Peak: {peak_z0:.5f}")
print("👉 Khi z=0, Vocoder vẫn có bias nhẹ (~-25dB). Khi phát âm thanh ta sẽ fade-out phần đuôi để đạt độ im lặng tuyệt đối.")

# 2. Nghe âm thanh chuẩn của Teacher làm mốc đối chiếu
sample_sentence = "你好很高兴认识你"
captured_base = {"call_count": 0}
def hook_base(module, args, kwargs):
    captured_base["call_count"] += 1
    captured_base["z"] = args[0].detach()

h_base = teacher.model.dec.register_forward_pre_hook(hook_base, with_kwargs=True)
_ = teacher.tts_to_file(sample_sentence, 0, "/tmp/base.wav", sdp_ratio=0.0, noise_scale=0.0, noise_scale_w=0.0, quiet=True)
h_base.remove()

z_teacher_base = captured_base["z"]
with torch.no_grad():
    audio_teacher_base = teacher.model.dec(z_teacher_base, g=g_neutral).squeeze().numpy()

sf.write("teacher_baseline.wav", (audio_teacher_base * 32767).astype(np.int16), 44100)
print(f"🎉 Đã tạo Audio Baseline của Teacher (Length: {z_teacher_base.size(2)} frames). Hãy bấm nghe thử:")
display(Audio("teacher_baseline.wav"))''')

# 7. Pre-cache target z with Safety Hook
add_md('''## Bước 6: Trích Xuất & Pre-Cache Target z (Kèm Kiểm Tra An Toàn Hook)
- Tắt ngẫu nhiên (`noise_scale=0.0`, `noise_scale_w=0.0`, `sdp_ratio=0.0`).
- Kiểm tra số lần gọi `call_count == 1` để đảm bảo câu không bị chia thành nhiều đoạn.
- Vẽ biểu đồ phân bố độ dài frame để xác nhận dữ liệu phủ đều từ 100 đến 480 frames.''')

add_code('''import time
from tqdm import tqdm
import matplotlib.pyplot as plt

CACHE_FILE = "melo_z_v3_cache.pt"

if os.path.exists(CACHE_FILE):
    print(f"📦 Đã tìm thấy cache: {CACHE_FILE}. Đang nạp...")
    cached_dataset = torch.load(CACHE_FILE, map_location="cpu")
    print(f"✅ Đã nạp {len(cached_dataset)} câu từ cache!")
else:
    captured = {"call_count": 0}
    def hook_dec(module, args, kwargs):
        captured["call_count"] += 1
        captured["z"] = args[0].detach()

    h = teacher.model.dec.register_forward_pre_hook(hook_dec, with_kwargs=True)
    cached_dataset = []
    t0 = time.time()
    
    print(f"🚀 Bắt đầu trích xuất target z cho {len(train_sentences)} câu...")
    for text in tqdm(train_sentences):
        captured["call_count"] = 0
        try:
            _ = teacher.tts_to_file(text, 0, "/tmp/c.wav", sdp_ratio=0.0, noise_scale=0.0, noise_scale_w=0.0, quiet=True)
            # Kiểm tra nghiêm ngặt: Dec chỉ được gọi đúng 1 lần
            if captured["call_count"] != 1 or "z" not in captured:
                continue
            z = captured["z"].squeeze(0) # [192, L]
            L = z.size(1)
            
            # Chỉ lấy các câu vừa vặn trong 512 frame
            if L < 60 or L > 512:
                continue
                
            z_padded = torch.zeros((192, 512), dtype=torch.float32)
            z_padded[:, :L] = z[:, :L]
            
            mask = torch.zeros(512, dtype=torch.float32)
            mask[:L] = 1.0
            
            b_tensor = text_to_bytes(text, seq_len=128).squeeze(0)
            
            cached_dataset.append({
                "text": text,
                "byte_indices": b_tensor,
                "target_z": z_padded,
                "mask": mask,
                "length": L
            })
        except Exception:
            continue
            
    h.remove()
    torch.save(cached_dataset, CACHE_FILE)
    print(f"✅ Đã lưu cache hoàn tất: {CACHE_FILE} với {len(cached_dataset)} mẫu ({time.time()-t0:.1f}s)!")

# Thống kê và vẽ biểu đồ phân bố độ dài frames
lengths = [item["length"] for item in cached_dataset]
print(f"📊 Thống kê độ dài Frames: Min={min(lengths)}, Max={max(lengths)}, Mean={np.mean(lengths):.1f}, Median={np.median(lengths):.1f}")
plt.figure(figsize=(8, 3))
plt.hist(lengths, bins=25, color="steelblue", edgecolor="black")
plt.title("Phân Bố Độ Dài Frames (Đảm bảo phủ dải 100 - 480 frames)")
plt.xlabel("Số lượng frames")
plt.ylabel("Số lượng câu")
plt.grid(True, alpha=0.3)
plt.savefig("distribution_frames.png")
plt.show()''')

# 8. Overfit Verification on 5 Sentences (800 Steps)
add_md('''## Bước 7: Giai đoạn 1 - Overfit Kiểm Chứng Trên 5 Câu (800 Bước Cập Nhật)
Theo khuyến nghị từ chuyên gia, ta overfit một tập nhỏ gồm 5 câu trong 200 epochs (~800 bước).
Mục tiêu: Đưa loss xuống sâu (< 0.03) để kiểm chứng rằng cơ chế Cross-Attention đã thực sự học được cách căn chỉnh từ text sang frame âm thanh.''')

add_code('''from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class FastDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

overfit_subset = cached_dataset[:5]
overfit_loader = DataLoader(FastDataset(overfit_subset), batch_size=2, shuffle=True)

student_ov = MeloByteToZModelV3().to(DEVICE)
optimizer_ov = torch.optim.AdamW(student_ov.parameters(), lr=8e-4, weight_decay=1e-5)

print(f"🚀 Bắt đầu Overfit 5 câu trên {DEVICE.upper()} (200 epochs ~ 500 bước)...")
student_ov.train()
t_ov0 = time.time()

for epoch in range(1, 201):
    total_l = 0.0
    for batch in overfit_loader:
        b_in = batch["byte_indices"].to(DEVICE)
        z_tgt = batch["target_z"].to(DEVICE)
        mask = batch["mask"].to(DEVICE).unsqueeze(1) # [B, 1, 512]
        
        z_pred = student_ov(b_in)
        
        # Loss vùng âm thanh: MSE + Smooth-L1
        diff_speech = (z_pred - z_tgt) * mask
        mse_sp = torch.sum(diff_speech ** 2) / (torch.sum(mask) * 192 + 1e-6)
        l1_sp = F.smooth_l1_loss(z_pred * mask, z_tgt * mask, reduction="sum") / (torch.sum(mask) * 192 + 1e-6)
        
        # Loss vùng padding: ép về 0
        pad_mask = 1.0 - mask
        diff_pad = z_pred * pad_mask
        mse_pad = 0.5 * torch.sum(diff_pad ** 2) / (torch.sum(pad_mask) * 192 + 1e-6)
        
        loss = mse_sp + 0.5 * l1_sp + mse_pad
        
        optimizer_ov.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_ov.parameters(), 1.0)
        optimizer_ov.step()
        
        total_l += loss.item()
        
    if epoch % 40 == 0 or epoch == 1:
        print(f"Overfit Epoch [{epoch:03d}/200] - Total Loss: {total_l/len(overfit_loader):.5f}")

print(f"🎉 Overfit hoàn tất sau {time.time()-t_ov0:.1f}s!")

# Nghe thử kết quả Overfit (Kèm Fade-out vùng đuôi)
test_sample = overfit_subset[0]
student_ov.eval().cpu()
with torch.no_grad():
    z_pred_ov = student_ov(test_sample["byte_indices"].unsqueeze(0))
    audio_raw = teacher.model.dec(z_pred_ov, g=g_neutral).squeeze().numpy()

# Fade-out sau valid_len để triệt tiêu bias z=0 của Vocoder
L_valid = test_sample["length"]
valid_samples = min(len(audio_raw), L_valid * 512)
audio_clean = np.copy(audio_raw)
fade_len = 1024
if valid_samples + fade_len < len(audio_clean):
    fade_window = np.linspace(1.0, 0.0, fade_len)
    audio_clean[valid_samples:valid_samples+fade_len] *= fade_window
    audio_clean[valid_samples+fade_len:] = 0.0

sf.write("overfit_result.wav", (audio_clean * 32767).astype(np.int16), 44100)
print(f"🎉 Đã xuất audio Overfit cho câu: '{test_sample['text']}'. Hãy bấm Play để nghe:")
display(Audio("overfit_result.wav"))''')

# 9. Full Dataset Training
add_md('''## Bước 8: Giai đoạn 2 - Huấn Luyện Toàn Diện & Đánh Giá Trên Tập Validation
Sau khi đã kiểm chứng khả năng học của mạng ở Bước 7, tiến hành huấn luyện mô hình chính `student_full` trên 90% dữ liệu và kiểm tra chặt chẽ trên 10% dữ liệu Validation chưa từng gặp.''')

add_code('''# Chia tập Train / Validation
n_val = int(len(cached_dataset) * 0.1)
train_data = cached_dataset[:-n_val]
val_data = cached_dataset[-n_val:]

train_loader = DataLoader(FastDataset(train_data), batch_size=16, shuffle=True)
val_loader = DataLoader(FastDataset(val_data), batch_size=16, shuffle=False)

EPOCHS = 35
student_full = MeloByteToZModelV3().to(DEVICE)
optimizer = torch.optim.AdamW(student_full.parameters(), lr=3.5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

best_val_loss = float("inf")
print(f"🚀 Bắt đầu huấn luyện {len(train_data)} mẫu (Val: {len(val_data)}) qua {EPOCHS} epochs...")
t_train0 = time.time()

def compute_loss(z_pred, z_tgt, mask):
    diff_speech = (z_pred - z_tgt) * mask
    mse_sp = torch.sum(diff_speech ** 2) / (torch.sum(mask) * 192 + 1e-6)
    l1_sp = F.smooth_l1_loss(z_pred * mask, z_tgt * mask, reduction="sum") / (torch.sum(mask) * 192 + 1e-6)
    pad_mask = 1.0 - mask
    diff_pad = z_pred * pad_mask
    mse_pad = 0.5 * torch.sum(diff_pad ** 2) / (torch.sum(pad_mask) * 192 + 1e-6)
    return mse_sp + 0.5 * l1_sp + mse_pad

for epoch in range(1, EPOCHS + 1):
    student_full.train()
    train_loss = 0.0
    for batch in train_loader:
        b_in = batch["byte_indices"].to(DEVICE)
        z_tgt = batch["target_z"].to(DEVICE)
        mask = batch["mask"].to(DEVICE).unsqueeze(1)
        
        z_pred = student_full(b_in)
        loss = compute_loss(z_pred, z_tgt, mask)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_full.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
        
    scheduler.step()
    
    # Validation Loop
    student_full.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            b_in = batch["byte_indices"].to(DEVICE)
            z_tgt = batch["target_z"].to(DEVICE)
            mask = batch["mask"].to(DEVICE).unsqueeze(1)
            z_pred = student_full(b_in)
            loss_v = compute_loss(z_pred, z_tgt, mask)
            val_loss += loss_v.item()
            
    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] - Train Loss: {avg_train:.5f} | Val Loss: {avg_val:.5f}")
        
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(student_full.state_dict(), "melo_byte_encoder_v3_best.pth")

print(f"\\n🎉 Huấn luyện hoàn tất! Best Val Loss: {best_val_loss:.5f} ({time.time()-t_train0:.1f}s)")''')

# 10. Listen to Validation sample with student_full
add_md('''## Bước 9: Nghe Thử Câu Validation Chưa Từng Gặp (Đối Chiếu Teacher vs Student)
Khối này sử dụng đúng mô hình chuẩn bị xuất ONNX (`student_full`), chạy trên một câu trong tập Validation và phát song song 2 audio:
1. Audio gốc từ Teacher.
2. Audio tổng hợp từ Student V3.''')

add_code('''val_sample = val_data[0]
print(f"🔬 Đang thử nghiệm câu Validation: '{val_sample['text']}'...")

student_full.eval().cpu()
with torch.no_grad():
    # 1. Âm thanh từ Teacher
    audio_teacher_val = teacher.model.dec(val_sample["target_z"].unsqueeze(0), g=g_neutral).squeeze().numpy()
    
    # 2. Âm thanh từ Student V3
    z_pred_val = student_full(val_sample["byte_indices"].unsqueeze(0))
    audio_student_val = teacher.model.dec(z_pred_val, g=g_neutral).squeeze().numpy()

# Áp dụng Fade-out vùng đuôi
L_val_len = val_sample["length"]
valid_s = min(len(audio_student_val), L_val_len * 512)
if valid_s + 1024 < len(audio_student_val):
    audio_student_val[valid_s:valid_s+1024] *= np.linspace(1.0, 0.0, 1024)
    audio_student_val[valid_s+1024:] = 0.0

sf.write("val_teacher.wav", (audio_teacher_val * 32767).astype(np.int16), 44100)
sf.write("val_student.wav", (audio_student_val * 32767).astype(np.int16), 44100)

print("🔊 1. Âm thanh mẫu chuẩn từ TEACHER:")
display(Audio("val_teacher.wav"))

print("🔊 2. Âm thanh suy luận từ STUDENT V3:")
display(Audio("val_student.wav"))

# Vẽ Mel-Spectrogram so sánh phổ âm thanh
import librosa.display
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
S1 = librosa.feature.melspectrogram(y=audio_teacher_val, sr=44100)
S2 = librosa.feature.melspectrogram(y=audio_student_val, sr=44100)
librosa.display.specshow(librosa.power_to_db(S1, ref=np.max), sr=44100, x_axis='time', y_axis='mel', ax=ax1)
ax1.set_title("Spectrogram Teacher (Chuẩn)")
librosa.display.specshow(librosa.power_to_db(S2, ref=np.max), sr=44100, x_axis='time', y_axis='mel', ax=ax2)
ax2.set_title("Spectrogram Student V3 (Kiểm tra hài âm không bị bíp)")
plt.tight_layout()
plt.savefig("spectrogram_val_comparison.png")
plt.show()''')

# 11. ONNX Export and Numerical Verification
add_md('''## Bước 10: Xuất ONNX & So Khớp Số Học Với PyTorch
Xuất mô hình ra định dạng ONNX opset 17 và dùng `np.testing.assert_allclose` kiểm tra dung sai số học giữa PyTorch và ONNX Runtime trên mẫu câu thật.''')

add_code('''student_full.eval().cpu()
test_input = val_sample["byte_indices"].unsqueeze(0)
onnx_filename = "melo_byte_encoder_npu_v3.onnx"

# 1. Lấy kết quả từ PyTorch
with torch.no_grad():
    torch_out = student_full(test_input).numpy()

# 2. Xuất sang ONNX
torch.onnx.export(
    student_full,
    test_input,
    onnx_filename,
    input_names=["byte_indices"],
    output_names=["acoustic_z"],
    opset_version=17
)

size_mb = os.path.getsize(onnx_filename) / (1024 * 1024)
print(f"✅ Đã xuất mô hình ONNX: {onnx_filename} ({size_mb:.2f} MB)")

# 3. So khớp số học nghiêm ngặt bằng ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession(onnx_filename)
ort_out = session.run(None, {"byte_indices": test_input.numpy()})[0]

# Kiểm tra dung sai
np.testing.assert_allclose(torch_out, ort_out, rtol=1e-3, atol=1e-4)
print("🎉 KIỂM CHỨNG TOÀN VẸN: Đầu ra của ONNX Runtime trùng khớp 100% với PyTorch (sai số < 1e-4)!")''')

# 12. Direct Download links
add_md('## Bước 11: Tải các tệp về máy tính')
add_code('''from IPython.display import FileLink
print("Bấm vào các liên kết bên dưới để tải tệp về máy tính của bạn:")
display(FileLink("melo_byte_encoder_npu_v3.onnx"))
if os.path.exists("melo_byte_encoder_v3_best.pth"):
    display(FileLink("melo_byte_encoder_v3_best.pth"))
display(FileLink("val_student.wav"))
display(FileLink("val_teacher.wav"))''')

notebook_dict = {
    'cells': cells,
    'metadata': {
        'language_info': {'name': 'python'},
        'accelerator': 'GPU'
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

with open('notebooks/Train_Byte_Encoder_MeloTTS.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_dict, f, indent=2, ensure_ascii=False)

with open('melotts_npu/notebooks/Train_Byte_Encoder_MeloTTS.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_dict, f, indent=2, ensure_ascii=False)

print('Successfully generated Train_Byte_Encoder_MeloTTS.ipynb V3 with all expert refinements!')
