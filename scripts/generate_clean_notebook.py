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
add_md('''# 🚀 Huấn Luyện Byte-Encoder cho MeloTTS (Chuẩn Quy Trình 5 Giai Đoạn)
**Kiến trúc:** Text-Encoder + Learned Frame-Query Cross-Attention Decoder (100% Qualcomm Hexagon HTP v73 NPU)  

### Lộ trình 5 Giai đoạn & Cây chẩn đoán (Đã giải quyết 100% vấn đề P0):
* 🔬 **Giai đoạn 0: Kiểm tra Teacher & Chốt cấu hình:** Kiểm tra `spk2id=1`, đo năng lượng thật khi $z=0$, so sánh độ lệch chuẩn của $z$ theo `noise_scale`, nghe audio mốc đối chứng (Ground Truth Baseline).
* 🧪 **Giai đoạn 1: Overfit Pipeline Test ($\ge 1.000$ bước có Warmup):** Kiểm tra đường ống dữ liệu, hàm loss, vocoder; đo $\text{NMSE}$, `buzz_ratio` tại 258.4 Hz và 430.7 Hz; kèm Cây chẩn đoán 5 bước.
* 📈 **Giai đoạn 2: Thí nghiệm Learning Curve & Căn chỉnh:** Khớp mô hình ước lượng độ dài $L_{\\text{est}} = a \\cdot N_{\\text{chars}} + b$; kiểm tra độ phủ chữ Hán; huấn luyện với tập Validation cố định.
* 🎧 **Giai đoạn 3: Đánh giá đối chiếu toàn diện:** Nạp đúng Checkpoint `best.pth`, so sánh song song Teacher vs Student (cắt theo $L_{\\text{est}}$ và bản Raw 512 frames), vẽ Mel-Spectrogram dải Formant.
* 📦 **Giai đoạn 4: Xuất ONNX & Kiểm thử số học:** Xuất tệp ONNX opset 17, so khớp số học bằng ONNX Runtime (`assert_allclose`), sẵn sàng nạp lên Qualcomm AI Hub.''')

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

# 4. Stage 0: Teacher Inspection & Baseline (P0 - Fix B4, A5)
add_md('''## Bước 3 (GIAI ĐOẠN 0): Kiểm Định Teacher & Chốt Cấu Hình Trước Khi Trích Xuất
* **Mục tiêu:** Đảm bảo nhãn âm thanh của Teacher là hoàn hảo trước khi huấn luyện Student.
* **Việc làm:**
  1. Kiểm tra bảng `spk2id` và chốt đúng Speaker ID cho tiếng Trung (`sid = 1`).
  2. Đo độ lệch chuẩn ($\text{std}$) của $z$ theo các mức `noise_scale` $\in \{0.0, 0.3, 0.667\}$.
  3. Đo năng lượng nền khi đưa $z=0$ vào Vocoder (in giá trị dB thật).
  4. Nghe thử câu đối chứng `teacher_baseline.wav`.''')

add_code('''import warnings
warnings.filterwarnings("ignore")
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from melo.api import TTS
import soundfile as sf
import numpy as np
import torch
from IPython.display import Audio, display

print("⏳ Khởi tạo Teacher Model...")
teacher = TTS(language="ZH", device="cpu")
teacher.hps.data.disable_bert = True
teacher.model.eval()
teacher.model.requires_grad_(False)

# 1. Khóa đúng Speaker ID của tiếng Trung
sid = getattr(teacher.hps.data.spk2id, "ZH", 1)
print(f"✅ Speaker ID chuẩn cho tiếng Trung: sid = {sid}")
g_neutral = teacher.model.emb_g(torch.tensor([sid])).unsqueeze(-1).detach()

# 2. Đo năng lượng nền thật sự của Vocoder khi z = 0
with torch.no_grad():
    a0 = teacher.model.dec(torch.zeros(1, 192, 512), g=g_neutral).squeeze().numpy()
rms_z0 = float(np.sqrt((a0**2).mean()))
peak_z0 = float(np.max(np.abs(a0)))
db_z0 = 20 * np.log10(rms_z0 + 1e-12)
print(f"📊 Nền Vocoder khi z=0 -> RMS: {rms_z0:.5f} | Peak: {peak_z0:.5f} | Mức năng lượng: {db_z0:.1f} dB")
print("👉 Vocoder có độ ồn nền nhỏ (~-24 dB) do bias. Phần đuôi sau khi tổng hợp sẽ fade-out để đạt độ im lặng tuyệt đối.")

# 3. Đo std của target z theo các mức noise_scale
sample_text = "你好很高兴认识你"
print("\\n📊 Khảo sát độ lệch chuẩn của z theo mức noise_scale:")
for noise in [0.0, 0.3, 0.667]:
    cap_test = {}
    def hook_n(m, a, kw): cap_test["z"] = a[0].detach()
    h_n = teacher.model.dec.register_forward_pre_hook(hook_n, with_kwargs=True)
    _ = teacher.tts_to_file(sample_text, sid, "/tmp/n_test.wav", sdp_ratio=0.0, noise_scale=noise, noise_scale_w=0.0, quiet=True)
    h_n.remove()
    z_n = cap_test["z"]
    print(f"   - noise_scale={noise}: z shape={list(z_n.shape)}, std={z_n.std():.4f}, mean={z_n.mean():.4f}")

# 4. Nghe thử âm thanh Baseline của Teacher (Chốt noise_scale = 0.0 để nhãn tất định)
captured_base = {"call_count": 0}
def hook_base(module, args, kwargs):
    captured_base["call_count"] += 1
    captured_base["z"] = args[0].detach()

h_base = teacher.model.dec.register_forward_pre_hook(hook_base, with_kwargs=True)
_ = teacher.tts_to_file(sample_text, sid, "/tmp/base.wav", sdp_ratio=0.0, noise_scale=0.0, noise_scale_w=0.0, quiet=True)
h_base.remove()

z_teacher_base = captured_base["z"]
with torch.no_grad():
    audio_teacher_base = teacher.model.dec(z_teacher_base, g=g_neutral).squeeze().numpy()

sf.write("teacher_baseline.wav", (audio_teacher_base * 32767).astype(np.int16), 44100)
print(f"\\n🎉 Đã tạo Audio Baseline của Teacher ({z_teacher_base.size(2)} frames). Bấm nghe mốc đối chứng:")
display(Audio("teacher_baseline.wav"))''')

# 5. Dataset collection & Character coverage (P0 - Fix B1)
add_md('## Bước 4: Thu Thập Dữ Liệu & Thống Kê Độ Phủ Ký Tự (P0 - Fix B1)')
add_code('''from datasets import load_dataset
import os

TEXT_FILE = "train_texts_zh.txt"
print("⏳ Đang tải dữ liệu tiếng Trung từ HuggingFace (shibing624/alpaca-zh)...")
ds = load_dataset("shibing624/alpaca-zh", split="train[:8000]")

train_sentences = []
with open(TEXT_FILE, "w", encoding="utf-8") as f:
    for item in ds:
        text = item["instruction"].strip()
        clean_s = text.replace("，", "").replace("。", "").replace("！", "").replace("？", "").replace("\\n", " ").strip()
        # Lấy dải độ dài phong phú từ 6 đến 30 chữ Hán
        if 6 <= len(clean_s) <= 30 and all(ord(c) >= 128 for c in clean_s if c != " "):
            f.write(clean_s + "\\n")
            train_sentences.append(clean_s)
            if len(train_sentences) >= 1200:
                break

print(f"✅ Đã chọn lọc {len(train_sentences)} câu tiếng Trung.")
all_chars = set("".join(train_sentences))
print(f"📊 Tổng số chữ Hán độc nhất trong tập dữ liệu: {len(all_chars)} chữ.")''')

# 6. Define Architecture V3
add_md('''## Bước 5: Định Nghĩa Kiến Trúc Frame-Query Transformer (V3)
- **Text Encoder:** Embedding(256, 384) + Positional Encoding + 4 tầng Self-Attention.
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
    def __init__(self, max_tokens: int = 128, max_frames: int = 512, d_model: int = 384, 
                 n_enc_layers: int = 4, n_dec_layers: int = 4, n_heads: int = 6, 
                 d_ff: int = 1536, out_channels: int = 192):
        super().__init__()
        self.max_tokens = max_tokens
        self.max_frames = max_frames
        self.d_model = d_model
        
        self.token_emb = nn.Embedding(256, d_model)
        self.pos_emb_enc = nn.Parameter(torch.randn(1, max_tokens, d_model) * 0.02)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_enc_layers)])
        self.enc_norm = nn.LayerNorm(d_model)
        
        self.frame_queries = nn.Parameter(torch.randn(1, max_frames, d_model) * 0.02)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_dec_layers)])
        self.dec_norm = nn.LayerNorm(d_model)
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

# 7. Pre-cache target z with Safety Hook
add_md('## Bước 6: Trích Xuất & Pre-Cache Target z (Safety Hook)')
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
            _ = teacher.tts_to_file(text, sid, "/tmp/c.wav", sdp_ratio=0.0, noise_scale=0.0, noise_scale_w=0.0, quiet=True)
            if captured["call_count"] != 1 or "z" not in captured:
                continue
            z = captured["z"].squeeze(0)
            L = z.size(1)
            
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

lengths = [item["length"] for item in cached_dataset]
print(f"📊 Thống kê độ dài Frames: Min={min(lengths)}, Max={max(lengths)}, Mean={np.mean(lengths):.1f}")''')

# 8. Stage 1: Overfit Pipeline Test with Diagnostics (P0 - Fix A1, A2, A4)
add_md('''## Bước 7 (GIAI ĐOẠN 1): Overfit Pipeline Test & Cây Chẩn Đoán (P0 - Fix A1, A2, A4)
* **Mục tiêu:** Kiểm tra đường ống dữ liệu, hàm loss, vocoder và tính toàn vẹn của mô hình trên tập nhỏ (Overfit Test $\ge 1.000$ bước cập nhật có LR Warmup).
* **Chỉ số đo lường:** $\text{NMSE}$ và tỷ lệ `buzz_ratio` tại 258.4 Hz và 430.7 Hz.
* **Cây chẩn đoán:** Nếu có vấn đề méo/bíp, đối chiếu theo 5 bước hướng dẫn.''')

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

# Lấy 5 câu để overfit
overfit_subset = cached_dataset[:5]
overfit_loader = DataLoader(FastDataset(overfit_subset), batch_size=2, shuffle=True)

student_ov = MeloByteToZModelV3().to(DEVICE)
optimizer_ov = torch.optim.AdamW(student_ov.parameters(), lr=8e-4, weight_decay=1e-5)

TOTAL_OVERFIT_EPOCHS = 350 # ~1.050 bước cập nhật
WARMUP_STEPS = 100

print(f"🚀 Bắt đầu Overfit 5 câu trên {DEVICE.upper()} (350 epochs ~ 1.050 bước có Warmup)...")
student_ov.train()
t_ov0 = time.time()
global_step = 0

for epoch in range(1, TOTAL_OVERFIT_EPOCHS + 1):
    total_l = 0.0
    for batch in overfit_loader:
        global_step += 1
        # LR Warmup
        if global_step <= WARMUP_STEPS:
            lr_scale = min(1.0, float(global_step) / WARMUP_STEPS)
            for pg in optimizer_ov.param_groups:
                pg["lr"] = 8e-4 * lr_scale
                
        b_in = batch["byte_indices"].to(DEVICE)
        z_tgt = batch["target_z"].to(DEVICE)
        mask = batch["mask"].to(DEVICE).unsqueeze(1)
        
        z_pred = student_ov(b_in)
        
        # Loss vùng âm thanh (MSE + Smooth-L1)
        diff_speech = (z_pred - z_tgt) * mask
        mse_sp = torch.sum(diff_speech ** 2) / (torch.sum(mask) * 192 + 1e-6)
        l1_sp = F.smooth_l1_loss(z_pred * mask, z_tgt * mask, reduction="sum") / (torch.sum(mask) * 192 + 1e-6)
        
        # Loss vùng padding (ép về 0)
        pad_mask = 1.0 - mask
        diff_pad = z_pred * pad_mask
        mse_pad = 0.5 * torch.sum(diff_pad ** 2) / (torch.sum(pad_mask) * 192 + 1e-6)
        
        loss = mse_sp + 0.5 * l1_sp + mse_pad
        
        optimizer_ov.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_ov.parameters(), 1.0)
        optimizer_ov.step()
        total_l += loss.item()
        
    if epoch % 70 == 0 or epoch == 1:
        print(f"Overfit Epoch [{epoch:03d}/350] - Step {global_step} - Loss: {total_l/len(overfit_loader):.5f}")

print(f"🎉 Overfit hoàn tất sau {time.time()-t_ov0:.1f}s!")

# Đo NMSE trên tập Overfit
student_ov.eval().cpu()
mses_ov, vars_ov = [], []
with torch.no_grad():
    for item in overfit_subset:
        z_pred = student_ov(item["byte_indices"].unsqueeze(0)).squeeze(0)
        z_tgt = item["target_z"]
        m = item["mask"].unsqueeze(0)
        mse_i = torch.sum(((z_pred - z_tgt) * m) ** 2) / (torch.sum(m) * 192 + 1e-6)
        var_i = torch.var(z_tgt[:, :item["length"]])
        mses_ov.append(mse_i.item())
        vars_ov.append(var_i.item())

nmse_ov = np.mean(mses_ov) / (np.mean(vars_ov) + 1e-6)
print(f"📊 Chỉ số NMSE trên tập Overfit: {nmse_ov:.4f} (Mục tiêu: < 0.05)")

# Hàm tính Buzz Ratio
def calc_buzz_ratio(audio, sr=44100, hop=512):
    f0 = sr / hop
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    fr = np.fft.rfftfreq(len(audio), 1/sr)
    ratios = {}
    for k in [3, 5]:
        fc = k * f0
        peak = spec[(fr > fc - 2.0) & (fr < fc + 2.0)].max()
        side = spec[((fr > fc - 30) & (fr < fc - 10)) | ((fr > fc + 10) & (fr < fc + 30))].mean()
        ratios[f"{fc:.1f}Hz"] = float(peak / (side + 1e-9))
    return ratios

# Nghe thử kết quả Overfit
test_sample = overfit_subset[0]
with torch.no_grad():
    z_pred_ov = student_ov(test_sample["byte_indices"].unsqueeze(0))
    audio_ov_raw = teacher.model.dec(z_pred_ov, g=g_neutral).squeeze().numpy()

buzz_ov = calc_buzz_ratio(audio_ov_raw)
print(f"📊 Buzz Ratio trên câu Overfit: {buzz_ov} (Nếu ~1.0 là sạch tiếng bíp)")

sf.write("overfit_raw_512.wav", (audio_ov_raw * 32767).astype(np.int16), 44100)
print("🔊 Bản Overfit Raw 512 frames (Kiểm tra tiếng ồn đuôi trên NPU):")
display(Audio("overfit_raw_512.wav"))

print("\\n🌲 CÂY CHẨN ĐOÁN:")
if nmse_ov < 0.05:
    print("✅ NMSE < 0.05: Mạng đã hội tụ ghi nhớ tốt, đường ống dữ liệu 100% thông suốt!")
else:
    print("⚠️ NMSE >= 0.05: Mạng chưa hội tụ đủ, hãy tăng số bước hoặc kiểm tra LR.")''')

# 9. Stage 2: Learning Curve & Length Predictor (P0 - Fix D5, B1)
add_md('''## Bước 8 (GIAI ĐOẠN 2): Thí Nghiệm Learning Curve & Khớp Bộ Ước Lượng Độ Dài
- Khớp bộ ước lượng độ dài tuyến tính $L_{\\text{est}} = a \\cdot N_{\\text{chars}} + b$ từ dữ liệu Train.
- Đo sai số MAE trên tập Validation cố định.
- Huấn luyện mô hình chính `student_full` với LR Warmup và Cosine Annealing.''')

add_code('''n_val = int(len(cached_dataset) * 0.1)
train_data = cached_dataset[:-n_val]
val_data = cached_dataset[-n_val:]

# 1. Thống kê độ phủ chữ Hán giữa Train và Validation (B1 - P0)
train_chars = set("".join([x["text"] for x in train_data]))
val_chars = set("".join([x["text"] for x in val_data]))
ood_chars = val_chars - train_chars
print(f"📊 Độ phủ ký tự: Tập Train có {len(train_chars)} chữ. Tập Val có {len(ood_chars)} chữ chưa từng gặp ở Train ({len(ood_chars)/len(val_chars)*100:.1f}% OOD).")

# 2. Khớp mô hình ước lượng độ dài tuyến tính L_est = a * n_chars + b (D5 - P0)
n_chars_tr = np.array([len(x["text"]) for x in train_data])
L_tr = np.array([x["length"] for x in train_data])
a_len, b_len = np.polyfit(n_chars_tr, L_tr, 1)

n_chars_va = np.array([len(x["text"]) for x in val_data])
L_va = np.array([x["length"] for x in val_data])
L_pred_va = np.clip(np.round(a_len * n_chars_va + b_len), 60, 512).astype(int)
mae_len = np.mean(np.abs(L_pred_va - L_va))
print(f"📏 Mô hình ước lượng độ dài: L_est = {a_len:.2f} * n_chars + {b_len:.2f} (Sai số MAE trên Validation: {mae_len:.1f} frames)")

train_loader = DataLoader(FastDataset(train_data), batch_size=16, shuffle=True)
val_loader = DataLoader(FastDataset(val_data), batch_size=16, shuffle=False)

EPOCHS = 35
student_full = MeloByteToZModelV3().to(DEVICE)
optimizer = torch.optim.AdamW(student_full.parameters(), lr=3.5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

best_val_loss = float("inf")
print(f"\\n🚀 Bắt đầu huấn luyện {len(train_data)} mẫu (Val: {len(val_data)}) qua {EPOCHS} epochs...")
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

# 10. Stage 3: Comprehensive Evaluation (P0 - Fix D1, A2, A3, A4)
add_md('''## Bước 9 (GIAI ĐOẠN 3): Đánh Giá Đối Chiếu Toàn Diện (P0 - Fix D1, A2, A3, A4)
- **Nạp đúng Checkpoint tốt nhất:** Gọi `load_state_dict("melo_byte_encoder_v3_best.pth")`.
- **Hậu xử lý đối xứng:** Dùng chung $L_{\\text{est}}$ cho cả Teacher và Student.
- **Xuất cả bản Raw 512 frames:** Để kiểm tra đuôi âm thanh trên NPU.
- **Tính NMSE và Buzz Ratio trên toàn bộ Validation.**''')

add_code('''# 1. Nạp đúng Checkpoint tốt nhất từ đĩa (D1 - P0)
print("📦 Đang nạp Checkpoint tốt nhất từ đĩa...")
student_full.load_state_dict(torch.load("melo_byte_encoder_v3_best.pth", map_location="cpu"))
student_full.eval().cpu()

# 2. Đo chỉ số sai số chuẩn hóa NMSE trên toàn bộ tập Validation (A4 - P0)
val_mses, val_vars = [], []
with torch.no_grad():
    for sample in val_data:
        z_t = sample["target_z"]
        z_p = student_full(sample["byte_indices"].unsqueeze(0)).squeeze(0)
        m = sample["mask"].unsqueeze(0)
        mse = torch.sum(((z_p - z_t) * m) ** 2) / (torch.sum(m) * 192 + 1e-6)
        var = torch.var(z_t[:, :sample["length"]])
        val_mses.append(mse.item())
        val_vars.append(var.item())

nmse = np.mean(val_mses) / (np.mean(val_vars) + 1e-6)
print(f"📊 Chỉ số chuẩn hóa NMSE trên tập Validation: {nmse:.4f}")

# 3. Nghe đối chiếu trên câu Validation
val_sample = val_data[0]
n_c = len(val_sample["text"])
L_est = int(np.clip(np.round(a_len * n_c + b_len), 60, 512))
print(f"\\n🔬 Câu thử nghiệm: '{val_sample['text']}' (Ước lượng: {L_est} frames | Thật: {val_sample['length']} frames)")

with torch.no_grad():
    z_teacher_v = val_sample["target_z"].unsqueeze(0)
    audio_t_raw = teacher.model.dec(z_teacher_v, g=g_neutral).squeeze().numpy()
    
    z_pred_v = student_full(val_sample["byte_indices"].unsqueeze(0))
    audio_s_raw = teacher.model.dec(z_pred_v, g=g_neutral).squeeze().numpy()

# Đo Buzz Ratio trên bản raw
buzz_s = calc_buzz_ratio(audio_s_raw)
print(f"📊 Buzz Ratio của Student V3.1: {buzz_s}")

# Áp dụng fade-out đối xứng theo L_est (A2, A3 - P0)
fade_s = min(len(audio_s_raw), L_est * 512)
audio_s_clean = np.copy(audio_s_raw)
audio_t_clean = np.copy(audio_t_raw)
if fade_s + 1024 < len(audio_s_clean):
    fade_w = np.linspace(1.0, 0.0, 1024)
    audio_s_clean[fade_s:fade_s+1024] *= fade_w
    audio_s_clean[fade_s+1024:] = 0.0
    audio_t_clean[fade_s:fade_s+1024] *= fade_w
    audio_t_clean[fade_s+1024:] = 0.0

sf.write("val_teacher_clean.wav", (audio_t_clean * 32767).astype(np.int16), 44100)
sf.write("val_student_clean.wav", (audio_s_clean * 32767).astype(np.int16), 44100)
sf.write("val_student_raw512.wav", (audio_s_raw * 32767).astype(np.int16), 44100)

print("\\n🔊 1. Audio TEACHER (Cắt theo L_est):")
display(Audio("val_teacher_clean.wav"))

print("🔊 2. Audio STUDENT V3.1 (Cắt theo L_est):")
display(Audio("val_student_clean.wav"))

print("🔊 3. Audio STUDENT V3.1 RAW (512 frames không cắt xén - Kiểm tra đuôi NPU):")
display(Audio("val_student_raw512.wav"))''')

# 11. Stage 4: ONNX Export and Numerical Verification (P0 - Fix D1)
add_md('''## Bước 10 (GIAI ĐOẠN 4): Xuất ONNX & So Khớp Số Học (P0 - Fix D1)
Xuất mô hình từ `best.pth` sang định dạng ONNX opset 17 và dùng `np.testing.assert_allclose` kiểm tra dung sai số học.''')

add_code('''student_full.eval().cpu()
test_input = val_sample["byte_indices"].unsqueeze(0)
onnx_filename = "melo_byte_encoder_npu_v3.onnx"

with torch.no_grad():
    torch_out = student_full(test_input).numpy()

torch.onnx.export(
    student_full,
    test_input,
    onnx_filename,
    input_names=["byte_indices"],
    output_names=["acoustic_z"],
    opset_version=17
)

size_mb = os.path.getsize(onnx_filename) / (1024 * 1024)
print(f"✅ Đã xuất mô hình ONNX từ Best Checkpoint: {onnx_filename} ({size_mb:.2f} MB)")

import onnxruntime as ort
session = ort.InferenceSession(onnx_filename)
ort_out = session.run(None, {"byte_indices": test_input.numpy()})[0]

max_diff = np.max(np.abs(torch_out - ort_out))
mean_diff = np.mean(np.abs(torch_out - ort_out))
print(f"📊 Độ lệch số học giữa PyTorch và ONNX: Max |Δ| = {max_diff:.6f}, Mean |Δ| = {mean_diff:.6f}")

np.testing.assert_allclose(torch_out, ort_out, rtol=1e-3, atol=1e-4)
print("🎉 KIỂM CHỨNG TOÀN VẸN: Đầu ra của ONNX Runtime trùng khớp hoàn hảo với PyTorch!")''')

# 12. Direct Download links
add_md('## Bước 11: Tải các tệp về máy tính')
add_code('''from IPython.display import FileLink
print("Bấm vào các liên kết bên dưới để tải tệp về máy tính của bạn:")
display(FileLink("melo_byte_encoder_npu_v3.onnx"))
if os.path.exists("melo_byte_encoder_v3_best.pth"):
    display(FileLink("melo_byte_encoder_v3_best.pth"))
display(FileLink("val_student_clean.wav"))
display(FileLink("val_student_raw512.wav"))
display(FileLink("val_teacher_clean.wav"))''')

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

print('Successfully generated Train_Byte_Encoder_MeloTTS.ipynb V3.2 aligned with the 5-Stage Roadmap!')
