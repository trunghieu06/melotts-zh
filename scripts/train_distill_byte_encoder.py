"""
Knowledge Distillation Training Script:
Byte-Level Text Encoder cho MeloTTS (100% NPU-Native)
======================================================
Mục tiêu:
- Teacher Model: MeloTTS-ZH gốc (sử dụng CPU / jieba / pypinyin) trích xuất đặc trưng âm học chuẩn [1, 192, L].
- Student Model: ByteTransformer (nhận UTF-8 byte integer tensor [1, 512] -> dự đoán đặc trưng [1, 192, L]).
- Hàm Loss: Mean Squared Error (MSE) giữa Student output và Teacher target.
- Xuất kết quả: Checkpoint .pth và file ONNX sẵn sàng nạp lên Qualcomm AI Hub / NPU.

Cách chạy:
  python scripts/train_distill_byte_encoder.py --epochs 10 --batch_size 4
"""

import os
import sys
import ssl
import time
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Bỏ qua xác thực SSL nếu tải weights tự động
ssl._create_default_https_context = ssl._create_unverified_context

# Import mô hình ByteTransformer từ src
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from step1_asr.byte_transformer import (
    ByteTransformer,
    TransformerBlock,
    text_to_byte_tensor,
    export_model_to_onnx
)


class MeloByteEncoder(nn.Module):
    """
    Byte-Level Text Encoder dành riêng cho MeloTTS.
    Nhận đầu vào: byte_indices [B, 512] (int32)
    Đầu ra: acoustic representation [B, 192, 512] (FP32)
    """
    def __init__(self, d_model: int = 512, n_layers: int = 8, n_heads: int = 8, d_ff: int = 2048, out_channels: int = 192):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(256, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        # Chiếu từ d_model (512) sang out_channels (192) của MeloTTS
        self.out_proj = nn.Linear(d_model, out_channels, bias=False)

    def forward(self, byte_indices: torch.Tensor) -> torch.Tensor:
        # byte_indices: [B, 512]
        x = self.embedding(byte_indices) # [B, 512, d_model]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        feat = self.out_proj(x) # [B, 512, 192]
        # Đổi chiều sang chuẩn [B, 192, 512] cho Vocoder
        return feat.transpose(1, 2)


# Tập dữ liệu mẫu nếu người dùng chưa truyền file text lớn
DEFAULT_SAMPLE_TEXTS = [
    "你好，很高兴认识你。",
    "今天天气非常好，阳光明媚。",
    "欢迎使用智能语音离线翻译系统。",
    "工业制造车间的噪音环境非常复杂。",
    "Qualcomm Dragonwing 芯片具备强大的神经网络计算能力。",
    "这是一条用于测试端到端语音合成的示例句子。",
    "深度学习模型正在边缘设备上高效运行。",
    "语音合成技术能够将文本直接转化为自然的人声。",
    "请确认设备电源连接正常，准备开始工作。",
    "系统已经成功启动，所有模块自检完毕。"
]


class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def extract_teacher_features(melo_model, text: str, device: str = "cpu") -> torch.Tensor:
    """
    Trích xuất ma trận đặc trưng âm học target từ Teacher (MeloTTS gốc).
    Kết quả trả về: [1, 192, L_target]
    """
    captured = {}
    def hook_fn(module, args, output):
        # output[0] chính là hidden representation [1, 192, L]
        captured['feat'] = output[0].detach()

    handle = melo_model.model.enc_p.register_forward_hook(hook_fn)
    speaker_id = 0
    try:
        # Chạy forward MeloTTS để kích hoạt TextEncoder
        _ = melo_model.tts_to_file(text, speaker_id, "/tmp/dummy_distill.wav", quiet=True)
    except Exception as e:
        pass
    finally:
        handle.remove()

    if 'feat' in captured:
        return captured['feat'].to(device)
    else:
        raise RuntimeError(f"Không trích xuất được đặc trưng từ Teacher cho câu: {text}")


def train_distillation(
    train_texts: List[str],
    epochs: int = 10,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "outputs/distill_melo"
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"🚀 Bắt đầu quá trình Chưng cất Tri thức (Knowledge Distillation) trên thiết bị: {device.upper()}")

    # 1. Khởi tạo Teacher Model (MeloTTS-ZH)
    print("⏳ Đang tải Teacher Model (MeloTTS-ZH PyTorch)...")
    from melo.api import TTS
    teacher = TTS(language="ZH", device="cpu")
    teacher.hps.data.disable_bert = True
    teacher.model.eval()
    for param in teacher.model.parameters():
        param.requires_grad = False
    print("✅ Teacher Model sẵn sàng (Đã đóng băng trọng số).")

    # 2. Khởi tạo Student Model (MeloByteEncoder)
    print("⏳ Khởi tạo Student Model (MeloByteEncoder, 25.45M params)...")
    student = MeloByteEncoder(d_model=512, n_layers=8, n_heads=8, d_ff=2048, out_channels=192).to(device)
    total_params = sum(p.numel() for p in student.parameters())
    print(f"✅ Student Model sẵn sàng. Tổng tham số: {total_params:,} ({total_params/1e6:.2f}M)")

    # 3. Optimizer & Loss Function
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    dataset = TextDataset(train_texts)
    print(f"📚 Số lượng câu huấn luyện: {len(dataset)}")

    # 4. Vòng lặp Huấn luyện (Training Loop)
    best_loss = float("inf")
    student.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        start_time = time.time()

        for idx, text in enumerate(dataset):
            # Lấy Teacher target [1, 192, L_target]
            with torch.no_grad():
                target_feat = extract_teacher_features(teacher, text, device=device) # [1, 192, L]
                L_target = target_feat.size(2)

            # Tạo input byte tensor cho Student [1, 512]
            byte_tensor, raw_bytes = text_to_byte_tensor(text, seq_len=512)
            byte_tensor = byte_tensor.to(device)

            # Student forward [1, 192, 512]
            student_feat = student(byte_tensor)

            # Cắt phần dự đoán tương ứng với chiều dài target để so sánh
            # Nếu target dài hơn 512 thì cắt target, nếu ngắn hơn thì so sánh đúng L_target
            min_len = min(L_target, 512)
            loss = criterion(student_feat[:, :, :min_len], target_feat[:, :, :min_len])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataset)
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.6f} - Thời gian: {elapsed:.2f}s")

        # Lưu checkpoint tốt nhất
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(output_dir, "byte_encoder_best.pth")
            torch.save(student.state_dict(), ckpt_path)

    print(f"\n🎉 Huấn luyện hoàn tất! Best Loss: {best_loss:.6f}")

    # 5. Export sang ONNX
    print("📦 Đang xuất mô hình Student sang ONNX...")
    student.eval().cpu()
    dummy_input = torch.zeros((1, 512), dtype=torch.int32)
    onnx_path = os.path.join(output_dir, "melo_byte_encoder.onnx")

    torch.onnx.export(
        student,
        dummy_input,
        onnx_path,
        input_names=["byte_indices"],
        output_names=["acoustic_features"],
        opset_version=17,
        dynamo=False
    )
    print(f"✅ Đã lưu file ONNX tại: {onnx_path}")
    print("👉 Bạn có thể tải file này lên Qualcomm AI Hub để chạy trên NPU IQ-9075 EVK!")
    return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chưng cất Byte Encoder cho MeloTTS")
    parser.add_argument("--text_file", type=str, default="", help="Đường dẫn đến file .txt chứa các câu tiếng Trung")
    parser.add_argument("--epochs", type=int, default=3, help="Số epoch huấn luyện")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    # Tải danh sách câu
    if args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"Đã đọc {len(texts)} câu từ {args.text_file}")
    else:
        print("Sử dụng tập dữ liệu câu mẫu mặc định...")
        texts = DEFAULT_SAMPLE_TEXTS

    train_distillation(texts, epochs=args.epochs, lr=args.lr)
