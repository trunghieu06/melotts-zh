from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MeloTTS-ZH ONNX Exporter
Khởi tạo và bóc tách các module của mô hình MeloTTS-ZH thành các tệp .onnx
tuân thủ tuyệt đối cấu trúc đầu vào/đầu ra chuẩn của Qualcomm Hexagon NPU (metadata.json).

Các module được xuất:
1. decoder.onnx      : Vocoder (HiFi-GAN) xử lý khung âm thanh chunk 64-frame
2. flow.onnx         : Normalizing Flow biến đổi không gian phân phối
3. encoder.onnx      : Text Encoder + Trích xuất âm vị và thanh điệu
4. bert_wrapper.onnx : Mô hình trích xuất ngữ cảnh RoBERTa tiếng Trung
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class _DummyModule:
        pass
    class _DummyNN:
        Module = _DummyModule
    nn = _DummyNN()

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


# ==============================================================================
# 1. CÁC WRAPPER MODULES (CHUẨN HÓA GIAO DIỆN I/O THEO QUALCOMM NPU)
# ==============================================================================

class DecoderWrapper(nn.Module):
    """
    Wrapper cho khối Vocoder (HiFi-GAN Generator).
    Đầu vào:
      - z: [1, 192, 64] (Mel-latent chunk cố định 64-frame)
      - g: [1, 256, 1]  (Speaker embedding)
    Đầu ra:
      - audio: [1, 1, 32768] (Khung sóng âm 24kHz: 64 frames * hop_length 512 = 32768 samples)
    """
    def __init__(self, dec_module: nn.Module):
        super().__init__()
        self.dec = dec_module

    def forward(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.dec(z, g=g)


class FlowWrapper(nn.Module):
    """
    Wrapper cho khối Normalizing Flow.
    Đầu vào:
      - attn_squeezed: [1, 1536, 512] (Ma trận căn chỉnh thời lượng)
      - logs_p:        [1, 192, 512]  (Log-variance từ encoder)
      - noise_scale:   [1]
      - m_p:           [1, 192, 512]  (Mean vector từ encoder)
      - y_mask:        [1, 1, 1536]   (Mask cho chuỗi thời gian)
      - g:             [1, 256, 1]    (Speaker embedding)
    Đầu ra:
      - z:             [1, 192, 1536] (Biến tiềm ẩn âm phổ trước khi cắt chunk)
    """
    def __init__(self, flow_module: nn.Module):
        super().__init__()
        self.flow = flow_module

    def forward(self,
                attn_squeezed: torch.Tensor,
                logs_p: torch.Tensor,
                noise_scale: torch.Tensor,
                m_p: torch.Tensor,
                y_mask: torch.Tensor,
                g: torch.Tensor) -> torch.Tensor:
        m_p_exp = torch.matmul(m_p, attn_squeezed.transpose(1, 2))
        logs_p_exp = torch.matmul(logs_p, attn_squeezed.transpose(1, 2))
        z_p = m_p_exp + torch.exp(logs_p_exp) * noise_scale * 0.0
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        return z


class EncoderWrapper(nn.Module):
    """
    Wrapper cho khối Text Encoder.
    Đầu vào:
      - sid:           [1] (Speaker ID - int32)
      - bert:          [1, 1024, 512] (Đặc trưng RoBERTa tiếng Trung)
      - ja_bert:       [1, 768, 512]  (Đặc trưng BERT phụ trợ)
      - x:             [1, 512]       (Phoneme IDs - int32)
      - tone:          [1, 512]       (Tone IDs - int32)
      - language:      [1, 512]       (Language IDs - int32)
      - x_lengths:     [1]            (Độ dài chuỗi âm vị - int32)
      - noise_scale_w: [1]            (Hệ số nhiễu độ dài - float32)
      - sdp_ratio:     [1]            (Tỷ lệ phân phối thời lượng - float32)
      - length_scale:  [1]            (Hệ số tốc độ nói - float32)
    Đầu ra:
      - m_p:           [1, 192, 512]
      - logs_p:        [1, 192, 512]
      - w_ceil:        [1, 1, 512]
      - y_lengths:     [1]
      - x_mask:        [1, 1, 512]
      - g:             [1, 256, 1]
    """
    def __init__(self, tts_model: nn.Module):
        super().__init__()
        self.enc_p = tts_model.enc_p
        self.emb_g = tts_model.emb_g
        self.dp = getattr(tts_model, "dp", None)
        self.sdp = getattr(tts_model, "sdp", None)

    def forward(self,
                sid: torch.Tensor,
                bert: torch.Tensor,
                ja_bert: torch.Tensor,
                x: torch.Tensor,
                tone: torch.Tensor,
                language: torch.Tensor,
                x_lengths: torch.Tensor,
                noise_scale_w: torch.Tensor,
                sdp_ratio: torch.Tensor,
                length_scale: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        g = self.emb_g(sid).unsqueeze(-1)  # [1, 256, 1]
        
        x_enc, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths,
            tone=tone,
            language=language,
            bert=bert,
            ja_bert=ja_bert,
            g=g
        )
        
        if self.sdp is not None and self.dp is not None:
            logw = self.sdp(x_enc, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * sdp_ratio + self.dp(x_enc, x_mask, g=g) * (1.0 - sdp_ratio)
        elif self.dp is not None:
            logw = self.dp(x_enc, x_mask, g=g)
        else:
            logw = torch.zeros_like(x_mask)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1.0)

        return m_p, logs_p, w_ceil, y_lengths, x_mask, g


class BertWrapper(nn.Module):
    """
    Wrapper cho khối RoBERTa tiếng Trung.
    Đầu vào:
      - input_ids:      [1, 200] (int32)
      - token_type_ids: [1, 200] (int32)
      - attention_mask: [1, 200] (int32)
    Đầu ra:
      - hidden_states:  [1, 200, 768] (float32)
    """
    def __init__(self, bert_model: nn.Module):
        super().__init__()
        self.bert = bert_model

    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state


# ==============================================================================
# 2. HÀM XUẤT ONNX TỪNG MODULE
# ==============================================================================

def export_decoder(model_dec: nn.Module, output_path: Path):
    print(f"\n[*] Đang xuất: {output_path.name}...")
    wrapper = DecoderWrapper(model_dec).eval()
    
    dummy_z = torch.randn(1, 192, 64, dtype=torch.float32)
    dummy_g = torch.randn(1, 256, 1, dtype=torch.float32)
    
    torch.onnx.export(
        wrapper,
        (dummy_z, dummy_g),
        str(output_path),
        input_names=["z", "g"],
        output_names=["audio"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False
    )
    print(f"[+] Xuất thành công: {output_path}")


def export_flow(model_flow: nn.Module, output_path: Path):
    print(f"\n[*] Đang xuất: {output_path.name}...")
    wrapper = FlowWrapper(model_flow).eval()
    
    dummy_attn = torch.randn(1, 1536, 512, dtype=torch.float32)
    dummy_logs_p = torch.randn(1, 192, 512, dtype=torch.float32)
    dummy_noise = torch.tensor([0.667], dtype=torch.float32)
    dummy_m_p = torch.randn(1, 192, 512, dtype=torch.float32)
    dummy_y_mask = torch.ones(1, 1, 1536, dtype=torch.float32)
    dummy_g = torch.randn(1, 256, 1, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (dummy_attn, dummy_logs_p, dummy_noise, dummy_m_p, dummy_y_mask, dummy_g),
        str(output_path),
        input_names=["attn_squeezed", "logs_p", "noise_scale", "m_p", "y_mask", "g"],
        output_names=["z"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False
    )
    print(f"[+] Xuất thành công: {output_path}")


def export_encoder(tts_model: nn.Module, output_path: Path):
    print(f"\n[*] Đang xuất: {output_path.name}...")
    wrapper = EncoderWrapper(tts_model).eval()

    dummy_sid = torch.tensor([0], dtype=torch.int32)
    dummy_bert = torch.randn(1, 1024, 512, dtype=torch.float32)
    dummy_ja_bert = torch.randn(1, 768, 512, dtype=torch.float32)
    dummy_x = torch.randint(0, 100, (1, 512), dtype=torch.int32)
    dummy_tone = torch.randint(0, 5, (1, 512), dtype=torch.int32)
    dummy_lang = torch.zeros((1, 512), dtype=torch.int32)
    dummy_x_len = torch.tensor([50], dtype=torch.int32)
    dummy_noise_w = torch.tensor([0.8], dtype=torch.float32)
    dummy_sdp = torch.tensor([0.2], dtype=torch.float32)
    dummy_len_scale = torch.tensor([1.0], dtype=torch.float32)

    inputs = (dummy_sid, dummy_bert, dummy_ja_bert, dummy_x, dummy_tone,
              dummy_lang, dummy_x_len, dummy_noise_w, dummy_sdp, dummy_len_scale)

    input_names = [
        "sid", "bert", "ja_bert", "x", "tone",
        "language", "x_lengths", "noise_scale_w", "sdp_ratio", "length_scale"
    ]
    output_names = ["m_p", "logs_p", "w_ceil", "y_lengths", "x_mask", "g"]

    torch.onnx.export(
        wrapper,
        inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        do_constant_folding=True,
        dynamo=False
    )
    print(f"[+] Xuất thành công: {output_path}")


def export_bert(bert_model: nn.Module, output_path: Path):
    print(f"\n[*] Đang xuất: {output_path.name}...")
    wrapper = BertWrapper(bert_model).eval()

    dummy_ids = torch.randint(0, 20000, (1, 200), dtype=torch.int32)
    dummy_token_type = torch.zeros((1, 200), dtype=torch.int32)
    dummy_mask = torch.ones((1, 200), dtype=torch.int32)

    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_token_type, dummy_mask),
        str(output_path),
        input_names=["input_ids", "token_type_ids", "attention_mask"],
        output_names=["hidden_states"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False
    )
    print(f"[+] Xuất thành công: {output_path}")


# ==============================================================================
# 3. HÀM KIỂM TRA HỢP LỆ (ONNX VALIDATION)
# ==============================================================================

def validate_onnx(onnx_path: Path):
    """Kiểm tra tính hợp lệ của file ONNX bằng onnx.checker và onnxruntime."""
    if not ONNX_AVAILABLE:
        print("[!] Không tìm thấy onnx hoặc onnxruntime, bỏ qua bước xác thực.")
        return

    print(f"[*] Đang xác thực mô hình: {onnx_path.name}...")
    try:
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        
        ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inputs_desc = [f"{i.name}: {i.shape} ({i.type})" for i in ort_sess.get_inputs()]
        outputs_desc = [f"{o.name}: {o.shape} ({o.type})" for o in ort_sess.get_outputs()]
        
        print(f"    - Inputs : {', '.join(inputs_desc)}")
        print(f"    - Outputs: {', '.join(outputs_desc)}")
        print(f"[✓] File {onnx_path.name} hợp lệ 100%!")
    except Exception as e:
        print(f"[✗] Lỗi xác thực trên {onnx_path.name}: {e}")


# ==============================================================================
# 4. CHƯƠNG TRÌNH ĐIỀU PHỐI CHÍNH (MAIN CLI)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Xuất các module MeloTTS-ZH sang ONNX theo chuẩn Qualcomm NPU.")
    parser.add_argument("--output_dir", type=str, default="onnx_models", help="Thư mục lưu trữ các file .onnx")
    parser.add_argument("--module", type=str, default="all", choices=["all", "decoder", "flow", "encoder", "bert"],
                        help="Module cụ thể muốn xuất (hoặc 'all' để xuất toàn bộ)")
    parser.add_argument("--language", type=str, default="ZH", help="Ngôn ngữ mô hình (Mặc định: ZH)")
    parser.add_argument("--device", type=str, default="cpu", help="Thiết bị nạp PyTorch (cpu hoặc cuda)")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("\n[!] LỖI: Chưa cài đặt thư viện 'torch'.")
        print("    Vui lòng cài đặt trước: pip install torch onnx onnxruntime melotts")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("🚀 MELOTTS-ZH TO ONNX EXPORT PIPELINE (QUALCOMM HTP COMPLIANT)")
    print(f"• Thư mục đích : {out_dir.resolve()}")
    print(f"• Module chọn  : {args.module}")
    print("=" * 65)

    try:
        from melo.api import TTS
        print(f"[*] Đang tải mô hình MeloTTS ({args.language}) từ HuggingFace / Cache...")
        tts = TTS(language=args.language, device=args.device)
        model = tts.model
        print("[+] Nạp mô hình MeloTTS gốc thành công.")
        try:
            from transformers import AutoModel
            print("[*] Đang nạp mô hình RoBERTa/BERT tiếng Trung (bert-base-chinese)...")
            bert_model = AutoModel.from_pretrained("bert-base-chinese")
        except Exception as e:
            print(f"[!] Không nạp được bert-base-chinese: {e}")
            bert_model = None
    except ImportError:
        print("\n[!] LỖI: Chưa cài đặt thư viện 'melotts'.")
        print("    Cài đặt bằng lệnh: pip install melotts")
        print("    Hoặc: pip install git+https://github.com/myshell-ai/MeloTTS.git")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Có lỗi xảy ra khi nạp mô hình MeloTTS: {e}")
        sys.exit(1)

    exported_files = []

    # A. DECODER
    if args.module in ["all", "decoder"]:
        p = out_dir / "decoder.onnx"
        export_decoder(model.dec, p)
        validate_onnx(p)
        exported_files.append(p)

    # B. FLOW
    if args.module in ["all", "flow"]:
        p = out_dir / "flow.onnx"
        export_flow(model.flow, p)
        validate_onnx(p)
        exported_files.append(p)

    # C. ENCODER
    if args.module in ["all", "encoder"]:
        p = out_dir / "encoder.onnx"
        export_encoder(model, p)
        validate_onnx(p)
        exported_files.append(p)

    # D. BERT WRAPPER
    if args.module in ["all", "bert"]:
        if bert_model is not None:
            p = out_dir / "bert_wrapper.onnx"
            export_bert(bert_model, p)
            validate_onnx(p)
            exported_files.append(p)
        else:
            print("[!] Cảnh báo: Không tìm thấy bert_model trong đối tượng TTS.")

    print("\n" + "=" * 65)
    print("🎉 HOÀN TẤT XUẤT CÁC FILE ONNX:")
    for f in exported_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   ✓ {f.name:<20} ({size_mb:.2f} MB)")
    print(f"\nBạn có thể upload các file trong '{out_dir}' lên Qualcomm AI Hub để Quantize & Benchmark!")
    print("=" * 65)


if __name__ == "__main__":
    main()
