"""
Script kiểm chứng Cosine Similarity của Vocoder (Decoder)
MeloTTS-ZH (PyTorch FP32 vs NPU QNN w8a16)
"""

import os
import torch
import numpy as np
import qai_hub as hub
from pathlib import Path

def cosine_similarity(a, b):
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

def main():
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    print("🚀 Khởi tạo test Cosine Similarity (Dummy Data) cho Decoder...")
    
    # 1. Khởi tạo PyTorch Reference
    print("⏳ Đang load PyTorch FP32 model...")
    from melo.api import TTS
    model = TTS(language="ZH", device="cpu")
    model.hps.data.disable_bert = True
    
    # 2. Tạo dữ liệu thật từ câu tiếng Trung
    print("📝 Trích xuất đặc trưng (z, g) từ câu tiếng Trung thật...")
    text = "你好，世界！这是一条测试语音。" # Xin chào thế giới, đây là giọng nói thử nghiệm
    speaker_ids = model.hps.data.spk2id
    spk_id = speaker_ids[list(speaker_ids.keys())[0]]

    # Biến để hứng dữ liệu
    captured_inputs = {}

    def hook_fn(module, args, kwargs):
        captured_inputs['z'] = args[0].detach().clone()
        if 'g' in kwargs:
            captured_inputs['g'] = kwargs['g'].detach().clone()
        elif len(args) > 1:
            captured_inputs['g'] = args[1].detach().clone()

    # Móc (hook) vào hàm forward của Decoder
    hook_handle = model.model.dec.register_forward_pre_hook(hook_fn, with_kwargs=True)
    
    # Chạy thử 1 lần để hứng z, g
    _ = model.tts_to_file(text, spk_id, "/tmp/dummy.wav")
    hook_handle.remove()

    real_z = captured_inputs['z'] # [1, 192, L]
    real_g = captured_inputs['g'] # [1, 256, 1]

    # QNN NPU yêu cầu sequence length = 64. 
    # Để nghe cả câu, ta chia (chunk) real_z thành nhiều đoạn 64 frames
    L = real_z.size(2)
    target_len = 64
    num_chunks = int(np.ceil(L / target_len))
    print(f"   - Chiều dài thật của z: {L} (Tự động băm thành {num_chunks} đoạn 64-frame)")
    
    pad_size = (num_chunks * target_len) - L
    if pad_size > 0:
        real_z_padded = torch.nn.functional.pad(real_z, (0, pad_size), "constant", 0)
    else:
        real_z_padded = real_z
        
    z_chunks = torch.split(real_z_padded, target_len, dim=2)

    # 3. Lấy Output chuẩn (FP32) trên từng chunk rồi nối lại
    print("🧠 Chạy PyTorch FP32 Inference (Chế độ Chunk-based)...")
    audio_fp32_chunks = []
    with torch.no_grad():
        for chunk in z_chunks:
            audio_fp32_chunks.append(model.model.dec(chunk, g=real_g).cpu().numpy())
    audio_fp32_np = np.concatenate(audio_fp32_chunks, axis=-1)
    
    # 4. Chạy qua NPU (QNN) trên AI Hub
    print("\n☁️ Gửi Batch Job (Nhiều đoạn) lên Qualcomm AI Hub...")
    decoder_bin_path = "melotts_zh-voice_ai-mixed_with_float-qualcomm_qcs9075/decoder.bin"
    if not os.path.exists(decoder_bin_path):
        print(f"❌ Không tìm thấy {decoder_bin_path}")
        return
        
    try:
        # Upload model
        hub_model = hub.upload_model(decoder_bin_path)
        
        # Submit Inference Job
        device = hub.Device("Dragonwing IQ-9075 EVK")
        
        inputs = {
            "g": [real_g.numpy()] * num_chunks,
            "z": [chunk.numpy() for chunk in z_chunks]
        }
        
        print(f"Đang chờ thiết bị NPU thực thi {num_chunks} đoạn (khoảng 1-3 phút)...")
        inference_job = hub.submit_inference_job(
            model=hub_model,
            device=device,
            inputs=inputs,
            name="MeloTTS_Decoder_Verify"
        )
        
        # Tải kết quả về
        output_data = inference_job.download_output_data()
        
        # Trích xuất các đoạn audio từ kết quả và nối lại
        out_keys = list(output_data.keys())
        audio_qnn_chunks = output_data[out_keys[0]] # Lấy list các mảng numpy
        audio_qnn_np = np.concatenate(audio_qnn_chunks, axis=-1)
        
        # Cắt bỏ phần âm thanh bị dư (tiếng bíp) do đoạn zero-padding sinh ra ở cuối
        hop_size = audio_qnn_np.shape[-1] // (num_chunks * target_len)
        actual_audio_len = L * hop_size
        audio_qnn_np = audio_qnn_np[..., :actual_audio_len]
        audio_fp32_np = audio_fp32_np[..., :actual_audio_len]
        
        # 5. So sánh
        sim = cosine_similarity(audio_fp32_np, audio_qnn_np)
        
        print("\n" + "="*50)
        print("🎉 KẾT QUẢ ĐÁNH GIÁ (COSINE SIMILARITY)")
        print("="*50)
        print(f"PyTorch FP32 Shape : {audio_fp32_np.shape}")
        print(f"QNN NPU Shape      : {audio_qnn_np.shape}")
        print(f"Cosine Similarity  : {sim:.5f}")
        print("="*50)
        
        if sim > 0.95:
            print("✅ ĐẠT YÊU CẦU! Mô hình Quantized hoạt động hoàn hảo, âm thanh không bị vỡ.")
        else:
            print("⚠️ CẢNH BÁO: Similarity thấp, có thể xảy ra hiện tượng méo tiếng.")
            
        # 6. Lưu file âm thanh (.wav) để nghe thử
        print("\n🎧 Đang xuất file âm thanh để nghe thử...")
        import scipy.io.wavfile as wavfile
        sr = model.hps.data.sampling_rate
        
        def save_wav(path, audio_np):
            audio_np = audio_np.flatten()
            max_val = np.abs(audio_np).max()
            if max_val > 0:
                audio_np = audio_np / max_val # Chuẩn hoá (Normalize)
            audio_int16 = (audio_np * 32767.0).astype(np.int16)
            wavfile.write(path, sr, audio_int16)
            print(f"   💾 Đã lưu: {path}")

        save_wav("output_1_fp32_macbook.wav", audio_fp32_np)
        save_wav("output_2_qnn_snapdragon.wav", audio_qnn_np)
            
    except Exception as e:
        print(f"❌ Có lỗi trong quá trình kết nối AI Hub: {e}")

if __name__ == "__main__":
    main()
