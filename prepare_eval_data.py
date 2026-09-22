import re
import json
import random
from pathlib import Path

def main():
    src_file = Path("/Users/htti/.gemini/antigravity/brain/cb9942a0-dd7b-47f0-b73e-286970a52b51/.system_generated/steps/103/content.md")
    output_dir = Path("/Users/htti/Documents/Code.nosync/melotts-zh/eval_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(src_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for pattern: 000001\tText
        match = re.match(r"^(\d{6})\t(.*)$", line)
        if match:
            item_id = match.group(1)
            raw_text = match.group(2)
            pinyin = ""
            if i + 1 < len(lines) and lines[i+1].startswith("\t"):
                pinyin = lines[i+1].strip()
                i += 1
            
            # Clean prosody tags #1, #2, #3, #4
            clean_text = re.sub(r"#\d", "", raw_text).strip()
            entries.append({
                "id": item_id,
                "text": clean_text,
                "raw_text": raw_text,
                "pinyin": pinyin
            })
        i += 1

    print(f"Tổng số câu trích xuất được từ Baker CSMSC: {len(entries)}")

    # Chọn ngẫu nhiên 500 câu với seed cố định để tái lập
    random.seed(42)
    sample_500 = random.sample(entries, min(500, len(entries)))

    # Sắp xếp theo ID để dễ quản lý
    sample_500.sort(key=lambda x: x["id"])

    # 1. Lưu dạng JSON đầy đủ thông tin
    json_path = output_dir / "baker_500_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sample_500, f, ensure_ascii=False, indent=2)

    # 2. Lưu dạng ID|TEXT (chuẩn TTS metadata)
    txt_path = output_dir / "baker_500_eval.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for item in sample_500:
            f.write(f"{item['id']}|{item['text']}\n")

    # 3. Lưu chỉ danh sách văn bản (mỗi dòng 1 câu)
    sentences_path = output_dir / "sentences_500.txt"
    with open(sentences_path, "w", encoding="utf-8") as f:
        for item in sample_500:
            f.write(f"{item['text']}\n")

    print(f"Đã lưu thành công 500 câu vào thư mục: {output_dir}")
    print(f"- {json_path.name}: Chứa ID, text sạch, text gốc prosody và pinyin")
    print(f"- {txt_path.name}: Định dạng chuẩn ID|Text")
    print(f"- {sentences_path.name}: Danh sách text thuần túy để đưa vào pipeline")

if __name__ == "__main__":
    main()
