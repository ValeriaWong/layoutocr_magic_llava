import shutil
from pathlib import Path
CLASSES = {
    0: "letter",
    1: "form",
    2: "email",
    3: "handwritten",
    4: "advertisement",
    5: "scientific report",
    6: "scientific publication",
    7: "specification",
    8: "file folder",
    9: "news article",
    10: "budget",
    11: "invoice",
    12: "presentation",
    13: "questionnaire",
    14: "resume",
    15: "memo",
}

result_path = Path("D:\\LLM\\来自：快传\\layoutocr_magic_llava_data\\LaytoutLM\\RVL-CDIP\\data\\rvl-cdip\\images")

train_file_path = Path("D:\\LLM\\来自：快传\\layoutocr_magic_llava_data\\LaytoutLM\\RVL-CDIP\\data\\train.txt")
with open(train_file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    # 分割路径和标签
    parts = line.strip().split()
    if len(parts) == 2:  # 确保行格式正确
        relative_img_path, img_label_str = parts
        img_label = int(img_label_str)  # 将标签转换为整数

        base_dir = Path("D:\\LLM\\来自：快传\\layoutocr_magic_llava_data\\LaytoutLM\\RVL-CDIP\\data\\rvl-cdip\\images")
        img_path = base_dir / relative_img_path

        if img_path.exists():
            # 根据标签创建目标目录
            target_dir = result_path / CLASSES[img_label]
            target_dir.mkdir(exist_ok=True)
            
            target_path = target_dir / img_path.name
          
            shutil.copy(img_path, target_path)
        else:
            print(f"Source file does not exist: {img_path}")
            
