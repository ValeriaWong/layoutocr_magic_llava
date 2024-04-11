# 关于如何把layoutlm数据集进行按类排放

# 类别标签映射字典
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

## 数据集存放位置的路径
result_path = Path("D:\\LLM\\来自：快传\\layoutocr_magic_llava_data\\LaytoutLM\\RVL-CDIP\\data\\rvl-cdip\\images")

## 包含标签的文件路径
train_file_path = Path("D:\\LLM\\来自：快传\\layoutocr_magic_llava_data\\LaytoutLM\\RVL-CDIP\\data\\train.txt")

## 读取包含标签的文件并处理每一行
with open(train_file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
## 分割路径和标签
    parts = line.strip().split()
    if len(parts) == 2:  # 确保行格式正确
        relative_img_path, img_label_str = parts
        img_label = int(img_label_str)  # 将标签转换为整数
        
## 构建图片的绝对路径
## 注意：这里我们假设 train.txt 中的路径是基于某个基础目录的相对路径
## 您需要根据实际情况调整基础目录的路径
        base_dir = Path("D:\\LLM\\来自：快传\\layoutocr_magic_llava_data\\LaytoutLM\\RVL-CDIP\\data\\rvl-cdip\\images")
        img_path = base_dir / relative_img_path
        
        # 检查图片文件是否存在
        if img_path.exists():
            # 根据标签创建目标目录
            target_dir = result_path / CLASSES[img_label]
            target_dir.mkdir(exist_ok=True)
            
            # 构建目标文件路径
            target_path = target_dir / img_path.name
            
            # 复制文件到目标路径
            shutil.copy(img_path, target_path)
        else:
            print(f"Source file does not exist: {img_path}")
            
### 分好类的数据
###链接：https://pan.baidu.com/s/1suShpOTr9xy2t7K3XG_LIg?pwd=1234 
提取码：1234 

