# 基于mmpretrain的微调指南
### 1. 环境准备
- 使用conda构建一个mmpretrain的环境

```bash  
conda create -n open-mmlab python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y  
conda activate open-mmlab  
pip3 install openmim  
git clone https://github.com/open-mmlab/mmpretrain.git  
cd mmpretrain  
mim install -e .
```

### 2.微调配置
``` 
python tools/train.py configs/vision_transformer/vit-base-p16_32xb128-mae_in1k.py
```

### 3.训练 
- vit是针对于目标分类的视觉模型
- 按照以下配置配置你的数据集
``` 
data_prefix/
├── class_x
│   ├── xxx.png
│   ├── xxy.png
│   └── ...
│       └── xxz.png
└── class_y
    ├── 123.png
    ├── nsdf3.png
    ├── ...
    └── asd932_.png
```
- 注意，如果如果你是使用你自己的数据的话，需要修改dataloader变成
```
train_dataloader = dict(
    ...
    # Training dataset configurations
    dataset=dict(
        type='CustomDataset',
        data_prefix='path/to/data_prefix',
        with_label=True,   # or False for unsupervised tasks
        pipeline=...
    )
)
```
## 最后你会在你的workdirs路径下得到你的权重文件

### 4. 测试
``` 
python tools/test.py configs/vision_transformer/vit-base-p32_64xb64_in1k-384px.py https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth

```
- 把你的路径输入进去即可测试
  
  

