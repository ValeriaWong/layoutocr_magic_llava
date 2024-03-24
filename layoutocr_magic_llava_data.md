# layoutocr_magic_llava_data

## 一、 [LaytoutLM](https://huggingface.co/docs/transformers/model_doc/layoutlm)

1. 表单理解：[FUNSD](https://guillaumejaume.github.io/FUNSD/)数据集（199 个带注释的表单的集合，包含超过 30,000 个单词）

2. 收据理解：[SROIE](https://rrc.cvc.uab.es/?ch=13)数据集（包含 626 个用于训练的收据和 347 个用于测试的收据的集合）

3. 文档图像分类：[RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/)数据集（属于 16 个类别之一的 400,000 张图像的集合）

## 二、 LayoutxLM

[XFUND](https://github.com/doc-analysis/XFUND/releases/tag/v1.0) 是一个多语言表单理解基准数据集，其中包含 7 种语言（中文、日语、西班牙语、法语、意大利语、德语、葡萄牙语）的带有键值对的人工标记表单。

#### [Statistics](https://github.com/doc-analysis/XFUND)

| lang | split    | header | question | answer | other | total  |
| ---- | -------- | ------ | -------- | ------ | ----- | ------ |
| ZH   | training | 441    | 3,266    | 2,808  | 896   | 7,411  |
|      | testing  | 122    | 1,077    | 821    | 312   | 2,332  |
| JA   | training | 229    | 3,692    | 4,641  | 1,666 | 10,228 |
|      | testing  | 58     | 1,253    | 1,732  | 586   | 3,629  |
| ES   | training | 253    | 3,013    | 4,254  | 3,929 | 11,449 |
|      | testing  | 90     | 909      | 1,218  | 1,196 | 3,413  |
| FR   | training | 183    | 2,497    | 3,427  | 2,709 | 8,816  |
|      | testing  | 66     | 1,023    | 1,281  | 1,131 | 3,501  |
| IT   | training | 166    | 3,762    | 4,932  | 3,355 | 12,215 |
|      | testing  | 65     | 1,230    | 1,599  | 1,135 | 4,029  |
| DE   | training | 155    | 2,609    | 3,992  | 1,876 | 8,632  |
|      | testing  | 59     | 858      | 1,322  | 650   | 2,889  |
| PT   | training | 185    | 3,510    | 5,428  | 2,531 | 11,654 |
|      | testing  | 59     | 1,288    | 1,940  | 882   | 4,169  |

## 三、 CogVLM

[CogVLM-SFT-311K](https://github.com/THUDM/CogVLM/blob/main/dataset_zh.md#cogvlm-sft-311kcogvlm-sft-%E4%B8%AD%E7%9A%84%E5%8F%8C%E8%AF%AD%E8%A7%86%E8%A7%89%E6%8C%87%E4%BB%A4%E6%95%B0%E6%8D%AE%E9%9B%86)：CogVLM SFT中的双语视觉指令数据集

[THUDM/CogVLM-SFT-311K](THUDM/CogVLM-SFT-311K)中含有数据集的信息、数量和格式



CogVLM-SFT-311K 是 CogVLM v1.0 初始训练中使用的主要对齐语料库。该数据集的构建过程如下：

1. [从开源MiniGPT-4](https://huggingface.co/datasets/Vision-CAIR/cc_sbu_align)中选取了大约 3500 个高质量数据样本，称为 minigpt4-3500。
2. [Minigpt4-3500与Llava-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)集成，并通过语言模型翻译成中文。
3. 我们在 minigpt4-3500 和 Llava-instruct 的详细描述部分发现了明显的噪音。因此，我们对这些中文语料库进行了修正，并将其重新翻译成英文。

