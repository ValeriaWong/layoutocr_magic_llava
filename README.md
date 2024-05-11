# layoutocr_magic_llava

一个可以发论文的idea，欢迎原来做nlp或者cv的佬进行尝试，该过程能够学习实践大模型的pt及sft（助手君说a100单卡算力给够，分布式训练尝试中）
如有兴趣欢迎戳[飞书](https://applink.feishu.cn/client/message/link/open?token=AmRoW1l1AMABZfhqPKJAgAM%3D)

从0到1复现llava架构（可能类似llavar，不过会针对视觉模型语言模型进行替换提升layout ocr能力）有可能的任务包括：
- 修改/微调/替换视觉模型以提升目标检测能力，
- 替换llm为internlm2等
进行两阶段的训练：
- 第一阶段，冻结视觉模型和文本模型，使用图像-文本对训练adapter; 
     - 线性/非线性/自定义网络
- 第二阶段，仅冻结视觉模型，使用图像-文本对端到端微调下游任务
     -  下游任务可增加其他类型

近期任务
- [ ]  分布式训练尝试（单卡+单卡，单卡+受限卡）
- [ ]  阅读相关论文
- [ ]  xtuner跑一下已有的internlm_llava_pt
- [ ]  llava更换vicuna为internlm2(或其他模型）进行两阶段训练（Obj:跑通流程，熟悉数据集格式）
- [ ]  使用layoutlmv3、cogvlm等数据集训练vit
- [ ]  垂直领域微调LLM
- [ ]  数据增强和合成
- [ ]  针对已魔改的视觉模型及文本模型进行两阶段训练
- [ ]  RAG
- [ ]  AGENT


后期可能还会实现
- [ ]  尝试更改adapter的架构，增强连接表示学习
- [ ]  借鉴AE的思想进行数据增强与合成
- [ ]  RLHF
