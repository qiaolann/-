医疗问答大模型微调项目
本项目通过 PEFT (Parameter-Efficient Fine-Tuning) 技术，在医疗对话数据集上微调了 Qwen 大语言模型，旨在使其能够对中文医疗问题提供专业、准确的回答。

🚀 主要特性
Qwen 模型: 基于 Qwen 模型，具备强大的语言理解与生成能力。

PEFT 高效微调: 采用参数高效微调，大幅降低训练资源消耗。

医疗领域专业化: 针对中文医疗问答场景进行优化。

支持量化加载: 支持 4-bit / 8-bit 量化，优化显存使用。

🛠️ 安装
克隆仓库:

Bash

git clone [你的仓库URL]
cd [你的项目文件夹]
安装依赖:
请确保已安装 Python 3.8+。核心依赖包括 datasets, transformers, peft, accelerate, torch 等。建议创建 requirements.txt 文件并安装：

requirements.txt 示例:

datasets>=3.6.0
transformers
peft
accelerate
torch
# 其他你在项目中使用的库，例如 fsspec, pandas, numpy 等
Bash

pip install -r requirements.txt
🚀 使用方法
数据格式
项目使用了类似以下格式的医疗对话数据：

JSON

{'conversations': [{'content': '感冒有什么症状？', 'role': 'user'}, {'content': '感冒的主要症状包括流鼻涕、咳嗽、喉咙痛和发热。', 'role': 'assistant'}]}
请确保你的数据集遵循此格式。

模型训练
项目通过微调 Qwen 模型来实现医疗问答能力。训练过程包括加载数据集，使用 PEFT 进行训练，并将微调后的模型权重和 tokenizer 保存到 ./qwen-medical-peft/ 目录。

训练日志概览:

读取数据条数：134

数据预处理 (Map: 100%)

训练 3 个 Epoch，总计 90 步

训练损失持续降低

模型推理
模型训练完成后，你可以加载 qwen-medical-peft 目录下的模型进行推理：

Python

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 加载分词器和基座模型（请替换为实际使用的Qwen基座模型名称）
base_model_name = "Qwen/Qwen-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained("./qwen-medical-peft/", trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16, # 根据你的训练配置调整
    device_map="auto",
    trust_remote_code=True
)

# 加载 PEFT 微调后的适配器权重
model = PeftModel.from_pretrained(base_model, "./qwen-medical-peft/")
model = model.eval() # 切换到评估模式

# 提问示例
user_query = "胃疼吃什么药？"
messages = [{"role": "user", "content": user_query}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
response = tokenizer.batch_decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)[0]

print(f"### 用户:\n{user_query}\n\n### 医生:\n{response}")
推理示例输出:

### 用户:
胃疼吃什么药？

### 医生:
清淡饮食、止痛药等。
