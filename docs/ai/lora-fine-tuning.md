---
sidebar_position: 10
title: 🚀 LoRA Fine-tuning（实战）
---

# LoRA Fine-tuning（实战）

LoRA (Low-Rank Adaptation) 是一种高效的微调方法，它通过冻结预训练模型权重，仅在每一层注入可训练的低秩矩阵，从而在显著减少可训练参数数量的同时，达到与全量微调相当的效果。

## 为什么选择 LoRA？

- **效率高**：训练参数量通常仅为原模型的 1% - 10%。
- **硬件门槛低**：显存占用大幅降低，单卡 3090/4090 即可微调 7B/13B 模型。
- **无延迟**：推理时可以将 Adapter 权重合并回基座模型，不增加推理延时。
- **灵活切换**：针对不同任务训练即使不同的 Adapter，切换时只需热插拔 Adapter 权重。

## 环境准备

我们需要安装 HuggingFace 生态的核心库：

```bash
pip install transformers peft bitsandbytes datasets accelerate
```

- `transformers`: 加载模型与 Tokenizer
- `peft`: LoRA 等微调库 (Parameter-Efficient Fine-Tuning)
- `bitsandbytes`: 4-bit/8-bit 量化支持
- `datasets`: 数据加载
- `accelerate`: 分布式训练与硬件加速

## 实战步骤

### 1. 加载基座模型（4-bit 量化）

为了节省显存，我们通常使用 QLoRA（4-bit 量化 + LoRA）。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "meta-llama/Llama-2-7b-chat-hf"  # 或其他模型

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # Llama 系列通常需要设置 pad_token
```

### 2. 配置 LoRA

使用 `peft` 库定义 LoRA 配置。

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,                    # 低秩矩阵的秩，越大参数越多但并非越好
    lora_alpha=32,           # 缩放系数，通常是 r 的 2 倍
    target_modules=["q_proj", "v_proj"], # 指定需要微调的层（通常是 attention 相关的层）
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 应用 LoRA 配置到模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出示例: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062
```

### 3. 准备数据

假设我们有一个 JSONL 文件 `train.jsonl`，格式如下：

```json
{ "text": "Human: 怎么做西红柿炒蛋？\nAssistant: 首先准备西红柿和鸡蛋..." }
```

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="train.jsonl", split="train")

def format_prompt(sample):
    return {"text": sample["text"]} # 确保字段名符合模型输入要求

dataset = dataset.map(format_prompt)
```

### 4. 开始训练

使用 `SFTTrainer` (来自 `trl` 库) 或标准的 `Trainer`。这里演示标准 `Trainer`。

```python
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling

# 数据收集器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 训练参数
training_args = TrainingArguments(
    output_dir="./lora_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=500,               # 快速演示，实际训练可以用 num_train_epochs
    fp16=True,                   # 开启混合精度
    optim="paged_adamw_32bit",   # 节省显存的优化器
    save_strategy="steps",
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()
```

### 5. 保存与合并

训练完成后，保存 Adapter 权重。

```python
trainer.save_model("my_lora_adapter")
```

#### 推理时加载

```python
from peft import PeftModel

# 1. 重新加载基座模型
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# 2. 加载 Adapter
model = PeftModel.from_pretrained(base_model, "my_lora_adapter")

# 3. 推理
inputs = tokenizer("Human: 你好\nAssistant:", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

#### 合并权重 (可选)

如果你想导出一个完整的模型文件用于部署（不再依赖 peft）：

```python
# 注意：合并时不能使用 4-bit/8-bit 量化加载基座模型，必须用 fp16/fp32
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "my_lora_adapter")

# 合并并卸载
model = model.merge_and_unload()

# 保存完整模型
model.save_pretrained("merged_model")
tokenizer.save_pretrained("merged_model")
```

## 常见问题

1.  **OOM (显存不足)**:

    - 减小 `batch_size`。
    - 增加 `gradient_accumulation_steps` 用于弥补 batch size 的减小。
    - 确保开启了 4-bit 量化和 `paged_adamw_32bit`。
    - 启用 `gradient_checkpointing` (在 TrainingArguments 中设置)。

2.  **Loss 不下降**:

    - 检查数据质量和格式。
    - 尝试调整 `learning_rate` (LoRA 通常比全量微调大，如 2e-4)。
    - 检查 `target_modules` 是否覆盖了关键层。

3.  **灾难性遗忘**:
    - LoRA 相对不容易发生灾难性遗忘，但如果发现基座通用能力下降严重，可以减小 `r` 或减少训练步数。
