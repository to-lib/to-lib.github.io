---
sidebar_position: 32
title: 🧬 模型蒸馏
---

# 模型蒸馏

模型蒸馏（Knowledge Distillation）是将大模型（教师模型）的知识迁移到小模型（学生模型）的技术，让小模型获得接近大模型的能力。

## 为什么需要蒸馏？

| 对比 | 大模型 | 蒸馏后小模型 |
|------|--------|-------------|
| 参数量 | 70B+ | 7B 或更小 |
| 推理成本 | 高 | 低 |
| 部署难度 | 需要多卡 | 单卡/CPU |
| 响应速度 | 慢 | 快 |
| 能力 | 通用强 | 特定任务强 |

## 蒸馏方法

```
┌─────────────────────────────────────────────────────────┐
│                    蒸馏方法                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 输出蒸馏：学习教师模型的输出分布                    │
│  2. 特征蒸馏：学习中间层表示                            │
│  3. 数据蒸馏：用教师模型生成训练数据                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 数据蒸馏（最常用）

使用大模型生成高质量训练数据，然后微调小模型。

### 生成训练数据

```python
from openai import OpenAI
import json

client = OpenAI()

def generate_training_data(task_description: str, num_samples: int = 100) -> list:
    """使用 GPT-4 生成训练数据"""
    samples = []
    
    for i in range(num_samples):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""你是一个数据生成专家。请为以下任务生成一个训练样本：
任务：{task_description}

生成格式：
{{"input": "输入文本", "output": "期望输出"}}

要求：
1. 输入要多样化
2. 输出要准确、高质量
3. 只返回 JSON，不要其他内容"""
                },
                {"role": "user", "content": f"生成第 {i+1} 个样本"}
            ],
            response_format={"type": "json_object"}
        )
        
        sample = json.loads(response.choices[0].message.content)
        samples.append(sample)
        
        if (i + 1) % 10 == 0:
            print(f"已生成 {i + 1}/{num_samples} 个样本")
    
    return samples

# 生成数据
samples = generate_training_data(
    task_description="中文文本情感分类（positive/negative/neutral）",
    num_samples=500
)

# 保存
with open("training_data.jsonl", "w") as f:
    for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
```

### 微调学生模型

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# 加载学生模型
model_name = "Qwen/Qwen2.5-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载蒸馏数据
dataset = load_dataset("json", data_files="training_data.jsonl")

def format_sample(sample):
    text = f"输入：{sample['input']}\n输出：{sample['output']}"
    return tokenizer(text, truncation=True, max_length=512)

dataset = dataset.map(format_sample)

# LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

model = get_peft_model(model, lora_config)

# 训练
training_args = TrainingArguments(
    output_dir="./distilled_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

trainer.train()
```

## 输出蒸馏

让学生模型学习教师模型的输出概率分布。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """蒸馏损失函数"""
    
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        # 软标签损失（KL 散度）
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)
        
        # 硬标签损失（交叉熵）
        hard_loss = self.ce_loss(student_logits, labels)
        
        # 组合损失
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

class DistillationTrainer:
    """蒸馏训练器"""
    
    def __init__(
        self,
        teacher_model,
        student_model,
        tokenizer,
        temperature: float = 2.0
    ):
        self.teacher = teacher_model.eval()
        self.student = student_model
        self.tokenizer = tokenizer
        self.loss_fn = DistillationLoss(temperature=temperature)
    
    def train_step(self, batch):
        # 教师模型推理（不计算梯度）
        with torch.no_grad():
            teacher_outputs = self.teacher(**batch)
            teacher_logits = teacher_outputs.logits
        
        # 学生模型推理
        student_outputs = self.student(**batch)
        student_logits = student_outputs.logits
        
        # 计算蒸馏损失
        loss = self.loss_fn(
            student_logits.view(-1, student_logits.size(-1)),
            teacher_logits.view(-1, teacher_logits.size(-1)),
            batch["labels"].view(-1)
        )
        
        return loss
```


## OpenAI 蒸馏 API

OpenAI 提供了官方的蒸馏功能。

```python
from openai import OpenAI

client = OpenAI()

# 1. 使用大模型生成带 metadata 的响应
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "解释量子计算"}],
    store=True,  # 存储用于蒸馏
    metadata={"task": "explanation", "domain": "physics"}
)

# 2. 创建蒸馏微调任务
# 使用存储的高质量响应微调小模型
fine_tune = client.fine_tuning.jobs.create(
    training_file="file-xxx",  # 包含蒸馏数据
    model="gpt-4o-mini",       # 学生模型
    method={
        "type": "supervised",
        "supervised": {
            "hyperparameters": {"n_epochs": 3}
        }
    }
)
```

## 实战：蒸馏代码助手

```python
class CodeAssistantDistiller:
    """代码助手蒸馏"""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_code_samples(self, num_samples: int = 200) -> list:
        """生成代码训练样本"""
        tasks = [
            "写一个 Python 函数实现快速排序",
            "实现一个 LRU 缓存",
            "写一个异步 HTTP 客户端",
            # ... 更多任务
        ]
        
        samples = []
        for task in tasks:
            # 使用 GPT-4 生成高质量代码
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的 Python 开发者。生成简洁、高效、有注释的代码。"
                    },
                    {"role": "user", "content": task}
                ]
            )
            
            samples.append({
                "instruction": task,
                "output": response.choices[0].message.content
            })
        
        return samples
    
    def format_for_training(self, samples: list) -> list:
        """格式化为训练数据"""
        formatted = []
        for sample in samples:
            formatted.append({
                "messages": [
                    {"role": "system", "content": "你是一个代码助手。"},
                    {"role": "user", "content": sample["instruction"]},
                    {"role": "assistant", "content": sample["output"]}
                ]
            })
        return formatted

# 使用
distiller = CodeAssistantDistiller()
samples = distiller.generate_code_samples(200)
training_data = distiller.format_for_training(samples)
```

## 蒸馏效果评估

```python
def evaluate_distillation(
    teacher_model,
    student_model,
    test_data: list,
    tokenizer
) -> dict:
    """评估蒸馏效果"""
    
    teacher_scores = []
    student_scores = []
    
    for sample in test_data:
        prompt = sample["input"]
        reference = sample["output"]
        
        # 教师模型输出
        teacher_output = generate(teacher_model, tokenizer, prompt)
        
        # 学生模型输出
        student_output = generate(student_model, tokenizer, prompt)
        
        # 评估（可以用 GPT-4 评分）
        teacher_score = evaluate_quality(teacher_output, reference)
        student_score = evaluate_quality(student_output, reference)
        
        teacher_scores.append(teacher_score)
        student_scores.append(student_score)
    
    return {
        "teacher_avg": sum(teacher_scores) / len(teacher_scores),
        "student_avg": sum(student_scores) / len(student_scores),
        "retention_rate": sum(student_scores) / sum(teacher_scores)
    }
```

## 最佳实践

1. **数据质量优先**：蒸馏数据的质量决定学生模型上限
2. **任务聚焦**：针对特定任务蒸馏效果更好
3. **多样性**：训练数据要覆盖各种场景
4. **迭代优化**：多轮蒸馏逐步提升
5. **评估验证**：确保学生模型达到预期效果

## 延伸阅读

- [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301)
- [OpenAI Model Distillation](https://platform.openai.com/docs/guides/distillation)
- [LLM Distillation Survey](https://arxiv.org/abs/2402.13116)