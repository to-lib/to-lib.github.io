---
sidebar_position: 37
title: 🎯 RLHF 与 DPO
---

# RLHF 与 DPO

RLHF（Reinforcement Learning from Human Feedback）和 DPO（Direct Preference Optimization）是让 LLM 与人类偏好对齐的核心技术。

## 为什么需要对齐？

```
预训练模型：预测下一个 token（可能生成有害/无用内容）
     │
     ▼
对齐后模型：生成有帮助、诚实、无害的内容
```

## RLHF 流程

```
┌─────────────────────────────────────────────────────────┐
│                    RLHF 三阶段                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  阶段1: SFT（监督微调）                                 │
│  └─> 用高质量数据微调基础模型                           │
│                                                         │
│  阶段2: 训练奖励模型                                    │
│  └─> 人类标注偏好数据，训练 RM                          │
│                                                         │
│  阶段3: PPO 强化学习                                    │
│  └─> 用 RM 指导模型优化                                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 偏好数据格式

```json
{
  "prompt": "写一首关于春天的诗",
  "chosen": "春风拂面暖意浓，\n桃花朵朵映日红。\n燕子归来寻旧巢，\n柳絮飘飘舞东风。",
  "rejected": "春天来了，花开了，很漂亮。"
}
```

## 奖励模型训练

```python
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RewardModel(nn.Module):
    """奖励模型"""
    
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

def reward_loss(chosen_rewards, rejected_rewards):
    """奖励模型损失函数"""
    # 希望 chosen 的奖励高于 rejected
    return -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()

# 训练循环
def train_reward_model(model, dataloader, optimizer, epochs=3):
    model.train()
    
    for epoch in range(epochs):
        for batch in dataloader:
            # 计算 chosen 和 rejected 的奖励
            chosen_rewards = model(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"]
            )
            rejected_rewards = model(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"]
            )
            
            loss = reward_loss(chosen_rewards, rejected_rewards)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## DPO（Direct Preference Optimization）

DPO 直接从偏好数据优化策略，无需训练奖励模型。

### DPO 原理

```
RLHF: 偏好数据 → 奖励模型 → PPO 训练 → 对齐模型
DPO:  偏好数据 → 直接优化 → 对齐模型（更简单！）
```

### DPO 损失函数

```python
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """DPO 损失函数"""
    # 计算 log ratio
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps
    
    # DPO 损失
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()
    
    return loss
```

### 使用 TRL 库训练 DPO

```bash
pip install trl
```

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

# 加载偏好数据
dataset = load_dataset("json", data_files="preferences.jsonl")

# DPO 配置
training_args = DPOConfig(
    output_dir="./dpo_model",
    beta=0.1,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-7,
    logging_steps=10
)

# 训练
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer
)

trainer.train()
```

## ORPO（无需参考模型）

```python
from trl import ORPOTrainer, ORPOConfig

# ORPO 不需要参考模型
training_args = ORPOConfig(
    output_dir="./orpo_model",
    beta=0.1,
    per_device_train_batch_size=4,
    num_train_epochs=3
)

trainer = ORPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer
)

trainer.train()
```

## 偏好数据收集

### 人工标注

```python
def collect_preferences(prompts: list, model) -> list:
    """收集人工偏好标注"""
    preferences = []
    
    for prompt in prompts:
        # 生成多个候选回复
        responses = []
        for _ in range(3):
            response = model.generate(prompt, temperature=0.8)
            responses.append(response)
        
        # 人工选择最佳和最差
        print(f"Prompt: {prompt}")
        for i, r in enumerate(responses):
            print(f"{i}: {r}")
        
        chosen_idx = int(input("Best response: "))
        rejected_idx = int(input("Worst response: "))
        
        preferences.append({
            "prompt": prompt,
            "chosen": responses[chosen_idx],
            "rejected": responses[rejected_idx]
        })
    
    return preferences
```

### AI 辅助标注

```python
from openai import OpenAI

client = OpenAI()

def ai_preference_labeling(prompt: str, response_a: str, response_b: str) -> dict:
    """使用 GPT-4 进行偏好标注"""
    judge_prompt = f"""
比较以下两个回复，选择更好的一个。

问题：{prompt}

回复 A：{response_a}

回复 B：{response_b}

哪个回复更好？只回答 "A" 或 "B"。
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=1
    )
    
    choice = response.choices[0].message.content.strip()
    
    if choice == "A":
        return {"prompt": prompt, "chosen": response_a, "rejected": response_b}
    else:
        return {"prompt": prompt, "chosen": response_b, "rejected": response_a}
```


## 方法对比

| 方法 | 复杂度 | 效果 | 适用场景 |
|------|--------|------|---------|
| RLHF | 高 | 最好 | 大规模对齐 |
| DPO | 中 | 很好 | 推荐首选 |
| ORPO | 低 | 好 | 资源有限 |
| KTO | 低 | 好 | 只有正/负样本 |

## 最佳实践

1. **数据质量优先**：偏好数据质量决定对齐效果
2. **从 DPO 开始**：比 RLHF 简单，效果接近
3. **多样化数据**：覆盖各种场景和边界情况
4. **迭代优化**：多轮对齐逐步提升
5. **评估验证**：用人工评估验证对齐效果

## 延伸阅读

- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [TRL Library](https://github.com/huggingface/trl)