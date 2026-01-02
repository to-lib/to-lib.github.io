---
sidebar_position: 40
title: 📚 持续学习
---

# 持续学习（Continual Learning）

持续学习是让模型能够不断学习新知识，同时保留旧知识的技术，解决"灾难性遗忘"问题。

## 灾难性遗忘

```
传统微调：
任务A训练 ──> 模型擅长A
    │
    ▼
任务B训练 ──> 模型擅长B，但忘记A ❌

持续学习：
任务A训练 ──> 模型擅长A
    │
    ▼
任务B训练 ──> 模型同时擅长A和B ✓
```

## 持续学习方法

| 方法 | 原理 | 适用场景 |
|------|------|---------|
| Replay | 重放旧数据 | 数据可存储 |
| EWC | 保护重要参数 | 参数级保护 |
| LoRA 累加 | 独立适配器 | LLM 微调 |
| 知识蒸馏 | 保留旧模型知识 | 模型更新 |

## 经验回放（Replay）

```python
import random
from collections import deque

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, samples: list):
        """添加新样本"""
        self.buffer.extend(samples)
    
    def sample(self, batch_size: int) -> list:
        """随机采样"""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

class ContinualTrainer:
    """持续学习训练器"""
    
    def __init__(self, model, replay_ratio: float = 0.3):
        self.model = model
        self.replay_buffer = ReplayBuffer()
        self.replay_ratio = replay_ratio
    
    def train_task(self, new_data: list, epochs: int = 3):
        """训练新任务"""
        for epoch in range(epochs):
            # 混合新数据和回放数据
            replay_size = int(len(new_data) * self.replay_ratio)
            replay_data = self.replay_buffer.sample(replay_size)
            
            combined_data = new_data + replay_data
            random.shuffle(combined_data)
            
            # 训练
            for batch in self._batch(combined_data):
                self._train_step(batch)
        
        # 将新数据加入回放缓冲区
        self.replay_buffer.add(new_data)
    
    def _batch(self, data, batch_size=32):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
    
    def _train_step(self, batch):
        # 训练逻辑
        pass
```

## EWC（弹性权重巩固）

```python
import torch
import torch.nn as nn
from copy import deepcopy

class EWC:
    """Elastic Weight Consolidation"""
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 1000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher = {}
        self.optimal_params = {}
    
    def compute_fisher(self, dataloader):
        """计算 Fisher 信息矩阵"""
        self.fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        
        self.model.eval()
        for batch in dataloader:
            self.model.zero_grad()
            output = self.model(batch["input_ids"])
            loss = output.loss
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.data ** 2
        
        # 归一化
        for n in self.fisher:
            self.fisher[n] /= len(dataloader)
        
        # 保存最优参数
        self.optimal_params = {n: p.clone() for n, p in self.model.named_parameters()}
    
    def ewc_loss(self) -> torch.Tensor:
        """计算 EWC 正则化损失"""
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.optimal_params[n]) ** 2).sum()
        return self.lambda_ewc * loss

def train_with_ewc(model, ewc, dataloader, optimizer, epochs=3):
    """带 EWC 的训练"""
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            
            # 任务损失
            output = model(batch["input_ids"], labels=batch["labels"])
            task_loss = output.loss
            
            # EWC 损失
            ewc_loss = ewc.ewc_loss()
            
            # 总损失
            total_loss = task_loss + ewc_loss
            total_loss.backward()
            optimizer.step()
```

## LoRA 累加

```python
from peft import LoraConfig, get_peft_model, PeftModel

class LoRAContinualLearning:
    """基于 LoRA 的持续学习"""
    
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path
        self.adapters = {}  # task_name -> adapter_path
    
    def train_task(self, task_name: str, train_data, output_dir: str):
        """为新任务训练 LoRA 适配器"""
        from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
        
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(self.base_model_path)
        
        # 添加 LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05
        )
        model = get_peft_model(model, lora_config)
        
        # 训练
        trainer = Trainer(
            model=model,
            train_dataset=train_data,
            args=TrainingArguments(output_dir=output_dir, num_train_epochs=3)
        )
        trainer.train()
        
        # 保存适配器
        model.save_pretrained(output_dir)
        self.adapters[task_name] = output_dir
    
    def load_for_task(self, task_name: str):
        """加载特定任务的模型"""
        model = AutoModelForCausalLM.from_pretrained(self.base_model_path)
        model = PeftModel.from_pretrained(model, self.adapters[task_name])
        return model
    
    def merge_adapters(self, task_names: list, weights: list = None):
        """合并多个适配器"""
        if weights is None:
            weights = [1.0 / len(task_names)] * len(task_names)
        
        model = AutoModelForCausalLM.from_pretrained(self.base_model_path)
        
        # 加载并加权合并适配器
        merged_state = None
        for task_name, weight in zip(task_names, weights):
            adapter = PeftModel.from_pretrained(model, self.adapters[task_name])
            adapter_state = adapter.state_dict()
            
            if merged_state is None:
                merged_state = {k: v * weight for k, v in adapter_state.items()}
            else:
                for k, v in adapter_state.items():
                    merged_state[k] += v * weight
        
        model.load_state_dict(merged_state, strict=False)
        return model
```

## 知识蒸馏保留

```python
class DistillationContinualLearning:
    """基于蒸馏的持续学习"""
    
    def __init__(self, model, temperature: float = 2.0, alpha: float = 0.5):
        self.current_model = model
        self.old_model = None
        self.temperature = temperature
        self.alpha = alpha
    
    def before_task(self):
        """任务开始前保存旧模型"""
        self.old_model = deepcopy(self.current_model)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad = False
    
    def compute_loss(self, inputs, labels):
        """计算带蒸馏的损失"""
        # 当前模型输出
        current_outputs = self.current_model(inputs)
        current_logits = current_outputs.logits
        
        # 任务损失
        task_loss = nn.CrossEntropyLoss()(
            current_logits.view(-1, current_logits.size(-1)),
            labels.view(-1)
        )
        
        if self.old_model is None:
            return task_loss
        
        # 蒸馏损失
        with torch.no_grad():
            old_logits = self.old_model(inputs).logits
        
        distill_loss = nn.KLDivLoss(reduction="batchmean")(
            nn.functional.log_softmax(current_logits / self.temperature, dim=-1),
            nn.functional.softmax(old_logits / self.temperature, dim=-1)
        ) * (self.temperature ** 2)
        
        return self.alpha * distill_loss + (1 - self.alpha) * task_loss
```


## LLM 持续预训练

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

def continual_pretrain(
    model_path: str,
    new_corpus_path: str,
    output_dir: str,
    replay_corpus_path: str = None
):
    """LLM 持续预训练"""
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 加载新语料
    from datasets import load_dataset
    new_data = load_dataset("text", data_files=new_corpus_path)
    
    # 可选：混合旧语料
    if replay_corpus_path:
        old_data = load_dataset("text", data_files=replay_corpus_path)
        # 混合比例 8:2
        combined = concatenate_datasets([
            new_data["train"].select(range(int(len(new_data["train"]) * 0.8))),
            old_data["train"].select(range(int(len(old_data["train"]) * 0.2)))
        ])
    else:
        combined = new_data["train"]
    
    # 训练配置
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=1e-5,  # 较小的学习率
        warmup_ratio=0.1,
        save_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined,
        tokenizer=tokenizer
    )
    
    trainer.train()
```

## 评估遗忘程度

```python
def evaluate_forgetting(model, task_datasets: dict) -> dict:
    """评估各任务的遗忘程度"""
    results = {}
    
    for task_name, dataset in task_datasets.items():
        # 评估当前模型在该任务上的表现
        score = evaluate_task(model, dataset)
        results[task_name] = score
    
    return results

def compute_forgetting_rate(
    scores_before: dict,
    scores_after: dict
) -> dict:
    """计算遗忘率"""
    forgetting = {}
    
    for task in scores_before:
        if task in scores_after:
            forgetting[task] = (scores_before[task] - scores_after[task]) / scores_before[task]
    
    return forgetting
```

## 方法对比

| 方法 | 优点 | 缺点 |
|------|------|------|
| Replay | 简单有效 | 需要存储数据 |
| EWC | 不需要旧数据 | 计算 Fisher 开销大 |
| LoRA 累加 | 灵活、可组合 | 多个适配器管理复杂 |
| 蒸馏 | 效果好 | 需要保存旧模型 |

## 最佳实践

1. **混合使用**：Replay + 正则化效果更好
2. **控制学习率**：持续学习用较小的学习率
3. **定期评估**：监控旧任务的性能
4. **数据平衡**：新旧数据比例要合理
5. **选择性更新**：只更新部分参数（如 LoRA）

## 延伸阅读

- [EWC Paper](https://arxiv.org/abs/1612.00796)
- [Continual Learning Survey](https://arxiv.org/abs/2302.00487)
- [LLM Continual Learning](https://arxiv.org/abs/2308.04014)