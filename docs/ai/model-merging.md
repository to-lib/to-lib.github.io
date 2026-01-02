---
sidebar_position: 38
title: 🔀 模型合并
---

# 模型合并

模型合并（Model Merging）是将多个微调模型的能力组合到一个模型中的技术，无需额外训练即可获得多种能力。

## 为什么要合并模型？

```
模型 A（擅长代码）──┐
模型 B（擅长数学）──┼──> 合并 ──> 新模型（代码+数学+写作）
模型 C（擅长写作）──┘
```

## 合并方法

| 方法 | 原理 | 效果 |
|------|------|------|
| Linear | 线性插值 | 简单有效 |
| SLERP | 球面插值 | 更平滑 |
| TIES | 修剪+合并 | 减少冲突 |
| DARE | 随机丢弃+缩放 | 保留多样性 |
| Task Arithmetic | 任务向量运算 | 灵活组合 |

## 使用 mergekit

```bash
pip install mergekit
```

### 线性合并

```yaml
# merge_config.yaml
merge_method: linear
slices:
  - sources:
      - model: model_a
        layer_range: [0, 32]
      - model: model_b
        layer_range: [0, 32]
    merge_method: linear
parameters:
  weight: 0.5  # 各 50%
base_model: model_a
dtype: float16
```

```bash
mergekit-yaml merge_config.yaml ./merged_model
```

### SLERP 合并

```yaml
merge_method: slerp
slices:
  - sources:
      - model: model_a
        layer_range: [0, 32]
      - model: model_b
        layer_range: [0, 32]
parameters:
  t: 0.5  # 插值参数
base_model: model_a
dtype: float16
```


### TIES 合并

```yaml
merge_method: ties
slices:
  - sources:
      - model: model_a
        layer_range: [0, 32]
        parameters:
          density: 0.5  # 保留 50% 的参数
          weight: 1.0
      - model: model_b
        layer_range: [0, 32]
        parameters:
          density: 0.5
          weight: 1.0
base_model: base_model
dtype: float16
```

### DARE 合并

```yaml
merge_method: dare_ties
slices:
  - sources:
      - model: model_a
        layer_range: [0, 32]
        parameters:
          density: 0.5
          weight: 1.0
      - model: model_b
        layer_range: [0, 32]
        parameters:
          density: 0.5
          weight: 1.0
base_model: base_model
dtype: float16
```

## Python 实现

### 线性合并

```python
import torch
from transformers import AutoModelForCausalLM

def linear_merge(model_a_path: str, model_b_path: str, alpha: float = 0.5):
    """线性合并两个模型"""
    model_a = AutoModelForCausalLM.from_pretrained(model_a_path)
    model_b = AutoModelForCausalLM.from_pretrained(model_b_path)
    
    merged_state_dict = {}
    
    for key in model_a.state_dict().keys():
        merged_state_dict[key] = (
            alpha * model_a.state_dict()[key] +
            (1 - alpha) * model_b.state_dict()[key]
        )
    
    model_a.load_state_dict(merged_state_dict)
    return model_a

# 使用
merged = linear_merge("model_a", "model_b", alpha=0.6)
merged.save_pretrained("./merged_model")
```

### SLERP 合并

```python
import torch
import numpy as np

def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, eps: float = 1e-8):
    """球面线性插值"""
    v0_norm = v0 / (torch.norm(v0) + eps)
    v1_norm = v1 / (torch.norm(v1) + eps)
    
    dot = torch.sum(v0_norm * v1_norm)
    dot = torch.clamp(dot, -1, 1)
    
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    if sin_theta < eps:
        return (1 - t) * v0 + t * v1
    
    s0 = torch.sin((1 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta
    
    return s0 * v0 + s1 * v1

def slerp_merge(model_a_path: str, model_b_path: str, t: float = 0.5):
    """SLERP 合并"""
    model_a = AutoModelForCausalLM.from_pretrained(model_a_path)
    model_b = AutoModelForCausalLM.from_pretrained(model_b_path)
    
    merged_state_dict = {}
    
    for key in model_a.state_dict().keys():
        merged_state_dict[key] = slerp(
            t,
            model_a.state_dict()[key].float(),
            model_b.state_dict()[key].float()
        ).to(model_a.state_dict()[key].dtype)
    
    model_a.load_state_dict(merged_state_dict)
    return model_a
```

### Task Arithmetic

```python
def task_arithmetic_merge(
    base_model_path: str,
    task_models: list[tuple[str, float]]  # [(model_path, weight), ...]
):
    """任务向量算术合并"""
    base = AutoModelForCausalLM.from_pretrained(base_model_path)
    base_state = base.state_dict()
    
    # 计算任务向量并加权求和
    task_vector_sum = {key: torch.zeros_like(val) for key, val in base_state.items()}
    
    for model_path, weight in task_models:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model_state = model.state_dict()
        
        for key in base_state.keys():
            # 任务向量 = 微调模型 - 基础模型
            task_vector = model_state[key] - base_state[key]
            task_vector_sum[key] += weight * task_vector
    
    # 合并：基础模型 + 任务向量和
    merged_state = {}
    for key in base_state.keys():
        merged_state[key] = base_state[key] + task_vector_sum[key]
    
    base.load_state_dict(merged_state)
    return base

# 使用
merged = task_arithmetic_merge(
    "base_model",
    [
        ("code_model", 0.5),
        ("math_model", 0.3),
        ("writing_model", 0.2)
    ]
)
```

## LoRA 合并

```python
from peft import PeftModel

def merge_lora_adapters(base_model_path: str, lora_paths: list[str]):
    """合并多个 LoRA 适配器"""
    base = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    for lora_path in lora_paths:
        base = PeftModel.from_pretrained(base, lora_path)
        base = base.merge_and_unload()
    
    return base
```

## 合并效果评估

```python
def evaluate_merged_model(model, test_sets: dict):
    """评估合并模型在各任务上的表现"""
    results = {}
    
    for task_name, test_data in test_sets.items():
        score = evaluate_task(model, test_data)
        results[task_name] = score
    
    return results

# 比较合并前后
original_scores = {
    "code": evaluate_task(code_model, code_test),
    "math": evaluate_task(math_model, math_test)
}

merged_scores = evaluate_merged_model(merged_model, {
    "code": code_test,
    "math": math_test
})

print("原始模型:", original_scores)
print("合并模型:", merged_scores)
```

## 最佳实践

1. **选择兼容模型**：相同架构、相同基础模型
2. **调整权重**：根据任务重要性调整合并权重
3. **评估验证**：合并后在各任务上评估
4. **TIES/DARE 优先**：比简单线性合并效果更好
5. **迭代优化**：尝试不同参数找到最佳组合

## 延伸阅读

- [mergekit](https://github.com/arcee-ai/mergekit)
- [TIES Paper](https://arxiv.org/abs/2306.01708)
- [DARE Paper](https://arxiv.org/abs/2311.03099)