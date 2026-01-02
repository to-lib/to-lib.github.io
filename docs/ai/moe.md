---
sidebar_position: 36
title: 🧩 Mixture of Experts
---

# Mixture of Experts (MoE)

Mixture of Experts 是一种稀疏激活的模型架构，通过路由机制选择性激活部分专家网络，在保持大模型能力的同时降低计算成本。

## MoE 原理

```
                    ┌─────────────┐
                    │   Router    │
                    │  (门控网络)  │
                    └──────┬──────┘
                           │
           ┌───────┬───────┼───────┬───────┐
           ▼       ▼       ▼       ▼       ▼
        ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
        │ E1  │ │ E2  │ │ E3  │ │ E4  │ │ E5  │
        │专家1│ │专家2│ │专家3│ │专家4│ │专家5│
        └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
           │       │       │       │       │
           └───────┴───────┼───────┴───────┘
                           ▼
                    ┌─────────────┐
                    │  加权求和   │
                    └─────────────┘

每次只激活 Top-K 个专家（通常 K=2）
```

## MoE vs Dense 模型

| 特性 | Dense 模型 | MoE 模型 |
|------|-----------|---------|
| 参数激活 | 100% | 10-20% |
| 总参数量 | 较小 | 较大 |
| 推理成本 | 高 | 低 |
| 训练难度 | 简单 | 复杂 |
| 代表模型 | LLaMA, GPT-4 | Mixtral, DeepSeek |

## 主流 MoE 模型

| 模型 | 总参数 | 激活参数 | 专家数 |
|------|--------|---------|--------|
| Mixtral 8x7B | 47B | 13B | 8 |
| Mixtral 8x22B | 141B | 39B | 8 |
| DeepSeek-V2 | 236B | 21B | 160 |
| Qwen2-MoE | 57B | 14B | 64 |
| DBRX | 132B | 36B | 16 |

## 使用 MoE 模型

### Ollama

```bash
# 下载 Mixtral
ollama pull mixtral

# 运行
ollama run mixtral "解释什么是 MoE 架构"
```


### vLLM

```python
from vllm import LLM, SamplingParams

# 加载 MoE 模型
llm = LLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    tensor_parallel_size=2,  # 多卡并行
    dtype="float16"
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=512
)

outputs = llm.generate(["解释 MoE 架构的优势"], sampling_params)
print(outputs[0].outputs[0].text)
```

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

messages = [{"role": "user", "content": "什么是 MoE？"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

## MoE 架构实现

### 简化版 MoE 层

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """单个专家网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.SiLU()
    
    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class Router(nn.Module):
    """路由网络"""
    
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        # 返回每个专家的权重
        return F.softmax(self.gate(x), dim=-1)

class MoELayer(nn.Module):
    """MoE 层"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 创建专家
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim)
            for _ in range(num_experts)
        ])
        
        # 路由器
        self.router = Router(input_dim, num_experts)
    
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        
        # 计算路由权重
        router_logits = self.router(x_flat)
        
        # 选择 Top-K 专家
        top_k_weights, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        # 计算输出
        output = torch.zeros_like(x_flat)
        
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_weight = top_k_weights[:, i].unsqueeze(-1)
            
            for j in range(self.num_experts):
                mask = expert_idx == j
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[j](expert_input)
                    output[mask] += expert_weight[mask] * expert_output
        
        return output.view(batch_size, seq_len, dim)
```

### 负载均衡损失

```python
def load_balancing_loss(router_logits: torch.Tensor, top_k: int = 2) -> torch.Tensor:
    """计算负载均衡损失，防止专家使用不均"""
    num_experts = router_logits.shape[-1]
    
    # 计算每个专家被选中的频率
    top_k_indices = torch.topk(router_logits, top_k, dim=-1).indices
    expert_counts = torch.zeros(num_experts, device=router_logits.device)
    
    for i in range(num_experts):
        expert_counts[i] = (top_k_indices == i).float().sum()
    
    # 归一化
    expert_probs = expert_counts / expert_counts.sum()
    
    # 计算与均匀分布的差异
    uniform_prob = 1.0 / num_experts
    loss = ((expert_probs - uniform_prob) ** 2).sum()
    
    return loss
```

## MoE 优化技巧

### 专家并行

```python
# 使用 DeepSpeed 进行专家并行
import deepspeed

config = {
    "train_batch_size": 32,
    "moe": {
        "enabled": True,
        "ep_size": 4,  # 专家并行度
        "moe_param_group": True
    }
}

model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=config
)
```

### 量化 MoE 模型

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)
```


## MoE 部署考虑

### 显存需求

```
Mixtral 8x7B (FP16):
- 总参数：47B × 2 bytes = 94 GB
- 但只需加载激活的专家
- 实际显存：~26 GB（单卡可运行量化版本）

DeepSeek-V2 (FP16):
- 总参数：236B
- 激活参数：21B
- 推理显存：~50 GB
```

### 推理优化

```python
# 使用 vLLM 的 MoE 优化
from vllm import LLM

llm = LLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    tensor_parallel_size=2,
    # MoE 特定优化
    enable_prefix_caching=True,  # 前缀缓存
    max_num_seqs=256,            # 批处理大小
)
```

## MoE vs Dense 选择

| 场景 | 推荐 |
|------|------|
| 资源有限 | MoE（激活参数少） |
| 追求极致性能 | Dense（更稳定） |
| 多任务场景 | MoE（专家专业化） |
| 简单部署 | Dense（架构简单） |

## 最佳实践

1. **选择合适的 Top-K**：通常 K=2 是好的平衡点
2. **注意负载均衡**：防止部分专家过载
3. **量化部署**：MoE 模型特别适合量化
4. **专家并行**：多卡部署时使用专家并行
5. **监控专家使用**：分析哪些专家被频繁使用

## 延伸阅读

- [Mixtral Paper](https://arxiv.org/abs/2401.04088)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [Switch Transformer](https://arxiv.org/abs/2101.03961)
- [GShard](https://arxiv.org/abs/2006.16668)