---
sidebar_position: 39
title: ⚡ Speculative Decoding
---

# Speculative Decoding（推测解码）

Speculative Decoding 是一种加速 LLM 推理的技术，使用小模型快速生成候选 token，再由大模型验证，在保持输出质量的同时提升速度。

## 原理

```
传统解码：
大模型 ──> token1 ──> token2 ──> token3 ──> token4
         (慢)      (慢)      (慢)      (慢)

推测解码：
小模型 ──> [token1, token2, token3, token4] (快速生成)
              │
              ▼
大模型 ──> 验证 ──> [✓token1, ✓token2, ✓token3, ✗token4]
              │
              ▼
         接受前3个，从 token4 重新生成
```

## 为什么有效？

- 大模型推理瓶颈在内存带宽，不在计算
- 验证多个 token 的成本 ≈ 生成 1 个 token
- 小模型生成的 token 大部分能被大模型接受

## 基础实现

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SpeculativeDecoder:
    """推测解码器"""
    
    def __init__(
        self,
        target_model_name: str,  # 大模型
        draft_model_name: str,   # 小模型
        gamma: int = 4           # 每次推测的 token 数
    ):
        self.target = AutoModelForCausalLM.from_pretrained(target_model_name)
        self.draft = AutoModelForCausalLM.from_pretrained(draft_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        self.gamma = gamma
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        generated = input_ids.clone()
        
        while generated.shape[1] - input_ids.shape[1] < max_tokens:
            # 1. 小模型生成 gamma 个候选 token
            draft_tokens = self._draft_generate(generated, self.gamma)
            
            # 2. 大模型验证
            accepted, next_token = self._verify(generated, draft_tokens)
            
            # 3. 接受验证通过的 token
            generated = torch.cat([generated, accepted, next_token], dim=1)
            
            # 检查是否生成了 EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
    
    def _draft_generate(self, input_ids: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """小模型生成候选 token"""
        draft_ids = input_ids.clone()
        
        for _ in range(num_tokens):
            with torch.no_grad():
                outputs = self.draft(draft_ids)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                draft_ids = torch.cat([draft_ids, next_token], dim=1)
        
        # 返回新生成的 token
        return draft_ids[:, input_ids.shape[1]:]
    
    def _verify(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """大模型验证候选 token"""
        # 拼接输入和候选
        full_ids = torch.cat([input_ids, draft_tokens], dim=1)
        
        with torch.no_grad():
            outputs = self.target(full_ids)
            logits = outputs.logits
        
        accepted = []
        
        for i in range(draft_tokens.shape[1]):
            pos = input_ids.shape[1] + i - 1
            target_probs = torch.softmax(logits[:, pos, :], dim=-1)
            draft_token = draft_tokens[:, i]
            
            # 检查大模型是否同意这个 token
            target_token = logits[:, pos, :].argmax(dim=-1)
            
            if target_token.item() == draft_token.item():
                accepted.append(draft_token)
            else:
                # 拒绝，返回大模型的选择
                return (
                    torch.stack(accepted, dim=1) if accepted else torch.tensor([[]]),
                    target_token.unsqueeze(0)
                )
        
        # 全部接受，生成下一个 token
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return torch.stack(accepted, dim=1), next_token

# 使用
decoder = SpeculativeDecoder(
    target_model_name="meta-llama/Llama-2-70b-hf",
    draft_model_name="meta-llama/Llama-2-7b-hf",
    gamma=4
)

output = decoder.generate("写一首关于春天的诗：")
```

## 带采样的推测解码

```python
def speculative_sampling(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token: int
) -> tuple[bool, int]:
    """推测采样（支持随机采样）"""
    p = target_probs[draft_token].item()
    q = draft_probs[draft_token].item()
    
    # 以 min(1, p/q) 的概率接受
    if torch.rand(1).item() < min(1, p / q):
        return True, draft_token
    else:
        # 从修正分布中采样
        adjusted_probs = torch.clamp(target_probs - draft_probs, min=0)
        adjusted_probs = adjusted_probs / adjusted_probs.sum()
        new_token = torch.multinomial(adjusted_probs, 1).item()
        return False, new_token
```

## vLLM 推测解码

```python
from vllm import LLM, SamplingParams

# vLLM 内置推测解码支持
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    speculative_model="meta-llama/Llama-2-7b-hf",
    num_speculative_tokens=5,
    use_v2_block_manager=True
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["写一首诗"], sampling_params)
```

## 自推测解码（Self-Speculative）

使用同一模型的早期层作为 draft model。

```python
class SelfSpeculativeDecoder:
    """自推测解码"""
    
    def __init__(self, model_name: str, draft_layers: int = 8):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.draft_layers = draft_layers
        self.total_layers = len(self.model.model.layers)
    
    def draft_forward(self, input_ids):
        """只使用前几层进行推测"""
        hidden = self.model.model.embed_tokens(input_ids)
        
        for i in range(self.draft_layers):
            hidden = self.model.model.layers[i](hidden)[0]
        
        # 直接用 LM head 预测
        logits = self.model.lm_head(hidden)
        return logits
    
    def full_forward(self, input_ids):
        """完整前向传播"""
        return self.model(input_ids).logits
```

## Medusa（多头推测）

```python
class MedusaHead(torch.nn.Module):
    """Medusa 多头预测"""
    
    def __init__(self, hidden_size: int, vocab_size: int, num_heads: int = 4):
        super().__init__()
        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size, vocab_size)
            for _ in range(num_heads)
        ])
    
    def forward(self, hidden_states):
        # 每个头预测未来的一个 token
        return [head(hidden_states) for head in self.heads]

# Medusa 可以一次预测多个未来 token
# 然后用树状验证找到最长的有效序列
```

## 加速效果

| 配置 | 加速比 |
|------|--------|
| Llama-70B + Llama-7B | 2-3x |
| 自推测（8层） | 1.5-2x |
| Medusa | 2-3x |

## 最佳实践

1. **选择合适的 draft 模型**：同系列小模型效果最好
2. **调整 gamma**：通常 4-8 是好的选择
3. **监控接受率**：接受率太低说明 draft 模型不匹配
4. **批处理优化**：推测解码对批处理不太友好
5. **使用 vLLM**：生产环境推荐使用 vLLM 的实现

## 延伸阅读

- [Speculative Decoding Paper](https://arxiv.org/abs/2211.17192)
- [Medusa Paper](https://arxiv.org/abs/2401.10774)
- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/models/spec_decode.html)