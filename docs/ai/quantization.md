---
sidebar_position: 31
title: 🗜️ 模型量化
---

# 模型量化

模型量化是将模型权重从高精度（FP32/FP16）转换为低精度（INT8/INT4）的技术，可以显著减少模型大小和内存占用，加速推理。

## 为什么需要量化？

| 精度 | 每参数大小 | 7B 模型大小 | 显存需求 |
|------|-----------|------------|---------|
| FP32 | 4 bytes | 28 GB | ~32 GB |
| FP16 | 2 bytes | 14 GB | ~16 GB |
| INT8 | 1 byte | 7 GB | ~8 GB |
| INT4 | 0.5 byte | 3.5 GB | ~4 GB |

## 量化方法对比

| 方法 | 精度损失 | 速度 | 适用场景 |
|------|---------|------|---------|
| GGUF | 低 | 快 | CPU/混合推理 |
| GPTQ | 低 | 快 | GPU 推理 |
| AWQ | 很低 | 快 | GPU 推理 |
| bitsandbytes | 中 | 中 | 训练/微调 |
| EETQ | 低 | 很快 | GPU 推理 |

## GGUF 格式

GGUF 是 llama.cpp 使用的量化格式，支持 CPU 和 GPU 混合推理。

### 量化级别

| 量化类型 | 大小比例 | 质量 | 说明 |
|---------|---------|------|------|
| Q2_K | ~29% | 较差 | 极限压缩 |
| Q3_K_M | ~37% | 可用 | 低端设备 |
| Q4_K_M | ~45% | 良好 | 推荐平衡 |
| Q5_K_M | ~52% | 很好 | 质量优先 |
| Q6_K | ~59% | 优秀 | 接近原始 |
| Q8_0 | ~75% | 极佳 | 几乎无损 |


### 使用 llama.cpp 量化

```bash
# 克隆 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# 编译
make -j

# 转换 HuggingFace 模型为 GGUF
python convert_hf_to_gguf.py /path/to/model --outfile model.gguf

# 量化
./llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M
```

### 使用 Ollama 运行 GGUF

```bash
# 创建 Modelfile
cat > Modelfile << EOF
FROM ./model-q4_k_m.gguf
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
SYSTEM "你是一个有帮助的助手。"
EOF

# 创建模型
ollama create my-model -f Modelfile

# 运行
ollama run my-model
```

### Python 使用 llama-cpp-python

```python
from llama_cpp import Llama

# 加载量化模型
llm = Llama(
    model_path="./model-q4_k_m.gguf",
    n_ctx=4096,        # 上下文长度
    n_gpu_layers=35,   # GPU 层数，-1 表示全部
    n_threads=8        # CPU 线程数
)

# 推理
output = llm(
    "用户：你好\n助手：",
    max_tokens=256,
    temperature=0.7,
    stop=["用户："]
)

print(output["choices"][0]["text"])
```

## GPTQ 量化

GPTQ 是一种基于校准数据的 Post-Training Quantization 方法。

### 使用 AutoGPTQ

```bash
pip install auto-gptq
```

```python
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_path = "meta-llama/Llama-2-7b-hf"
quantized_path = "./llama2-7b-gptq"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 准备校准数据
calibration_data = [
    tokenizer("这是一段用于校准的文本。" * 10, return_tensors="pt"),
    # 添加更多样本...
]

# 量化配置
quantize_config = BaseQuantizeConfig(
    bits=4,              # 量化位数
    group_size=128,      # 分组大小
    desc_act=True        # 激活值排序
)

# 加载模型
model = AutoGPTQForCausalLM.from_pretrained(
    model_path,
    quantize_config=quantize_config
)

# 执行量化
model.quantize(calibration_data)

# 保存
model.save_quantized(quantized_path)
tokenizer.save_pretrained(quantized_path)
```

### 加载 GPTQ 模型

```python
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "./llama2-7b-gptq",
    device="cuda:0",
    use_safetensors=True
)

tokenizer = AutoTokenizer.from_pretrained("./llama2-7b-gptq")

# 推理
inputs = tokenizer("你好，", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## AWQ 量化

AWQ (Activation-aware Weight Quantization) 通过保护重要权重来减少精度损失。

### 使用 AutoAWQ

```bash
pip install autoawq
```

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-2-7b-hf"
quant_path = "./llama2-7b-awq"

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 量化配置
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# 执行量化
model.quantize(tokenizer, quant_config=quant_config)

# 保存
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### vLLM 使用 AWQ 模型

```python
from vllm import LLM, SamplingParams

# 加载 AWQ 模型
llm = LLM(
    model="./llama2-7b-awq",
    quantization="awq",
    dtype="half"
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256
)

outputs = llm.generate(["你好，请介绍一下自己。"], sampling_params)
print(outputs[0].outputs[0].text)
```


## bitsandbytes 量化

bitsandbytes 支持训练时量化，常用于 QLoRA 微调。

### 8-bit 量化

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 8-bit 配置
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### 4-bit 量化 (QLoRA)

```python
from transformers import BitsAndBytesConfig
import torch

# 4-bit 配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # nf4 或 fp4
    bnb_4bit_compute_dtype=torch.float16, # 计算精度
    bnb_4bit_use_double_quant=True        # 双重量化
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 配合 LoRA 微调
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

## HuggingFace 量化模型

直接使用社区量化好的模型。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 使用 TheBloke 的量化模型
model_id = "TheBloke/Llama-2-7B-GPTQ"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
```

### 常用量化模型来源

| 来源 | 格式 | 说明 |
|------|------|------|
| TheBloke | GPTQ/AWQ/GGUF | 最全面的量化模型 |
| Qwen | GPTQ/AWQ | 官方量化版本 |
| unsloth | 4-bit | 优化的训练模型 |

## 量化质量评估

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def evaluate_perplexity(model, tokenizer, dataset_name="wikitext", split="test"):
    """评估困惑度"""
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
    
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    max_length = 2048
    stride = 512
    
    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)
    
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

# 比较原始模型和量化模型
original_ppl = evaluate_perplexity(original_model, tokenizer)
quantized_ppl = evaluate_perplexity(quantized_model, tokenizer)

print(f"原始模型 PPL: {original_ppl:.2f}")
print(f"量化模型 PPL: {quantized_ppl:.2f}")
print(f"精度损失: {(quantized_ppl - original_ppl) / original_ppl * 100:.2f}%")
```

## 量化选择指南

```
┌─────────────────────────────────────────────────────────┐
│                    量化方法选择                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  场景：CPU 推理 / 低显存                                │
│  └─> GGUF (Q4_K_M 或 Q5_K_M)                           │
│                                                         │
│  场景：GPU 推理，追求速度                               │
│  └─> AWQ 或 GPTQ                                       │
│                                                         │
│  场景：微调训练                                         │
│  └─> bitsandbytes (4-bit QLoRA)                        │
│                                                         │
│  场景：质量优先                                         │
│  └─> AWQ > GPTQ > GGUF Q6_K                            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 最佳实践

1. **选择合适的量化级别**：Q4 平衡大小和质量，Q5/Q6 质量优先
2. **使用校准数据**：GPTQ/AWQ 需要代表性的校准数据
3. **评估质量**：量化后测试困惑度和任务性能
4. **考虑硬件**：CPU 用 GGUF，GPU 用 AWQ/GPTQ
5. **使用社区模型**：优先使用 TheBloke 等已量化的模型

## 延伸阅读

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)