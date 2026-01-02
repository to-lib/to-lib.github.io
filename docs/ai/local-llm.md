---
sidebar_position: 19
title: 🏠 本地部署 LLM
---

# 本地部署 LLM

本地部署 LLM 可以保护数据隐私、降低成本、减少延迟。本文介绍主流的本地部署方案。

## 部署方案对比

| 方案         | 特点                   | 适用场景           |
| ------------ | ---------------------- | ------------------ |
| **Ollama**   | 简单易用，一键部署     | 开发测试、个人使用 |
| **vLLM**     | 高性能，生产级         | 生产环境、高并发   |
| **llama.cpp** | 轻量，CPU 友好        | 边缘设备、低资源   |
| **TGI**      | HuggingFace 官方       | 企业部署           |

## Ollama

Ollama 是最简单的本地 LLM 部署方案，支持 macOS、Linux、Windows。

### 安装

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# 或使用 Homebrew (macOS)
brew install ollama
```

### 基础使用

```bash
# 启动服务
ollama serve

# 运行模型（自动下载）
ollama run llama3.2
ollama run qwen2.5:7b
ollama run deepseek-coder:6.7b

# 列出已下载模型
ollama list

# 删除模型
ollama rm llama3.2
```

### API 调用

```python
import requests

# Ollama 兼容 OpenAI API 格式
response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "你好"}],
        "stream": False
    }
)

print(response.json()["message"]["content"])
```

### 使用 OpenAI SDK

```python
from openai import OpenAI

# 指向 Ollama 服务
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # 任意值
)

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "解释什么是机器学习"}]
)

print(response.choices[0].message.content)
```

### 流式输出

```python
stream = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "写一首诗"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 自定义模型

```bash
# 创建 Modelfile
cat > Modelfile << 'EOF'
FROM llama3.2

# 设置系统提示
SYSTEM """
你是一个专业的代码助手，擅长 Python 和 JavaScript。
回答要简洁、准确，并提供代码示例。
"""

# 调整参数
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# 创建自定义模型
ollama create code-assistant -f Modelfile

# 运行
ollama run code-assistant
```

## vLLM

vLLM 是高性能的 LLM 推理引擎，支持 PagedAttention、连续批处理等优化技术。

### 安装

```bash
pip install vllm
```

### 启动服务

```bash
# 启动 OpenAI 兼容服务
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1
```

### API 调用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="vllm"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "你好"}]
)

print(response.choices[0].message.content)
```

### 离线批量推理

```python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# 批量推理
prompts = [
    "什么是机器学习？",
    "Python 有什么优点？",
    "解释 RESTful API"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Output: {output.outputs[0].text}\n")
```

### 高级配置

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=2,      # 多 GPU 并行
    gpu_memory_utilization=0.9,  # GPU 内存使用率
    max_model_len=8192,          # 最大上下文长度
    quantization="awq",          # 量化方式
    dtype="float16"              # 数据类型
)
```

## llama.cpp

llama.cpp 是纯 C/C++ 实现的 LLM 推理库，支持 CPU 推理，资源占用低。

### 安装 Python 绑定

```bash
pip install llama-cpp-python

# 支持 GPU 加速
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

### 下载模型

```bash
# 从 HuggingFace 下载 GGUF 格式模型
# 例如：https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
```

### 基础使用

```python
from llama_cpp import Llama

# 加载模型
llm = Llama(
    model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=4096,      # 上下文长度
    n_threads=8,     # CPU 线程数
    n_gpu_layers=35  # GPU 加速层数（0 = 纯 CPU）
)

# 生成
output = llm(
    "Q: 什么是人工智能？\nA:",
    max_tokens=256,
    temperature=0.7,
    stop=["Q:", "\n\n"]
)

print(output["choices"][0]["text"])
```

### Chat 格式

```python
output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "解释什么是深度学习"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(output["choices"][0]["message"]["content"])
```

### 启动 OpenAI 兼容服务

```bash
python -m llama_cpp.server \
    --model ./models/llama-2-7b-chat.Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8000 \
    --n_ctx 4096
```

## 模型量化

量化可以显著减少模型大小和内存占用。

### 常见量化格式

| 格式   | 说明                     | 大小（7B） |
| ------ | ------------------------ | ---------- |
| FP16   | 半精度浮点               | ~14GB      |
| INT8   | 8位整数                  | ~7GB       |
| INT4   | 4位整数                  | ~4GB       |
| GGUF Q4 | llama.cpp 4位量化       | ~4GB       |
| AWQ    | 激活感知量化             | ~4GB       |
| GPTQ   | 后训练量化               | ~4GB       |

### 使用 AutoGPTQ 量化

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 加载模型
model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 量化配置
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False
)

# 量化
model = AutoGPTQForCausalLM.from_pretrained(
    model_id,
    quantize_config=quantize_config
)

# 准备校准数据
examples = [tokenizer(text) for text in calibration_texts]
model.quantize(examples)

# 保存
model.save_quantized("./qwen2.5-7b-gptq")
```

## LangChain 集成

### Ollama

```python
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama

# LLM
llm = Ollama(model="llama3.2")
response = llm.invoke("你好")

# Chat Model
chat = ChatOllama(model="llama3.2")
response = chat.invoke([{"role": "user", "content": "你好"}])
```

### llama.cpp

```python
from langchain_community.llms import LlamaCpp

llm = LlamaCpp(
    model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=35,
    temperature=0.7
)

response = llm.invoke("什么是机器学习？")
```

## Docker 部署

### Ollama Docker

```yaml
# docker-compose.yml
version: '3.8'
services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  ollama_data:
```

### vLLM Docker

```yaml
version: '3.8'
services:
  vllm:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    command: >
      --model Qwen/Qwen2.5-7B-Instruct
      --host 0.0.0.0
      --port 8000
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## 硬件要求

### GPU 显存需求（推理）

| 模型大小 | FP16   | INT8   | INT4   |
| -------- | ------ | ------ | ------ |
| 7B       | 14GB   | 7GB    | 4GB    |
| 13B      | 26GB   | 13GB   | 7GB    |
| 70B      | 140GB  | 70GB   | 35GB   |

### 推荐配置

| 场景       | GPU                | 内存   | 推荐模型      |
| ---------- | ------------------ | ------ | ------------- |
| 个人开发   | RTX 3060 12GB      | 16GB   | 7B INT4       |
| 小团队     | RTX 4090 24GB      | 32GB   | 7B-13B        |
| 生产环境   | A100 40GB/80GB     | 64GB+  | 13B-70B       |

## 性能优化

### 1. 批处理

```python
# vLLM 自动批处理
# 多个请求会被合并处理，提高吞吐量
```

### 2. KV Cache 优化

```python
# vLLM PagedAttention
llm = LLM(
    model="...",
    gpu_memory_utilization=0.9,  # 更多内存用于 KV cache
)
```

### 3. 投机解码

```python
# 使用小模型加速大模型
llm = LLM(
    model="large-model",
    speculative_model="small-model",
    num_speculative_tokens=5
)
```

## 延伸阅读

- [Ollama 官网](https://ollama.com/)
- [vLLM 文档](https://docs.vllm.ai/)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [HuggingFace TGI](https://huggingface.co/docs/text-generation-inference)
