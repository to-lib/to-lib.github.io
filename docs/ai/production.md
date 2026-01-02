---
sidebar_position: 11
title: 🚀 Production（生产化与部署）
---

# Production（生产化与部署）

把 Demo 变成可用的生产系统，关键是"稳定、可观测、可控成本"。本页从延迟、稳定性、缓存、观测与发布策略总结常用做法。

## 典型生产架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway                               │
│              (鉴权、限流、配额、负载均衡)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌──────────┐    ┌──────────┐    ┌──────────┐
       │ LLM      │    │ RAG      │    │ Agent    │
       │ Service  │    │ Service  │    │ Service  │
       └────┬─────┘    └────┬─────┘    └────┬─────┘
            │               │               │
            ▼               ▼               ▼
       ┌──────────────────────────────────────────┐
       │           Observability Layer            │
       │     (日志、指标、追踪、评估回放)           │
       └──────────────────────────────────────────┘
```

## 延迟优化

### 1. 模型选择策略

```python
class ModelRouter:
    """根据任务复杂度选择模型"""
    
    def __init__(self):
        self.models = {
            "simple": "gpt-4o-mini",      # 简单任务
            "complex": "gpt-4o",           # 复杂任务
            "code": "gpt-4o",              # 代码生成
        }
    
    def classify_task(self, prompt: str) -> str:
        """简单的任务分类"""
        if len(prompt) < 100:
            return "simple"
        if any(kw in prompt.lower() for kw in ["代码", "code", "function", "class"]):
            return "code"
        return "complex"
    
    def get_model(self, prompt: str) -> str:
        task_type = self.classify_task(prompt)
        return self.models[task_type]

router = ModelRouter()
model = router.get_model(user_prompt)
```

### 2. 流式输出

```python
from openai import OpenAI

client = OpenAI()

def stream_response(prompt: str):
    """流式输出，提升用户体验"""
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# FastAPI 流式响应
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(prompt: str):
    return StreamingResponse(
        stream_response(prompt),
        media_type="text/event-stream"
    )
```

### 3. 并行化处理

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def parallel_requests(prompts: list[str]) -> list[str]:
    """并行处理多个请求"""
    tasks = [
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": p}]
        )
        for p in prompts
    ]
    
    responses = await asyncio.gather(*tasks)
    return [r.choices[0].message.content for r in responses]

# RAG 场景：并行检索和生成
async def rag_parallel(query: str):
    """并行执行检索和查询重写"""
    rewrite_task = rewrite_query(query)
    retrieve_task = retrieve_documents(query)
    
    rewritten, docs = await asyncio.gather(rewrite_task, retrieve_task)
    
    # 使用重写后的查询再次检索
    more_docs = await retrieve_documents(rewritten)
    
    return generate_answer(query, docs + more_docs)
```

### 4. 上下文压缩

```python
def compress_context(messages: list[dict], max_tokens: int = 4000) -> list[dict]:
    """压缩对话历史"""
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o")
    
    # 保留系统消息
    system_msg = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]
    
    # 从最新消息开始保留
    compressed = []
    total_tokens = sum(len(enc.encode(m["content"])) for m in system_msg)
    
    for msg in reversed(other_msgs):
        msg_tokens = len(enc.encode(msg["content"]))
        if total_tokens + msg_tokens > max_tokens:
            break
        compressed.insert(0, msg)
        total_tokens += msg_tokens
    
    return system_msg + compressed

def summarize_history(messages: list[dict]) -> str:
    """将历史对话总结为摘要"""
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"请用 100 字以内总结以下对话的要点：\n{history_text}"
        }],
        max_tokens=150
    )
    
    return response.choices[0].message.content
```

## 稳定性与容错

### 1. 重试策略

```python
import time
from functools import wraps
from openai import RateLimitError, APIError

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """指数退避重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limited, retrying in {delay}s...")
                    time.sleep(delay)
                except APIError as e:
                    if e.status_code >= 500:  # 服务端错误才重试
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(base_delay)
                    else:
                        raise
            return None
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3)
def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### 2. 超时与取消

```python
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def timeout_context(seconds: float):
    """超时上下文管理器"""
    try:
        yield asyncio.wait_for(asyncio.sleep(0), timeout=seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {seconds}s")

async def call_with_timeout(prompt: str, timeout: float = 30.0) -> str:
    """带超时的 LLM 调用"""
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            ),
            timeout=timeout
        )
        return response.choices[0].message.content
    except asyncio.TimeoutError:
        return "请求超时，请稍后重试"
```

### 3. 降级策略

```python
class LLMService:
    """带降级策略的 LLM 服务"""
    
    def __init__(self):
        self.primary_model = "gpt-4o"
        self.fallback_model = "gpt-4o-mini"
        self.error_count = 0
        self.error_threshold = 5
        self.circuit_open = False
    
    async def call(self, prompt: str) -> str:
        # 熔断器打开时直接使用降级模型
        if self.circuit_open:
            return await self._call_fallback(prompt)
        
        try:
            response = await self._call_primary(prompt)
            self.error_count = 0  # 成功则重置错误计数
            return response
        except Exception as e:
            self.error_count += 1
            if self.error_count >= self.error_threshold:
                self.circuit_open = True
                print("⚠️ 熔断器打开，切换到降级模型")
            return await self._call_fallback(prompt)
    
    async def _call_primary(self, prompt: str) -> str:
        response = await client.chat.completions.create(
            model=self.primary_model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    async def _call_fallback(self, prompt: str) -> str:
        response = await client.chat.completions.create(
            model=self.fallback_model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
```

## 缓存策略

### 1. 语义缓存

```python
import hashlib
from typing import Optional
import numpy as np

class SemanticCache:
    """基于语义相似度的缓存"""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.cache = {}  # {embedding_hash: (embedding, response)}
        self.threshold = similarity_threshold
    
    def _get_embedding(self, text: str) -> np.ndarray:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def get(self, query: str) -> Optional[str]:
        """查找语义相似的缓存"""
        query_embedding = self._get_embedding(query)
        
        for key, (cached_embedding, response) in self.cache.items():
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            if similarity >= self.threshold:
                return response
        
        return None
    
    def set(self, query: str, response: str):
        """设置缓存"""
        embedding = self._get_embedding(query)
        key = hashlib.md5(query.encode()).hexdigest()
        self.cache[key] = (embedding, response)

# 使用示例
cache = SemanticCache(similarity_threshold=0.95)

def cached_llm_call(prompt: str) -> str:
    # 先查缓存
    cached = cache.get(prompt)
    if cached:
        return cached
    
    # 缓存未命中，调用 LLM
    response = call_llm(prompt)
    cache.set(prompt, response)
    return response
```

### 2. Redis 缓存

```python
import redis
import json
import hashlib

class RedisCache:
    """Redis 缓存"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, ttl: int = 3600):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = ttl
    
    def _make_key(self, prompt: str, model: str) -> str:
        content = f"{model}:{prompt}"
        return f"llm_cache:{hashlib.sha256(content.encode()).hexdigest()}"
    
    def get(self, prompt: str, model: str) -> Optional[str]:
        key = self._make_key(prompt, model)
        return self.client.get(key)
    
    def set(self, prompt: str, model: str, response: str):
        key = self._make_key(prompt, model)
        self.client.setex(key, self.ttl, response)
```

## 可观测性（Observability）

### 1. 结构化日志

```python
import logging
import json
from datetime import datetime

class LLMLogger:
    """LLM 调用日志记录器"""
    
    def __init__(self):
        self.logger = logging.getLogger("llm")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_request(self, request_id: str, model: str, prompt: str, 
                    response: str, latency_ms: float, tokens: dict):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "model": model,
            "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "response_preview": response[:100] + "..." if len(response) > 100 else response,
            "latency_ms": latency_ms,
            "prompt_tokens": tokens.get("prompt_tokens", 0),
            "completion_tokens": tokens.get("completion_tokens", 0),
            "total_tokens": tokens.get("total_tokens", 0)
        }
        self.logger.info(json.dumps(log_entry, ensure_ascii=False))

llm_logger = LLMLogger()
```

### 2. 指标收集

```python
from prometheus_client import Counter, Histogram, Gauge

# 定义指标
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['model', 'status']
)

llm_latency_seconds = Histogram(
    'llm_latency_seconds',
    'LLM request latency',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total tokens used',
    ['model', 'type']  # type: prompt/completion
)

# 使用示例
def call_llm_with_metrics(prompt: str, model: str) -> str:
    import time
    start = time.time()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        latency = time.time() - start
        llm_requests_total.labels(model=model, status="success").inc()
        llm_latency_seconds.labels(model=model).observe(latency)
        llm_tokens_total.labels(model=model, type="prompt").inc(response.usage.prompt_tokens)
        llm_tokens_total.labels(model=model, type="completion").inc(response.usage.completion_tokens)
        
        return response.choices[0].message.content
    except Exception as e:
        llm_requests_total.labels(model=model, status="error").inc()
        raise
```

### 3. 分布式追踪

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# 初始化追踪
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# 添加导出器
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))

# 使用追踪
@tracer.start_as_current_span("llm_call")
def traced_llm_call(prompt: str) -> str:
    span = trace.get_current_span()
    span.set_attribute("llm.model", "gpt-4o")
    span.set_attribute("llm.prompt_length", len(prompt))
    
    response = call_llm(prompt)
    
    span.set_attribute("llm.response_length", len(response))
    return response
```

## 发布与回滚

### 1. 灰度发布

```python
class GradualRollout:
    """灰度发布管理"""
    
    def __init__(self):
        self.rollout_percentage = 0.0
        self.new_config = None
        self.old_config = None
    
    def start_rollout(self, new_config: dict, initial_percentage: float = 0.05):
        """开始灰度发布"""
        self.old_config = self.get_current_config()
        self.new_config = new_config
        self.rollout_percentage = initial_percentage
    
    def increase_rollout(self, percentage: float):
        """增加灰度比例"""
        self.rollout_percentage = min(1.0, percentage)
    
    def get_config_for_request(self, user_id: str) -> dict:
        """获取请求应使用的配置"""
        import hashlib
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        if (hash_value % 100) / 100 < self.rollout_percentage:
            return self.new_config
        return self.old_config
    
    def rollback(self):
        """回滚到旧配置"""
        self.rollout_percentage = 0.0
        self.new_config = None
```

### 2. Feature Flag

```python
class FeatureFlags:
    """功能开关管理"""
    
    def __init__(self):
        self.flags = {
            "use_new_prompt": False,
            "enable_rag": True,
            "use_gpt4": False,
            "enable_caching": True
        }
    
    def is_enabled(self, flag_name: str, user_id: str = None) -> bool:
        """检查功能是否启用"""
        if flag_name not in self.flags:
            return False
        
        flag_value = self.flags[flag_name]
        
        # 支持百分比灰度
        if isinstance(flag_value, float):
            if user_id:
                import hashlib
                hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
                return (hash_value % 100) / 100 < flag_value
            return False
        
        return flag_value
    
    def set_flag(self, flag_name: str, value):
        """设置功能开关"""
        self.flags[flag_name] = value

# 使用示例
flags = FeatureFlags()

def process_request(prompt: str, user_id: str) -> str:
    if flags.is_enabled("use_new_prompt", user_id):
        prompt = apply_new_prompt_template(prompt)
    
    if flags.is_enabled("enable_rag", user_id):
        context = retrieve_context(prompt)
        prompt = f"Context: {context}\n\nQuestion: {prompt}"
    
    model = "gpt-4o" if flags.is_enabled("use_gpt4", user_id) else "gpt-4o-mini"
    
    return call_llm(prompt, model)
```

## 成本控制

### 预算管理

```python
from datetime import datetime, timedelta
from collections import defaultdict

class BudgetManager:
    """预算管理器"""
    
    # 模型价格 (USD per 1M tokens)
    PRICES = {
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    }
    
    def __init__(self, daily_budget_usd: float = 100.0):
        self.daily_budget = daily_budget_usd
        self.usage = defaultdict(float)  # {date: cost}
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """计算单次调用成本"""
        prices = self.PRICES.get(model, {"input": 0, "output": 0})
        input_cost = (input_tokens / 1_000_000) * prices["input"]
        output_cost = (output_tokens / 1_000_000) * prices["output"]
        return input_cost + output_cost
    
    def record_usage(self, model: str, input_tokens: int, output_tokens: int):
        """记录使用量"""
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        today = datetime.now().strftime("%Y-%m-%d")
        self.usage[today] += cost
    
    def can_proceed(self) -> bool:
        """检查是否超出预算"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.usage[today] < self.daily_budget
    
    def get_remaining_budget(self) -> float:
        """获取剩余预算"""
        today = datetime.now().strftime("%Y-%m-%d")
        return max(0, self.daily_budget - self.usage[today])

budget = BudgetManager(daily_budget_usd=50.0)

def call_with_budget_check(prompt: str) -> str:
    if not budget.can_proceed():
        raise Exception("Daily budget exceeded")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    budget.record_usage(
        "gpt-4o",
        response.usage.prompt_tokens,
        response.usage.completion_tokens
    )
    
    return response.choices[0].message.content
```

## 延伸阅读

- [OpenAI 延迟优化指南](https://platform.openai.com/docs/guides/latency-optimization)
- [LangChain 生产化指南](https://python.langchain.com/docs/guides/productionization/)
- [OpenTelemetry 文档](https://opentelemetry.io/docs/)
