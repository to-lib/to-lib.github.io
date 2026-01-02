---
sidebar_position: 28
title: 📊 AI 可观测性
---

# AI 可观测性

AI 可观测性是指对 AI 应用进行监控、追踪和调试的能力，帮助理解模型行为、定位问题、优化性能。

## 可观测性三支柱

```
┌─────────────────────────────────────────────────────────┐
│                    AI 可观测性                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Tracing（追踪）                                        │
│  └─> 完整调用链路、每步耗时、Token 使用                 │
│                                                         │
│  Logging（日志）                                        │
│  └─> 输入输出记录、错误信息、调试数据                   │
│                                                         │
│  Metrics（指标）                                        │
│  └─> 延迟、成功率、成本、质量评分                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 基础实现

### 简单日志

```python
import logging
import json
from datetime import datetime
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_app")

def log_llm_call(func):
    """LLM 调用日志装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        
        # 记录输入
        logger.info(json.dumps({
            "event": "llm_call_start",
            "function": func.__name__,
            "timestamp": start_time.isoformat(),
            "kwargs": {k: str(v)[:200] for k, v in kwargs.items()}
        }, ensure_ascii=False))
        
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            
            # 记录成功
            logger.info(json.dumps({
                "event": "llm_call_success",
                "function": func.__name__,
                "duration_seconds": duration,
                "output_preview": str(result)[:200]
            }, ensure_ascii=False))
            
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            # 记录错误
            logger.error(json.dumps({
                "event": "llm_call_error",
                "function": func.__name__,
                "duration_seconds": duration,
                "error": str(e)
            }, ensure_ascii=False))
            
            raise
    
    return wrapper

@log_llm_call
def chat(message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": message}]
    )
    return response.choices[0].message.content
```

### 追踪系统

```python
import uuid
from dataclasses import dataclass, field
from typing import Any
from contextlib import contextmanager

@dataclass
class Span:
    """追踪 Span"""
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: str | None = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    attributes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    
    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: dict = None):
        self.events.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {}
        })
    
    def end(self):
        self.end_time = datetime.now()
    
    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0

class Tracer:
    """简单追踪器"""
    
    def __init__(self):
        self.spans: list[Span] = []
        self._current_trace_id: str | None = None
        self._current_span: Span | None = None
    
    @contextmanager
    def start_trace(self, name: str):
        """开始新的追踪"""
        self._current_trace_id = str(uuid.uuid4())
        span = Span(name=name, trace_id=self._current_trace_id)
        self._current_span = span
        
        try:
            yield span
        finally:
            span.end()
            self.spans.append(span)
            self._current_span = None
    
    @contextmanager
    def start_span(self, name: str):
        """开始子 Span"""
        parent = self._current_span
        span = Span(
            name=name,
            trace_id=self._current_trace_id,
            parent_id=parent.span_id if parent else None
        )
        self._current_span = span
        
        try:
            yield span
        finally:
            span.end()
            self.spans.append(span)
            self._current_span = parent
    
    def export(self) -> list[dict]:
        """导出追踪数据"""
        return [
            {
                "name": s.name,
                "trace_id": s.trace_id,
                "span_id": s.span_id,
                "parent_id": s.parent_id,
                "duration_ms": s.duration_ms,
                "attributes": s.attributes,
                "events": s.events
            }
            for s in self.spans
        ]

tracer = Tracer()

# 使用示例
def rag_query(question: str) -> str:
    with tracer.start_trace("rag_query") as trace:
        trace.set_attribute("question", question)
        
        # 检索
        with tracer.start_span("retrieval") as span:
            docs = retrieve_documents(question)
            span.set_attribute("num_docs", len(docs))
        
        # 生成
        with tracer.start_span("generation") as span:
            response = generate_answer(question, docs)
            span.set_attribute("response_length", len(response))
        
        return response
```


## LangSmith

LangSmith 是 LangChain 官方的可观测性平台。

### 配置

```bash
pip install langsmith
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_api_key
export LANGCHAIN_PROJECT=my_project
```

### 自动追踪

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 配置后自动追踪所有 LangChain 调用
llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手。"),
    ("user", "{input}")
])

chain = prompt | llm
response = chain.invoke({"input": "你好"})
# 自动记录到 LangSmith
```

### 手动追踪

```python
from langsmith import traceable

@traceable(name="my_function")
def process_query(query: str) -> str:
    # 函数内的所有 LLM 调用都会被追踪
    response = llm.invoke(query)
    return response.content

@traceable(run_type="retriever")
def search_documents(query: str) -> list:
    # 标记为检索器类型
    return vector_store.similarity_search(query)
```

### 评估

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# 创建数据集
dataset = client.create_dataset("qa_dataset")
client.create_examples(
    inputs=[{"question": "什么是 RAG？"}],
    outputs=[{"answer": "RAG 是检索增强生成..."}],
    dataset_id=dataset.id
)

# 定义评估函数
def correctness(run, example):
    # 比较预测和参考答案
    prediction = run.outputs["output"]
    reference = example.outputs["answer"]
    # 返回评分
    return {"score": 0.8}

# 运行评估
results = evaluate(
    lambda x: chain.invoke(x),
    data=dataset.name,
    evaluators=[correctness]
)
```

## OpenTelemetry 集成

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# 配置 OpenTelemetry
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

def llm_call_with_otel(prompt: str) -> str:
    with tracer.start_as_current_span("llm_call") as span:
        span.set_attribute("prompt", prompt[:100])
        span.set_attribute("model", "gpt-4o")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.choices[0].message.content
        span.set_attribute("response_length", len(result))
        span.set_attribute("tokens_used", response.usage.total_tokens)
        
        return result
```

## 指标收集

```python
from dataclasses import dataclass, field
from collections import defaultdict
import time

@dataclass
class MetricsCollector:
    """指标收集器"""
    
    latencies: list = field(default_factory=list)
    token_counts: list = field(default_factory=list)
    error_counts: dict = field(default_factory=lambda: defaultdict(int))
    success_count: int = 0
    total_cost: float = 0.0
    
    def record_latency(self, latency_ms: float):
        self.latencies.append(latency_ms)
    
    def record_tokens(self, input_tokens: int, output_tokens: int):
        self.token_counts.append({
            "input": input_tokens,
            "output": output_tokens
        })
    
    def record_error(self, error_type: str):
        self.error_counts[error_type] += 1
    
    def record_success(self):
        self.success_count += 1
    
    def record_cost(self, cost: float):
        self.total_cost += cost
    
    def get_stats(self) -> dict:
        total_requests = self.success_count + sum(self.error_counts.values())
        
        return {
            "total_requests": total_requests,
            "success_rate": self.success_count / total_requests if total_requests > 0 else 0,
            "avg_latency_ms": sum(self.latencies) / len(self.latencies) if self.latencies else 0,
            "p95_latency_ms": sorted(self.latencies)[int(len(self.latencies) * 0.95)] if self.latencies else 0,
            "total_tokens": sum(t["input"] + t["output"] for t in self.token_counts),
            "total_cost": self.total_cost,
            "error_breakdown": dict(self.error_counts)
        }

metrics = MetricsCollector()

def monitored_chat(message: str) -> str:
    start = time.time()
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": message}]
        )
        
        latency = (time.time() - start) * 1000
        metrics.record_latency(latency)
        metrics.record_tokens(
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )
        metrics.record_success()
        
        return response.choices[0].message.content
        
    except Exception as e:
        metrics.record_error(type(e).__name__)
        raise
```


## 质量监控

```python
class QualityMonitor:
    """质量监控"""
    
    def __init__(self):
        self.client = OpenAI()
        self.scores = []
    
    def evaluate_response(
        self,
        question: str,
        response: str,
        context: str = ""
    ) -> dict:
        """评估回复质量"""
        eval_prompt = f"""
评估以下 AI 回复的质量（1-5 分）：

问题：{question}
{"上下文：" + context if context else ""}
回复：{response}

评估维度：
1. 相关性：回复是否切题
2. 准确性：信息是否正确
3. 完整性：是否完整回答问题
4. 清晰度：表达是否清晰

返回 JSON：{{"relevance": 1-5, "accuracy": 1-5, "completeness": 1-5, "clarity": 1-5}}
"""
        
        eval_response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": eval_prompt}],
            response_format={"type": "json_object"}
        )
        
        scores = json.loads(eval_response.choices[0].message.content)
        scores["overall"] = sum(scores.values()) / len(scores)
        self.scores.append(scores)
        
        return scores
    
    def get_average_scores(self) -> dict:
        if not self.scores:
            return {}
        
        keys = self.scores[0].keys()
        return {
            k: sum(s[k] for s in self.scores) / len(self.scores)
            for k in keys
        }
```

## 告警系统

```python
from enum import Enum

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertSystem:
    """告警系统"""
    
    def __init__(self):
        self.thresholds = {
            "latency_p95_ms": 5000,
            "error_rate": 0.05,
            "cost_daily": 100.0
        }
        self.handlers = []
    
    def add_handler(self, handler):
        self.handlers.append(handler)
    
    def check(self, metrics: dict):
        """检查指标并触发告警"""
        alerts = []
        
        # 延迟告警
        if metrics.get("p95_latency_ms", 0) > self.thresholds["latency_p95_ms"]:
            alerts.append({
                "level": AlertLevel.WARNING,
                "message": f"P95 延迟过高：{metrics['p95_latency_ms']:.0f}ms"
            })
        
        # 错误率告警
        error_rate = 1 - metrics.get("success_rate", 1)
        if error_rate > self.thresholds["error_rate"]:
            alerts.append({
                "level": AlertLevel.CRITICAL,
                "message": f"错误率过高：{error_rate:.2%}"
            })
        
        # 成本告警
        if metrics.get("total_cost", 0) > self.thresholds["cost_daily"]:
            alerts.append({
                "level": AlertLevel.WARNING,
                "message": f"日成本超限：${metrics['total_cost']:.2f}"
            })
        
        # 触发处理器
        for alert in alerts:
            for handler in self.handlers:
                handler(alert)
        
        return alerts

# 使用
alert_system = AlertSystem()
alert_system.add_handler(lambda a: print(f"[{a['level'].value}] {a['message']}"))
```

## 可观测性平台对比

| 平台 | 特点 | 适用场景 |
|------|------|---------|
| LangSmith | LangChain 原生支持 | LangChain 项目 |
| Arize | 强大的数据分析 | 生产环境监控 |
| Weights & Biases | ML 实验追踪 | 模型训练 |
| Helicone | 简单易用 | 快速集成 |
| OpenTelemetry | 标准化、可扩展 | 企业级应用 |

## 最佳实践

1. **全链路追踪**：记录从输入到输出的完整链路
2. **采样策略**：高流量时使用采样减少开销
3. **敏感信息脱敏**：日志中不记录敏感数据
4. **实时告警**：关键指标异常时及时通知
5. **定期回顾**：分析历史数据优化系统

## 延伸阅读

- [LangSmith 文档](https://docs.smith.langchain.com/)
- [OpenTelemetry](https://opentelemetry.io/)
- [Arize AI](https://arize.com/)