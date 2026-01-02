---
sidebar_position: 10
title: 📏 Evaluation（评估与测试）
---

# Evaluation（评估与测试）

LLM 应用的难点之一是"看起来能用"，但线上表现不稳定。一个可持续迭代的 AI 系统，需要把评估当成工程能力：可复现、可对比、可回归。

## 评估目标

| 维度         | 说明                               |
| ------------ | ---------------------------------- |
| **正确性**   | 答案是否符合事实/业务规则          |
| **相关性**   | 回答是否切中用户问题               |
| **完整性**   | 关键点是否遗漏                     |
| **安全性**   | 是否泄露敏感信息、是否遵守策略     |
| **格式/结构** | JSON/表格/字段是否符合约束         |
| **成本与延迟** | 是否满足 SLA                       |

## 评估体系架构

```
┌─────────────────────────────────────────────────────────┐
│                    评估体系                              │
├─────────────────────────────────────────────────────────┤
│  离线评估 (Offline)          │  在线评估 (Online)        │
│  ├─ 黄金数据集               │  ├─ A/B 测试              │
│  ├─ 自动化评分               │  ├─ 业务指标监控          │
│  ├─ LLM-as-Judge            │  ├─ 用户反馈收集          │
│  └─ 人工抽检                 │  └─ 异常检测              │
└─────────────────────────────────────────────────────────┘
```

## 离线评估（Offline Eval）

### 1. 构建黄金数据集（Golden Set）

```python
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class EvalSample:
    """评估样本"""
    id: str
    input: str
    expected_output: str
    context: Optional[str] = None  # RAG 场景的检索结果
    category: str = "general"
    difficulty: str = "medium"
    
    def to_dict(self):
        return {
            "id": self.id,
            "input": self.input,
            "expected_output": self.expected_output,
            "context": self.context,
            "category": self.category,
            "difficulty": self.difficulty
        }

# 创建评估数据集
eval_dataset = [
    EvalSample(
        id="001",
        input="什么是 RAG？",
        expected_output="RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的 AI 技术...",
        category="concept",
        difficulty="easy"
    ),
    EvalSample(
        id="002",
        input="如何优化 LLM 的响应延迟？",
        expected_output="优化 LLM 响应延迟的方法包括：1. 使用更小的模型...",
        category="optimization",
        difficulty="hard"
    )
]

# 保存为 JSONL
def save_eval_dataset(samples: list[EvalSample], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
```

### 2. 自动化评分

#### 规则/断言评分

```python
import json
import re

class RuleBasedEvaluator:
    """基于规则的评估器"""
    
    @staticmethod
    def check_json_valid(output: str) -> bool:
        """检查 JSON 格式是否有效"""
        try:
            json.loads(output)
            return True
        except json.JSONDecodeError:
            return False
    
    @staticmethod
    def check_contains_keywords(output: str, keywords: list[str]) -> float:
        """检查是否包含关键词"""
        found = sum(1 for kw in keywords if kw.lower() in output.lower())
        return found / len(keywords) if keywords else 0
    
    @staticmethod
    def check_length(output: str, min_len: int = 10, max_len: int = 1000) -> bool:
        """检查长度是否在范围内"""
        return min_len <= len(output) <= max_len
    
    @staticmethod
    def check_no_sensitive_info(output: str) -> bool:
        """检查是否包含敏感信息"""
        patterns = [
            r'\b\d{11}\b',  # 手机号
            r'\b\d{18}\b',  # 身份证号
            r'sk-[a-zA-Z0-9]+',  # API Key
        ]
        for pattern in patterns:
            if re.search(pattern, output):
                return False
        return True

# 使用示例
evaluator = RuleBasedEvaluator()
output = '{"result": "success", "data": [1, 2, 3]}'

results = {
    "json_valid": evaluator.check_json_valid(output),
    "length_ok": evaluator.check_length(output),
    "no_sensitive": evaluator.check_no_sensitive_info(output)
}
```

#### 指标计算

```python
from collections import Counter
import numpy as np

def calculate_accuracy(predictions: list[str], labels: list[str]) -> float:
    """计算准确率"""
    correct = sum(1 for p, l in zip(predictions, labels) if p.strip() == l.strip())
    return correct / len(predictions)

def calculate_f1(predictions: list[str], labels: list[str], positive_label: str) -> dict:
    """计算 F1 分数"""
    tp = sum(1 for p, l in zip(predictions, labels) if p == positive_label and l == positive_label)
    fp = sum(1 for p, l in zip(predictions, labels) if p == positive_label and l != positive_label)
    fn = sum(1 for p, l in zip(predictions, labels) if p != positive_label and l == positive_label)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}

def calculate_retrieval_metrics(retrieved_ids: list[list[str]], relevant_ids: list[list[str]], k: int = 5) -> dict:
    """计算检索指标"""
    recall_at_k = []
    mrr = []
    
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        # Recall@K
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        recall = len(retrieved_k & relevant_set) / len(relevant_set) if relevant_set else 0
        recall_at_k.append(recall)
        
        # MRR
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                mrr.append(1 / (i + 1))
                break
        else:
            mrr.append(0)
    
    return {
        f"recall@{k}": np.mean(recall_at_k),
        "mrr": np.mean(mrr)
    }
```

### 3. LLM-as-Judge

```python
from openai import OpenAI

client = OpenAI()

def llm_judge(question: str, answer: str, reference: str, criteria: str = "accuracy") -> dict:
    """使用 LLM 作为评判者"""
    
    judge_prompt = f"""你是一个专业的评估专家。请根据以下标准评估回答的质量。

评估标准：{criteria}

问题：{question}

参考答案：{reference}

待评估答案：{answer}

请按以下格式输出评估结果：
1. 分数 (1-5)：
2. 理由：
3. 改进建议：

只输出 JSON 格式：
{{"score": <1-5>, "reason": "<理由>", "suggestion": "<建议>"}}
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

# 批量评估
def batch_evaluate(samples: list[dict], judge_model: str = "gpt-4o") -> list[dict]:
    """批量评估"""
    results = []
    for sample in samples:
        result = llm_judge(
            question=sample["input"],
            answer=sample["output"],
            reference=sample["expected"]
        )
        results.append({
            "id": sample["id"],
            **result
        })
    return results
```

### 4. RAG 专项评估

```python
def evaluate_rag_faithfulness(answer: str, context: str) -> dict:
    """评估 RAG 回答的忠实度（是否基于检索内容）"""
    
    prompt = f"""评估以下回答是否忠实于给定的上下文。

上下文：
{context}

回答：
{answer}

评估标准：
- 回答中的所有事实是否都能在上下文中找到依据
- 是否有编造或臆测的内容

输出 JSON：{{"faithfulness_score": <0-1>, "unsupported_claims": ["<不支持的声明>"]}}
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

def evaluate_rag_relevancy(question: str, answer: str) -> dict:
    """评估回答与问题的相关性"""
    
    prompt = f"""评估以下回答与问题的相关性。

问题：{question}

回答：{answer}

评估标准：
- 回答是否直接回应了问题
- 是否有无关的内容

输出 JSON：{{"relevancy_score": <0-1>, "irrelevant_parts": ["<无关部分>"]}}
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)
```

## 在线评估（Online Eval）

### A/B 测试框架

```python
import random
import hashlib
from datetime import datetime

class ABTestManager:
    """A/B 测试管理器"""
    
    def __init__(self):
        self.experiments = {}
    
    def create_experiment(self, name: str, variants: dict[str, float]):
        """创建实验，variants 为变体名称和流量比例"""
        self.experiments[name] = {
            "variants": variants,
            "created_at": datetime.now(),
            "metrics": {v: [] for v in variants}
        }
    
    def get_variant(self, experiment_name: str, user_id: str) -> str:
        """根据用户 ID 分配变体（确保同一用户始终分到同一组）"""
        exp = self.experiments[experiment_name]
        
        # 使用 hash 确保分配一致性
        hash_value = int(hashlib.md5(f"{experiment_name}:{user_id}".encode()).hexdigest(), 16)
        random_value = (hash_value % 10000) / 10000
        
        cumulative = 0
        for variant, ratio in exp["variants"].items():
            cumulative += ratio
            if random_value < cumulative:
                return variant
        
        return list(exp["variants"].keys())[-1]
    
    def record_metric(self, experiment_name: str, variant: str, metric_name: str, value: float):
        """记录指标"""
        self.experiments[experiment_name]["metrics"][variant].append({
            "metric": metric_name,
            "value": value,
            "timestamp": datetime.now()
        })

# 使用示例
ab_manager = ABTestManager()
ab_manager.create_experiment("prompt_v2", {
    "control": 0.5,    # 50% 使用旧 prompt
    "treatment": 0.5   # 50% 使用新 prompt
})

# 获取用户分组
variant = ab_manager.get_variant("prompt_v2", user_id="user123")
```

### 业务指标监控

```python
from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class RequestMetrics:
    """请求指标"""
    request_id: str
    user_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    success: bool
    error_type: Optional[str] = None
    user_rating: Optional[int] = None  # 1-5
    timestamp: float = field(default_factory=time.time)

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.metrics: list[RequestMetrics] = []
    
    def record(self, metrics: RequestMetrics):
        self.metrics.append(metrics)
    
    def get_summary(self, time_window_hours: int = 24) -> dict:
        """获取指标摘要"""
        cutoff = time.time() - time_window_hours * 3600
        recent = [m for m in self.metrics if m.timestamp > cutoff]
        
        if not recent:
            return {}
        
        return {
            "total_requests": len(recent),
            "success_rate": sum(1 for m in recent if m.success) / len(recent),
            "avg_latency_ms": sum(m.latency_ms for m in recent) / len(recent),
            "p95_latency_ms": sorted([m.latency_ms for m in recent])[int(len(recent) * 0.95)],
            "avg_tokens": sum(m.prompt_tokens + m.completion_tokens for m in recent) / len(recent),
            "avg_rating": sum(m.user_rating for m in recent if m.user_rating) / 
                         sum(1 for m in recent if m.user_rating) if any(m.user_rating for m in recent) else None
        }
```

## 评估工具推荐

### LangSmith

```python
from langsmith import Client
from langsmith.evaluation import evaluate

# 初始化客户端
client = Client()

# 创建数据集
dataset = client.create_dataset("my-eval-dataset")

# 添加样本
client.create_examples(
    inputs=[{"question": "什么是 RAG？"}],
    outputs=[{"answer": "RAG 是检索增强生成..."}],
    dataset_id=dataset.id
)

# 定义评估函数
def my_evaluator(run, example):
    prediction = run.outputs["answer"]
    reference = example.outputs["answer"]
    # 自定义评估逻辑
    score = calculate_similarity(prediction, reference)
    return {"score": score}

# 运行评估
results = evaluate(
    lambda inputs: my_llm_app(inputs["question"]),
    data=dataset.name,
    evaluators=[my_evaluator]
)
```

### DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

# 创建测试用例
test_case = LLMTestCase(
    input="什么是机器学习？",
    actual_output="机器学习是人工智能的一个分支...",
    retrieval_context=["机器学习是让计算机从数据中学习的技术..."]
)

# 定义指标
relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
faithfulness_metric = FaithfulnessMetric(threshold=0.7)

# 运行评估
evaluate([test_case], [relevancy_metric, faithfulness_metric])
```

### RAGAS

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

# 准备数据
data = {
    "question": ["什么是 RAG？"],
    "answer": ["RAG 是检索增强生成技术..."],
    "contexts": [["RAG 全称 Retrieval-Augmented Generation..."]],
    "ground_truth": ["RAG 是一种结合检索和生成的 AI 技术"]
}

dataset = Dataset.from_dict(data)

# 评估
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)

print(results)
```

## 最小可行评估体系（MVP）

### 第一阶段：基础评估

```python
# 1. 10-20 条黄金样本
golden_set = load_golden_set("golden_samples.jsonl")

# 2. 3 个硬指标
def basic_eval(model_output: str, expected: str) -> dict:
    return {
        "format_correct": check_format(model_output),
        "latency_ms": measure_latency(),
        "cost_usd": calculate_cost()
    }

# 3. 人工 spot-check
def spot_check(samples: list, n: int = 20):
    """随机抽取 n 条进行人工检查"""
    import random
    return random.sample(samples, min(n, len(samples)))
```

### 第二阶段：自动化评估

```python
# 添加 LLM-as-Judge
# 添加 CI/CD 集成
# 添加回归测试
```

### 第三阶段：在线评估

```python
# 添加 A/B 测试
# 添加业务指标监控
# 添加异常检测
```

## 延伸阅读

- [LangSmith 文档](https://docs.smith.langchain.com/)
- [DeepEval 文档](https://github.com/confident-ai/deepeval)
- [RAGAS 文档](https://docs.ragas.io/)
- [OpenAI Evals](https://github.com/openai/evals)
