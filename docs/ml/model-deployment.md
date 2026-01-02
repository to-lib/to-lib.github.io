---
sidebar_position: 17
title: 🚀 模型部署
---

# 模型部署 (MLOps)

将训练好的模型部署到生产环境，提供实时推理服务。

## 模型保存与加载

### scikit-learn

```python
import joblib

# 保存
joblib.dump(model, 'model.joblib')

# 加载
model = joblib.load('model.joblib')
```

### PyTorch

```python
import torch

# 保存模型权重
torch.save(model.state_dict(), 'model.pth')

# 加载
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# TorchScript (生产推荐)
scripted = torch.jit.script(model)
scripted.save('model_scripted.pt')
```

### ONNX (跨框架)

```python
import torch.onnx

# PyTorch → ONNX
dummy_input = torch.randn(1, 10)
torch.onnx.export(model, dummy_input, 'model.onnx')

# 使用 ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession('model.onnx')
result = session.run(None, {'input': input_data})
```

## FastAPI 服务

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('model.joblib')

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    prediction: float
    probability: list[float] = None

@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    X = [request.features]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0].tolist() if hasattr(model, 'predict_proba') else None
    return PredictResponse(prediction=pred, probability=proba)

# 运行: uvicorn app:app --host 0.0.0.0 --port 8000
```

## Docker 容器化

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.joblib .
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# 构建并运行
docker build -t ml-service .
docker run -p 8000:8000 ml-service
```

## 批量预测

```python
import pandas as pd
from multiprocessing import Pool

def predict_batch(df, model_path, batch_size=1000):
    model = joblib.load(model_path)
    results = []

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        preds = model.predict(batch)
        results.extend(preds)

    return results

# 并行批处理
def parallel_predict(df, n_workers=4):
    chunks = np.array_split(df, n_workers)
    with Pool(n_workers) as p:
        results = p.map(predict_chunk, chunks)
    return np.concatenate(results)
```

## 性能优化

### 模型量化

```python
# PyTorch 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### 模型蒸馏

```python
# 用大模型指导训练小模型
def distillation_loss(student_logits, teacher_logits, labels, T=3, alpha=0.5):
    soft_loss = nn.KLDivLoss()(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1)
    ) * T * T
    hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

## 监控与告警

```python
from prometheus_client import Counter, Histogram, start_http_server

# 定义指标
prediction_counter = Counter('predictions_total', '预测总数')
latency_histogram = Histogram('prediction_latency_seconds', '预测延迟')

@app.post('/predict')
def predict(request: PredictRequest):
    with latency_histogram.time():
        result = model.predict([request.features])
    prediction_counter.inc()
    return {'prediction': result[0]}

# 启动 Prometheus 指标端点
start_http_server(8001)
```

## A/B 测试

```python
import random

models = {
    'A': joblib.load('model_a.joblib'),
    'B': joblib.load('model_b.joblib')
}

@app.post('/predict')
def predict(request: PredictRequest, user_id: str):
    # 根据用户 ID 分流
    variant = 'A' if hash(user_id) % 100 < 50 else 'B'
    model = models[variant]
    result = model.predict([request.features])[0]

    # 记录实验数据
    log_experiment(user_id, variant, result)

    return {'prediction': result, 'variant': variant}
```

## MLOps 工具

| 工具             | 用途               |
| ---------------- | ------------------ |
| MLflow           | 实验追踪、模型注册 |
| DVC              | 数据版本控制       |
| Kubeflow         | K8s 上的 ML 流水线 |
| BentoML          | 模型打包部署       |
| Seldon           | K8s 模型服务       |
| Weights & Biases | 实验追踪可视化     |
