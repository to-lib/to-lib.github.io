---
sidebar_position: 34
title: 🔒 联邦学习
---

# 联邦学习

联邦学习（Federated Learning）是一种分布式机器学习方法，允许多方在不共享原始数据的情况下协作训练模型，保护数据隐私。

## 为什么需要联邦学习？

```
传统机器学习：
数据方 A ──┐
数据方 B ──┼──> 中心服务器（汇集所有数据）──> 训练模型
数据方 C ──┘
           ⚠️ 隐私风险

联邦学习：
数据方 A ──> 本地训练 ──> 模型更新 ──┐
数据方 B ──> 本地训练 ──> 模型更新 ──┼──> 聚合服务器 ──> 全局模型
数据方 C ──> 本地训练 ──> 模型更新 ──┘
                                    ✅ 数据不出域
```

## 联邦学习类型

| 类型 | 特点 | 适用场景 |
|------|------|---------|
| 横向联邦 | 样本不同，特征相同 | 多家医院的病历数据 |
| 纵向联邦 | 样本相同，特征不同 | 银行+电商的用户数据 |
| 联邦迁移 | 样本和特征都不同 | 跨领域协作 |

## 基础实现

### 联邦平均算法 (FedAvg)

```python
import torch
import torch.nn as nn
from typing import List, Dict
import copy

class FederatedServer:
    """联邦学习服务器"""
    
    def __init__(self, model: nn.Module):
        self.global_model = model
        self.client_updates = []
    
    def distribute_model(self) -> Dict:
        """分发全局模型"""
        return copy.deepcopy(self.global_model.state_dict())
    
    def receive_update(self, client_update: Dict, num_samples: int):
        """接收客户端更新"""
        self.client_updates.append({
            "weights": client_update,
            "num_samples": num_samples
        })
    
    def aggregate(self):
        """聚合客户端更新（FedAvg）"""
        if not self.client_updates:
            return
        
        total_samples = sum(u["num_samples"] for u in self.client_updates)
        
        # 加权平均
        new_weights = {}
        for key in self.client_updates[0]["weights"].keys():
            weighted_sum = sum(
                u["weights"][key] * u["num_samples"]
                for u in self.client_updates
            )
            new_weights[key] = weighted_sum / total_samples
        
        self.global_model.load_state_dict(new_weights)
        self.client_updates = []

class FederatedClient:
    """联邦学习客户端"""
    
    def __init__(self, client_id: str, local_data, model: nn.Module):
        self.client_id = client_id
        self.local_data = local_data
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
    
    def receive_model(self, global_weights: Dict):
        """接收全局模型"""
        self.model.load_state_dict(global_weights)
    
    def local_train(self, epochs: int = 5) -> Dict:
        """本地训练"""
        self.model.train()
        
        for epoch in range(epochs):
            for batch_x, batch_y in self.local_data:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
        
        return self.model.state_dict()
    
    def get_num_samples(self) -> int:
        return len(self.local_data.dataset)

def federated_training(
    server: FederatedServer,
    clients: List[FederatedClient],
    rounds: int = 10,
    local_epochs: int = 5
):
    """联邦训练主循环"""
    for round_num in range(rounds):
        print(f"Round {round_num + 1}/{rounds}")
        
        # 1. 分发全局模型
        global_weights = server.distribute_model()
        
        # 2. 客户端本地训练
        for client in clients:
            client.receive_model(global_weights)
            local_weights = client.local_train(epochs=local_epochs)
            server.receive_update(local_weights, client.get_num_samples())
        
        # 3. 聚合更新
        server.aggregate()
        
        print(f"Round {round_num + 1} completed")
```

## 差分隐私

添加噪声保护模型更新。

```python
import numpy as np

class DifferentialPrivacy:
    """差分隐私"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
    
    def add_noise(self, gradients: Dict, sensitivity: float = 1.0) -> Dict:
        """添加高斯噪声"""
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
        noisy_gradients = {}
        for key, value in gradients.items():
            noise = torch.normal(0, sigma, size=value.shape)
            noisy_gradients[key] = value + noise
        
        return noisy_gradients
    
    def clip_gradients(self, gradients: Dict, max_norm: float = 1.0) -> Dict:
        """梯度裁剪"""
        total_norm = 0
        for grad in gradients.values():
            total_norm += grad.norm() ** 2
        total_norm = total_norm ** 0.5
        
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for key in gradients:
                gradients[key] = gradients[key] * clip_coef
        
        return gradients

class PrivateFederatedClient(FederatedClient):
    """带差分隐私的客户端"""
    
    def __init__(self, *args, epsilon: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp = DifferentialPrivacy(epsilon=epsilon)
    
    def local_train(self, epochs: int = 5) -> Dict:
        weights = super().local_train(epochs)
        
        # 计算更新差值
        global_weights = self.model.state_dict()
        updates = {k: weights[k] - global_weights[k] for k in weights}
        
        # 裁剪和加噪
        updates = self.dp.clip_gradients(updates)
        updates = self.dp.add_noise(updates)
        
        # 返回加噪后的权重
        return {k: global_weights[k] + updates[k] for k in weights}
```

## 安全聚合

防止服务器看到单个客户端的更新。

```python
import secrets
from typing import Tuple

class SecureAggregation:
    """安全聚合"""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.masks = {}
    
    def generate_masks(self, client_ids: List[str]) -> Dict[str, Dict]:
        """生成成对掩码"""
        masks = {cid: {} for cid in client_ids}
        
        for i, cid1 in enumerate(client_ids):
            for cid2 in client_ids[i+1:]:
                # 生成随机掩码
                seed = secrets.randbits(256)
                mask = self._generate_mask_from_seed(seed)
                
                masks[cid1][cid2] = mask
                masks[cid2][cid1] = -mask  # 相反的掩码
        
        return masks
    
    def _generate_mask_from_seed(self, seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randn(1000)  # 根据模型大小调整
    
    def mask_update(self, update: Dict, masks: Dict) -> Dict:
        """应用掩码"""
        total_mask = sum(masks.values())
        masked = {}
        for key, value in update.items():
            masked[key] = value + total_mask[:value.numel()].reshape(value.shape)
        return masked
```


## Flower 框架

Flower 是流行的联邦学习框架。

```bash
pip install flwr
```

### 服务端

```python
import flwr as fl

# 定义聚合策略
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.5,           # 每轮参与训练的客户端比例
    fraction_evaluate=0.5,      # 每轮参与评估的客户端比例
    min_fit_clients=2,          # 最少训练客户端数
    min_evaluate_clients=2,     # 最少评估客户端数
    min_available_clients=2,    # 最少可用客户端数
)

# 启动服务器
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)
```

### 客户端

```python
import flwr as fl
import torch

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.parameters()]
    
    def set_parameters(self, parameters):
        for param, new_val in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_val)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=5)
        return self.get_parameters(config), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), len(self.testloader.dataset), {"accuracy": accuracy}

# 启动客户端
fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=FlowerClient(model, trainloader, testloader)
)
```

## LLM 联邦微调

```python
class FederatedLLMClient:
    """LLM 联邦微调客户端"""
    
    def __init__(self, model_name: str, local_data_path: str):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 只训练 LoRA 参数
        lora_config = LoraConfig(r=8, lora_alpha=16)
        self.model = get_peft_model(self.model, lora_config)
        
        self.local_data = self._load_data(local_data_path)
    
    def get_lora_parameters(self) -> Dict:
        """只返回 LoRA 参数"""
        return {
            k: v for k, v in self.model.state_dict().items()
            if "lora" in k
        }
    
    def set_lora_parameters(self, parameters: Dict):
        """设置 LoRA 参数"""
        state_dict = self.model.state_dict()
        state_dict.update(parameters)
        self.model.load_state_dict(state_dict)
    
    def local_train(self, epochs: int = 1):
        """本地微调"""
        # 训练逻辑...
        pass
```

## 应用场景

| 场景 | 说明 |
|------|------|
| 医疗 | 多家医院协作训练诊断模型 |
| 金融 | 银行间反欺诈模型训练 |
| 移动设备 | 手机键盘预测、语音识别 |
| 企业协作 | 跨公司数据协作 |

## 最佳实践

1. **通信效率**：压缩模型更新减少通信
2. **异构处理**：处理客户端数据不均衡
3. **隐私保护**：结合差分隐私和安全聚合
4. **容错机制**：处理客户端掉线
5. **模型验证**：防止恶意客户端攻击

## 延伸阅读

- [Flower](https://flower.dev/)
- [PySyft](https://github.com/OpenMined/PySyft)
- [TensorFlow Federated](https://www.tensorflow.org/federated)