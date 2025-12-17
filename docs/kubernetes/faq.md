---
sidebar_position: 12
title: 常见问题
description: Kubernetes 常见问题与解决方案
---

# Kubernetes 常见问题

## Pod 问题

### Pod 一直处于 Pending 状态

**可能原因**：

- 资源不足（CPU/内存）
- 没有符合条件的节点
- PVC 未绑定

**排查方法**：

```bash
kubectl describe pod <pod-name>
kubectl get events --sort-by='.lastTimestamp'
```

### Pod 一直处于 CrashLoopBackOff

**可能原因**：

- 应用启动失败
- 配置错误
- 健康检查失败

**排查方法**：

```bash
kubectl logs <pod-name> --previous
kubectl describe pod <pod-name>
```

### ImagePullBackOff

**可能原因**：

- 镜像不存在
- 仓库认证失败
- 网络问题

**解决方法**：

```bash
# 检查镜像名称
kubectl describe pod <pod-name>

# 添加镜像拉取凭证
kubectl create secret docker-registry my-registry \
  --docker-server=registry.example.com \
  --docker-username=user \
  --docker-password=pass
```

## 网络问题

### Pod 无法访问 Service

```bash
# 检查 Service
kubectl get svc
kubectl get endpoints <service-name>

# DNS 测试
kubectl run dns-test --image=busybox -it --rm -- nslookup <service-name>
```

### 无法从外部访问

1. 检查 Service 类型（NodePort/LoadBalancer）
2. 检查防火墙规则
3. 检查 Ingress 配置

## 存储问题

### PVC 一直处于 Pending

```bash
# 检查 StorageClass
kubectl get storageclass

# 查看 PVC 事件
kubectl describe pvc <pvc-name>
```

## 资源限制

### OOMKilled

容器内存超出限制被杀死。

**解决方法**：增加内存限制或优化应用内存使用。

```yaml
resources:
  limits:
    memory: "512Mi" # 增加限制
```

## 调试技巧

```bash
# 查看事件
kubectl get events -A --sort-by='.lastTimestamp'

# 进入临时调试容器
kubectl debug <pod-name> -it --image=busybox

# 网络调试
kubectl run netshoot --image=nicolaka/netshoot -it --rm
```
