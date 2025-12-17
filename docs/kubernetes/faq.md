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

## 安全问题

### 如何限制容器权限？

```yaml
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
  containers:
    - name: app
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop:
            - ALL
```

### Secret 如何安全管理？

1. **启用 etcd 加密**：配置 EncryptionConfiguration
2. **使用外部 Secret 管理器**：Vault、AWS Secrets Manager
3. **限制 Secret 访问权限**：RBAC 最小权限原则
4. **避免环境变量传递**：使用 Volume 挂载

## 性能问题

### 如何优化 Pod 启动速度？

1. **使用较小的基础镜像**：Alpine、Distroless
2. **预拉取镜像**：使用 DaemonSet 预热镜像
3. **优化健康检查**：合理设置 initialDelaySeconds
4. **减少 Init Container**：合并初始化逻辑

### 如何处理节点资源压力？

```bash
# 查看节点资源状况
kubectl describe node <node-name> | grep -A 10 "Conditions"

# 查找资源消耗大的 Pod
kubectl top pods -A --sort-by=memory | head -10

# 驱逐节点上的 Pod
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data
```

## Helm 问题

### Helm install 失败如何排查？

```bash
# 查看 Release 状态
helm status my-release

# 查看安装历史
helm history my-release

# 使用 --debug 获取详细信息
helm install my-release ./chart --dry-run --debug

# 模板渲染测试
helm template my-release ./chart
```

### 如何回滚 Helm Release？

```bash
# 查看历史版本
helm history my-release

# 回滚到指定版本
helm rollback my-release 2

# 回滚并等待完成
helm rollback my-release 2 --wait
```

## 集群升级

### 如何安全升级 Kubernetes 集群？

1. **阅读发布说明**：了解废弃 API 和破坏性变更
2. **备份 etcd**：`etcdctl snapshot save backup.db`
3. **升级控制平面**：`kubeadm upgrade apply v1.x.x`
4. **升级工作节点**：逐个 drain 后升级
5. **验证集群状态**：`kubectl get nodes`

### API 废弃如何处理？

```bash
# 检查废弃 API 使用情况
kubectl get --raw /metrics | grep apiserver_requested_deprecated_apis

# 使用 kubent 工具检查
kubent

# 更新资源版本
kubectl convert -f old.yaml --output-version apps/v1 > new.yaml
```

## 云平台特定问题

### EKS 节点无法加入集群？

1. 检查 IAM 角色权限
2. 验证安全组规则
3. 检查 aws-auth ConfigMap

```bash
kubectl get configmap aws-auth -n kube-system -o yaml
```

### AKS Pod 无法拉取 ACR 镜像？

```bash
# 检查 ACR 集成
az aks check-acr --name myAKSCluster --resource-group myResourceGroup --acr myACR.azurecr.io

# 手动附加 ACR
az aks update -n myAKSCluster -g myResourceGroup --attach-acr myACR
```

### GKE 负载均衡器未创建？

1. 检查服务账号权限
2. 验证配额限制
3. 查看 Cloud Console 事件

```bash
# 查看 GKE 服务事件
kubectl describe svc <service-name>
```

## 监控告警

### Prometheus 抓取失败？

```bash
# 检查 ServiceMonitor
kubectl get servicemonitor -A

# 验证标签匹配
kubectl get svc -l <label> --show-labels

# 查看 Prometheus targets
kubectl port-forward svc/prometheus 9090:9090
# 访问 http://localhost:9090/targets
```

### 如何设置有效的告警规则？

1. **避免告警风暴**：使用合适的 for 持续时间
2. **设置优先级**：critical/warning/info 分级
3. **提供上下文**：在 annotations 中添加 runbook 链接
4. **减少噪音**：聚合相关告警

```yaml
- alert: PodCrashLooping
  expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
  for: 5m # 避免短暂重启触发告警
  labels:
    severity: warning
  annotations:
    summary: "Pod {{ $labels.pod }} 频繁重启"
    runbook_url: "https://wiki.example.com/runbooks/pod-crash"
```
