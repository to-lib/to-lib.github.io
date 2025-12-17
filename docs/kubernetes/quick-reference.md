---
sidebar_position: 11
title: 快速参考
description: Kubernetes 命令与配置速查表
---

# Kubernetes 快速参考

## kubectl 常用命令

### 集群信息

```bash
kubectl cluster-info                 # 集群信息
kubectl get nodes -o wide           # 查看节点
kubectl top nodes                   # 节点资源使用
kubectl api-resources               # API 资源列表
```

### 资源操作

```bash
# 查看
kubectl get pods -A                  # 所有命名空间
kubectl get pods -l app=nginx        # 按标签筛选
kubectl describe pod <name>
kubectl get pod <name> -o yaml

# 创建/更新
kubectl apply -f manifest.yaml
kubectl create deployment nginx --image=nginx
kubectl set image deployment/nginx nginx=nginx:1.25
kubectl scale deployment nginx --replicas=5

# 删除
kubectl delete pod <name>
kubectl delete -f manifest.yaml
kubectl delete pod <name> --force --grace-period=0
```

### 调试

```bash
kubectl logs <pod> -f                # 实时日志
kubectl exec -it <pod> -- /bin/bash  # 进入容器
kubectl port-forward pod/<pod> 8080:80
kubectl cp local-file <pod>:/path
```

## 资源缩写

| 资源         | 缩写   | 资源                   | 缩写 |
| ------------ | ------ | ---------------------- | ---- |
| pods         | po     | services               | svc  |
| deployments  | deploy | configmaps             | cm   |
| statefulsets | sts    | persistentvolumeclaims | pvc  |
| namespaces   | ns     | serviceaccounts        | sa   |

## Pod 配置模板

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
  labels:
    app: my-app
spec:
  containers:
    - name: app
      image: nginx:1.24
      ports:
        - containerPort: 80
      resources:
        requests:
          memory: "64Mi"
          cpu: "250m"
        limits:
          memory: "128Mi"
          cpu: "500m"
      livenessProbe:
        httpGet:
          path: /healthz
          port: 80
        initialDelaySeconds: 30
      readinessProbe:
        httpGet:
          path: /ready
          port: 80
```

## Deployment 模板

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
        - name: app
          image: my-app:1.0
```

## Service 模板

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  type: ClusterIP # NodePort / LoadBalancer
  selector:
    app: my-app
  ports:
    - port: 80
      targetPort: 8080
```

## Helm 命令

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install my-release bitnami/nginx -f values.yaml
helm upgrade my-release bitnami/nginx
helm rollback my-release 1
helm uninstall my-release
```

## 调试镜像

| 镜像              | 用途      |
| ----------------- | --------- |
| busybox           | 基础调试  |
| nicolaka/netshoot | 网络调试  |
| curlimages/curl   | HTTP 测试 |
