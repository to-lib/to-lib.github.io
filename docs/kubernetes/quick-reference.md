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

## StatefulSet 模板

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql
spec:
  serviceName: mysql
  replicas: 3
  selector:
    matchLabels:
      app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
        - name: mysql
          image: mysql:8.0
          ports:
            - containerPort: 3306
          volumeMounts:
            - name: data
              mountPath: /var/lib/mysql
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
```

## Job 模板

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: my-job
spec:
  completions: 1
  parallelism: 1
  backoffLimit: 3
  activeDeadlineSeconds: 600
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: job
          image: busybox
          command: ["echo", "Hello, World!"]
```

## CronJob 模板

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: my-cronjob
spec:
  schedule: "0 2 * * *" # 每天凌晨 2 点
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
            - name: job
              image: busybox
              command: ["echo", "Scheduled task"]
```

## DaemonSet 模板

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: my-daemonset
spec:
  selector:
    matchLabels:
      app: my-daemon
  template:
    metadata:
      labels:
        app: my-daemon
    spec:
      containers:
        - name: daemon
          image: my-agent:1.0
          resources:
            limits:
              memory: 200Mi
              cpu: 100m
```

## Ingress 模板

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - example.com
      secretName: tls-secret
  rules:
    - host: example.com
      http:
        paths:
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: api-service
                port:
                  number: 80
```

## 资源限制推荐值

| 应用类型 | CPU requests | CPU limits  | 内存 requests | 内存 limits |
| -------- | ------------ | ----------- | ------------- | ----------- |
| 微服务   | 100m-250m    | 500m-1000m  | 128Mi-256Mi   | 256Mi-512Mi |
| 数据库   | 500m-1000m   | 2000m-4000m | 1Gi-2Gi       | 2Gi-4Gi     |
| 缓存     | 250m-500m    | 1000m-2000m | 512Mi-1Gi     | 1Gi-2Gi     |
| 批处理   | 500m-1000m   | 2000m-4000m | 512Mi-1Gi     | 2Gi-4Gi     |

## 常用标签

```yaml
metadata:
  labels:
    app.kubernetes.io/name: my-app
    app.kubernetes.io/instance: my-app-prod
    app.kubernetes.io/version: "1.0.0"
    app.kubernetes.io/component: frontend
    app.kubernetes.io/part-of: my-platform
    app.kubernetes.io/managed-by: helm
    environment: production
    team: platform
```

## 调试镜像

| 镜像                                            | 用途      | 示例命令                                                                                                      |
| ----------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------------- |
| busybox                                         | 基础调试  | `kubectl run debug --image=busybox -it --rm -- sh`                                                            |
| nicolaka/netshoot                               | 网络调试  | `kubectl run netshoot --image=nicolaka/netshoot -it --rm`                                                     |
| curlimages/curl                                 | HTTP 测试 | `kubectl run curl --image=curlimages/curl -it --rm -- curl http://svc`                                        |
| registry.k8s.io/e2e-test-images/jessie-dnsutils | DNS 调试  | `kubectl run dns --image=registry.k8s.io/e2e-test-images/jessie-dnsutils:1.3 -it --rm -- nslookup kubernetes` |

## 高频调试命令

```bash
# 查看所有问题 Pod
kubectl get pods -A | grep -v Running | grep -v Completed

# 查看最近事件
kubectl get events -A --sort-by='.lastTimestamp' | tail -20

# 按资源使用排序
kubectl top pods -A --sort-by=memory | head -20
kubectl top pods -A --sort-by=cpu | head -20

# 导出资源定义
kubectl get deployment my-app -o yaml > deployment.yaml

# 临时运行调试 Pod
kubectl run debug --image=busybox -it --rm -- sh

# 端口转发
kubectl port-forward svc/my-service 8080:80

# 查看容器进程
kubectl exec -it <pod> -- ps aux

# 检查网络连通性
kubectl exec -it <pod> -- curl -v http://service:port
```
