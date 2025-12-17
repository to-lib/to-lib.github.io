---
sidebar_position: 3
title: 基础命令
description: kubectl 常用命令详解
---

# kubectl 基础命令

## 命令结构

```bash
kubectl [command] [TYPE] [NAME] [flags]
```

- **command**：要执行的操作（get、create、delete 等）
- **TYPE**：资源类型（pod、service、deployment 等）
- **NAME**：资源名称
- **flags**：可选标志

## 集群信息

```bash
# 查看集群信息
kubectl cluster-info

# 查看 API 版本
kubectl api-versions

# 查看 API 资源
kubectl api-resources

# 查看组件状态
kubectl get componentstatuses  # 或 kubectl get cs
```

## 资源查看 (get)

```bash
# 查看 Pod
kubectl get pods                      # 当前命名空间
kubectl get pods -A                   # 所有命名空间
kubectl get pods -n kube-system       # 指定命名空间
kubectl get pods -o wide              # 显示更多信息
kubectl get pods -o yaml              # YAML 格式输出
kubectl get pods -o json              # JSON 格式输出
kubectl get pods --show-labels        # 显示标签
kubectl get pods -l app=nginx         # 按标签筛选
kubectl get pods --field-selector status.phase=Running  # 按字段筛选

# 查看多种资源
kubectl get pods,services,deployments
kubectl get all                       # 查看常用资源

# 监视资源变化
kubectl get pods -w                   # watch 模式

# 自定义输出列
kubectl get pods -o custom-columns=NAME:.metadata.name,STATUS:.status.phase
```

## 资源详情 (describe)

```bash
# 查看 Pod 详情
kubectl describe pod <pod-name>

# 查看节点详情
kubectl describe node <node-name>

# 查看 Service 详情
kubectl describe service <service-name>

# 查看事件
kubectl get events --sort-by='.lastTimestamp'
kubectl get events -w
```

## 创建资源 (create/apply)

```bash
# 从文件创建
kubectl create -f pod.yaml
kubectl apply -f deployment.yaml      # 推荐：支持更新

# 从目录创建
kubectl apply -f ./manifests/

# 从 URL 创建
kubectl apply -f https://example.com/manifest.yaml

# 快速创建资源
kubectl create namespace dev
kubectl create deployment nginx --image=nginx
kubectl create service clusterip nginx --tcp=80:80
kubectl create configmap my-config --from-literal=key=value
kubectl create secret generic my-secret --from-literal=password=123456

# 试运行（不实际创建）
kubectl create deployment nginx --image=nginx --dry-run=client -o yaml
```

## 更新资源

```bash
# 应用更新
kubectl apply -f deployment.yaml

# 编辑资源
kubectl edit deployment nginx

# 替换资源
kubectl replace -f deployment.yaml

# 打补丁
kubectl patch deployment nginx -p '{"spec":{"replicas":3}}'
kubectl patch deployment nginx --type='json' -p='[{"op":"replace","path":"/spec/replicas","value":5}]'

# 更新镜像
kubectl set image deployment/nginx nginx=nginx:1.25

# 扩缩容
kubectl scale deployment nginx --replicas=5

# 自动扩缩容
kubectl autoscale deployment nginx --min=2 --max=10 --cpu-percent=50
```

## 删除资源 (delete)

```bash
# 删除指定资源
kubectl delete pod <pod-name>
kubectl delete deployment nginx

# 从文件删除
kubectl delete -f deployment.yaml

# 删除所有 Pod
kubectl delete pods --all

# 按标签删除
kubectl delete pods -l app=nginx

# 强制删除
kubectl delete pod <pod-name> --force --grace-period=0

# 删除命名空间（及其所有资源）
kubectl delete namespace dev
```

## 执行命令 (exec)

```bash
# 进入容器
kubectl exec -it <pod-name> -- /bin/bash
kubectl exec -it <pod-name> -- /bin/sh

# 执行命令
kubectl exec <pod-name> -- ls /app
kubectl exec <pod-name> -- cat /etc/nginx/nginx.conf

# 多容器 Pod 指定容器
kubectl exec -it <pod-name> -c <container-name> -- /bin/bash
```

## 查看日志 (logs)

```bash
# 查看 Pod 日志
kubectl logs <pod-name>

# 查看最近 100 行
kubectl logs <pod-name> --tail=100

# 实时查看
kubectl logs <pod-name> -f

# 查看前一个容器的日志
kubectl logs <pod-name> --previous

# 指定容器
kubectl logs <pod-name> -c <container-name>

# 按时间筛选
kubectl logs <pod-name> --since=1h
kubectl logs <pod-name> --since-time=2024-01-01T00:00:00Z

# 查看多个 Pod 日志
kubectl logs -l app=nginx
```

## 端口转发 (port-forward)

```bash
# 转发 Pod 端口
kubectl port-forward pod/<pod-name> 8080:80

# 转发 Service 端口
kubectl port-forward service/nginx 8080:80

# 转发 Deployment 端口
kubectl port-forward deployment/nginx 8080:80

# 后台运行
kubectl port-forward pod/<pod-name> 8080:80 &
```

## 文件传输 (cp)

```bash
# 从本地复制到容器
kubectl cp ./local-file.txt <pod-name>:/path/in/container/

# 从容器复制到本地
kubectl cp <pod-name>:/path/in/container/file.txt ./local-file.txt

# 指定容器
kubectl cp ./file.txt <pod-name>:/path/ -c <container-name>
```

## 标签管理 (label)

```bash
# 添加标签
kubectl label pod <pod-name> env=production

# 更新标签
kubectl label pod <pod-name> env=staging --overwrite

# 删除标签
kubectl label pod <pod-name> env-

# 查看标签
kubectl get pods --show-labels
kubectl get pods -L env,tier
```

## 注解管理 (annotate)

```bash
# 添加注解
kubectl annotate pod <pod-name> description="This is my pod"

# 更新注解
kubectl annotate pod <pod-name> description="Updated description" --overwrite

# 删除注解
kubectl annotate pod <pod-name> description-
```

## 调试命令

```bash
# 查看资源使用
kubectl top nodes
kubectl top pods

# 运行调试容器
kubectl run debug --image=busybox -it --rm -- /bin/sh
kubectl run debug --image=nicolaka/netshoot -it --rm -- /bin/bash

# 调试现有 Pod
kubectl debug <pod-name> -it --image=busybox

# 查看 Pod 事件
kubectl describe pod <pod-name> | grep -A 20 Events
```

## 输出格式

| 格式                | 说明            |
| ------------------- | --------------- |
| `-o wide`           | 显示更多列      |
| `-o yaml`           | YAML 格式       |
| `-o json`           | JSON 格式       |
| `-o name`           | 仅显示名称      |
| `-o jsonpath`       | JSONPath 表达式 |
| `-o custom-columns` | 自定义列        |

```bash
# JSONPath 示例
kubectl get pods -o jsonpath='{.items[*].metadata.name}'
kubectl get pods -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
kubectl get pod <pod-name> -o jsonpath='{.status.podIP}'

# 自定义列
kubectl get pods -o custom-columns=\
'POD:metadata.name,NODE:spec.nodeName,IP:status.podIP'
```

## 常用别名

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
alias k='kubectl'
alias kg='kubectl get'
alias kd='kubectl describe'
alias kl='kubectl logs'
alias ke='kubectl exec -it'
alias ka='kubectl apply -f'
alias kdel='kubectl delete'
alias kgp='kubectl get pods'
alias kgs='kubectl get services'
alias kgd='kubectl get deployments'
alias kgn='kubectl get nodes'
```
