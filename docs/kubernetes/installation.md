---
sidebar_position: 2
title: 安装配置
description: Kubernetes 集群安装与 kubectl 配置指南
---

# Kubernetes 安装配置

## 本地开发环境

### Minikube

Minikube 是本地运行 Kubernetes 的最简单方式。

```bash
# macOS 安装
brew install minikube

# Linux 安装
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# 启动集群
minikube start

# 指定资源和驱动
minikube start --cpus=4 --memory=8192 --driver=docker

# 常用命令
minikube status          # 查看状态
minikube dashboard       # 打开 Dashboard
minikube stop            # 停止集群
minikube delete          # 删除集群
```

### Kind (Kubernetes in Docker)

Kind 使用 Docker 容器作为节点，适合 CI/CD 测试。

```bash
# 安装 Kind
go install sigs.k8s.io/kind@v0.20.0
# 或使用 brew
brew install kind

# 创建简单集群
kind create cluster

# 使用配置文件创建多节点集群
cat <<EOF | kind create cluster --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
- role: worker
- role: worker
EOF

# 查看集群
kind get clusters

# 删除集群
kind delete cluster
```

### Docker Desktop

Docker Desktop 内置 Kubernetes 支持（仅 macOS/Windows）。

1. 打开 Docker Desktop 设置
2. 进入 Kubernetes 选项卡
3. 勾选 "Enable Kubernetes"
4. 点击 Apply & Restart

## kubectl 安装

### macOS

```bash
# 使用 Homebrew
brew install kubectl

# 或直接下载
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/darwin/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/
```

### Linux

```bash
# 下载最新版本
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# 或使用包管理器 (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.31/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.31/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubectl
```

### Windows

```powershell
# 使用 Chocolatey
choco install kubernetes-cli

# 或使用 Scoop
scoop install kubectl

# 或直接下载
curl.exe -LO "https://dl.k8s.io/release/v1.31.0/bin/windows/amd64/kubectl.exe"
```

## kubectl 配置

### kubeconfig 文件

kubectl 使用 kubeconfig 文件进行集群认证，默认位置为 `~/.kube/config`。

```yaml
apiVersion: v1
kind: Config
clusters:
  - name: minikube
    cluster:
      certificate-authority: /path/to/ca.crt
      server: https://192.168.49.2:8443
contexts:
  - name: minikube
    context:
      cluster: minikube
      user: minikube
      namespace: default
current-context: minikube
users:
  - name: minikube
    user:
      client-certificate: /path/to/client.crt
      client-key: /path/to/client.key
```

### 多集群管理

```bash
# 查看所有上下文
kubectl config get-contexts

# 切换上下文
kubectl config use-context minikube

# 设置默认命名空间
kubectl config set-context --current --namespace=my-namespace

# 合并多个 kubeconfig
export KUBECONFIG=~/.kube/config:~/.kube/config-prod
kubectl config view --merge --flatten > ~/.kube/config-merged
```

### 自动补全

```bash
# Bash
echo 'source <(kubectl completion bash)' >> ~/.bashrc
echo 'alias k=kubectl' >> ~/.bashrc
echo 'complete -F __start_kubectl k' >> ~/.bashrc

# Zsh
echo 'source <(kubectl completion zsh)' >> ~/.zshrc
echo 'alias k=kubectl' >> ~/.zshrc

# 使配置生效
source ~/.bashrc  # 或 source ~/.zshrc
```

## 生产集群部署

### kubeadm 安装

```bash
# 1. 安装容器运行时 (containerd)
sudo apt-get update
sudo apt-get install -y containerd
sudo mkdir -p /etc/containerd
containerd config default | sudo tee /etc/containerd/config.toml
sudo systemctl restart containerd

# 2. 安装 kubeadm, kubelet, kubectl
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl gpg
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.31/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.31/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl

# 3. 初始化控制平面
sudo kubeadm init --pod-network-cidr=10.244.0.0/16

# 4. 配置 kubectl
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# 5. 安装网络插件 (Flannel)
kubectl apply -f https://raw.githubusercontent.com/flannel-io/flannel/master/Documentation/kube-flannel.yml
```

### 节点加入集群

```bash
# 在 master 节点获取 join 命令
kubeadm token create --print-join-command

# 在 worker 节点执行
sudo kubeadm join <master-ip>:6443 --token <token> --discovery-token-ca-cert-hash sha256:<hash>
```

## 验证安装

```bash
# 检查 kubectl 版本
kubectl version

# 查看集群信息
kubectl cluster-info

# 查看节点状态
kubectl get nodes -o wide

# 查看系统组件
kubectl get pods -n kube-system

# 创建测试部署
kubectl create deployment nginx --image=nginx
kubectl expose deployment nginx --port=80 --type=NodePort
kubectl get services
```

## 常见问题

### kubectl 无法连接集群

```bash
# 检查 kubeconfig 路径
echo $KUBECONFIG

# 检查配置文件
kubectl config view

# 测试 API Server 连通性
curl -k https://<api-server>:6443/healthz
```

### 节点 NotReady

```bash
# 查看节点详情
kubectl describe node <node-name>

# 检查 kubelet 状态
sudo systemctl status kubelet
sudo journalctl -u kubelet -f

# 检查网络插件
kubectl get pods -n kube-system | grep -E 'flannel|calico|weave'
```
