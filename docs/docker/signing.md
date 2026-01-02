---
sidebar_position: 36
title: 镜像签名
description: Docker Content Trust、Notary 与 Cosign 镜像签名验证
---

# 镜像签名

镜像签名用于验证镜像的来源和完整性，确保供应链安全。

## Docker Content Trust (DCT)

Docker 内置的镜像签名机制，基于 Notary。

### 启用 DCT

```bash
# 临时启用
export DOCKER_CONTENT_TRUST=1

# 永久启用（添加到 ~/.bashrc）
echo 'export DOCKER_CONTENT_TRUST=1' >> ~/.bashrc
```

### 签名和推送

```bash
# 启用 DCT 后，push 自动签名
export DOCKER_CONTENT_TRUST=1
docker push myregistry/myapp:v1.0

# 首次推送会生成密钥
# - Root key: 离线保存，用于创建其他密钥
# - Repository key: 用于签名特定仓库
```

### 拉取验证

```bash
# 启用 DCT 后，只能拉取已签名镜像
export DOCKER_CONTENT_TRUST=1
docker pull myregistry/myapp:v1.0

# 未签名镜像会报错
# Error: remote trust data does not exist
```

### 密钥管理

```bash
# 查看密钥
docker trust key list

# 生成新密钥
docker trust key generate mykey

# 添加签名者
docker trust signer add --key mykey.pub myname myregistry/myapp

# 查看签名信息
docker trust inspect myregistry/myapp

# 撤销签名
docker trust revoke myregistry/myapp:v1.0
```

## Cosign（推荐）

Sigstore 项目的镜像签名工具，更现代、更易用。

### 安装

```bash
# macOS
brew install cosign

# Linux
curl -LO https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64
chmod +x cosign-linux-amd64
sudo mv cosign-linux-amd64 /usr/local/bin/cosign
```

### 生成密钥对

```bash
# 生成密钥对
cosign generate-key-pair

# 输出：
# - cosign.key (私钥，需要密码保护)
# - cosign.pub (公钥)
```

### 签名镜像

```bash
# 使用本地密钥签名
cosign sign --key cosign.key myregistry/myapp:v1.0

# 使用 Keyless 签名（推荐，无需管理密钥）
cosign sign myregistry/myapp:v1.0
# 会打开浏览器进行 OIDC 认证
```

### 验证签名

```bash
# 使用公钥验证
cosign verify --key cosign.pub myregistry/myapp:v1.0

# Keyless 验证
cosign verify \
  --certificate-identity=user@example.com \
  --certificate-oidc-issuer=https://accounts.google.com \
  myregistry/myapp:v1.0
```

### 附加证明 (Attestations)

```bash
# 附加 SBOM
cosign attest --key cosign.key --predicate sbom.json myregistry/myapp:v1.0

# 附加漏洞扫描结果
cosign attest --key cosign.key --predicate vuln-scan.json myregistry/myapp:v1.0

# 验证证明
cosign verify-attestation --key cosign.pub myregistry/myapp:v1.0
```

## CI/CD 集成

### GitHub Actions + Cosign

```yaml
name: Build and Sign

on:
  push:
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      id-token: write  # Keyless 签名需要

    steps:
      - uses: actions/checkout@v4

      - name: Install Cosign
        uses: sigstore/cosign-installer@v3

      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and Push
        id: build
        uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.ref_name }}

      - name: Sign Image (Keyless)
        run: |
          cosign sign --yes ghcr.io/${{ github.repository }}@${{ steps.build.outputs.digest }}
```

### 使用私钥签名

```yaml
- name: Sign Image
  env:
    COSIGN_KEY: ${{ secrets.COSIGN_KEY }}
    COSIGN_PASSWORD: ${{ secrets.COSIGN_PASSWORD }}
  run: |
    echo "$COSIGN_KEY" > cosign.key
    cosign sign --key cosign.key ghcr.io/${{ github.repository }}:${{ github.ref_name }}
```

## Kubernetes 策略

### 使用 Kyverno 验证签名

```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: verify-image-signature
spec:
  validationFailureAction: Enforce
  rules:
    - name: verify-signature
      match:
        resources:
          kinds:
            - Pod
      verifyImages:
        - imageReferences:
            - "ghcr.io/myorg/*"
          attestors:
            - entries:
                - keyless:
                    subject: "user@example.com"
                    issuer: "https://accounts.google.com"
```

### 使用 Sigstore Policy Controller

```yaml
apiVersion: policy.sigstore.dev/v1alpha1
kind: ClusterImagePolicy
metadata:
  name: require-signature
spec:
  images:
    - glob: "ghcr.io/myorg/**"
  authorities:
    - keyless:
        identities:
          - issuer: https://accounts.google.com
            subject: user@example.com
```

## 对比

| 特性 | DCT/Notary | Cosign |
|------|------------|--------|
| 易用性 | 复杂 | 简单 |
| Keyless | 不支持 | 支持 |
| 透明日志 | 无 | Rekor |
| SBOM 支持 | 无 | 支持 |
| 社区活跃度 | 低 | 高 |
| 推荐程度 | 遗留系统 | 新项目首选 |

## 最佳实践

```bash
# 1. 使用 Keyless 签名（无需管理密钥）
cosign sign myregistry/myapp:v1.0

# 2. 签名时附加 SBOM
syft myregistry/myapp:v1.0 -o spdx-json > sbom.json
cosign attest --predicate sbom.json myregistry/myapp:v1.0

# 3. 在 CI/CD 中自动签名
# 4. 在 Kubernetes 中强制验证签名
# 5. 定期轮换密钥（如使用本地密钥）
```
