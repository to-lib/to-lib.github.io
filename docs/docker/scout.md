---
sidebar_position: 31
title: Docker Scout
description: Docker 官方镜像安全扫描与漏洞分析工具
---

# Docker Scout

Docker Scout 是 Docker 官方的软件供应链安全工具，用于分析镜像漏洞、提供修复建议。

## 功能概述

- **漏洞扫描** - 检测已知 CVE 漏洞
- **SBOM 生成** - 软件物料清单
- **策略评估** - 安全合规检查
- **修复建议** - 提供升级路径

## 基本使用

### 快速扫描

```bash
# 扫描本地镜像
docker scout cves myapp:latest

# 扫描远程镜像
docker scout cves nginx:latest

# 扫描 Dockerfile（构建前分析）
docker scout cves --file Dockerfile .
```

### 输出示例

```
    ✓ Image stored for indexing
    ✓ Indexed 187 packages

  Target     │  myapp:latest  │    0C     4H    12M    23L
    digest   │  sha256:abc... │
  Base image │  node:18       │    0C     2H     8M    15L

## Packages and Vulnerabilities

   0C     2H     0M     0L  lodash 4.17.20
                            pkg:npm/lodash@4.17.20

    ✗ HIGH CVE-2021-23337
      https://scout.docker.com/v/CVE-2021-23337
      Affected range : <4.17.21
      Fixed version  : 4.17.21
```

### 详细报告

```bash
# 显示所有漏洞详情
docker scout cves --details myapp:latest

# 只显示高危和严重漏洞
docker scout cves --only-severity critical,high myapp:latest

# JSON 格式输出
docker scout cves --format json myapp:latest > report.json

# SARIF 格式（用于 CI/CD）
docker scout cves --format sarif myapp:latest > scout.sarif
```

## SBOM 生成

```bash
# 生成软件物料清单
docker scout sbom myapp:latest

# 指定格式
docker scout sbom --format spdx myapp:latest
docker scout sbom --format cyclonedx myapp:latest

# 输出到文件
docker scout sbom --output sbom.json myapp:latest
```

## 策略评估

```bash
# 评估镜像是否符合策略
docker scout policy myapp:latest

# 查看可用策略
docker scout policy --list
```

### 内置策略

| 策略 | 说明 |
|------|------|
| No critical vulnerabilities | 无严重漏洞 |
| No high vulnerabilities | 无高危漏洞 |
| Packages with fixable CVEs | 可修复的漏洞 |
| Base image up to date | 基础镜像是否最新 |
| Supply chain attestations | 供应链证明 |

## 修复建议

```bash
# 获取修复建议
docker scout recommendations myapp:latest

# 查看基础镜像更新建议
docker scout recommendations --only-base myapp:latest
```

输出示例：

```
  Recommended fixes for image myapp:latest

  Base image is node:18.19.0-alpine

    │ Updating the base image would fix:
    │ - 2 high vulnerabilities
    │ - 5 medium vulnerabilities

  Recommended base image: node:18.20.0-alpine

  To update, change your Dockerfile:
    - FROM node:18.19.0-alpine
    + FROM node:18.20.0-alpine
```

## CI/CD 集成

### GitHub Actions

```yaml
name: Docker Scout

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  scout:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build image
        run: docker build -t myapp:${{ github.sha }} .

      - name: Docker Scout Scan
        uses: docker/scout-action@v1
        with:
          command: cves
          image: myapp:${{ github.sha }}
          only-severities: critical,high
          exit-code: true  # 发现漏洞时失败

      - name: Docker Scout Compare
        if: github.event_name == 'pull_request'
        uses: docker/scout-action@v1
        with:
          command: compare
          image: myapp:${{ github.sha }}
          to: myapp:latest
          only-severities: critical,high
```

### GitLab CI

```yaml
scout-scan:
  image: docker:latest
  services:
    - docker:dind
  variables:
    DOCKER_SCOUT_HUB_USER: $DOCKER_HUB_USER
    DOCKER_SCOUT_HUB_PASSWORD: $DOCKER_HUB_TOKEN
  script:
    - docker build -t myapp:$CI_COMMIT_SHA .
    - docker scout cves myapp:$CI_COMMIT_SHA --exit-code --only-severity critical,high
```

## 比较镜像

```bash
# 比较两个版本的漏洞变化
docker scout compare myapp:v2 --to myapp:v1

# 比较本地与远程
docker scout compare local://myapp:latest --to registry://myapp:prod
```

## 配置

### 登录 Docker Hub

```bash
# Scout 需要 Docker Hub 账号
docker login

# 或使用 PAT
docker login -u username -p token
```

### 环境变量

```bash
# CI/CD 中使用
export DOCKER_SCOUT_HUB_USER=username
export DOCKER_SCOUT_HUB_PASSWORD=token
```

## 与其他工具对比

| 工具 | 优势 | 劣势 |
|------|------|------|
| Docker Scout | 官方集成、修复建议 | 需要 Docker Hub |
| Trivy | 开源、快速 | 无修复建议 |
| Grype | 开源、准确 | 功能较少 |
| Snyk | 功能全面 | 商业产品 |

## 最佳实践

```bash
# 构建流程中集成扫描
docker build -t myapp:latest .
docker scout cves myapp:latest --exit-code --only-severity critical

# 定期扫描生产镜像
docker scout cves myregistry/myapp:prod

# PR 中比较变化
docker scout compare myapp:pr-123 --to myapp:main
```
