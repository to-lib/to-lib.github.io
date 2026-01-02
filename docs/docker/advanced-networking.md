---
sidebar_position: 20
title: 高级网络
description: IPVLAN、容器 DNS 原理与 IPv6 双栈配置
---

# 高级网络

深入理解 Docker 网络的高级特性和配置。

## IPVLAN 网络

IPVLAN 允许容器直接使用主机网络接口的 IP 地址，无需 NAT。

### IPVLAN vs Macvlan

| 特性 | IPVLAN | Macvlan |
|------|--------|---------|
| MAC 地址 | 共享主机 MAC | 每个容器独立 MAC |
| 交换机兼容 | 更好 | 可能触发端口安全 |
| 性能 | 略高 | 高 |
| 适用场景 | 云环境、MAC 限制场景 | 需要独立 MAC 的场景 |

### IPVLAN L2 模式

```bash
# 创建 IPVLAN L2 网络
docker network create -d ipvlan \
  --subnet=192.168.1.0/24 \
  --gateway=192.168.1.1 \
  -o parent=eth0 \
  -o ipvlan_mode=l2 \
  ipvlan-l2

# 运行容器
docker run -d --network ipvlan-l2 --ip 192.168.1.100 nginx
```

### IPVLAN L3 模式

```bash
# 创建 IPVLAN L3 网络
docker network create -d ipvlan \
  --subnet=10.10.10.0/24 \
  -o parent=eth0 \
  -o ipvlan_mode=l3 \
  ipvlan-l3

# L3 模式需要在路由器上配置路由
```

## 容器 DNS 原理

Docker 内置 DNS 服务器为容器提供名称解析。

### 内置 DNS 服务器

```
┌─────────────────────────────────────────────────┐
│                  Container                       │
│  /etc/resolv.conf:                              │
│  nameserver 127.0.0.11                          │
│  options ndots:0                                │
└─────────────────────┬───────────────────────────┘
                      │ DNS Query
                      ▼
┌─────────────────────────────────────────────────┐
│           Docker Embedded DNS (127.0.0.11)       │
│  ┌─────────────────────────────────────────┐    │
│  │ 1. 检查容器名/服务名/网络别名            │    │
│  │ 2. 检查 --link 配置                     │    │
│  │ 3. 转发到外部 DNS                       │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

### DNS 解析流程

```bash
# 查看容器 DNS 配置
docker exec container cat /etc/resolv.conf

# 测试 DNS 解析
docker exec container nslookup other-container

# 查看 DNS 解析详情
docker exec container dig other-container
```

### 自定义 DNS 配置

```bash
# 运行时指定 DNS
docker run --dns 8.8.8.8 --dns 8.8.4.4 nginx

# 指定 DNS 搜索域
docker run --dns-search example.com nginx

# 添加 hosts 记录
docker run --add-host db.local:192.168.1.100 nginx
```

### 全局 DNS 配置

```json
// /etc/docker/daemon.json
{
  "dns": ["8.8.8.8", "8.8.4.4"],
  "dns-search": ["example.com"],
  "dns-opts": ["ndots:1"]
}
```

### Docker Compose DNS

```yaml
services:
  app:
    image: myapp
    dns:
      - 8.8.8.8
      - 8.8.4.4
    dns_search:
      - example.com
    extra_hosts:
      - "db.local:192.168.1.100"
    networks:
      mynet:
        aliases:
          - app.local
          - api.local
```

## IPv6 支持

### 启用 IPv6

```json
// /etc/docker/daemon.json
{
  "ipv6": true,
  "fixed-cidr-v6": "2001:db8:1::/64"
}
```

重启 Docker：
```bash
sudo systemctl restart docker
```

### 创建 IPv6 网络

```bash
# 创建双栈网络
docker network create \
  --ipv6 \
  --subnet=172.20.0.0/16 \
  --subnet=2001:db8:2::/64 \
  dual-stack

# 运行容器
docker run -d --network dual-stack nginx

# 验证 IPv6
docker exec container ip -6 addr
```

### Docker Compose IPv6

```yaml
version: "3.8"

services:
  web:
    image: nginx
    networks:
      dual-stack:
        ipv4_address: 172.20.0.10
        ipv6_address: 2001:db8:2::10

networks:
  dual-stack:
    enable_ipv6: true
    ipam:
      config:
        - subnet: 172.20.0.0/16
        - subnet: 2001:db8:2::/64
```

## 网络性能优化

### 使用 host 网络

```bash
# 最高性能，无网络隔离
docker run --network host nginx
```

### 调整 MTU

```json
// /etc/docker/daemon.json
{
  "mtu": 9000  // Jumbo frames
}
```

### 网络命名空间共享

```bash
# 多个容器共享网络命名空间
docker run -d --name web nginx
docker run -d --network container:web redis
# redis 与 web 共享网络，可通过 localhost 通信
```

## 网络故障排查

### 常用诊断命令

```bash
# 查看网络详情
docker network inspect bridge

# 查看容器网络配置
docker inspect --format '{{json .NetworkSettings}}' container | jq

# 容器内网络诊断
docker exec container ip addr
docker exec container ip route
docker exec container netstat -tlnp
docker exec container ss -tlnp

# 抓包分析
docker run --net=host --privileged -v /tmp:/tmp \
  nicolaka/netshoot tcpdump -i docker0 -w /tmp/capture.pcap
```

### 使用 netshoot 工具

```bash
# 运行网络诊断容器
docker run -it --net container:target_container \
  nicolaka/netshoot

# 可用工具：ping, curl, dig, nslookup, tcpdump, iperf, netstat 等
```

### 检查 iptables 规则

```bash
# 查看 Docker 相关的 iptables 规则
sudo iptables -L -n -t nat
sudo iptables -L -n -t filter

# 查看 DOCKER 链
sudo iptables -L DOCKER -n -v
```
