---
sidebar_position: 23
title: nftables 防火墙
---

# nftables 防火墙

`nftables` 是 Linux 新一代包过滤/防火墙框架，用于替代 `iptables`/`ip6tables`/`arptables`/`ebtables`。在不少发行版上（尤其较新的 Debian/Ubuntu/RHEL 系列），它已经成为推荐方案。

适合：

- 统一管理 IPv4/IPv6 规则
- 更好的规则表达能力（集合、映射等）
- 更容易做“最小可用规则集”与可维护的策略

## 快速概念

- **table**：规则容器（可按协议族 `ip`/`ip6`/`inet`）
- **chain**：规则链（挂载在 hook 上，如 `input`/`forward`/`output`）
- **rule**：具体匹配与动作（accept/drop/reject/log）

建议优先使用 `inet` 族表：同一套规则同时覆盖 IPv4/IPv6。

## 基本命令

```bash
# 查看规则（建议加 -a 显示 handle）
sudo nft list ruleset
sudo nft -a list ruleset

# 查看表/链
sudo nft list tables
sudo nft list table inet filter

# 测试语法（不修改系统）
sudo nft -c -f /etc/nftables.conf
```

## 一个最小可用的防火墙示例（推荐起点）

以下示例包含：

- 默认拒绝入站与转发
- 允许已建立连接
- 允许本机回环
- 允许 SSH（可按需限制来源）
- 允许 ICMP（用于基本诊断）

```nft
table inet filter {
  chain input {
    type filter hook input priority 0;
    policy drop;

    # 已建立/相关连接放行
    ct state established,related accept

    # 回环
    iif "lo" accept

    # 允许 ping / 基本 ICMP（IPv4/IPv6）
    ip protocol icmp accept
    ip6 nexthdr icmpv6 accept

    # SSH（示例：放开 22）
    tcp dport 22 accept

    # 可选：记录被丢弃的包（注意日志量）
    # log prefix "nft drop: " flags all counter drop
  }

  chain forward {
    type filter hook forward priority 0;
    policy drop;
  }

  chain output {
    type filter hook output priority 0;
    policy accept;
  }
}
```

如果你希望只允许特定来源 SSH：

```nft
tcp dport 22 ip saddr { 10.0.0.0/8, 192.168.0.0/16 } accept
```

## 集合（set）：让规则更易维护

例如把允许 SSH 的来源单独做成集合：

```nft
table inet filter {
  set ssh_allow {
    type ipv4_addr
    flags interval
    elements = { 10.0.0.0/8, 192.168.0.0/16 }
  }

  chain input {
    type filter hook input priority 0;
    policy drop;

    ct state established,related accept
    iif "lo" accept

    tcp dport 22 ip saddr @ssh_allow accept
  }
}
```

## 与发行版集成（持久化）

### Debian/Ubuntu（常见）

- 常见配置文件：`/etc/nftables.conf`
- 服务：`nftables.service`

```bash
# 启用开机加载
sudo systemctl enable --now nftables

# 检查服务状态
sudo systemctl status nftables

# 加载配置
sudo nft -f /etc/nftables.conf
```

### RHEL/CentOS/Fedora

不同版本与发行版对 nftables 的集成方式不同；有些场景可能更推荐 `firewalld` 作为上层管理工具（底层可以使用 nftables）。

## 常见排错

- **规则不生效**：确认是否加载了正确的配置文件（`nft list ruleset`）
- **SSH 把自己锁外面了**：
  - 先在控制台/带外登录测试
  - 修改前保留一个可回滚窗口（或用 `at`/`systemd-run` 定时恢复规则）
- **日志爆炸**：`log` 规则要加条件/采样，必要时加 `limit rate`

## 与现有文档的关系

- 网络与传统防火墙概述见：[网络配置](/docs/linux/networking)
- 安全加固建议见：[系统安全](/docs/linux/security)
