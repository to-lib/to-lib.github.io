---
sidebar_position: 11
title: 常见问题
description: 计算机网络常见问题与解决方案
keywords: [网络问题, 排查, TCP, HTTP, DNS]
---

# 常见问题 (FAQ)

## TCP 相关

### Q: 为什么是三次握手？

三次握手的作用：

1. 同步双方初始序列号
2. 确认双方的收发能力
3. 防止历史重复连接

两次不够：无法确认客户端的接收能力。四次多余：三次已足够。

---

### Q: TIME_WAIT 为什么等待 2MSL？

1. 确保最后的 ACK 能到达
2. 让旧连接的数据包消失

---

### Q: 大量 TIME_WAIT 如何处理？

```bash
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 30
```

根本方案：使用连接池、HTTP 长连接。

---

### Q: TCP 粘包如何解决？

1. 固定长度
2. 分隔符（如 `\n`）
3. 长度前缀（推荐）

---

## HTTP 相关

### Q: GET 和 POST 的区别？

| 对比项   | GET    | POST   |
| -------- | ------ | ------ |
| 参数位置 | URL    | 请求体 |
| 缓存     | 可缓存 | 不缓存 |
| 幂等性   | 幂等   | 非幂等 |

---

### Q: 301 和 302 的区别？

- 301：永久重定向，浏览器缓存
- 302：临时重定向，不缓存

---

## DNS 相关

### Q: DNS 解析顺序？

浏览器缓存 → 系统缓存 → hosts → 本地 DNS → 根 DNS → 顶级域 → 权威 DNS

---

## 排查命令

```bash
# 检查连通性
ping <host>

# 路由跟踪
traceroute <host>

# DNS 查询
nslookup <domain>

# 端口检查
nc -zv <host> <port>
```
