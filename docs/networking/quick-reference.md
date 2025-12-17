---
sidebar_position: 10
title: 快速参考
description: 计算机网络协议、端口、报文格式快速查询
keywords: [网络协议, 端口号, TCP, UDP, HTTP, 速查表]
---

# 计算机网络快速参考

## 常用端口号

### 系统端口 (0-1023)

| 端口  | 协议    | 服务             |
| ----- | ------- | ---------------- |
| 20/21 | TCP     | FTP（数据/控制） |
| 22    | TCP     | SSH              |
| 23    | TCP     | Telnet           |
| 25    | TCP     | SMTP             |
| 53    | TCP/UDP | DNS              |
| 67/68 | UDP     | DHCP             |
| 80    | TCP     | HTTP             |
| 110   | TCP     | POP3             |
| 143   | TCP     | IMAP             |
| 443   | TCP     | HTTPS            |

### 注册端口 (1024-49151)

| 端口  | 协议 | 服务       |
| ----- | ---- | ---------- |
| 1433  | TCP  | SQL Server |
| 3306  | TCP  | MySQL      |
| 3389  | TCP  | RDP        |
| 5432  | TCP  | PostgreSQL |
| 5672  | TCP  | RabbitMQ   |
| 6379  | TCP  | Redis      |
| 8080  | TCP  | HTTP 代理  |
| 9092  | TCP  | Kafka      |
| 27017 | TCP  | MongoDB    |

## TCP 速查

### TCP 标志位

| 标志 | 含义 | 用途     |
| ---- | ---- | -------- |
| SYN  | 同步 | 建立连接 |
| ACK  | 确认 | 确认收到 |
| FIN  | 结束 | 关闭连接 |
| RST  | 重置 | 异常终止 |
| PSH  | 推送 | 立即传递 |
| URG  | 紧急 | 紧急数据 |

### TCP 连接状态

```
CLOSED → LISTEN → SYN_RCVD → ESTABLISHED
CLOSED → SYN_SENT → ESTABLISHED
ESTABLISHED → FIN_WAIT_1 → FIN_WAIT_2 → TIME_WAIT → CLOSED
ESTABLISHED → CLOSE_WAIT → LAST_ACK → CLOSED
```

### 三次握手口诀

```
客 → 服: SYN (请求建立连接)
服 → 客: SYN + ACK (同意，请确认)
客 → 服: ACK (确认)
```

### 四次挥手口诀

```
主动方: FIN (我要关了)
被动方: ACK (知道了)
被动方: FIN (我也关了)
主动方: ACK (好的) → TIME_WAIT(2MSL) → CLOSED
```

## HTTP 速查

### 请求方法

| 方法   | 作用 | 幂等 | 安全 | 有请求体 |
| ------ | ---- | :--: | :--: | :------: |
| GET    | 获取 |  ✅  |  ✅  |    ❌    |
| POST   | 创建 |  ❌  |  ❌  |    ✅    |
| PUT    | 替换 |  ✅  |  ❌  |    ✅    |
| DELETE | 删除 |  ✅  |  ❌  |    ❌    |
| PATCH  | 更新 |  ❌  |  ❌  |    ✅    |

### 状态码速记

```
1xx: 继续
2xx: 成功 (200 OK, 201 Created, 204 No Content)
3xx: 重定向 (301 永久, 302 临时, 304 未修改)
4xx: 客户端错误 (400 请求错误, 401 未认证, 403 禁止, 404 未找到)
5xx: 服务端错误 (500 内部错误, 502 网关错误, 503 服务不可用)
```

### 常用请求头

| 头部          | 说明           | 示例                          |
| ------------- | -------------- | ----------------------------- |
| Host          | 主机名         | `www.example.com`             |
| User-Agent    | 客户端信息     | `Mozilla/5.0...`              |
| Accept        | 接受的内容类型 | `text/html, application/json` |
| Content-Type  | 请求体类型     | `application/json`            |
| Authorization | 认证信息       | `Bearer token123`             |
| Cookie        | 客户端 Cookie  | `sessionId=abc`               |
| Cache-Control | 缓存控制       | `max-age=3600`                |

### 常用响应头

| 头部                        | 说明        | 示例                              |
| --------------------------- | ----------- | --------------------------------- |
| Content-Type                | 响应体类型  | `application/json; charset=utf-8` |
| Content-Length              | 响应体长度  | `1234`                            |
| Set-Cookie                  | 设置 Cookie | `sessionId=abc; HttpOnly`         |
| Location                    | 重定向地址  | `https://new-url.com`             |
| Cache-Control               | 缓存策略    | `no-cache`                        |
| Access-Control-Allow-Origin | CORS 跨域   | `*`                               |

## IP 速查

### IPv4 地址分类

| 类别 | 范围                      | 默认子网掩码  | 用途     |
| ---- | ------------------------- | ------------- | -------- |
| A    | 1.0.0.0-126.255.255.255   | 255.0.0.0     | 大型网络 |
| B    | 128.0.0.0-191.255.255.255 | 255.255.0.0   | 中型网络 |
| C    | 192.0.0.0-223.255.255.255 | 255.255.255.0 | 小型网络 |
| D    | 224.0.0.0-239.255.255.255 | -             | 组播     |
| E    | 240.0.0.0-255.255.255.255 | -             | 保留     |

### 私有 IP 地址范围

| 类别 | 范围                          |
| ---- | ----------------------------- |
| A 类 | 10.0.0.0 - 10.255.255.255     |
| B 类 | 172.16.0.0 - 172.31.255.255   |
| C 类 | 192.168.0.0 - 192.168.255.255 |

### 特殊地址

| 地址            | 含义                 |
| --------------- | -------------------- |
| 127.0.0.1       | 本地回环 (localhost) |
| 0.0.0.0         | 任意地址             |
| 255.255.255.255 | 广播地址             |

## 子网掩码速算

| CIDR | 子网掩码        | 主机数     |
| ---- | --------------- | ---------- |
| /8   | 255.0.0.0       | 16,777,214 |
| /16  | 255.255.0.0     | 65,534     |
| /24  | 255.255.255.0   | 254        |
| /25  | 255.255.255.128 | 126        |
| /26  | 255.255.255.192 | 62         |
| /27  | 255.255.255.224 | 30         |
| /28  | 255.255.255.240 | 14         |
| /29  | 255.255.255.248 | 6          |
| /30  | 255.255.255.252 | 2          |

## 网络命令

### Linux/macOS

```bash
# 查看网络配置
ip addr / ifconfig

# 测试连通性
ping <host>

# 路由跟踪
traceroute <host>

# DNS 查询
nslookup <domain>
dig <domain>

# 查看端口
netstat -tlnp
ss -tlnp

# 抓包
tcpdump -i eth0 port 80
```

### Windows

```powershell
# 查看网络配置
ipconfig /all

# 测试连通性
ping <host>

# 路由跟踪
tracert <host>

# DNS 查询
nslookup <domain>

# 查看端口
netstat -ano
```

## OSI 模型速记

```
Please Do Not Throw Sausage Pizza Away
(物理 → 数据链路 → 网络 → 传输 → 会话 → 表示 → 应用)

All People Seem To Need Data Processing
(应用 → 表示 → 会话 → 传输 → 网络 → 数据链路 → 物理)
```

| 层         | 数据单元 | 设备         | 协议           |
| ---------- | -------- | ------------ | -------------- |
| 应用层     | 数据     | -            | HTTP, FTP, DNS |
| 表示层     | 数据     | -            | SSL, JPEG      |
| 会话层     | 数据     | -            | RPC            |
| 传输层     | 段       | -            | TCP, UDP       |
| 网络层     | 包       | 路由器       | IP, ICMP       |
| 数据链路层 | 帧       | 交换机       | Ethernet       |
| 物理层     | 比特     | 网卡、集线器 | -              |
