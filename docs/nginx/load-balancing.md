---
sidebar_position: 6
title: 负载均衡
description: Nginx 负载均衡策略与配置详解
---

# 负载均衡

## 负载均衡策略

### 轮询（默认）

```nginx
upstream backend {
    server 192.168.1.10:8080;
    server 192.168.1.11:8080;
    server 192.168.1.12:8080;
}
```

### 加权轮询

```nginx
upstream backend {
    server 192.168.1.10:8080 weight=5;
    server 192.168.1.11:8080 weight=3;
    server 192.168.1.12:8080 weight=2;
}
```

### IP Hash

```nginx
upstream backend {
    ip_hash;
    server 192.168.1.10:8080;
    server 192.168.1.11:8080;
}
```

### 最少连接

```nginx
upstream backend {
    least_conn;
    server 192.168.1.10:8080;
    server 192.168.1.11:8080;
}
```

### 一致性哈希

```nginx
upstream backend {
    hash $request_uri consistent;
    server 192.168.1.10:8080;
    server 192.168.1.11:8080;
}
```

## 服务器参数

```nginx
upstream backend {
    server 192.168.1.10:8080 weight=5 max_fails=3 fail_timeout=30s;
    server 192.168.1.11:8080 backup;  # 备用服务器
    server 192.168.1.12:8080 down;    # 标记下线
}
```

| 参数           | 说明           |
| -------------- | -------------- |
| `weight`       | 权重，默认 1   |
| `max_fails`    | 最大失败次数   |
| `fail_timeout` | 失败后暂停时间 |
| `backup`       | 备用服务器     |
| `down`         | 标记为下线     |

## 健康检查

```nginx
upstream backend {
    server 192.168.1.10:8080 max_fails=3 fail_timeout=30s;
    server 192.168.1.11:8080 max_fails=3 fail_timeout=30s;
}

location / {
    proxy_pass http://backend;
    proxy_next_upstream error timeout http_500 http_502 http_503;
    proxy_next_upstream_tries 3;
}
```

## Session 保持

```nginx
upstream backend {
    ip_hash;  # 基于 IP 保持会话
    server 192.168.1.10:8080;
    server 192.168.1.11:8080;
}

# 或使用 Cookie
upstream backend {
    hash $cookie_sessionid consistent;
    server 192.168.1.10:8080;
    server 192.168.1.11:8080;
}
```

## 完整配置示例

```nginx
upstream api_servers {
    least_conn;
    server 10.0.0.1:8080 weight=5 max_fails=3 fail_timeout=30s;
    server 10.0.0.2:8080 weight=3 max_fails=3 fail_timeout=30s;
    server 10.0.0.3:8080 backup;
    keepalive 32;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://api_servers;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Connection "";

        proxy_next_upstream error timeout http_500 http_502 http_503;
    }
}
```
