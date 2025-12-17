---
sidebar_position: 5
title: 反向代理
description: Nginx 反向代理配置详解
---

# 反向代理

## 基本配置

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## Proxy 指令详解

```nginx
location /api/ {
    proxy_pass http://backend:8080/;

    # 请求头设置
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # 超时设置
    proxy_connect_timeout 60s;
    proxy_send_timeout 60s;
    proxy_read_timeout 60s;

    # 缓冲设置
    proxy_buffering on;
    proxy_buffer_size 4k;
    proxy_buffers 8 32k;

    # HTTP 版本
    proxy_http_version 1.1;
}
```

## WebSocket 代理

```nginx
map $http_upgrade $connection_upgrade {
    default upgrade;
    '' close;
}

location /ws/ {
    proxy_pass http://websocket_server;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection $connection_upgrade;
    proxy_read_timeout 3600s;
}
```

## Upstream 配置

```nginx
upstream backend {
    server 192.168.1.10:8080;
    server 192.168.1.11:8080;
    keepalive 32;
}

server {
    location / {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
```

## 代理缓存

```nginx
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=my_cache:10m max_size=1g;

location /api/ {
    proxy_pass http://backend;
    proxy_cache my_cache;
    proxy_cache_valid 200 10m;
    add_header X-Cache-Status $upstream_cache_status;
}
```

## 传递真实 IP

```nginx
set_real_ip_from 10.0.0.0/8;
real_ip_header X-Forwarded-For;
real_ip_recursive on;
```
