---
sidebar_position: 9
title: 性能优化
description: Nginx 性能调优与最佳实践
---

# 性能优化

## Worker 进程优化

```nginx
# 工作进程数（通常设为 CPU 核心数）
worker_processes auto;

# 每个 worker 的最大连接数
worker_connections 10240;

# 文件描述符限制
worker_rlimit_nofile 65535;

events {
    use epoll;
    multi_accept on;
}
```

## 连接优化

```nginx
http {
    # 开启长连接
    keepalive_timeout 65;
    keepalive_requests 1000;

    # 文件传输优化
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
}
```

## Gzip 压缩

```nginx
gzip on;
gzip_comp_level 5;
gzip_min_length 1k;
gzip_types text/plain text/css application/json application/javascript text/xml;
gzip_vary on;
```

## 缓冲区优化

```nginx
# 客户端请求缓冲
client_body_buffer_size 16k;
client_header_buffer_size 1k;
large_client_header_buffers 4 8k;

# 代理缓冲
proxy_buffer_size 4k;
proxy_buffers 8 32k;
proxy_busy_buffers_size 64k;
```

## 静态资源缓存

```nginx
location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
    expires 30d;
    add_header Cache-Control "public, no-transform";
    access_log off;
}
```

## 开启 HTTP/2

```nginx
server {
    listen 443 ssl http2;
    # ...
}
```

## Open File Cache

```nginx
open_file_cache max=10000 inactive=20s;
open_file_cache_valid 30s;
open_file_cache_min_uses 2;
open_file_cache_errors on;
```

## Upstream 保持连接

```nginx
upstream backend {
    server 127.0.0.1:8080;
    keepalive 32;
}

location / {
    proxy_pass http://backend;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
}
```

## 系统级优化

```bash
# /etc/sysctl.conf
net.core.somaxconn = 65535
net.ipv4.tcp_max_tw_buckets = 5000
net.ipv4.tcp_fin_timeout = 30
net.ipv4.tcp_keepalive_time = 1200
```

## 监控指标

```nginx
location /nginx_status {
    stub_status on;
    allow 127.0.0.1;
    deny all;
}
```

访问 `/nginx_status` 查看：

- Active connections: 活跃连接数
- accepts: 接受的连接数
- handled: 处理的连接数
- requests: 请求数
