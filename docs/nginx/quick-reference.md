---
sidebar_position: 11
title: 快速参考
description: Nginx 常用配置与命令速查表
---

# Nginx 快速参考

## 常用命令

```bash
# 启动/停止
nginx                    # 启动
nginx -s stop            # 快速停止
nginx -s quit            # 优雅停止
nginx -s reload          # 重载配置
nginx -s reopen          # 重新打开日志

# 测试配置
nginx -t                 # 测试语法
nginx -T                 # 测试并打印

# 查看信息
nginx -v                 # 版本
nginx -V                 # 版本和编译参数

# systemd
systemctl start nginx
systemctl stop nginx
systemctl reload nginx
systemctl status nginx
```

## 配置结构

```nginx
# 全局配置
user nginx;
worker_processes auto;

events {
    worker_connections 1024;
}

http {
    # HTTP 配置
    server {
        # 虚拟主机
        location / {
            # 位置匹配
        }
    }
}
```

## 常用配置模板

### 静态网站

```nginx
server {
    listen 80;
    server_name example.com;
    root /var/www/html;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }
}
```

### 反向代理

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### HTTPS

```nginx
server {
    listen 443 ssl http2;
    server_name example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
}
```

### 负载均衡

```nginx
upstream backend {
    server 10.0.0.1:8080;
    server 10.0.0.2:8080;
}

server {
    location / {
        proxy_pass http://backend;
    }
}
```

## Location 匹配优先级

| 语法       | 优先级 | 说明                 |
| ---------- | ------ | -------------------- |
| `= /path`  | 1      | 精确匹配             |
| `^~ /path` | 2      | 前缀匹配（优先）     |
| `~ regex`  | 3      | 正则（区分大小写）   |
| `~* regex` | 3      | 正则（不区分大小写） |
| `/path`    | 4      | 前缀匹配             |
| `/`        | 5      | 默认匹配             |

## 常用变量

| 变量              | 说明         |
| ----------------- | ------------ |
| `$host`           | 请求主机名   |
| `$uri`            | 请求 URI     |
| `$args`           | 查询参数     |
| `$remote_addr`    | 客户端 IP    |
| `$request_method` | 请求方法     |
| `$status`         | 响应状态码   |
| `$request_time`   | 请求处理时间 |

## 目录结构

```
/etc/nginx/
├── nginx.conf           # 主配置
├── conf.d/              # 站点配置
├── sites-available/     # 可用站点
├── sites-enabled/       # 启用站点
└── ssl/                 # 证书目录

/var/log/nginx/
├── access.log           # 访问日志
└── error.log            # 错误日志
```

## 常见端口

| 端口 | 用途      |
| ---- | --------- |
| 80   | HTTP      |
| 443  | HTTPS     |
| 8080 | 备用 HTTP |
