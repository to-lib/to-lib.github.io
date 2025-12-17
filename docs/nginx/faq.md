---
sidebar_position: 12
title: 常见问题
description: Nginx 常见问题与解决方案
---

# 常见问题

## 安装与启动

### 端口 80 被占用

```bash
# 查看端口占用
sudo lsof -i :80
sudo netstat -tlnp | grep :80

# 停止占用进程或修改 Nginx 端口
listen 8080;
```

### 权限不足

```bash
# 确保目录权限
sudo chown -R nginx:nginx /var/www/html
sudo chmod -R 755 /var/www/html
```

### 配置语法错误

```bash
# 测试配置
nginx -t

# 输出示例
nginx: [emerg] unknown directive "servrr" in /etc/nginx/nginx.conf:10
```

## 代理问题

### 502 Bad Gateway

常见原因：

- 后端服务未启动
- 后端地址配置错误
- 超时时间太短

```nginx
proxy_connect_timeout 60s;
proxy_read_timeout 60s;
proxy_send_timeout 60s;
```

### 504 Gateway Timeout

```nginx
# 增加超时时间
proxy_read_timeout 300s;

# 检查后端服务响应时间
```

### 无法获取真实 IP

```nginx
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

# 多层代理
set_real_ip_from 10.0.0.0/8;
real_ip_header X-Forwarded-For;
real_ip_recursive on;
```

## SSL 问题

### 证书链不完整

```nginx
# 合并证书
cat domain.crt intermediate.crt > fullchain.crt

ssl_certificate /path/to/fullchain.crt;
```

### 混合内容警告

确保所有资源都使用 HTTPS。

## 性能问题

### Nginx 进程 CPU 占用高

```nginx
# 检查 worker 配置
worker_processes auto;
worker_connections 1024;
```

### 连接数耗尽

```bash
# 增加系统限制
ulimit -n 65535

# nginx 配置
worker_rlimit_nofile 65535;
```

## 日志相关

### 日志增长过快

```bash
# 配置日志轮转
# /etc/logrotate.d/nginx
/var/log/nginx/*.log {
    daily
    rotate 7
    compress
}
```

### 关闭特定路径日志

```nginx
location /health {
    access_log off;
}
```

## 常用排查命令

```bash
# 查看 Nginx 进程
ps aux | grep nginx

# 查看错误日志
tail -f /var/log/nginx/error.log

# 测试连接
curl -I http://localhost

# 查看连接状态
netstat -an | grep :80 | wc -l
```
