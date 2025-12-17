---
sidebar_position: 2
title: 安装配置
description: Nginx 在各平台的安装与基本配置指南
---

# Nginx 安装配置

## 各平台安装

### macOS

```bash
# 使用 Homebrew 安装
brew install nginx

# 启动 Nginx
brew services start nginx

# 或手动启动
nginx

# 默认配置路径
/opt/homebrew/etc/nginx/nginx.conf  # Apple Silicon
/usr/local/etc/nginx/nginx.conf      # Intel Mac
```

### Ubuntu / Debian

```bash
# 更新包索引
sudo apt update

# 安装 Nginx
sudo apt install nginx

# 启动并设置开机启动
sudo systemctl start nginx
sudo systemctl enable nginx

# 查看状态
sudo systemctl status nginx
```

### CentOS / RHEL / Rocky Linux

```bash
# 安装 EPEL 仓库（如需要）
sudo yum install epel-release

# 安装 Nginx
sudo yum install nginx

# 启动并设置开机启动
sudo systemctl start nginx
sudo systemctl enable nginx
```

### Docker

```bash
# 拉取官方镜像
docker pull nginx:latest

# 运行 Nginx 容器
docker run -d \
  --name nginx \
  -p 80:80 \
  -p 443:443 \
  -v /path/to/nginx.conf:/etc/nginx/nginx.conf:ro \
  -v /path/to/html:/usr/share/nginx/html:ro \
  nginx:latest
```

```yaml title="docker-compose.yml"
version: "3.8"
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./html:/usr/share/nginx/html:ro
      - ./ssl:/etc/nginx/ssl:ro
    restart: unless-stopped
```

### 源码编译安装

```bash
# 安装依赖（Ubuntu/Debian）
sudo apt install build-essential libpcre3 libpcre3-dev zlib1g zlib1g-dev libssl-dev

# 下载源码
wget https://nginx.org/download/nginx-1.24.0.tar.gz
tar -xzf nginx-1.24.0.tar.gz
cd nginx-1.24.0

# 配置（包含常用模块）
./configure \
  --prefix=/usr/local/nginx \
  --with-http_ssl_module \
  --with-http_v2_module \
  --with-http_realip_module \
  --with-http_gzip_static_module \
  --with-http_stub_status_module \
  --with-stream \
  --with-stream_ssl_module

# 编译安装
make && sudo make install
```

## 目录结构

### 标准安装目录

```
/etc/nginx/                    # 配置目录
├── nginx.conf                 # 主配置文件
├── conf.d/                    # 额外配置目录
│   └── default.conf           # 默认站点配置
├── sites-available/           # 可用站点配置（Debian 风格）
├── sites-enabled/             # 已启用站点配置
├── mime.types                 # MIME 类型映射
├── fastcgi_params             # FastCGI 参数
├── proxy_params               # 代理参数
└── ssl/                       # SSL 证书目录

/var/log/nginx/                # 日志目录
├── access.log                 # 访问日志
└── error.log                  # 错误日志

/var/www/html/                 # 默认网站根目录
/usr/share/nginx/html/         # 部分系统的默认目录
```

## 常用命令

```bash
# 启动 Nginx
sudo nginx

# 停止 Nginx
sudo nginx -s stop      # 快速停止
sudo nginx -s quit      # 优雅停止（等待请求完成）

# 重载配置（不中断服务）
sudo nginx -s reload

# 重新打开日志文件
sudo nginx -s reopen

# 测试配置文件语法
sudo nginx -t
sudo nginx -T          # 测试并打印配置

# 查看版本和编译信息
nginx -v               # 版本号
nginx -V               # 版本和编译参数

# 指定配置文件启动
sudo nginx -c /path/to/nginx.conf

# 查看 Nginx 进程
ps aux | grep nginx

# 查看监听端口
sudo netstat -tlnp | grep nginx
# 或
sudo ss -tlnp | grep nginx
```

## systemd 管理

```bash
# 启动/停止/重启
sudo systemctl start nginx
sudo systemctl stop nginx
sudo systemctl restart nginx

# 重载配置
sudo systemctl reload nginx

# 查看状态
sudo systemctl status nginx

# 开机启动
sudo systemctl enable nginx
sudo systemctl disable nginx

# 查看日志
sudo journalctl -u nginx
sudo journalctl -u nginx -f  # 实时查看
```

## 基础配置结构

```nginx title="/etc/nginx/nginx.conf"
# 全局块
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

# 事件块
events {
    worker_connections 1024;
    use epoll;  # Linux 推荐
    multi_accept on;
}

# HTTP 块
http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # 日志格式
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;

    # 性能优化
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Gzip 压缩
    gzip on;
    gzip_disable "msie6";
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css application/json application/javascript
               text/xml application/xml application/xml+rss text/javascript;

    # 包含其他配置
    include /etc/nginx/conf.d/*.conf;
    include /etc/nginx/sites-enabled/*;
}
```

## 验证安装

```bash
# 1. 检查 Nginx 是否运行
sudo systemctl status nginx

# 2. 测试配置语法
sudo nginx -t

# 3. 浏览器访问
curl http://localhost
# 或访问 http://your-server-ip

# 4. 查看版本
nginx -v
```

## 常见问题

### 端口被占用

```bash
# 查看端口占用
sudo lsof -i :80
sudo netstat -tlnp | grep :80

# 停止占用进程或修改 Nginx 端口
listen 8080;
```

### 权限问题

```bash
# 确保 Nginx 用户有权限访问网站目录
sudo chown -R nginx:nginx /var/www/html
sudo chmod -R 755 /var/www/html

# 确保日志目录可写
sudo chown -R nginx:nginx /var/log/nginx
```

### SELinux 问题（CentOS/RHEL）

```bash
# 临时禁用
sudo setenforce 0

# 永久禁用（不推荐生产环境）
sudo sed -i 's/SELINUX=enforcing/SELINUX=disabled/' /etc/selinux/config

# 正确方式：配置 SELinux 策略
sudo setsebool -P httpd_can_network_connect 1
```

### 防火墙配置

```bash
# Ubuntu (UFW)
sudo ufw allow 'Nginx Full'
# 或
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# CentOS (firewalld)
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```
