---
sidebar_position: 4
title: 静态文件服务
description: 使用 Nginx 提供静态文件服务的配置指南
---

# 静态文件服务

## 基本配置

```nginx
server {
    listen 80;
    server_name static.example.com;

    # 网站根目录
    root /var/www/static;

    # 默认首页
    index index.html index.htm;

    location / {
        try_files $uri $uri/ =404;
    }
}
```

## 静态资源优化

### Gzip 压缩

```nginx
http {
    # 开启 Gzip
    gzip on;

    # 最小压缩文件大小
    gzip_min_length 1k;

    # 压缩级别 1-9，建议 4-6
    gzip_comp_level 5;

    # 压缩缓冲区
    gzip_buffers 16 8k;

    # 压缩的 MIME 类型
    gzip_types
        text/plain
        text/css
        text/javascript
        text/xml
        application/json
        application/javascript
        application/xml
        application/xml+rss
        application/x-javascript
        image/svg+xml;

    # 代理请求也压缩
    gzip_proxied any;

    # 添加 Vary 头
    gzip_vary on;

    # 禁用 IE6 Gzip
    gzip_disable "msie6";
}
```

### 预压缩（Gzip Static）

```nginx
# 需要编译时 --with-http_gzip_static_module
location / {
    gzip_static on;
    # 会优先查找 .gz 文件
    # 如 /style.css 会先找 /style.css.gz
}
```

### Brotli 压缩

```nginx
# 需要第三方模块
load_module modules/ngx_http_brotli_filter_module.so;
load_module modules/ngx_http_brotli_static_module.so;

http {
    brotli on;
    brotli_comp_level 6;
    brotli_types text/plain text/css application/json application/javascript
                 text/xml application/xml application/xml+rss text/javascript
                 image/svg+xml;

    # 预压缩
    brotli_static on;
}
```

## 缓存控制

### 浏览器缓存

```nginx
# 图片缓存 30 天
location ~* \.(jpg|jpeg|png|gif|ico|webp|svg)$ {
    expires 30d;
    add_header Cache-Control "public, no-transform";
    access_log off;
}

# CSS/JS 缓存 7 天
location ~* \.(css|js)$ {
    expires 7d;
    add_header Cache-Control "public, no-transform";
}

# 字体文件缓存 1 年
location ~* \.(woff|woff2|ttf|otf|eot)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
    access_log off;
}

# HTML 不缓存或短缓存
location ~* \.html$ {
    expires -1;
    add_header Cache-Control "no-cache, no-store, must-revalidate";
}
```

### 使用 ETag 和 Last-Modified

```nginx
location /static/ {
    # 默认开启
    etag on;

    # 如果文件未修改，返回 304
    if_modified_since before;
}
```

### 版本化资源（长期缓存）

```nginx
# 带 hash 的资源，缓存一年
location ~* \.[a-f0-9]{8,}\.(css|js)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}

# 或使用查询参数版本
location ~* ^.+\.(css|js)$ {
    if ($args ~* "v=") {
        expires 1y;
    }
}
```

## Sendfile 优化

```nginx
http {
    # 开启 sendfile
    sendfile on;

    # 配合 sendfile 使用
    tcp_nopush on;    # 在一个数据包中发送 HTTP 头
    tcp_nodelay on;   # 禁用 Nagle 算法

    # 直接 IO（大文件场景）
    # directio 4m;

    # AIO（异步 IO，需要系统支持）
    # aio on;
}
```

## 目录浏览

```nginx
location /files/ {
    root /var/www;

    # 开启目录浏览
    autoindex on;

    # 显示确切文件大小
    autoindex_exact_size off;

    # 显示本地时间
    autoindex_localtime on;

    # JSON 格式（用于 API）
    # autoindex_format json;
}
```

### 美化目录列表

```nginx
location /files/ {
    autoindex on;
    autoindex_exact_size off;
    autoindex_localtime on;

    # 添加自定义样式
    add_before_body /autoindex-header.html;
    add_after_body /autoindex-footer.html;
}
```

## 文件下载

```nginx
location /download/ {
    root /var/www;

    # 强制下载而非预览
    add_header Content-Disposition 'attachment';

    # 或根据文件类型
    if ($request_filename ~* \.(pdf|zip|rar)$) {
        add_header Content-Disposition 'attachment';
    }
}

# 限制下载速度
location /download/ {
    root /var/www;
    limit_rate 1m;           # 限速 1MB/s
    limit_rate_after 10m;    # 前 10MB 不限速
}
```

## 防盗链

```nginx
location ~* \.(jpg|jpeg|png|gif|webp|svg)$ {
    # 合法来源
    valid_referers none blocked server_names
        *.example.com
        example.com
        ~\.google\.
        ~\.baidu\.;

    # 非法来源返回 403 或替换图片
    if ($invalid_referer) {
        return 403;
        # 或 rewrite ^ /images/forbidden.png break;
    }
}
```

## 跨域资源共享（CORS）

```nginx
location /api/ {
    # 允许的源
    add_header Access-Control-Allow-Origin "*";
    # 或指定域名
    # add_header Access-Control-Allow-Origin "https://example.com";

    add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
    add_header Access-Control-Allow-Headers "Origin, Content-Type, Accept";
    add_header Access-Control-Max-Age 3600;

    # 处理预检请求
    if ($request_method = OPTIONS) {
        return 204;
    }
}

# 动态 CORS
map $http_origin $cors_origin {
    default "";
    "~^https?://(.+\.)?example\.com$" $http_origin;
    "~^https?://localhost(:\d+)?$" $http_origin;
}

server {
    location /api/ {
        add_header Access-Control-Allow-Origin $cors_origin always;
        add_header Access-Control-Allow-Credentials true always;
    }
}
```

## SPA 应用配置

```nginx
server {
    listen 80;
    server_name app.example.com;
    root /var/www/spa;

    # 静态资源
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;
    }

    # HTML 不缓存
    location ~* \.html$ {
        expires -1;
        add_header Cache-Control "no-cache, no-store, must-revalidate";
    }

    # 所有路由回退到 index.html
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

## 多网站配置

```nginx
# 网站 1
server {
    listen 80;
    server_name site1.example.com;
    root /var/www/site1;

    location / {
        try_files $uri $uri/ =404;
    }
}

# 网站 2
server {
    listen 80;
    server_name site2.example.com;
    root /var/www/site2;

    location / {
        try_files $uri $uri/ =404;
    }
}

# 默认服务器
server {
    listen 80 default_server;
    server_name _;
    return 444;  # 关闭连接
}
```

## 媒体文件处理

### 图片处理

```nginx
# 需要 ngx_http_image_filter_module
load_module modules/ngx_http_image_filter_module.so;

location ~* ^/images/(.+)_(\d+)x(\d+)\.(jpg|png|gif)$ {
    set $image_path /images/$1.$4;
    set $width $2;
    set $height $3;

    image_filter resize $width $height;
    image_filter_jpeg_quality 85;
    image_filter_buffer 10M;

    try_files $image_path =404;
}
```

### 视频流（MP4）

```nginx
# 需要 --with-http_mp4_module
location ~* \.mp4$ {
    mp4;
    mp4_buffer_size 1m;
    mp4_max_buffer_size 5m;
}

# FLV
location ~* \.flv$ {
    flv;
}
```

## 安全配置

```nginx
location /static/ {
    root /var/www;

    # 禁止访问隐藏文件
    location ~ /\. {
        deny all;
    }

    # 禁止访问备份文件
    location ~* \.(bak|swp|~)$ {
        deny all;
    }

    # 禁止执行脚本
    location ~* \.(php|pl|py|cgi)$ {
        deny all;
    }
}
```
