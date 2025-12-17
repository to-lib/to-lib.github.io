---
sidebar_position: 8
title: 安全配置
description: Nginx 安全加固与防护配置
---

# 安全配置

## 隐藏版本信息

```nginx
http {
    server_tokens off;
}
```

## 访问控制

```nginx
# IP 黑白名单
location /admin/ {
    allow 192.168.1.0/24;
    allow 10.0.0.1;
    deny all;
}

# 基本认证
location /protected/ {
    auth_basic "Restricted";
    auth_basic_user_file /etc/nginx/.htpasswd;
}
```

生成密码文件：

```bash
htpasswd -c /etc/nginx/.htpasswd username
```

## 请求限制

```nginx
# 限制请求频率
http {
    limit_req_zone $binary_remote_addr zone=req_limit:10m rate=10r/s;

    server {
        location /api/ {
            limit_req zone=req_limit burst=20 nodelay;
        }
    }
}

# 限制并发连接
http {
    limit_conn_zone $binary_remote_addr zone=conn_limit:10m;

    server {
        location / {
            limit_conn conn_limit 10;
        }
    }
}
```

## 防止常见攻击

```nginx
# 防止点击劫持
add_header X-Frame-Options "SAMEORIGIN" always;

# 防止 XSS
add_header X-XSS-Protection "1; mode=block" always;

# 防止 MIME 类型嗅探
add_header X-Content-Type-Options "nosniff" always;

# 内容安全策略
add_header Content-Security-Policy "default-src 'self'" always;

# 禁用服务器方法
if ($request_method !~ ^(GET|POST|HEAD)$) {
    return 405;
}

# 禁止访问隐藏文件
location ~ /\. {
    deny all;
}
```

## 请求体大小限制

```nginx
client_max_body_size 10m;
client_body_buffer_size 128k;
```

## 超时设置

```nginx
client_body_timeout 10s;
client_header_timeout 10s;
send_timeout 10s;
keepalive_timeout 65s;
```

## SSL 安全配置

```nginx
ssl_protocols TLSv1.2 TLSv1.3;
ssl_prefer_server_ciphers on;
ssl_session_tickets off;

add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
```
