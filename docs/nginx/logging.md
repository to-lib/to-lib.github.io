---
sidebar_position: 10
title: 日志配置
description: Nginx 访问日志与错误日志配置
---

# 日志配置

## 访问日志

### 默认格式

```nginx
log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                '$status $body_bytes_sent "$http_referer" '
                '"$http_user_agent" "$http_x_forwarded_for"';

access_log /var/log/nginx/access.log main;
```

### JSON 格式

```nginx
log_format json escape=json '{'
    '"time":"$time_iso8601",'
    '"remote_addr":"$remote_addr",'
    '"method":"$request_method",'
    '"uri":"$uri",'
    '"status":"$status",'
    '"body_bytes_sent":"$body_bytes_sent",'
    '"request_time":"$request_time",'
    '"upstream_response_time":"$upstream_response_time",'
    '"user_agent":"$http_user_agent"'
'}';

access_log /var/log/nginx/access.json.log json;
```

## 错误日志

```nginx
# 错误级别：debug, info, notice, warn, error, crit, alert, emerg
error_log /var/log/nginx/error.log warn;

# 调试模式
error_log /var/log/nginx/error.log debug;
```

## 按站点分离日志

```nginx
server {
    server_name site1.example.com;
    access_log /var/log/nginx/site1.access.log main;
    error_log /var/log/nginx/site1.error.log warn;
}

server {
    server_name site2.example.com;
    access_log /var/log/nginx/site2.access.log main;
    error_log /var/log/nginx/site2.error.log warn;
}
```

## 条件日志

```nginx
# 排除健康检查
map $request_uri $loggable {
    /health 0;
    /ping 0;
    default 1;
}

access_log /var/log/nginx/access.log main if=$loggable;

# 排除静态资源
location ~* \.(jpg|png|gif|css|js)$ {
    access_log off;
}
```

## 日志轮转

```bash
# /etc/logrotate.d/nginx
/var/log/nginx/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 nginx adm
    sharedscripts
    postrotate
        [ -f /var/run/nginx.pid ] && kill -USR1 `cat /var/run/nginx.pid`
    endscript
}
```

## 常用日志变量

| 变量                      | 说明         |
| ------------------------- | ------------ |
| `$remote_addr`            | 客户端 IP    |
| `$time_local`             | 本地时间     |
| `$request`                | 请求行       |
| `$status`                 | 响应状态码   |
| `$body_bytes_sent`        | 发送字节数   |
| `$request_time`           | 请求处理时间 |
| `$upstream_response_time` | 上游响应时间 |
| `$http_user_agent`        | User-Agent   |
| `$http_referer`           | Referer      |

## 缓冲日志

```nginx
access_log /var/log/nginx/access.log main buffer=32k flush=5s;
```
