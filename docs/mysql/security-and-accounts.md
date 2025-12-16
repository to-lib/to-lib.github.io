---
sidebar_position: 8
title: 账号、权限与安全
---

# MySQL 账号、权限与安全

> [!IMPORTANT] > **原则**：业务账号遵循“最小权限”，运维账号可审计可追踪，root 只用于必要的管理操作。

## 账号体系设计

建议至少区分三类账号：

- **DBA/运维账号**：具备管理能力，但仍建议限定来源 IP，并开启审计。
- **应用账号**：只授予业务库的 DML 权限（必要时再加 DDL）。
- **只读账号**：用于报表/查询/只读 API。

## 用户与权限基础

### 创建用户

```sql
CREATE USER 'app_user'@'%' IDENTIFIED BY 'ChangeMe_App_123!';
CREATE USER 'report_user'@'%' IDENTIFIED BY 'ChangeMe_Report_123!';
```

限制来源更安全（优先）：

```sql
CREATE USER 'app_user'@'10.0.%' IDENTIFIED BY 'ChangeMe_App_123!';
```

### 授权与回收

常见应用授权：

```sql
GRANT SELECT, INSERT, UPDATE, DELETE ON app_db.* TO 'app_user'@'%';
```

只读授权：

```sql
GRANT SELECT ON app_db.* TO 'report_user'@'%';
```

回收权限：

```sql
REVOKE DELETE ON app_db.* FROM 'app_user'@'%';
```

刷新权限：

```sql
FLUSH PRIVILEGES;
```

### 查看权限

```sql
SHOW GRANTS FOR 'app_user'@'%';
```

## 角色（Role）

在 MySQL 8.0+ 中，可以用角色管理权限，减少重复授权。

```sql
CREATE ROLE 'role_app_rw';
GRANT SELECT, INSERT, UPDATE, DELETE ON app_db.* TO 'role_app_rw';
GRANT 'role_app_rw' TO 'app_user'@'%';
SET DEFAULT ROLE 'role_app_rw' TO 'app_user'@'%';
```

## 密码与认证安全

### 密码策略

- 使用强密码（长度、复杂度、不可复用）。
- 不在代码仓库中明文保存密码。
- 定期轮换，并确保应用支持无缝切换（双账号/灰度）。

### 认证插件（MySQL 8.0）

检查认证插件：

```sql
SELECT user, host, plugin FROM mysql.user;
```

一般建议保持默认 `caching_sha2_password`。

## 连接安全

### TLS/SSL

如果是跨机器访问生产库，建议开启 TLS，避免明文传输。

排查客户端是否使用 SSL：

```sql
SHOW STATUS LIKE 'Ssl_cipher';
```

## 数据库安全基线

### 1) 禁止 root 远程登录

- root 限定为 `'root'@'localhost'`
- 管理操作走跳板机或 VPN

检查 root 的 host：

```sql
SELECT user, host FROM mysql.user WHERE user = 'root';
```

### 2) 限制高危权限

业务账号避免：

- `GRANT OPTION`
- `SUPER` / `SYSTEM_USER`
- `FILE`
- `SHUTDOWN`

### 3) 限制 FILE 与导入导出

如果确实需要导入导出，建议使用受控目录并设置：

```sql
SHOW VARIABLES LIKE 'secure_file_priv';
```

### 4) 防注入与审计

- 应用侧使用预编译/参数绑定（参见：[/docs/mysql/best-practices](/docs/mysql/best-practices)）。
- 重要操作开启审计（企业版/第三方方案）。

## 常见安全操作

### 修改密码

```sql
ALTER USER 'app_user'@'%' IDENTIFIED BY 'New_ChangeMe_App_123!';
```

### 锁定/解锁用户

```sql
ALTER USER 'app_user'@'%' ACCOUNT LOCK;
ALTER USER 'app_user'@'%' ACCOUNT UNLOCK;
```

### 删除用户

```sql
DROP USER 'app_user'@'%';
```

## 推荐实践清单

- **应用账号**：只授予业务库必要权限
- **账号来源**：尽量限制网段/固定 IP
- **root 使用**：最小化且可审计
- **备份**：备份文件加密、异地存放（参考：[/docs/mysql/backup-recovery](/docs/mysql/backup-recovery)）
- **日志**：保留错误日志、慢日志

## 下一步

- 连接与初始化：[/docs/mysql/installation-and-connection](/docs/mysql/installation-and-connection)
- 慢查询与监控排障：[/docs/mysql/monitoring-and-troubleshooting](/docs/mysql/monitoring-and-troubleshooting)
