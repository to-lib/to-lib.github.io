---
sidebar_position: 15
title: 安全管理
---

# PostgreSQL 安全管理

数据库安全是保护敏感数据的关键，包括用户管理、权限控制、连接加密等方面。

## 👤 用户和角色管理

### 创建用户/角色

```sql
-- 创建用户（带登录权限的角色）
CREATE USER myuser WITH PASSWORD 'secure_password';

-- 创建角色（无登录权限）
CREATE ROLE myrole;

-- 带更多选项
CREATE USER admin WITH
    PASSWORD 'admin_pass'
    SUPERUSER
    CREATEDB
    CREATEROLE
    VALID UNTIL '2025-12-31';
```

### 用户属性

```sql
-- 修改密码
ALTER USER myuser WITH PASSWORD 'new_password';

-- 设置过期时间
ALTER USER myuser VALID UNTIL '2025-06-30';

-- 授予/撤销权限
ALTER USER myuser CREATEDB;
ALTER USER myuser NOCREATEDB;

-- 删除用户
DROP USER myuser;
```

### 角色继承

```sql
-- 创建角色组
CREATE ROLE readonly;
CREATE ROLE readwrite;

-- 授予角色
GRANT readonly TO myuser;
GRANT readwrite TO admin_user;

-- 设置默认角色
SET ROLE readonly;
RESET ROLE;
```

## 🔐 权限控制

### 数据库权限

```sql
-- 授予连接权限
GRANT CONNECT ON DATABASE mydb TO myuser;

-- 授予创建模式权限
GRANT CREATE ON DATABASE mydb TO myuser;

-- 撤销权限
REVOKE CONNECT ON DATABASE mydb FROM myuser;
```

### 模式权限

```sql
-- 授予模式使用权限
GRANT USAGE ON SCHEMA public TO myuser;

-- 授予创建对象权限
GRANT CREATE ON SCHEMA public TO myuser;
```

### 表权限

```sql
-- 授予单表权限
GRANT SELECT ON users TO myuser;
GRANT INSERT, UPDATE, DELETE ON users TO myuser;
GRANT ALL PRIVILEGES ON users TO admin_user;

-- 授予所有表权限
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly;

-- 设置默认权限（未来创建的表）
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT SELECT ON TABLES TO readonly;
```

### 列级权限

```sql
-- 只允许查看特定列
GRANT SELECT (id, name, email) ON users TO myuser;

-- 只允许更新特定列
GRANT UPDATE (email, phone) ON users TO myuser;
```

### 行级安全（RLS）

```sql
-- 启用行级安全
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

-- 创建策略
CREATE POLICY user_orders ON orders
    FOR ALL
    TO myuser
    USING (user_id = current_user_id());

-- 创建只读策略
CREATE POLICY read_own_orders ON orders
    FOR SELECT
    USING (user_id = current_user_id());

-- 删除策略
DROP POLICY user_orders ON orders;
```

## 🔒 连接安全

### pg_hba.conf 配置

```conf
# TYPE  DATABASE  USER  ADDRESS        METHOD

# 本地连接
local   all       postgres               peer
local   all       all                    md5

# IPv4 连接
host    all       all    127.0.0.1/32   scram-sha-256
host    all       all    10.0.0.0/8     scram-sha-256

# IPv6 连接
host    all       all    ::1/128        scram-sha-256

# SSL 连接
hostssl all       all    0.0.0.0/0      scram-sha-256

# 拒绝特定用户
host    all       baduser 0.0.0.0/0     reject
```

### 认证方式

| 方式              | 描述                 |
| ----------------- | -------------------- |
| **trust**         | 无需密码（不推荐）   |
| **md5**           | MD5 密码加密         |
| **scram-sha-256** | 更安全的加密（推荐） |
| **peer**          | 操作系统用户验证     |
| **cert**          | SSL 证书验证         |

### 修改默认加密方式

```conf
# postgresql.conf
password_encryption = scram-sha-256
```

## 🔐 SSL/TLS 加密

### 生成证书

```bash
# 生成私钥
openssl genrsa -out server.key 2048

# 生成证书
openssl req -new -key server.key -out server.csr
openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt

# 设置权限
chmod 600 server.key
chown postgres:postgres server.key server.crt
```

### 配置 SSL

```conf
# postgresql.conf
ssl = on
ssl_cert_file = '/var/lib/postgresql/server.crt'
ssl_key_file = '/var/lib/postgresql/server.key'
ssl_ca_file = '/var/lib/postgresql/ca.crt'  # 可选
```

### 强制 SSL 连接

```conf
# pg_hba.conf
hostssl all all 0.0.0.0/0 scram-sha-256
```

### 客户端连接

```bash
psql "host=server sslmode=require dbname=mydb user=myuser"
```

## 📝 审计日志

### 启用日志

```conf
# postgresql.conf
logging_collector = on
log_directory = '/var/log/postgresql'
log_filename = 'postgresql-%Y-%m-%d.log'
log_rotation_age = 1d
log_rotation_size = 100MB

# 记录内容
log_connections = on
log_disconnections = on
log_statement = 'all'  # none, ddl, mod, all
log_duration = on
```

### pgAudit 扩展

```sql
-- 安装扩展
CREATE EXTENSION pgaudit;

-- 配置审计
-- postgresql.conf
pgaudit.log = 'read, write, ddl'
```

## 🛡️ 安全最佳实践

1. **使用强密码和 scram-sha-256**
2. **启用 SSL 加密连接**
3. **限制网络访问（pg_hba.conf）**
4. **使用最小权限原则**
5. **定期轮换密码**
6. **启用审计日志**
7. **使用行级安全策略**
8. **定期备份**

## 📊 安全检查

```sql
-- 查看用户权限
SELECT grantee, privilege_type
FROM information_schema.role_table_grants
WHERE table_name = 'users';

-- 查看用户属性
SELECT usename, usecreatedb, usesuper
FROM pg_user;

-- 查看活动连接
SELECT usename, client_addr, ssl
FROM pg_stat_activity;
```

## 📚 相关资源

- [基础概念](/docs/postgres/basic-concepts)
- [常见问题](/docs/postgres/faq)
