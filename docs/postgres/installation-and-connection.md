---
sidebar_position: 100
title: 安装与连接
---

# PostgreSQL 安装与连接

## 版本选择

- **PostgreSQL 16+**：推荐优先使用，功能与性能持续改进。
- **生产环境**：尽量避免跨大版本“原地升级”，建议走备份恢复或逻辑复制/迁移方案。

## 安装方式

### macOS（Homebrew）

```bash
# 安装
brew install postgresql@16

# 启动服务
brew services start postgresql@16

# 检查版本
psql --version
```

### Docker（推荐：环境隔离、可复现）

```bash
docker run --name postgres16 \
  -e POSTGRES_PASSWORD=ChangeMe_123! \
  -e POSTGRES_DB=app_db \
  -p 5432:5432 \
  -d postgres:16

# 进入容器
docker exec -it postgres16 psql -U postgres -d app_db
```

### Linux（通用思路）

不同发行版包管理与服务管理命令不同，但核心流程一致：

- 安装 PostgreSQL
- 启动服务并设置开机自启
- 确认数据目录权限、监听地址与端口
- 初始化业务账号与权限

安装完成后建议优先确认：

- `postgres` 进程是否存在
- 5432 端口是否监听
- `postgresql.conf` / `pg_hba.conf` 的实际加载路径

## 连接方式

### 1) 使用 psql 连接

```bash
# 本地 TCP
psql -h 127.0.0.1 -p 5432 -U postgres -d postgres

# 指定密码（不推荐长期使用，适合临时排障）
PGPASSWORD='ChangeMe_123!' psql -h 127.0.0.1 -U postgres -d postgres
```

### 2) URL 连接串

```text
postgresql://postgres:ChangeMe_123!@127.0.0.1:5432/app_db
```

常用参数：

- `sslmode=require`：强制 SSL
- `connect_timeout=5`：连接超时

示例：

```text
postgresql://postgres:ChangeMe_123!@127.0.0.1:5432/app_db?sslmode=disable&connect_timeout=5
```

### 3) JDBC

```text
jdbc:postgresql://127.0.0.1:5432/app_db
```

如需 SSL：

```text
jdbc:postgresql://127.0.0.1:5432/app_db?sslmode=require
```

## 初始化：创建数据库与业务账号（推荐最小权限）

```sql
-- 创建数据库
CREATE DATABASE app_db;

-- 创建业务账号
CREATE USER app_user WITH PASSWORD 'ChangeMe_App_123!';

-- 授予库级权限（连接 + 临时对象）
GRANT CONNECT, TEMP ON DATABASE app_db TO app_user;

-- 进入目标库
\c app_db

-- 建议：创建专用 schema，避免直接把对象放 public
CREATE SCHEMA IF NOT EXISTS app AUTHORIZATION app_user;

-- schema 使用与创建权限
GRANT USAGE, CREATE ON SCHEMA app TO app_user;

-- 如仍使用 public（不推荐），至少显式授权
-- GRANT USAGE, CREATE ON SCHEMA public TO app_user;
```

## psql 常用命令速记

```sql
-- 列出数据库
\l

-- 切换数据库
\c app_db

-- 列出 schema
\dn

-- 列出表
\dt

-- 查看表结构
\d app.users

-- 查看当前连接信息
\conninfo

-- 打开执行时间统计
\timing on
```

## 常见连接问题与排查

### 1) 连接被拒绝（Connection refused）

- 服务是否启动
- 端口是否监听（默认 5432）
- Docker 是否映射端口（`-p 5432:5432`）

### 2) 密码认证失败

- 检查用户名、密码、数据库名
- 检查 `pg_hba.conf` 认证方式（`md5` / `scram-sha-256`）
- 检查账号是否被禁用/过期

```sql
-- 查看用户
SELECT usename FROM pg_user;
```

### 3) 远程连不上

重点看两处配置：

- `postgresql.conf`：

```conf
listen_addresses = '*'
```

- `pg_hba.conf`：

```conf
# 允许某个网段访问
host  all  all  10.0.0.0/8  scram-sha-256
```

修改后通常需要重载/重启生效（方式依环境而定）。

## 下一步

- 学习基础概念：[/docs/postgres/basic-concepts](/docs/postgres/basic-concepts)
- 熟悉 SQL 语法：[/docs/postgres/sql-syntax](/docs/postgres/sql-syntax)
- 开始性能优化：[/docs/postgres/performance-optimization](/docs/postgres/performance-optimization)
