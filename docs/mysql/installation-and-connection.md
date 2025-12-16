---
sidebar_position: 2
title: 安装与连接
---

# MySQL 安装与连接

> [!TIP] > **目标**：完成 MySQL 安装、初始化与基础连接配置，建立一套“可复现、可迁移”的本地/服务器环境。

## 版本与发行版选择

- **MySQL Community Server**：最常用的社区版。
- **MySQL 8.0+**：建议优先使用，默认认证插件、字符集、性能与可观测性均更完善。

## 安装方式

### macOS（Homebrew）

```bash
brew install mysql
brew services start mysql
```

常用检查：

```bash
mysql --version
brew services list | grep mysql
```

首次安装后可执行安全初始化（可选，但建议）：

```bash
mysql_secure_installation
```

### Docker（推荐：环境隔离、可复现）

```bash
docker run -d \
  --name mysql8 \
  -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=ChangeMe_123! \
  -e MYSQL_DATABASE=app_db \
  -e TZ=Asia/Shanghai \
  mysql:8.0
```

进入容器：

```bash
docker exec -it mysql8 mysql -uroot -p
```

### Linux（通用思路）

不同发行版的包管理命令不同，但核心点一致：

- 安装服务端
- 启动服务
- 设置开机自启
- 初始化 root 账号/密码

安装完成后重点关注：

- `mysqld` 是否已启动
- 数据目录（datadir）与权限
- 配置文件加载路径

## 初始化与登录

### 使用 mysql 客户端连接

```bash
mysql -h 127.0.0.1 -P 3306 -u root -p
```

### 创建数据库与用户（推荐：业务账号最小权限）

```sql
CREATE DATABASE IF NOT EXISTS app_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'app_user'@'%' IDENTIFIED BY 'ChangeMe_App_123!';
GRANT SELECT, INSERT, UPDATE, DELETE ON app_db.* TO 'app_user'@'%';
FLUSH PRIVILEGES;
```

## 关键基础配置（建议放到 my.cnf / my.ini）

以下是更偏“通用”的基础配置方向（示例仅供参考，生产环境要结合内存与负载评估）：

- 字符集与排序规则
- 时区
- 慢查询
- Binlog（如果要做复制/增量恢复）

```ini
[mysqld]
character-set-server = utf8mb4
collation-server = utf8mb4_unicode_ci

default-time-zone = +08:00

slow_query_log = 1
long_query_time = 1

# 如需主从复制/增量恢复再开启
# log_bin = mysql-bin
# binlog_format = ROW
```

## 连接与认证常见问题

### 1) Host 与 localhost 的区别

- `-h localhost`：很多环境会走 **Unix Socket**。
- `-h 127.0.0.1`：走 **TCP**。

排障时建议明确使用 `127.0.0.1`，避免 socket 路径不一致导致误判。

### 2) MySQL 8.0 认证插件

MySQL 8.0 默认常见为 `caching_sha2_password`。如果你的客户端过旧可能连不上。

检查用户认证方式：

```sql
SELECT user, host, plugin FROM mysql.user;
```

如果确实需要兼容旧客户端，可为某个用户设置 `mysql_native_password`（不建议对 root 这么做）：

```sql
ALTER USER 'app_user'@'%' IDENTIFIED WITH mysql_native_password BY 'ChangeMe_App_123!';
FLUSH PRIVILEGES;
```

## 应用连接串示例

### JDBC

```text
jdbc:mysql://127.0.0.1:3306/app_db?useUnicode=true&characterEncoding=utf8&useSSL=false&serverTimezone=Asia/Shanghai
```

### 常见排错顺序

- **账号/密码**是否正确
- **权限**是否包含目标库/表
- **端口**是否监听（3306）
- **防火墙/安全组**是否放行
- **bind-address** 是否限制

## 下一步

- 学习表结构与数据类型：[/docs/mysql/data-types](/docs/mysql/data-types)
- 熟悉 SQL 语法：[/docs/mysql/sql-syntax](/docs/mysql/sql-syntax)
- 账号与权限与安全：[/docs/mysql/security-and-accounts](/docs/mysql/security-and-accounts)
