---
sidebar_position: 9
title: 视图与触发器
---

# MySQL 视图与触发器

## 视图 (View)

### 什么是视图

视图是虚拟表，基于 SQL 查询结果集。视图不存储数据，数据来自基表。

### 创建视图

```sql
-- 基本语法
CREATE VIEW view_name AS
SELECT column1, column2
FROM table_name
WHERE condition;

-- 示例：创建用户信息视图
CREATE VIEW v_user_info AS
SELECT id, username, email, created_at
FROM users
WHERE status = 1;

-- 使用视图
SELECT * FROM v_user_info;
```

### 视图的优点

- ✅ 简化复杂查询
- ✅ 提高数据安全性（隐藏敏感列）
- ✅ 数据独立性
- ✅ 代码复用

### 复杂视图示例

```sql
-- 订单统计视图
CREATE VIEW v_order_statistics AS
SELECT
    u.id AS user_id,
    u.username,
    COUNT(o.id) AS total_orders,
    SUM(o.amount) AS total_amount,
    AVG(o.amount) AS avg_amount
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.username;

-- 查询视图
SELECT * FROM v_order_statistics WHERE total_orders > 10;
```

### 可更新视图

```sql
-- 满足条件的视图可以更新
CREATE VIEW v_active_users AS
SELECT id, username, email FROM users WHERE status = 1;

-- 可以更新
UPDATE v_active_users SET email = 'new@example.com' WHERE id = 1;
INSERT INTO v_active_users (username, email) VALUES ('张三', 'zhangsan@example.com');
DELETE FROM v_active_users WHERE id = 1;
```

> [!WARNING]
> 不可更新的视图包括：包含聚合函数、DISTINCT、GROUP BY、UNION、子查询等的视图。

### 管理视图

```sql
-- 查看所有视图
SHOW FULL TABLES WHERE table_type = 'VIEW';

-- 查看视图定义
SHOW CREATE VIEW v_user_info;

-- 修改视图
CREATE OR REPLACE VIEW v_user_info AS
SELECT id, username, email, age FROM users;

-- 删除视图
DROP VIEW IF EXISTS v_user_info;
```

## 触发器 (Trigger)

### 什么是触发器

触发器是在特定事件发生时自动执行的 SQL 语句。

### 触发器类型

- **BEFORE INSERT** - 插入前触发
- **AFTER INSERT** - 插入后触发
- **BEFORE UPDATE** - 更新前触发
- **AFTER UPDATE** - 更新后触发
- **BEFORE DELETE** - 删除前触发
- **AFTER DELETE** - 删除后触发

### 创建触发器

```sql
DELIMITER //

CREATE TRIGGER trigger_name
{BEFORE | AFTER} {INSERT | UPDATE | DELETE}
ON table_name
FOR EACH ROW
BEGIN
    -- SQL 语句
END//

DELIMITER ;
```

### AFTER INSERT 触发器

```sql
-- 插入订单后，自动更新用户订单数
DELIMITER //

CREATE TRIGGER after_order_insert
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    UPDATE users
    SET total_orders = total_orders + 1
    WHERE id = NEW.user_id;
END//

DELIMITER ;
```

### BEFORE UPDATE 触发器

```sql
-- 更新前记录修改时间
DELIMITER //

CREATE TRIGGER before_user_update
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    SET NEW.updated_at = NOW();
END//

DELIMITER ;
```

### AFTER DELETE 触发器

```sql
-- 删除用户后，删除相关订单
DELIMITER //

CREATE TRIGGER after_user_delete
AFTER DELETE ON users
FOR EACH ROW
BEGIN
    DELETE FROM orders WHERE user_id = OLD.id;
END//

DELIMITER ;
```

### NEW 和 OLD 关键字

- **NEW** - 新数据（INSERT 和 UPDATE）
- **OLD** - 旧数据（UPDATE 和 DELETE）

```sql
-- 示例：记录价格变更
CREATE TABLE price_history (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    product_id BIGINT,
    old_price DECIMAL(10,2),
    new_price DECIMAL(10,2),
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

DELIMITER //

CREATE TRIGGER after_price_update
AFTER UPDATE ON products
FOR EACH ROW
BEGIN
    IF OLD.price != NEW.price THEN
        INSERT INTO price_history (product_id, old_price, new_price)
        VALUES (NEW.id, OLD.price, NEW.price);
    END IF;
END//

DELIMITER ;
```

### 管理触发器

```sql
-- 查看所有触发器
SHOW TRIGGERS;

-- 查看触发器定义
SHOW CREATE TRIGGER trigger_name;

-- 删除触发器
DROP TRIGGER IF EXISTS trigger_name;
```

## 实用示例

### 审计日志触发器

```sql
-- 创建审计日志表
CREATE TABLE audit_log (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    table_name VARCHAR(50),
    operation VARCHAR(10),
    old_data JSON,
    new_data JSON,
    user VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建审计触发器
DELIMITER //

CREATE TRIGGER audit_user_changes
AFTER UPDATE ON users
FOR EACH ROW
BEGIN
    INSERT INTO audit_log (table_name, operation, old_data, new_data, user)
    VALUES (
        'users',
        'UPDATE',
        JSON_OBJECT('id', OLD.id, 'username', OLD.username, 'email', OLD.email),
        JSON_OBJECT('id', NEW.id, 'username', NEW.username, 'email', NEW.email),
        USER()
    );
END//

DELIMITER ;
```

### 数据验证触发器

```sql
DELIMITER //

CREATE TRIGGER validate_user_age
BEFORE INSERT ON users
FOR EACH ROW
BEGIN
    IF NEW.age < 0 OR NEW.age > 150 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Invalid age value';
    END IF;
END//

DELIMITER ;
```

## 最佳实践

> [!IMPORTANT] > **视图和触发器最佳实践**:
>
> **视图**
>
> - ✅ 用于简化复杂查询
> - ✅ 隐藏敏感数据
> - ✅ 避免过于复杂的视图
> - ❌ 不要嵌套太多层视图
>
> **触发器**
>
> - ✅ 保持触发器简单
> - ✅ 避免触发器间的级联
> - ✅ 谨慎使用（影响性能）
> - ❌ 不要在触发器中执行耗时操作

## 总结

本文介绍了 MySQL 视图与触发器：

- ✅ 视图创建和管理
- ✅ 可更新视图
- ✅ 触发器类型（BEFORE/AFTER）
- ✅ NEW 和 OLD 关键字
- ✅ 实用示例（审计日志、数据验证）

继续学习 [备份与恢复](/docs/mysql/backup-recovery) 和 [面试题集](/docs/interview/mysql-interview-questions)！
