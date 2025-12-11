---
sidebar_position: 8
title: 视图与触发器
---

# PostgreSQL 视图与触发器

## 视图（View）

### 什么是视图

视图是基于 SQL 查询结果的虚拟表。视图不存储数据，数据来自基表。

### 创建视图

```sql
-- 基本语法
CREATE VIEW view_name AS
SELECT column1, column2
FROM table_name
WHERE condition;

-- 示例：创建活跃用户视图
CREATE VIEW v_active_users AS
SELECT id, username, email, created_at
FROM users
WHERE is_active = true;

-- 使用视图
SELECT * FROM v_active_users;
```

### 视图的优势

- ✅ **简化复杂查询**：将复杂的 JOIN 和子查询封装
- ✅ **提高安全性**：隐藏敏感列，只暴露必要数据
- ✅ **数据独立性**：修改底层表结构时，视图可保持接口稳定
- ✅ **代码复用**：避免重复编写相同的查询逻辑

### 复杂视图示例

```sql
-- 订单统计视图
CREATE VIEW v_order_statistics AS
SELECT
    u.id AS user_id,
    u.username,
    COUNT(o.id) AS total_orders,
    COALESCE(SUM(o.total), 0) AS total_amount,
    COALESCE(AVG(o.total), 0) AS avg_amount,
    MAX(o.created_at) AS last_order_date
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.username;

-- 使用视图
SELECT * FROM v_order_statistics
WHERE total_orders > 10
ORDER BY total_amount DESC;
```

### 可更新视图

PostgreSQL 支持自动可更新的视图：

```sql
-- 简单视图（可更新）
CREATE VIEW v_user_contact AS
SELECT id, username, email, phone
FROM users;

-- 可以直接更新视图
UPDATE v_user_contact
SET email = 'newemail@example.com'
WHERE id = 1;

-- 可以插入
INSERT INTO v_user_contact (username, email, phone)
VALUES ('john_doe', 'john@example.com', '1234567890');

-- 可以删除
DELETE FROM v_user_contact WHERE id = 1;
```

> [!WARNING] > **不可自动更新的视图**包含：
>
> - 聚合函数（SUM, COUNT, AVG 等）
> - DISTINCT、GROUP BY、HAVING
> - UNION、INTERSECT、EXCEPT
> - 窗口函数
> - 多表 JOIN（某些情况）

### 使用规则使视图可更新

```sql
-- 创建复杂视图
CREATE VIEW v_user_orders AS
SELECT u.id, u.username, o.order_number, o.total
FROM users u
JOIN orders o ON u.id = o.user_id;

-- 创建 INSTEAD OF 规则使其可插入
CREATE RULE insert_user_order AS
ON INSERT TO v_user_orders
DO INSTEAD
INSERT INTO orders (user_id, order_number, total)
VALUES (NEW.id, NEW.order_number, NEW.total);
```

### 管理视图

```sql
-- 查看所有视图
SELECT table_name, view_definition
FROM information_schema.views
WHERE table_schema = 'public';

-- 或使用 psql 命令
\dv

-- 查看视图定义
\d+ view_name

-- 修改视图（CREATE OR REPLACE）
CREATE OR REPLACE VIEW v_active_users AS
SELECT id, username, email, phone, created_at
FROM users
WHERE is_active = true AND last_login > NOW() - INTERVAL '30 days';

-- 删除视图
DROP VIEW IF EXISTS v_active_users;

-- 级联删除依赖该视图的对象
DROP VIEW v_active_users CASCADE;
```

## 物化视图（Materialized View）

### 什么是物化视图

物化视图会实际存储查询结果，需要手动刷新数据。

### 创建物化视图

```sql
-- 创建物化视图
CREATE MATERIALIZED VIEW mv_daily_sales AS
SELECT
    DATE(created_at) AS sale_date,
    COUNT(*) AS order_count,
    SUM(total) AS total_sales,
    AVG(total) AS avg_sales
FROM orders
GROUP BY DATE(created_at);

-- 刷新物化视图
REFRESH MATERIALIZED VIEW mv_daily_sales;

-- 并发刷新（不锁定视图）
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_sales;

-- 为物化视图创建索引
CREATE UNIQUE INDEX idx_mv_daily_sales_date ON mv_daily_sales(sale_date);
```

### 视图 vs 物化视图

| 特性       | 普通视图     | 物化视图           |
| ---------- | ------------ | ------------------ |
| 数据存储   | 不存储       | 存储查询结果       |
| 查询性能   | 每次执行查询 | 直接读取存储的数据 |
| 数据实时性 | 实时         | 需要手动刷新       |
| 磁盘空间   | 不占用       | 占用空间           |
| 索引支持   | 不支持       | 支持               |
| 使用场景   | 简化查询语法 | 复杂聚合统计       |

## 触发器（Trigger）

### 什么是触发器

触发器是在特定数据库事件（INSERT、UPDATE、DELETE）发生时自动执行的函数。

### 触发器类型

**按时机分类：**

- `BEFORE`：在操作执行前触发
- `AFTER`：在操作执行后触发
- `INSTEAD OF`：替代操作执行（仅用于视图）

**按级别分类：**

- 行级触发器（`FOR EACH ROW`）：每行触发一次
- 语句级触发器（`FOR EACH STATEMENT`）：每个语句触发一次

### 创建触发器

```sql
-- 1. 先创建触发器函数
CREATE OR REPLACE FUNCTION update_modified_time()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 2. 创建触发器
CREATE TRIGGER trigger_update_user_time
BEFORE UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION update_modified_time();
```

### BEFORE INSERT 触发器

```sql
-- 自动生成 UUID
CREATE OR REPLACE FUNCTION generate_uuid()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.id IS NULL THEN
        NEW.id = gen_random_uuid();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_generate_user_uuid
BEFORE INSERT ON users
FOR EACH ROW
EXECUTE FUNCTION generate_uuid();
```

### AFTER INSERT 触发器

```sql
-- 插入订单后更新统计
CREATE OR REPLACE FUNCTION update_user_order_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE users
    SET total_orders = total_orders + 1,
        total_spent = total_spent + NEW.total
    WHERE id = NEW.user_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_after_order_insert
AFTER INSERT ON orders
FOR EACH ROW
EXECUTE FUNCTION update_user_order_stats();
```

### BEFORE UPDATE 触发器

```sql
-- 记录修改前的数据
CREATE OR REPLACE FUNCTION log_price_change()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.price != NEW.price THEN
        INSERT INTO price_history (product_id, old_price, new_price, changed_at)
        VALUES (NEW.id, OLD.price, NEW.price, NOW());
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_log_price_change
BEFORE UPDATE ON products
FOR EACH ROW
EXECUTE FUNCTION log_price_change();
```

### AFTER DELETE 触发器

```sql
-- 删除用户后清理相关数据
CREATE OR REPLACE FUNCTION cleanup_user_data()
RETURNS TRIGGER AS $$
BEGIN
    DELETE FROM user_sessions WHERE user_id = OLD.id;
    DELETE FROM user_preferences WHERE user_id = OLD.id;

    -- 记录删除日志
    INSERT INTO deletion_log (table_name, record_id, deleted_at)
    VALUES ('users', OLD.id, NOW());

    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_cleanup_after_user_delete
AFTER DELETE ON users
FOR EACH ROW
EXECUTE FUNCTION cleanup_user_data();
```

### NEW 和 OLD 变量

| 触发器类型 | NEW       | OLD       |
| ---------- | --------- | --------- |
| INSERT     | ✅ 可用   | ❌ 不可用 |
| UPDATE     | ✅ 可用   | ✅ 可用   |
| DELETE     | ❌ 不可用 | ✅ 可用   |

```sql
-- 使用 NEW 和 OLD
CREATE OR REPLACE FUNCTION audit_user_changes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (operation, table_name, new_data)
        VALUES ('INSERT', 'users', row_to_json(NEW));
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (operation, table_name, old_data, new_data)
        VALUES ('UPDATE', 'users', row_to_json(OLD), row_to_json(NEW));
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (operation, table_name, old_data)
        VALUES ('DELETE', 'users', row_to_json(OLD));
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_audit_users
AFTER INSERT OR UPDATE OR DELETE ON users
FOR EACH ROW
EXECUTE FUNCTION audit_user_changes();
```

### 条件触发器

```sql
-- 仅在特定条件下触发
CREATE TRIGGER trigger_high_value_order
AFTER INSERT ON orders
FOR EACH ROW
WHEN (NEW.total > 1000)
EXECUTE FUNCTION notify_high_value_order();
```

### INSTEAD OF 触发器（视图）

```sql
-- 为视图创建 INSTEAD OF 触发器
CREATE VIEW v_user_summary AS
SELECT u.id, u.username, COUNT(o.id) AS order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.username;

CREATE OR REPLACE FUNCTION insert_user_via_view()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO users (username, email)
    VALUES (NEW.username, NEW.username || '@example.com');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_insert_user_view
INSTEAD OF INSERT ON v_user_summary
FOR EACH ROW
EXECUTE FUNCTION insert_user_via_view();
```

### 管理触发器

```sql
-- 查看表的所有触发器
SELECT
    trigger_name,
    event_manipulation,
    event_object_table,
    action_timing
FROM information_schema.triggers
WHERE event_object_schema = 'public';

-- 或使用 psql 命令
\dft

-- 查看触发器详细信息
\d+ table_name

-- 禁用触发器
ALTER TABLE users DISABLE TRIGGER trigger_name;

-- 启用触发器
ALTER TABLE users ENABLE TRIGGER trigger_name;

-- 删除触发器
DROP TRIGGER IF EXISTS trigger_name ON table_name;
```

## 实用示例

### 审计日志系统

```sql
-- 创建审计日志表
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(50),
    operation VARCHAR(10),
    user_name VARCHAR(50),
    old_data JSONB,
    new_data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 通用审计触发器函数
CREATE OR REPLACE FUNCTION audit_trigger_func()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_log (
        table_name,
        operation,
        user_name,
        old_data,
        new_data
    )
    VALUES (
        TG_TABLE_NAME,
        TG_OP,
        current_user,
        CASE WHEN TG_OP IN ('UPDATE', 'DELETE') THEN row_to_json(OLD) END,
        CASE WHEN TG_OP IN ('INSERT', 'UPDATE') THEN row_to_json(NEW) END
    );
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- 为多个表添加审计
CREATE TRIGGER audit_users
AFTER INSERT OR UPDATE OR DELETE ON users
FOR EACH ROW EXECUTE FUNCTION audit_trigger_func();

CREATE TRIGGER audit_orders
AFTER INSERT OR UPDATE OR DELETE ON orders
FOR EACH ROW EXECUTE FUNCTION audit_trigger_func();
```

### 数据验证触发器

```sql
CREATE OR REPLACE FUNCTION validate_user_email()
RETURNS TRIGGER AS $$
BEGIN
    -- 验证邮箱格式
    IF NEW.email !~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$' THEN
        RAISE EXCEPTION 'Invalid email format: %', NEW.email;
    END IF;

    -- 验证年龄范围
    IF NEW.age IS NOT NULL AND (NEW.age < 0 OR NEW.age > 150) THEN
        RAISE EXCEPTION 'Invalid age: %', NEW.age;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER validate_user_data
BEFORE INSERT OR UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION validate_user_email();
```

### 自动缓存失效

```sql
-- 当数据变化时通知应用层
CREATE OR REPLACE FUNCTION notify_cache_invalidation()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('cache_invalidate', TG_TABLE_NAME || ':' || NEW.id::TEXT);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER notify_user_changes
AFTER INSERT OR UPDATE OR DELETE ON users
FOR EACH ROW
EXECUTE FUNCTION notify_cache_invalidation();
```

## 最佳实践

> [!IMPORTANT] > **视图最佳实践：**
>
> - ✅ 用视图简化复杂查询和隐藏敏感数据
> - ✅ 使用物化视图优化复杂统计查询
> - ✅ 为物化视图创建索引提升性能
> - ✅ 定期刷新物化视图保持数据新鲜
> - ❌ 避免视图嵌套过深（影响性能）
> - ❌ 不要在高频查询中使用包含复杂 JOIN 的视图

> [!IMPORTANT] > **触发器最佳实践：**
>
> - ✅ 保持触发器函数简单快速
> - ✅ 使用条件触发（WHEN 子句）减少不必要的执行
> - ✅ 在触发器中添加适当的错误处理
> - ✅ 记录触发器逻辑，便于维护
> - ❌ 避免触发器中的循环依赖
> - ❌ 不要在触发器中执行耗时操作
> - ❌ 慎用 BEFORE 触发器修改数据

## 性能考虑

### 视图性能优化

```sql
-- 为常用过滤条件创建索引
CREATE INDEX idx_users_active ON users(is_active)
WHERE is_active = true;

-- 使用 EXPLAIN 分析视图查询
EXPLAIN ANALYZE SELECT * FROM v_active_users WHERE created_at > '2024-01-01';
```

### 触发器性能优化

```sql
-- 使用语句级触发器而非行级触发器（适用场景）
CREATE TRIGGER trigger_statement_level
AFTER INSERT ON orders
FOR EACH STATEMENT
EXECUTE FUNCTION update_statistics();

-- 使用条件减少触发次数
CREATE TRIGGER trigger_conditional
AFTER UPDATE ON products
FOR EACH ROW
WHEN (OLD.price IS DISTINCT FROM NEW.price)
EXECUTE FUNCTION log_price_change();
```

## 总结

PostgreSQL 的视图和触发器功能强大：

- ✅ 视图提供数据抽象和安全性
- ✅ 物化视图优化复杂查询性能
- ✅ 触发器实现自动化业务逻辑
- ✅ 支持行级和语句级触发器
- ✅ 丰富的触发器时机选择
- ✅ 完善的审计和验证能力

继续学习 [存储过程与函数](/docs/postgres/stored-procedures) 和 [最佳实践](/docs/postgres/best-practices)！
