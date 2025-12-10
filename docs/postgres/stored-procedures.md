---
sidebar_position: 7
title: 存储过程与函数
---

# PostgreSQL 存储过程与函数

> [!TIP] > **存储过程与函数优势**：提高性能、减少网络传输、增强安全性、代码复用。PostgreSQL 支持多种过程语言，其中 PL/pgSQL 最为常用。

## 函数基础

### 什么是函数

PostgreSQL 中的函数是可重用的代码块，可以接受参数并返回值。

### 创建简单函数

```sql
-- 创建返回整数的函数
CREATE FUNCTION add_numbers(a INTEGER, b INTEGER)
RETURNS INTEGER
AS $$
BEGIN
    RETURN a + b;
END;
$$ LANGUAGE plpgsql;

-- 调用函数
SELECT add_numbers(3, 5);  -- 8
```

### 函数返回类型

```sql
-- 返回单个值
CREATE FUNCTION get_user_count()
RETURNS BIGINT
AS $$
BEGIN
    RETURN (SELECT COUNT(*) FROM users);
END;
$$ LANGUAGE plpgsql;

-- 返回表
CREATE FUNCTION get_active_users()
RETURNS TABLE(id INT, username VARCHAR, email VARCHAR)
AS $$
BEGIN
    RETURN QUERY
    SELECT id, username, email FROM users WHERE is_active = true;
END;
$$ LANGUAGE plpgsql;

-- 调用返回表的函数
SELECT * FROM get_active_users();
```

## PL/pgSQL 语言

### 变量声明

```sql
CREATE FUNCTION calculate_discount(
    original_price NUMERIC,
    discount_rate NUMERIC
)
RETURNS NUMERIC
AS $$
DECLARE
    final_price NUMERIC;
    tax_rate NUMERIC := 0.08;
BEGIN
    final_price := original_price * (1 - discount_rate);
    final_price := final_price * (1 + tax_rate);
    RETURN final_price;
END;
$$ LANGUAGE plpgsql;
```

### 参数模式

PostgreSQL 支持 IN、OUT、INOUT 参数：

```sql
-- OUT 参数
CREATE FUNCTION get_user_info(
    user_id INT,
    OUT username VARCHAR,
    OUT email VARCHAR,
    OUT age INT
)
AS $$
BEGIN
    SELECT u.username, u.email, EXTRACT(YEAR FROM AGE(u.birth_date))
    INTO username, email, age
    FROM users u
    WHERE u.id = user_id;
END;
$$ LANGUAGE plpgsql;

-- 调用
SELECT * FROM get_user_info(1);

-- INOUT 参数
CREATE FUNCTION square_number(INOUT num INTEGER)
AS $$
BEGIN
    num := num * num;
END;
$$ LANGUAGE plpgsql;

-- 调用
SELECT square_number(5);  -- 25
```

## 流程控制

### IF 条件语句

```sql
CREATE FUNCTION check_age_category(age INT)
RETURNS VARCHAR
AS $$
BEGIN
    IF age < 18 THEN
        RETURN '未成年';
    ELSIF age >= 18 AND age < 60 THEN
        RETURN '成年';
    ELSE
        RETURN '老年';
    END IF;
END;
$$ LANGUAGE plpgsql;
```

### CASE 表达式

```sql
CREATE FUNCTION get_grade(score INT)
RETURNS CHAR
AS $$
BEGIN
    RETURN CASE
        WHEN score >= 90 THEN 'A'
        WHEN score >= 80 THEN 'B'
        WHEN score >= 70 THEN 'C'
        WHEN score >= 60 THEN 'D'
        ELSE 'F'
    END;
END;
$$ LANGUAGE plpgsql;
```

### 循环

```sql
-- LOOP
CREATE FUNCTION sum_to_n(n INT)
RETURNS INT
AS $$
DECLARE
    i INT := 1;
    total INT := 0;
BEGIN
    LOOP
        EXIT WHEN i > n;
        total := total + i;
        i := i + 1;
    END LOOP;
    RETURN total;
END;
$$ LANGUAGE plpgsql;

-- WHILE
CREATE FUNCTION factorial(n INT)
RETURNS BIGINT
AS $$
DECLARE
    result BIGINT := 1;
    i INT := 1;
BEGIN
    WHILE i <= n LOOP
        result := result * i;
        i := i + 1;
    END LOOP;
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- FOR
CREATE FUNCTION fibonacci(n INT)
RETURNS INT[]
AS $$
DECLARE
    fib INT[] := ARRAY[0, 1];
    i INT;
BEGIN
    FOR i IN 3..n LOOP
        fib := fib || (fib[i-1] + fib[i-2]);
    END LOOP;
    RETURN fib;
END;
$$ LANGUAGE plpgsql;
```

## 存储过程（PROCEDURE）

PostgreSQL 11+ 支持真正的存储过程。

### 创建存储过程

```sql
-- 存储过程可以包含事务控制
CREATE PROCEDURE transfer_money(
    from_account INT,
    to_account INT,
    amount NUMERIC
)
LANGUAGE plpgsql
AS $$
BEGIN
    -- 减少转出账户余额
    UPDATE accounts SET balance = balance - amount
    WHERE account_id = from_account;

    -- 增加转入账户余额
    UPDATE accounts SET balance = balance + amount
    WHERE account_id = to_account;

    COMMIT;
END;
$$;

-- 调用存储过程
CALL transfer_money(1, 2, 100.00);
```

### 函数 vs 存储过程

| 特性     | 函数（FUNCTION）         | 存储过程（PROCEDURE）     |
| -------- | ------------------------ | ------------------------- |
| 返回值   | 必须返回值               | 不返回值（可用 OUT 参数） |
| 调用方式 | SELECT 或表达式中        | CALL 语句                 |
| 事务控制 | 不能使用 COMMIT/ROLLBACK | 可以使用 COMMIT/ROLLBACK  |
| 参数     | IN, OUT, INOUT           | IN, OUT, INOUT            |
| 使用场景 | 计算、查询               | 复杂业务逻辑、批量处理    |

## 触发器函数

### 创建触发器函数

```sql
-- 创建审计日志触发器函数
CREATE OR REPLACE FUNCTION audit_user_changes()
RETURNS TRIGGER
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, operation, user_id, timestamp)
        VALUES ('users', 'INSERT', NEW.id, NOW());
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, operation, user_id, old_data, new_data, timestamp)
        VALUES ('users', 'UPDATE', NEW.id, row_to_json(OLD), row_to_json(NEW), NOW());
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, operation, user_id, old_data, timestamp)
        VALUES ('users', 'DELETE', OLD.id, row_to_json(OLD), NOW());
        RETURN OLD;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- 创建触发器
CREATE TRIGGER user_audit_trigger
AFTER INSERT OR UPDATE OR DELETE ON users
FOR EACH ROW
EXECUTE FUNCTION audit_user_changes();
```

### 触发器特殊变量

- `NEW`：INSERT/UPDATE 操作中的新行
- `OLD`：UPDATE/DELETE 操作中的老行
- `TG_OP`：操作类型（INSERT、UPDATE、DELETE）
- `TG_TABLE_NAME`：触发器所在的表名
- `TG_WHEN`：BEFORE 或 AFTER

## 错误处理

### 异常处理

```sql
CREATE FUNCTION safe_divide(a NUMERIC, b NUMERIC)
RETURNS NUMERIC
AS $$
BEGIN
    RETURN a / b;
EXCEPTION
    WHEN division_by_zero THEN
        RAISE NOTICE 'Division by zero detected';
        RETURN NULL;
    WHEN OTHERS THEN
        RAISE NOTICE 'An error occurred: %', SQLERRM;
        RETURN NULL;
END;
$$ LANGUAGE plpgsql;
```

### 自定义异常

```sql
CREATE FUNCTION check_balance(account_id INT, withdraw_amount NUMERIC)
RETURNS VOID
AS $$
DECLARE
    current_balance NUMERIC;
BEGIN
    SELECT balance INTO current_balance
    FROM accounts
    WHERE id = account_id;

    IF current_balance < withdraw_amount THEN
        RAISE EXCEPTION 'Insufficient funds. Balance: %, Withdraw: %',
            current_balance, withdraw_amount;
    END IF;
END;
$$ LANGUAGE plpgsql;
```

## 动态 SQL

### EXECUTE 语句

```sql
CREATE FUNCTION create_user_table(table_name TEXT)
RETURNS VOID
AS $$
BEGIN
    EXECUTE format('
        CREATE TABLE %I (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50),
            created_at TIMESTAMP DEFAULT NOW()
        )', table_name);
END;
$$ LANGUAGE plpgsql;

-- 动态查询
CREATE FUNCTION get_table_count(table_name TEXT)
RETURNS BIGINT
AS $$
DECLARE
    row_count BIGINT;
BEGIN
    EXECUTE format('SELECT COUNT(*) FROM %I', table_name)
    INTO row_count;
    RETURN row_count;
END;
$$ LANGUAGE plpgsql;
```

## 实用示例

### 批量数据处理

```sql
CREATE PROCEDURE batch_update_user_status()
LANGUAGE plpgsql
AS $$
DECLARE
    batch_size INT := 1000;
    processed INT := 0;
    total INT;
BEGIN
    SELECT COUNT(*) INTO total
    FROM users
    WHERE last_login < NOW() - INTERVAL '1 year';

    WHILE processed < total LOOP
        UPDATE users
        SET is_active = false
        WHERE id IN (
            SELECT id FROM users
            WHERE last_login < NOW() - INTERVAL '1 year'
            AND is_active = true
            LIMIT batch_size
        );

        processed := processed + batch_size;
        RAISE NOTICE 'Processed % of % users', processed, total;
        COMMIT;
    END LOOP;
END;
$$;
```

### 数据统计函数

```sql
CREATE FUNCTION user_statistics()
RETURNS TABLE(
    total_users BIGINT,
    active_users BIGINT,
    inactive_users BIGINT,
    avg_age NUMERIC
)
AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*) as total,
        COUNT(*) FILTER (WHERE is_active = true) as active,
        COUNT(*) FILTER (WHERE is_active = false) as inactive,
        AVG(EXTRACT(YEAR FROM AGE(birth_date))) as avg_age
    FROM users;
END;
$$ LANGUAGE plpgsql;

-- 调用
SELECT * FROM user_statistics();
```

## 函数管理

### 查看函数

```sql
-- 查看所有函数
SELECT
    n.nspname as schema,
    p.proname as name,
    pg_get_function_arguments(p.oid) as arguments,
    pg_get_function_result(p.oid) as result_type
FROM pg_proc p
JOIN pg_namespace n ON p.pronamespace = n.oid
WHERE n.nspname = 'public';

-- 查看函数定义
\df+ function_name

-- 查看函数源码
SELECT pg_get_functiondef(oid)
FROM pg_proc
WHERE proname = 'function_name';
```

### 删除和修改

```sql
-- 删除函数
DROP FUNCTION IF EXISTS function_name;
DROP FUNCTION IF EXISTS function_name(INT, VARCHAR);  -- 指定参数

-- 替换函数（CREATE OR REPLACE）
CREATE OR REPLACE FUNCTION add_numbers(a INT, b INT)
RETURNS INT
AS $$
BEGIN
    RETURN a + b;
END;
$$ LANGUAGE plpgsql;
```

### 函数属性

```sql
CREATE FUNCTION expensive_calculation(n INT)
RETURNS INT
PARALLEL SAFE          -- 可并行执行
IMMUTABLE             -- 相同输入总返回相同结果
STRICT                -- NULL 输入返回 NULL
AS $$
BEGIN
    RETURN n * n;
END;
$$ LANGUAGE plpgsql;
```

## 最佳实践

> [!IMPORTANT] > **存储过程与函数最佳实践：**
>
> 1. **命名规范**：使用描述性名称，如 `get_user_by_id`
> 2. **参数验证**：在函数开始处验证参数
> 3. **错误处理**：使用 EXCEPTION 处理可能的错误
> 4. **避免过度使用**：复杂业务逻辑可能更适合应用层
> 5. **性能考虑**：使用 `IMMUTABLE`、`STABLE`、`VOLATILE` 正确标记
> 6. **安全性**：使用 `SECURITY DEFINER` 时要特别小心
> 7. **文档注释**：使用 `COMMENT ON FUNCTION` 添加说明

```sql
-- 添加函数注释
COMMENT ON FUNCTION add_numbers(INT, INT) IS '计算两个整数的和';
```

## 性能优化

### 使用 STABLE/IMMUTABLE

```sql
-- IMMUTABLE：相同输入始终返回相同结果
CREATE FUNCTION circle_area(radius NUMERIC)
RETURNS NUMERIC
IMMUTABLE
AS $$
BEGIN
    RETURN 3.14159 * radius * radius;
END;
$$ LANGUAGE plpgsql;

-- STABLE：同一事务内相同输入返回相同结果
CREATE FUNCTION get_current_date_formatted()
RETURNS TEXT
STABLE
AS $$
BEGIN
    RETURN TO_CHAR(CURRENT_DATE, 'YYYY-MM-DD');
END;
$$ LANGUAGE plpgsql;
```

### 使用 SQL 函数

对于简单逻辑，SQL 函数比 PL/pgSQL 更快：

```sql
-- SQL 函数（更快）
CREATE FUNCTION get_user_email(user_id INT)
RETURNS VARCHAR
AS $$
    SELECT email FROM users WHERE id = user_id;
$$ LANGUAGE sql;

-- PL/pgSQL 函数（灵活但稍慢）
CREATE FUNCTION get_user_email_plpgsql(user_id INT)
RETURNS VARCHAR
AS $$
DECLARE
    user_email VARCHAR;
BEGIN
    SELECT email INTO user_email FROM users WHERE id = user_id;
    RETURN user_email;
END;
$$ LANGUAGE plpgsql;
```

## 总结

PostgreSQL 的函数和存储过程功能强大：

- ✅ 支持多种过程语言（PL/pgSQL、PL/Python、PL/Perl 等）
- ✅ 函数可返回标量值、记录或表
- ✅ 存储过程支持事务控制
- ✅ 触发器函数实现自动化操作
- ✅ 异常处理和动态 SQL
- ✅ 丰富的性能优化选项

继续学习 [视图与触发器](./views-triggers) 和 [最佳实践](./best-practices)！
