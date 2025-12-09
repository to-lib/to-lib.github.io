---
sidebar_position: 8
title: 存储过程与函数
---

# MySQL 存储过程与函数

> [!TIP] > **存储过程优势**: 减少网络传输、提高性能、增强安全性、代码复用。本文介绍存储过程和函数的创建与使用。

## 存储过程基础

### 什么是存储过程

存储过程是一组预编译的 SQL 语句集合，存储在数据库中，可以被重复调用。

### 创建存储过程

```sql
DELIMITER //

CREATE PROCEDURE procedure_name(
    IN param1 INT,
    OUT param2 VARCHAR(50)
)
BEGIN
    -- SQL 语句
    SELECT column INTO param2 FROM table WHERE id = param1;
END//

DELIMITER ;
```

### 调用存储过程

```sql
CALL procedure_name(1, @result);
SELECT @result;
```

## 参数类型

### IN 参数（输入）

```sql
DELIMITER //

CREATE PROCEDURE get_user_by_id(IN user_id BIGINT)
BEGIN
    SELECT * FROM users WHERE id = user_id;
END//

DELIMITER ;

-- 调用
CALL get_user_by_id(1);
```

### OUT 参数（输出）

```sql
DELIMITER //

CREATE PROCEDURE get_user_count(OUT total INT)
BEGIN
    SELECT COUNT(*) INTO total FROM users;
END//

DELIMITER ;

-- 调用
CALL get_user_count(@count);
SELECT @count;
```

### INOUT 参数（输入输出）

```sql
DELIMITER //

CREATE PROCEDURE square_number(INOUT num INT)
BEGIN
    SET num = num * num;
END//

DELIMITER ;

-- 调用
SET @number = 5;
CALL square_number(@number);
SELECT @number;  -- 25
```

## 变量

### 局部变量

```sql
DELIMITER //

CREATE PROCEDURE calculate_discount()
BEGIN
    DECLARE original_price DECIMAL(10,2);
    DECLARE discount_rate DECIMAL(4,2);
    DECLARE final_price DECIMAL(10,2);

    SET original_price = 100.00;
    SET discount_rate = 0.20;
    SET final_price = original_price * (1 - discount_rate);

    SELECT final_price;
END//

DELIMITER ;
```

### 用户变量

```sql
-- 使用 @ 前缀
SET @name = '张三';
SET @age = 25;

SELECT @name, @age;
```

## 流程控制

### IF 条件语句

```sql
DELIMITER //

CREATE PROCEDURE check_age(IN age INT)
BEGIN
    IF age < 18 THEN
        SELECT '未成年' AS result;
    ELSEIF age >= 18 AND age < 60 THEN
        SELECT '成年' AS result;
    ELSE
        SELECT '老年' AS result;
    END IF;
END//

DELIMITER ;
```

### CASE 语句

```sql
DELIMITER //

CREATE PROCEDURE get_grade(IN score INT, OUT grade CHAR(1))
BEGIN
    CASE
        WHEN score >= 90 THEN SET grade = 'A';
        WHEN score >= 80 THEN SET grade = 'B';
        WHEN score >= 70 THEN SET grade = 'C';
        WHEN score >= 60 THEN SET grade = 'D';
        ELSE SET grade = 'F';
    END CASE;
END//

DELIMITER ;
```

### WHILE 循环

```sql
DELIMITER //

CREATE PROCEDURE sum_to_n(IN n INT, OUT total INT)
BEGIN
    DECLARE i INT DEFAULT 1;
    SET total = 0;

    WHILE i <= n DO
        SET total = total + i;
        SET i = i + 1;
    END WHILE;
END//

DELIMITER ;
```

### REPEAT 循环

```sql
DELIMITER //

CREATE PROCEDURE repeat_example(IN n INT)
BEGIN
    DECLARE i INT DEFAULT 1;

    REPEAT
        SELECT i;
        SET i = i + 1;
    UNTIL i > n
    END REPEAT;
END//

DELIMITER ;
```

### LOOP 循环

```sql
DELIMITER //

CREATE PROCEDURE loop_example(IN n INT)
BEGIN
    DECLARE i INT DEFAULT 1;

    my_loop: LOOP
        IF i > n THEN
            LEAVE my_loop;
        END IF;

        SELECT i;
        SET i = i + 1;
    END LOOP;
END//

DELIMITER ;
```

## 游标

### 游标使用

```sql
DELIMITER //

CREATE PROCEDURE cursor_example()
BEGIN
    DECLARE done INT DEFAULT FALSE;
    DECLARE user_id BIGINT;
    DECLARE user_name VARCHAR(50);

    -- 声明游标
    DECLARE user_cursor CURSOR FOR
        SELECT id, username FROM users;

    -- 声明 CONTINUE HANDLER
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;

    -- 打开游标
    OPEN user_cursor;

    -- 循环读取
    read_loop: LOOP
        FETCH user_cursor INTO user_id, user_name;

        IF done THEN
            LEAVE read_loop;
        END IF;

        -- 处理数据
        SELECT user_id, user_name;
    END LOOP;

    -- 关闭游标
    CLOSE user_cursor;
END//

DELIMITER ;
```

## 自定义函数

### 创建函数

```sql
DELIMITER //

CREATE FUNCTION calculate_age(birth_date DATE)
RETURNS INT
DETERMINISTIC
BEGIN
    RETURN YEAR(CURDATE()) - YEAR(birth_date);
END//

DELIMITER ;

-- 使用函数
SELECT username, calculate_age(birth_date) AS age FROM users;
```

### 函数 vs 存储过程

| 特性     | 函数             | 存储过程               |
| -------- | ---------------- | ---------------------- |
| 返回值   | 必须返回单个值   | 可以返回多个值或不返回 |
| 调用方式 | 可在 SQL 中使用  | 使用 CALL 调用         |
| 参数     | 只支持 IN        | 支持 IN、OUT、INOUT    |
| 事务     | 不能包含事务控制 | 可以包含事务控制       |

## 错误处理

### HANDLER 声明

```sql
DELIMITER //

CREATE PROCEDURE safe_insert(IN username VARCHAR(50))
BEGIN
    -- 声明错误处理
    DECLARE CONTINUE HANDLER FOR SQLEXCEPTION
    BEGIN
        SELECT 'Error occurred' AS message;
    END;

    INSERT INTO users (username) VALUES (username);
    SELECT 'Success' AS message;
END//

DELIMITER ;
```

### 事务处理示例

```sql
DELIMITER //

CREATE PROCEDURE transfer_money(
    IN from_user_id BIGINT,
    IN to_user_id BIGINT,
    IN amount DECIMAL(10,2)
)
BEGIN
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        SELECT 'Transaction failed' AS message;
    END;

    START TRANSACTION;

    UPDATE accounts SET balance = balance - amount WHERE user_id = from_user_id;
    UPDATE accounts SET balance = balance + amount WHERE user_id = to_user_id;

    COMMIT;
    SELECT 'Transaction successful' AS message;
END//

DELIMITER ;
```

## 实用示例

### 批量插入数据

```sql
DELIMITER //

CREATE PROCEDURE batch_insert_users(IN count INT)
BEGIN
    DECLARE i INT DEFAULT 1;

    WHILE i <= count DO
        INSERT INTO users (username, email)
        VALUES (CONCAT('user', i), CONCAT('user', i, '@example.com'));
        SET i = i + 1;
    END WHILE;
END//

DELIMITER ;

CALL batch_insert_users(1000);
```

### 数据统计

```sql
DELIMITER //

CREATE PROCEDURE user_statistics(
    OUT total_users INT,
    OUT active_users INT,
    OUT avg_age FLOAT
)
BEGIN
    SELECT COUNT(*) INTO total_users FROM users;
    SELECT COUNT(*) INTO active_users FROM users WHERE status = 1;
    SELECT AVG(age) INTO avg_age FROM users;
END//

DELIMITER ;

CALL user_statistics(@total, @active, @avg);
SELECT @total, @active, @avg;
```

## 管理存储过程和函数

### 查看

```sql
-- 查看所有存储过程
SHOW PROCEDURE STATUS WHERE db = 'database_name';

-- 查看所有函数
SHOW FUNCTION STATUS WHERE db = 'database_name';

-- 查看存储过程定义
SHOW CREATE PROCEDURE procedure_name;

-- 查看函数定义
SHOW CREATE FUNCTION function_name;
```

### 删除

```sql
DROP PROCEDURE IF EXISTS procedure_name;
DROP FUNCTION IF EXISTS function_name;
```

### 修改

```sql
-- MySQL 不支持直接修改，需要先删除再重建
DROP PROCEDURE IF EXISTS procedure_name;

DELIMITER //
CREATE PROCEDURE procedure_name() BEGIN ... END//
DELIMITER ;
```

## 最佳实践

> [!IMPORTANT] > **存储过程最佳实践**:
>
> 1. 合理使用参数（IN、OUT、INOUT）
> 2. 添加错误处理（HANDLER）
> 3. 使用事务保证数据一致性
> 4. 避免过于复杂的逻辑
> 5. 添加注释说明
> 6. 定期检查和优化

## 总结

本文介绍了 MySQL 存储过程与函数：

- ✅ 存储过程创建和调用
- ✅ 参数类型（IN、OUT、INOUT）
- ✅ 变量和流程控制
- ✅ 游标使用
- ✅ 自定义函数
- ✅ 错误处理和事务

继续学习 [视图与触发器](./views-triggers) 和 [备份恢复](./backup-recovery)！
