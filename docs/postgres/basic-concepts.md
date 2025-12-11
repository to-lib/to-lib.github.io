---
sidebar_position: 2
title: 基础概念
---

# PostgreSQL 基础概念

## 📚 数据库架构

### 1. 数据库集群（Database Cluster）

PostgreSQL 的一个实例称为数据库集群，包含：

- **多个数据库**：一个集群可包含多个独立的数据库
- **共享的系统目录**：所有数据库共享 `pg_catalog` 等系统表
- **单一的服务器进程**：一个 PostgreSQL 实例对应一个服务器进程

```bash
# 初始化数据库集群
initdb -D /usr/local/pgsql/data

# 查看所有数据库
\l
```

### 2. 数据库（Database）

数据库是表、视图、索引等对象的集合。

```sql
-- 创建数据库
CREATE DATABASE myapp
    WITH
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TEMPLATE = template0;

-- 删除数据库
DROP DATABASE myapp;

-- 切换数据库
\c myapp
```

### 3. 模式（Schema）

模式是数据库对象的逻辑分组，类似于命名空间。

```sql
-- 创建模式
CREATE SCHEMA sales;
CREATE SCHEMA hr;

-- 在模式中创建表
CREATE TABLE sales.orders (
    order_id SERIAL PRIMARY KEY,
    product_name VARCHAR(100)
);

-- 设置搜索路径
SET search_path TO sales, public;

-- 查看当前搜索路径
SHOW search_path;
```

**默认模式：**

- `public`：默认模式，所有用户都可访问
- `pg_catalog`：系统目录模式
- `information_schema`：SQL 标准的系统视图

### 4. 表（Table）

表是存储数据的基本单位。

```sql
-- 创建表
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    department VARCHAR(50),
    salary NUMERIC(10, 2),
    hire_date DATE DEFAULT CURRENT_DATE,
    is_active BOOLEAN DEFAULT true
);

-- 查看表结构
\d employees

-- 修改表
ALTER TABLE employees ADD COLUMN phone VARCHAR(20);
ALTER TABLE employees DROP COLUMN phone;
ALTER TABLE employees RENAME COLUMN name TO full_name;

-- 删除表
DROP TABLE employees;
```

## 🔑 约束（Constraints）

### 1. 主键约束（PRIMARY KEY）

唯一标识表中的每一行。

```sql
-- 单列主键
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50)
);

-- 复合主键
CREATE TABLE order_items (
    order_id INT,
    product_id INT,
    quantity INT,
    PRIMARY KEY (order_id, product_id)
);

-- 添加主键
ALTER TABLE users ADD PRIMARY KEY (id);
```

### 2. 外键约束（FOREIGN KEY）

维护表之间的引用完整性。

```sql
CREATE TABLE departments (
    dept_id SERIAL PRIMARY KEY,
    dept_name VARCHAR(50)
);

CREATE TABLE employees (
    emp_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    dept_id INT,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);
```

**外键选项：**

- `ON DELETE CASCADE`：删除父行时，自动删除子行
- `ON DELETE SET NULL`：删除父行时，子行外键设为 NULL
- `ON DELETE RESTRICT`：如果有子行，禁止删除父行（默认）
- `ON UPDATE CASCADE`：更新父行时，自动更新子行

### 3. 唯一约束（UNIQUE）

确保列中的值唯一。

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(100) UNIQUE,
    username VARCHAR(50) UNIQUE
);

-- 复合唯一约束
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    category VARCHAR(50),
    UNIQUE (name, category)
);
```

### 4. 检查约束（CHECK）

验证列值满足特定条件。

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    price NUMERIC(10, 2) CHECK (price > 0),
    stock INT CHECK (stock >= 0),
    discount NUMERIC(3, 2) CHECK (discount BETWEEN 0 AND 1)
);

-- 表级检查约束
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    salary NUMERIC(10, 2),
    bonus NUMERIC(10, 2),
    CHECK (salary > bonus)
);
```

### 5. 非空约束（NOT NULL）

确保列不能为 NULL。

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,
    phone VARCHAR(20)  -- 可以为 NULL
);
```

## 🎯 数据完整性

### 1. 实体完整性

通过主键约束实现，确保每一行都可唯一标识。

```sql
CREATE TABLE students (
    student_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);
```

### 2. 参照完整性

通过外键约束实现，确保表之间的关系一致。

```sql
CREATE TABLE courses (
    course_id SERIAL PRIMARY KEY,
    course_name VARCHAR(100)
);

CREATE TABLE enrollments (
    student_id INT,
    course_id INT,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id) REFERENCES courses(course_id),
    PRIMARY KEY (student_id, course_id)
);
```

### 3. 域完整性

通过数据类型、CHECK 约束、NOT NULL 等实现。

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    price NUMERIC(10, 2) CHECK (price > 0),
    category VARCHAR(50) CHECK (category IN ('Electronics', 'Clothing', 'Food'))
);
```

## 📊 系统目录

PostgreSQL 使用系统目录存储元数据。

```sql
-- 查询所有表
SELECT tablename FROM pg_tables WHERE schemaname = 'public';

-- 查询表的列信息
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'employees';

-- 查询所有索引
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'employees';

-- 查询数据库大小
SELECT pg_size_pretty(pg_database_size('myapp'));

-- 查询表大小
SELECT pg_size_pretty(pg_total_relation_size('employees'));
```

## 🔧 命名规范

### 推荐的命名约定

1. **表名**：使用复数形式，小写，下划线分隔

   ```sql
   CREATE TABLE user_profiles (...);
   CREATE TABLE order_items (...);
   ```

2. **列名**：小写，下划线分隔

   ```sql
   CREATE TABLE users (
       user_id INT,
       first_name VARCHAR(50),
       created_at TIMESTAMP
   );
   ```

3. **索引名**：`idx_<表名>_<列名>`

   ```sql
   CREATE INDEX idx_users_email ON users(email);
   ```

4. **外键名**：`fk_<表名>_<引用表名>`

   ```sql
   ALTER TABLE orders
   ADD CONSTRAINT fk_orders_users
   FOREIGN KEY (user_id) REFERENCES users(id);
   ```

5. **主键名**：`pk_<表名>`
   ```sql
   ALTER TABLE users
   ADD CONSTRAINT pk_users PRIMARY KEY (id);
   ```

## 💡 最佳实践

1. **使用 SERIAL 或 IDENTITY 作为主键**

   ```sql
   CREATE TABLE users (
       id SERIAL PRIMARY KEY,
       -- 或
       id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
   );
   ```

2. **合理使用约束**

   - 在数据层面保证数据完整性
   - 避免在应用层重复验证

3. **使用模式组织对象**

   ```sql
   CREATE SCHEMA app;
   CREATE SCHEMA analytics;
   CREATE SCHEMA staging;
   ```

4. **添加注释**
   ```sql
   COMMENT ON TABLE users IS '用户信息表';
   COMMENT ON COLUMN users.email IS '用户邮箱地址';
   ```

## 🎓 练习题

1. 创建一个博客系统的数据库结构，包括：

   - 用户表（users）
   - 文章表（posts）
   - 评论表（comments）
   - 标签表（tags）
   - 文章标签关联表（post_tags）

2. 为上述表添加适当的约束：
   - 主键
   - 外键
   - 唯一约束
   - 检查约束

## 📚 相关资源

- [数据类型](/docs/postgres/data-types) - 了解 PostgreSQL 的数据类型
- [SQL 语法](/docs/postgres/sql-syntax) - 学习 SQL 查询语法
- [索引优化](/docs/postgres/indexes) - 理解索引的使用

下一节：[数据类型](/docs/postgres/data-types)
