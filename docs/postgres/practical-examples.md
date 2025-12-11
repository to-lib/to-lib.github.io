---
sidebar_position: 11
title: 实战案例
---

# PostgreSQL 实战案例

> [!TIP] > **实战演练**：本文档通过实际项目案例，展示 PostgreSQL 在不同业务场景下的应用和最佳实践，充分利用 PostgreSQL 的高级特性。

## 用户系统设计

### 需求分析

设计一个支持用户注册、登录、权限管理的用户系统，使用 PostgreSQL 的 JSONB 和数组特性。

### 数据库设计

```sql
-- 用户表
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    phone VARCHAR(20),
    avatar_url VARCHAR(255),
    -- 使用 JSONB 存储额外属性
    metadata JSONB DEFAULT '{}',
    -- 使用数组存储角色
    roles TEXT[] DEFAULT '{}',
    status SMALLINT NOT NULL DEFAULT 1,  -- 1-正常 2-禁用
    last_login_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 创建索引
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_metadata ON users USING GIN(metadata);
CREATE INDEX idx_users_roles ON users USING GIN(roles);

-- 添加注释
COMMENT ON TABLE users IS '用户表';
COMMENT ON COLUMN users.metadata IS 'JSON格式的用户额外信息';
COMMENT ON COLUMN users.roles IS '用户角色数组';

-- 角色权限表
CREATE TABLE role_permissions (
    id SERIAL PRIMARY KEY,
    role_name VARCHAR(50) NOT NULL UNIQUE,
    permissions JSONB NOT NULL DEFAULT '[]',
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_role_permissions_jsonb ON role_permissions USING GIN(permissions);

-- 插入示例角色
INSERT INTO role_permissions (role_name, permissions, description) VALUES
('admin', '["user:create", "user:read", "user:update", "user:delete", "post:*"]', '管理员'),
('editor', '["post:create", "post:read", "post:update", "comment:*"]', '编辑'),
('viewer', '["post:read", "comment:read"]', '访客');
```

### 常用查询

```sql
-- 用户注册
INSERT INTO users (username, email, password_hash, phone, metadata, roles)
VALUES (
    'zhangsan',
    'zhangsan@example.com',
    '$2a$10$...',
    '13800138000',
    '{"city": "Beijing", "age": 25}',
    ARRAY['viewer']
);

-- 用户登录验证
SELECT id, username, password_hash, status, roles
FROM users
WHERE email = 'zhangsan@example.com' AND status = 1;

-- 分配角色（添加到数组）
UPDATE users
SET roles = array_append(roles, 'editor')
WHERE id = 1 AND NOT ('editor' = ANY(roles));

-- 移除角色
UPDATE users
SET roles = array_remove(roles, 'viewer')
WHERE id = 1;

-- 检查用户是否有特定角色
SELECT * FROM users
WHERE id = 1 AND 'admin' = ANY(roles);

-- 查询用户的所有权限
SELECT DISTINCT jsonb_array_elements_text(rp.permissions) AS permission
FROM users u
CROSS JOIN LATERAL unnest(u.roles) AS role_name
INNER JOIN role_permissions rp ON role_name = rp.role_name
WHERE u.id = 1;

-- 使用 JSONB 查询用户
SELECT * FROM users
WHERE metadata->>'city' = 'Beijing';

SELECT * FROM users
WHERE (metadata->>'age')::int > 20;
```

## 电商订单系统

### 需求分析

设计一个支持商品、订单、支付的电商系统，使用 PostgreSQL 事务和触发器。

### 数据库设计

```sql
-- 商品表
CREATE TABLE products (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    price NUMERIC(10,2) NOT NULL CHECK (price > 0),
    stock INT NOT NULL DEFAULT 0 CHECK (stock >= 0),
    category_id INT,
    -- 使用 JSONB 存储规格
    specifications JSONB DEFAULT '{}',
    status SMALLINT NOT NULL DEFAULT 1,  -- 1-上架 2-下架
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_products_status ON products(status);
CREATE INDEX idx_products_name ON products USING GIN(to_tsvector('english', name));

-- 订单表
CREATE TABLE orders (
    id BIGSERIAL PRIMARY KEY,
    order_no VARCHAR(32) NOT NULL UNIQUE,
    user_id BIGINT NOT NULL,
    total_amount NUMERIC(10,2) NOT NULL,
    status SMALLINT NOT NULL DEFAULT 1,  -- 1-待支付 2-已支付 3-已发货 4-已完成 5-已取消
    payment_method SMALLINT,
    -- 使用 JSONB 存储收货地址
    shipping_address JSONB,
    paid_at TIMESTAMPTZ,
    shipped_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created_at ON orders(created_at);

-- 订单明细表
CREATE TABLE order_items (
    id BIGSERIAL PRIMARY KEY,
    order_id BIGINT NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    product_id BIGINT NOT NULL,
    product_name VARCHAR(200) NOT NULL,
    product_snapshot JSONB,  -- 商品快照（价格、规格等）
    price NUMERIC(10,2) NOT NULL,
    quantity INT NOT NULL CHECK (quantity > 0),
    subtotal NUMERIC(10,2) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
```

### 创建订单（使用事务和存储过程）

```sql
CREATE OR REPLACE FUNCTION create_order(
    p_user_id BIGINT,
    p_items JSONB,  -- 格式: [{"product_id": 1, "quantity": 2}, ...]
    p_shipping_address JSONB
)
RETURNS TABLE(order_id BIGINT, order_no VARCHAR)
LANGUAGE plpgsql
AS $$
DECLARE
    v_order_id BIGINT;
    v_order_no VARCHAR(32);
    v_total NUMERIC(10,2) := 0;
    v_item JSONB;
    v_product_id BIGINT;
    v_quantity INT;
    v_price NUMERIC(10,2);
    v_stock INT;
BEGIN
    -- 生成订单号
    v_order_no := 'ORD' || to_char(NOW(), 'YYYYMMDD') || LPAD(nextval('orders_id_seq')::TEXT, 6, '0');

    -- 创建订单
    INSERT INTO orders (order_no, user_id, total_amount, shipping_address, status)
    VALUES (v_order_no, p_user_id, 0, p_shipping_address, 1)
    RETURNING id INTO v_order_id;

    -- 处理每个商品
    FOR v_item IN SELECT * FROM jsonb_array_elements(p_items)
    LOOP
        v_product_id := (v_item->>'product_id')::BIGINT;
        v_quantity := (v_item->>'quantity')::INT;

        -- 获取商品信息并锁定行
        SELECT price, stock INTO v_price, v_stock
        FROM products
        WHERE id = v_product_id AND status = 1
        FOR UPDATE;

        -- 检查库存
        IF v_stock < v_quantity THEN
            RAISE EXCEPTION '商品 % 库存不足，当前库存：%', v_product_id, v_stock;
        END IF;

        -- 扣减库存
        UPDATE products
        SET stock = stock - v_quantity
        WHERE id = v_product_id;

        -- 添加订单明细
        INSERT INTO order_items (
            order_id, product_id, product_name, price, quantity, subtotal,
            product_snapshot
        )
        SELECT
            v_order_id,
            p.id,
            p.name,
            v_price,
            v_quantity,
            v_price * v_quantity,
            jsonb_build_object('price', p.price, 'specifications', p.specifications)
        FROM products p
        WHERE p.id = v_product_id;

        -- 累加总金额
        v_total := v_total + (v_price * v_quantity);
    END LOOP;

    -- 更新订单总金额
    UPDATE orders SET total_amount = v_total WHERE id = v_order_id;

    RETURN QUERY SELECT v_order_id, v_order_no;
END;
$$;

-- 使用示例
SELECT * FROM create_order(
    1,
    '[{"product_id": 101, "quantity": 2}, {"product_id": 102, "quantity": 1}]'::jsonb,
    '{"name": "张三", "phone": "13800138000", "address": "北京市朝阳区xxx"}'::jsonb
);
```

### 订单统计查询

```sql
-- 每日订单统计（使用窗口函数）
SELECT
    DATE(created_at) AS order_date,
    COUNT(*) AS order_count,
    SUM(total_amount) AS total_sales,
    AVG(total_amount) AS avg_order_value,
    -- 累计销售额
    SUM(SUM(total_amount)) OVER (ORDER BY DATE(created_at)) AS cumulative_sales
FROM orders
WHERE status IN (2, 3, 4)  -- 已支付、已发货、已完成
  AND created_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY order_date DESC;

-- 热销商品 TOP 10
WITH product_sales AS (
    SELECT
        oi.product_id,
        oi.product_name,
        SUM(oi.quantity) AS total_sold,
        SUM(oi.subtotal) AS total_revenue,
        COUNT(DISTINCT oi.order_id) AS order_count
    FROM order_items oi
    INNER JOIN orders o ON oi.order_id = o.id
    WHERE o.status IN (2, 3, 4)
      AND o.created_at >= NOW() - INTERVAL '30 days'
    GROUP BY oi.product_id, oi.product_name
)
SELECT
    ps.*,
    RANK() OVER (ORDER BY ps.total_sold DESC) AS rank
FROM product_sales ps
ORDER BY ps.total_sold DESC
LIMIT 10;
```

## 博客系统设计

### 数据库设计（使用全文搜索）

```sql
-- 文章表
CREATE TABLE posts (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    summary VARCHAR(500),
    cover_image VARCHAR(255),
    tags TEXT[] DEFAULT '{}',  -- 使用数组存储标签
    view_count INT NOT NULL DEFAULT 0,
    like_count INT NOT NULL DEFAULT 0,
    comment_count INT NOT NULL DEFAULT 0,
    status SMALLINT NOT NULL DEFAULT 1,  -- 1-草稿 2-已发布 3-已删除
    -- 全文搜索向量
    search_vector tsvector,
    published_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 创建全文搜索索引
CREATE INDEX idx_posts_search ON posts USING GIN(search_vector);
CREATE INDEX idx_posts_tags ON posts USING GIN(tags);
CREATE INDEX idx_posts_published_at ON posts(published_at);

-- 自动更新搜索向量的触发器
CREATE OR REPLACE FUNCTION update_post_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', COALESCE(NEW.title, '') || ' ' || COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_post_search
BEFORE INSERT OR UPDATE OF title, content ON posts
FOR EACH ROW
EXECUTE FUNCTION update_post_search_vector();

-- 评论表（支持树形结构）
CREATE TABLE comments (
    id BIGSERIAL PRIMARY KEY,
    post_id BIGINT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL,
    parent_id BIGINT REFERENCES comments(id) ON DELETE CASCADE,
    -- 存储完整路径，便于查询整个评论树
    path TEXT,
    content TEXT NOT NULL,
    like_count INT NOT NULL DEFAULT 0,
    status SMALLINT NOT NULL DEFAULT 1,  -- 1-正常 2-已删除
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_comments_post_id ON comments(post_id);
CREATE INDEX idx_comments_path ON comments(path);

-- 自动更新评论路径的触发器
CREATE OR REPLACE FUNCTION update_comment_path()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.parent_id IS NULL THEN
        NEW.path := NEW.id::TEXT;
    ELSE
        SELECT path || '.' || NEW.id INTO NEW.path
        FROM comments WHERE id = NEW.parent_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_comment_path
BEFORE INSERT ON comments
FOR EACH ROW
EXECUTE FUNCTION update_comment_path();
```

### 全文搜索查询

```sql
-- 基本全文搜索
SELECT
    id,
    title,
    summary,
    ts_rank(search_vector, query) AS rank
FROM posts, to_tsquery('english', 'postgresql & database') AS query
WHERE search_vector @@ query AND status = 2
ORDER BY rank DESC
LIMIT 20;

-- 高级全文搜索（带高亮）
SELECT
    id,
    title,
    ts_headline('english', content, query, 'MaxWords=50, MinWords=25') AS snippet,
    ts_rank(search_vector, query) AS rank
FROM posts, to_tsquery('english', 'postgresql <-> optimization') AS query
WHERE search_vector @@ query AND status = 2
ORDER BY rank DESC
LIMIT 20;

-- 标签搜索（使用数组）
SELECT * FROM posts
WHERE 'postgresql' = ANY(tags) AND status = 2
ORDER BY published_at DESC;

-- 多标签搜索
SELECT * FROM posts
WHERE tags @> ARRAY['postgresql', 'optimization'] AND status = 2;
```

### 评论树查询

```sql
-- 获取文章的所有评论（树形结构）
WITH RECURSIVE comment_tree AS (
    -- 根评论
    SELECT
        c.*,
        u.username,
        0 AS depth
    FROM comments c
    INNER JOIN users u ON c.user_id = u.id
    WHERE c.post_id = 1 AND c.parent_id IS NULL

    UNION ALL

    -- 子评论
    SELECT
        c.*,
        u.username,
        ct.depth + 1
    FROM comments c
    INNER JOIN users u ON c.user_id = u.id
    INNER JOIN comment_tree ct ON c.parent_id = ct.id
)
SELECT * FROM comment_tree
ORDER BY path;
```

## 地理位置应用（PostGIS）

### 启用 PostGIS 扩展

```sql
-- 启用 PostGIS
CREATE EXTENSION IF NOT EXISTS postgis;

-- 创建地点表
CREATE TABLE locations (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    address TEXT,
    -- 地理坐标点（经度，纬度）
    geom GEOMETRY(Point, 4326),
    category VARCHAR(50),
    rating NUMERIC(2,1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 创建空间索引
CREATE INDEX idx_locations_geom ON locations USING GIST(geom);
```

### 地理位置查询

```sql
-- 插入地点（北京天安门）
INSERT INTO locations (name, address, geom, category)
VALUES (
    '天安门',
    '北京市东城区',
    ST_SetSRID(ST_MakePoint(116.397428, 39.909187), 4326),
    'landmark'
);

-- 查找附近的地点（5公里内）
SELECT
    name,
    address,
    ST_Distance(
        geom::geography,
        ST_SetSRID(ST_MakePoint(116.404, 39.915), 4326)::geography
    ) / 1000 AS distance_km
FROM locations
WHERE ST_DWithin(
    geom::geography,
    ST_SetSRID(ST_MakePoint(116.404, 39.915), 4326)::geography,
    5000  -- 5000米
)
ORDER BY geom <-> ST_SetSRID(ST_MakePoint(116.404, 39.915), 4326)
LIMIT 10;

-- 在多边形区域内查找
SELECT name, address
FROM locations
WHERE ST_Within(
    geom,
    ST_GeomFromText('POLYGON((
        116.3 39.8,
        116.5 39.8,
        116.5 40.0,
        116.3 40.0,
        116.3 39.8
    ))', 4326)
);
```

## 时序数据分析

### 物联网传感器数据

```sql
-- 传感器数据表（使用 TimescaleDB 扩展会更好）
CREATE TABLE sensor_data (
    time TIMESTAMPTZ NOT NULL,
    sensor_id INT NOT NULL,
    temperature NUMERIC(5,2),
    humidity NUMERIC(5,2),
    pressure NUMERIC(7,2)
);

-- 创建超表（需要 TimescaleDB）
-- SELECT create_hypertable('sensor_data', 'time');

-- 创建索引
CREATE INDEX idx_sensor_data_time ON sensor_data(time DESC);
CREATE INDEX idx_sensor_data_sensor_id ON sensor_data(sensor_id, time DESC);

-- 时间范围聚合查询
SELECT
    time_bucket('1 hour', time) AS hour,
    sensor_id,
    AVG(temperature) AS avg_temp,
    MAX(temperature) AS max_temp,
    MIN(temperature) AS min_temp
FROM sensor_data
WHERE time >= NOW() - INTERVAL '24 hours'
GROUP BY hour, sensor_id
ORDER BY hour DESC;

-- 移动平均
SELECT
    time,
    sensor_id,
    temperature,
    AVG(temperature) OVER (
        PARTITION BY sensor_id
        ORDER BY time
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    ) AS moving_avg
FROM sensor_data
WHERE sensor_id = 1 AND time >= NOW() - INTERVAL '24 hours'
ORDER BY time DESC;
```

## 分区表实战

### 按时间范围分区

```sql
-- 创建分区主表
CREATE TABLE orders_partitioned (
    id BIGSERIAL,
    order_no VARCHAR(32) NOT NULL,
    user_id BIGINT NOT NULL,
    total_amount NUMERIC(10,2) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
) PARTITION BY RANGE (created_at);

-- 创建分区
CREATE TABLE orders_2024_q1 PARTITION OF orders_partitioned
FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE orders_2024_q2 PARTITION OF orders_partitioned
FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');

CREATE TABLE orders_2024_q3 PARTITION OF orders_partitioned
FOR VALUES FROM ('2024-07-01') TO ('2024-10-01');

CREATE TABLE orders_2024_q4 PARTITION OF orders_partitioned
FOR VALUES FROM ('2024-10-01') TO ('2025-01-01');

-- 为每个分区创建索引
CREATE INDEX idx_orders_2024_q1_created ON orders_2024_q1(created_at);
CREATE INDEX idx_orders_2024_q2_created ON orders_2024_q2(created_at);

-- 查询会自动路由到对应分区
SELECT * FROM orders_partitioned
WHERE created_at >= '2024-03-01' AND created_at < '2024-04-01';
```

## 总结

本文档通过实际案例展示了 PostgreSQL 的强大特性：

- ✅ JSONB 和数组类型的灵活应用
- ✅ 全文搜索和 GIN 索引
- ✅ 递归查询处理树形结构
- ✅ PostGIS 地理位置查询
- ✅ 窗口函数和时序数据分析
- ✅ 分区表优化大数据量场景
- ✅ 存储过程和触发器实现业务逻辑

这些案例充分展示了 PostgreSQL 相比传统关系数据库的优势，可以直接应用到实际项目中。

继续学习 [性能优化](/docs/postgres/performance-optimization) 和 [最佳实践](/docs/postgres/best-practices)！
