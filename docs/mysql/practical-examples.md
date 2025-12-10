---
sidebar_position: 12
title: 实战案例
---

# MySQL 实战案例

> [!TIP] > **实战演练**: 本文档通过实际项目案例，展示 MySQL 在不同业务场景下的应用和最佳实践。

## 用户系统设计

### 需求分析

设计一个支持用户注册、登录、权限管理的用户系统。

### 数据库设计

```sql
-- 用户表
CREATE TABLE users (
    id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT COMMENT '用户ID',
    username VARCHAR(50) NOT NULL UNIQUE COMMENT '用户名',
    email VARCHAR(100) NOT NULL UNIQUE COMMENT '邮箱',
    password_hash VARCHAR(255) NOT NULL COMMENT '密码哈希',
    phone VARCHAR(20) COMMENT '手机号',
    avatar_url VARCHAR(255) COMMENT '头像URL',
    status TINYINT NOT NULL DEFAULT 1 COMMENT '状态: 1-正常 2-禁用',
    last_login_at TIMESTAMP NULL COMMENT '最后登录时间',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_email (email),
    INDEX idx_phone (phone),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户表';

-- 角色表
CREATE TABLE roles (
    id INT UNSIGNED PRIMARY KEY AUTO_INCREMENT COMMENT '角色ID',
    name VARCHAR(50) NOT NULL UNIQUE COMMENT '角色名称',
    description VARCHAR(255) COMMENT '角色描述',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='角色表';

-- 权限表
CREATE TABLE permissions (
    id INT UNSIGNED PRIMARY KEY AUTO_INCREMENT COMMENT '权限ID',
    name VARCHAR(50) NOT NULL UNIQUE COMMENT '权限名称',
    resource VARCHAR(100) NOT NULL COMMENT '资源',
    action VARCHAR(50) NOT NULL COMMENT '操作',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_resource (resource)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='权限表';

-- 用户角色关联表
CREATE TABLE user_roles (
    id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT UNSIGNED NOT NULL COMMENT '用户ID',
    role_id INT UNSIGNED NOT NULL COMMENT '角色ID',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    UNIQUE KEY uk_user_role (user_id, role_id),
    INDEX idx_user_id (user_id),
    INDEX idx_role_id (role_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户角色关联表';

-- 角色权限关联表
CREATE TABLE role_permissions (
    id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
    role_id INT UNSIGNED NOT NULL COMMENT '角色ID',
    permission_id INT UNSIGNED NOT NULL COMMENT '权限ID',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    UNIQUE KEY uk_role_permission (role_id, permission_id),
    INDEX idx_role_id (role_id),
    INDEX idx_permission_id (permission_id),
    FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE,
    FOREIGN KEY (permission_id) REFERENCES permissions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='角色权限关联表';
```

### 常用查询

```sql
-- 用户注册
INSERT INTO users (username, email, password_hash, phone)
VALUES ('zhangsan', 'zhangsan@example.com', '$2a$10$...', '13800138000');

-- 用户登录验证
SELECT id, username, password_hash, status
FROM users
WHERE email = 'zhangsan@example.com' AND status = 1;

-- 分配角色
INSERT INTO user_roles (user_id, role_id) VALUES (1, 2);

-- 查询用户的所有权限
SELECT DISTINCT p.name, p.resource, p.action
FROM users u
INNER JOIN user_roles ur ON u.id = ur.user_id
INNER JOIN roles r ON ur.role_id = r.id
INNER JOIN role_permissions rp ON r.id = rp.role_id
INNER JOIN permissions p ON rp.permission_id = p.id
WHERE u.id = 1;

-- 检查用户是否有特定权限
SELECT COUNT(*) > 0 AS has_permission
FROM users u
INNER JOIN user_roles ur ON u.id = ur.user_id
INNER JOIN role_permissions rp ON ur.role_id = rp.role_id
INNER JOIN permissions p ON rp.permission_id = p.id
WHERE u.id = 1 AND p.resource = 'articles' AND p.action = 'create';
```

## 电商订单系统

### 需求分析

设计一个支持商品、订单、支付的电商系统。

### 数据库设计

```sql
-- 商品表
CREATE TABLE products (
    id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT COMMENT '商品ID',
    name VARCHAR(200) NOT NULL COMMENT '商品名称',
    description TEXT COMMENT '商品描述',
    price DECIMAL(10,2) NOT NULL COMMENT '价格',
    stock INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '库存',
    category_id INT UNSIGNED COMMENT '分类ID',
    status TINYINT NOT NULL DEFAULT 1 COMMENT '状态: 1-上架 2-下架',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_category (category_id),
    INDEX idx_status (status),
    INDEX idx_name (name(50))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='商品表';

-- 订单表
CREATE TABLE orders (
    id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT COMMENT '订单ID',
    order_no VARCHAR(32) NOT NULL UNIQUE COMMENT '订单号',
    user_id BIGINT UNSIGNED NOT NULL COMMENT '用户ID',
    total_amount DECIMAL(10,2) NOT NULL COMMENT '订单总金额',
    status TINYINT NOT NULL DEFAULT 1 COMMENT '订单状态: 1-待支付 2-已支付 3-已发货 4-已完成 5-已取消',
    payment_method TINYINT COMMENT '支付方式: 1-支付宝 2-微信 3-银行卡',
    paid_at TIMESTAMP NULL COMMENT '支付时间',
    shipped_at TIMESTAMP NULL COMMENT '发货时间',
    completed_at TIMESTAMP NULL COMMENT '完成时间',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_order_no (order_no),
    INDEX idx_user_id (user_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='订单表';

-- 订单明细表
CREATE TABLE order_items (
    id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT COMMENT '明细ID',
    order_id BIGINT UNSIGNED NOT NULL COMMENT '订单ID',
    product_id BIGINT UNSIGNED NOT NULL COMMENT '商品ID',
    product_name VARCHAR(200) NOT NULL COMMENT '商品名称',
    price DECIMAL(10,2) NOT NULL COMMENT '商品价格',
    quantity INT UNSIGNED NOT NULL COMMENT '购买数量',
    subtotal DECIMAL(10,2) NOT NULL COMMENT '小计',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_order_id (order_id),
    INDEX idx_product_id (product_id),
    FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='订单明细表';
```

### 创建订单（带事务）

```sql
START TRANSACTION;

-- 1. 创建订单
INSERT INTO orders (order_no, user_id, total_amount, status)
VALUES ('ORD20251210001', 1, 299.00, 1);

SET @order_id = LAST_INSERT_ID();

-- 2. 添加订单明细
INSERT INTO order_items (order_id, product_id, product_name, price, quantity, subtotal)
VALUES
    (@order_id, 101, 'iPhone 15', 5999.00, 1, 5999.00),
    (@order_id, 102, 'AirPods Pro', 1999.00, 1, 1999.00);

-- 3. 扣减库存
UPDATE products SET stock = stock - 1 WHERE id = 101 AND stock >= 1;
UPDATE products SET stock = stock - 1 WHERE id = 102 AND stock >= 1;

-- 4. 检查库存是否充足
SELECT COUNT(*) INTO @stock_ok FROM products
WHERE id IN (101, 102) AND stock >= 0;

IF @stock_ok = 2 THEN
    COMMIT;
ELSE
    ROLLBACK;
END IF;
```

### 订单统计查询

```sql
-- 每日订单统计
SELECT
    DATE(created_at) AS order_date,
    COUNT(*) AS order_count,
    SUM(total_amount) AS total_sales,
    AVG(total_amount) AS avg_order_value
FROM orders
WHERE status IN (2, 3, 4)  -- 已支付、已发货、已完成
  AND created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY DATE(created_at)
ORDER BY order_date DESC;

-- 热销商品 TOP 10
SELECT
    p.id,
    p.name,
    SUM(oi.quantity) AS total_sold,
    SUM(oi.subtotal) AS total_revenue
FROM products p
INNER JOIN order_items oi ON p.id = oi.product_id
INNER JOIN orders o ON oi.order_id = o.id
WHERE o.status IN (2, 3, 4)
  AND o.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY p.id, p.name
ORDER BY total_sold DESC
LIMIT 10;
```

## 博客系统设计

### 数据库设计

```sql
-- 文章表
CREATE TABLE posts (
    id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT COMMENT '文章ID',
    user_id BIGINT UNSIGNED NOT NULL COMMENT '作者ID',
    title VARCHAR(200) NOT NULL COMMENT '标题',
    content LONGTEXT NOT NULL COMMENT '内容',
    summary VARCHAR(500) COMMENT '摘要',
    cover_image VARCHAR(255) COMMENT '封面图',
    view_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '浏览量',
    like_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '点赞数',
    comment_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '评论数',
    status TINYINT NOT NULL DEFAULT 1 COMMENT '状态: 1-草稿 2-已发布 3-已删除',
    published_at TIMESTAMP NULL COMMENT '发布时间',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_user_id (user_id),
    INDEX idx_status (status),
    INDEX idx_published_at (published_at),
    FULLTEXT INDEX ft_title_content (title, content)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='文章表';

-- 标签表
CREATE TABLE tags (
    id INT UNSIGNED PRIMARY KEY AUTO_INCREMENT COMMENT '标签ID',
    name VARCHAR(50) NOT NULL UNIQUE COMMENT '标签名称',
    post_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '文章数',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='标签表';

-- 文章标签关联表
CREATE TABLE post_tags (
    id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
    post_id BIGINT UNSIGNED NOT NULL COMMENT '文章ID',
    tag_id INT UNSIGNED NOT NULL COMMENT '标签ID',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    UNIQUE KEY uk_post_tag (post_id, tag_id),
    INDEX idx_post_id (post_id),
    INDEX idx_tag_id (tag_id),
    FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='文章标签关联表';

-- 评论表
CREATE TABLE comments (
    id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT COMMENT '评论ID',
    post_id BIGINT UNSIGNED NOT NULL COMMENT '文章ID',
    user_id BIGINT UNSIGNED NOT NULL COMMENT '用户ID',
    parent_id BIGINT UNSIGNED COMMENT '父评论ID',
    content TEXT NOT NULL COMMENT '评论内容',
    like_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '点赞数',
    status TINYINT NOT NULL DEFAULT 1 COMMENT '状态: 1-正常 2-已删除',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_post_id (post_id),
    INDEX idx_user_id (user_id),
    INDEX idx_parent_id (parent_id),
    FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='评论表';
```

### 常用查询

```sql
-- 获取文章列表（带分页）
SELECT
    p.id,
    p.title,
    p.summary,
    p.cover_image,
    p.view_count,
    p.like_count,
    p.comment_count,
    p.published_at,
    u.username AS author
FROM posts p
INNER JOIN users u ON p.user_id = u.id
WHERE p.status = 2  -- 已发布
ORDER BY p.published_at DESC
LIMIT 20 OFFSET 0;

-- 获取文章详情（带标签）
SELECT
    p.*,
    u.username AS author,
    GROUP_CONCAT(t.name) AS tags
FROM posts p
INNER JOIN users u ON p.user_id = u.id
LEFT JOIN post_tags pt ON p.id = pt.post_id
LEFT JOIN tags t ON pt.tag_id = t.id
WHERE p.id = 1 AND p.status = 2
GROUP BY p.id;

-- 增加浏览量（非事务）
UPDATE posts SET view_count = view_count + 1 WHERE id = 1;

-- 全文搜索
SELECT id, title, summary
FROM posts
WHERE MATCH(title, content) AGAINST('MySQL 优化' IN NATURAL LANGUAGE MODE)
  AND status = 2
LIMIT 20;

-- 热门标签 TOP 10
SELECT t.name, t.post_count
FROM tags t
ORDER BY t.post_count DESC
LIMIT 10;
```

## 分库分表实战

### 场景：订单表数据量过大

当订单表数据超过千万级别时，需要进行分表。

### 水平分表方案

```sql
-- 按订单ID进行哈希分表（4张表）
CREATE TABLE orders_0 LIKE orders;
CREATE TABLE orders_1 LIKE orders;
CREATE TABLE orders_2 LIKE orders;
CREATE TABLE orders_3 LIKE orders;

-- 路由规则：order_id % 4
-- order_id = 1 -> orders_1
-- order_id = 2 -> orders_2
-- order_id = 3 -> orders_3
-- order_id = 4 -> orders_0
```

### 应用层路由

```java
public class OrderShardingStrategy {
    private static final int SHARD_COUNT = 4;

    public String getTableName(Long orderId) {
        int shardIndex = (int)(orderId % SHARD_COUNT);
        return "orders_" + shardIndex;
    }

    public void insertOrder(Order order) {
        String tableName = getTableName(order.getId());
        String sql = "INSERT INTO " + tableName + " ...";
        // 执行插入
    }

    public Order getOrder(Long orderId) {
        String tableName = getTableName(orderId);
        String sql = "SELECT * FROM " + tableName + " WHERE id = ?";
        // 执行查询
    }
}
```

## 主从复制与读写分离

### 配置主从复制

**主库配置** (`master`):

```ini
# my.cnf
[mysqld]
server-id=1
log-bin=mysql-bin
binlog-format=ROW
binlog-do-db=mydb
```

**从库配置** (`slave`):

```ini
# my.cnf
[mysqld]
server-id=2
relay-log=mysql-relay-bin
read_only=1
```

**配置从库**:

```sql
-- 从库执行
CHANGE MASTER TO
    MASTER_HOST='192.168.1.100',
    MASTER_PORT=3306,
    MASTER_USER='repl',
    MASTER_PASSWORD='password',
    MASTER_LOG_FILE='mysql-bin.000001',
    MASTER_LOG_POS=154;

START SLAVE;

-- 查看状态
SHOW SLAVE STATUS\G
```

### 应用层读写分离

```java
public class DataSourceRouter {
    private DataSource masterDataSource;  // 主库
    private List<DataSource> slaveDataSources;  // 从库列表
    private AtomicInteger counter = new AtomicInteger(0);

    public DataSource getDataSource() {
        // 判断当前操作类型
        if (isWriteOperation()) {
            return masterDataSource;  // 写操作用主库
        } else {
            // 读操作从多个从库中轮询选择
            int index = counter.getAndIncrement() % slaveDataSources.size();
            return slaveDataSources.get(index);
        }
    }
}
```

## 总结

本文档通过实际案例展示了：

- ✅ 用户系统：表设计、RBAC 权限模型
- ✅ 电商订单：订单流程、库存扣减、事务处理
- ✅ 博客系统：文章管理、标签系统、评论系统
- ✅ 分库分表：水平分表、路由策略
- ✅ 主从复制：配置方法、读写分离

这些案例可以直接应用到实际项目中，继续学习 [性能优化](./performance-optimization) 和 [最佳实践](./best-practices)！
