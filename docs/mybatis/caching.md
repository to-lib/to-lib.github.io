---
sidebar_position: 7
title: MyBatis 缓存机制
---

# MyBatis 缓存机制

MyBatis 提供了强大的缓存机制来提升查询性能。本章详细介绍一级缓存、二级缓存的原理和使用方法。

## 缓存概述

MyBatis 包含两级缓存：

| 缓存级别 | 作用域 | 默认状态 | 生命周期 |
|----------|--------|----------|----------|
| 一级缓存 | SqlSession | 默认开启 | SqlSession 生命周期 |
| 二级缓存 | Mapper (namespace) | 默认关闭 | 应用生命周期 |

```
┌─────────────────────────────────────────────────────────┐
│                      应用程序                            │
├─────────────────────────────────────────────────────────┤
│  SqlSession 1          SqlSession 2          SqlSession 3│
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐│
│  │ 一级缓存    │      │ 一级缓存    │      │ 一级缓存    ││
│  └─────────────┘      └─────────────┘      └─────────────┘│
├─────────────────────────────────────────────────────────┤
│                    二级缓存 (Mapper 级别)                 │
│  ┌─────────────────────────────────────────────────────┐│
│  │  UserMapper Cache    │    OrderMapper Cache         ││
│  └─────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────┤
│                        数据库                            │
└─────────────────────────────────────────────────────────┘
```

## 一级缓存

### 工作原理

一级缓存是 SqlSession 级别的缓存，同一个 SqlSession 中执行相同的查询会直接从缓存获取结果。

```java
try (SqlSession session = sqlSessionFactory.openSession()) {
    UserMapper mapper = session.getMapper(UserMapper.class);
    
    // 第一次查询，执行 SQL，结果放入一级缓存
    User user1 = mapper.selectById(1L);
    
    // 第二次查询，直接从一级缓存获取，不执行 SQL
    User user2 = mapper.selectById(1L);
    
    System.out.println(user1 == user2); // true，同一个对象
}
```

### 缓存 Key

一级缓存的 Key 由以下元素组成：
- Statement ID (namespace + id)
- 分页参数 (offset, limit)
- SQL 语句
- 参数值

```java
// 这两个查询使用不同的缓存 Key
User user1 = mapper.selectById(1L);  // Key: selectById + 1
User user2 = mapper.selectById(2L);  // Key: selectById + 2
```

### 缓存失效场景

一级缓存在以下情况会失效：

#### 1. 执行增删改操作

```java
try (SqlSession session = sqlSessionFactory.openSession()) {
    UserMapper mapper = session.getMapper(UserMapper.class);
    
    User user1 = mapper.selectById(1L);  // 查询，放入缓存
    
    mapper.updateName(1L, "新名字");      // 更新，清空缓存
    
    User user2 = mapper.selectById(1L);  // 重新查询数据库
    
    System.out.println(user1 == user2);  // false
}
```

#### 2. 调用 clearCache()

```java
try (SqlSession session = sqlSessionFactory.openSession()) {
    UserMapper mapper = session.getMapper(UserMapper.class);
    
    User user1 = mapper.selectById(1L);
    
    session.clearCache();  // 手动清空缓存
    
    User user2 = mapper.selectById(1L);  // 重新查询
}
```

#### 3. 调用 commit() 或 rollback()

```java
try (SqlSession session = sqlSessionFactory.openSession()) {
    UserMapper mapper = session.getMapper(UserMapper.class);
    
    User user1 = mapper.selectById(1L);
    
    session.commit();  // 提交事务，清空缓存
    
    User user2 = mapper.selectById(1L);  // 重新查询
}
```

#### 4. 不同的 SqlSession

```java
// SqlSession 1
try (SqlSession session1 = sqlSessionFactory.openSession()) {
    User user1 = session1.getMapper(UserMapper.class).selectById(1L);
}

// SqlSession 2 - 不共享一级缓存
try (SqlSession session2 = sqlSessionFactory.openSession()) {
    User user2 = session2.getMapper(UserMapper.class).selectById(1L);
}
```

### 一级缓存作用域配置

```xml
<settings>
    <!-- SESSION: 默认，整个 SqlSession 有效 -->
    <!-- STATEMENT: 仅当前语句有效，相当于禁用一级缓存 -->
    <setting name="localCacheScope" value="SESSION"/>
</settings>
```

### Spring 中的一级缓存

在 Spring 中，默认每次调用 Mapper 方法都会创建新的 SqlSession，因此一级缓存通常不生效。

```java
@Service
public class UserService {
    @Autowired
    private UserMapper userMapper;
    
    public void test() {
        // 每次调用都是新的 SqlSession，一级缓存不生效
        User user1 = userMapper.selectById(1L);  // 执行 SQL
        User user2 = userMapper.selectById(1L);  // 再次执行 SQL
    }
    
    @Transactional
    public void testWithTransaction() {
        // 在事务中，使用同一个 SqlSession，一级缓存生效
        User user1 = userMapper.selectById(1L);  // 执行 SQL
        User user2 = userMapper.selectById(1L);  // 从缓存获取
    }
}
```

## 二级缓存

### 工作原理

二级缓存是 Mapper (namespace) 级别的缓存，多个 SqlSession 可以共享同一个 Mapper 的二级缓存。

```
SqlSession 1 ──┐
               ├──> UserMapper 二级缓存 ──> 数据库
SqlSession 2 ──┘
```

### 开启二级缓存

#### 1. 全局开启（默认已开启）

```xml
<!-- mybatis-config.xml -->
<settings>
    <setting name="cacheEnabled" value="true"/>
</settings>
```

#### 2. Mapper 中配置

**XML 方式：**

```xml
<!-- UserMapper.xml -->
<mapper namespace="com.example.mapper.UserMapper">
    
    <!-- 开启二级缓存 -->
    <cache/>
    
    <!-- 或者详细配置 -->
    <cache
        eviction="LRU"
        flushInterval="60000"
        size="512"
        readOnly="false"/>
    
</mapper>
```

**注解方式：**

```java
@CacheNamespace(
    eviction = LruCache.class,
    flushInterval = 60000,
    size = 512,
    readWrite = true
)
public interface UserMapper {
    // ...
}
```

### cache 属性

| 属性 | 描述 | 默认值 |
|------|------|--------|
| `eviction` | 淘汰策略 | LRU |
| `flushInterval` | 刷新间隔（毫秒） | 无（不定时刷新） |
| `size` | 缓存对象数量 | 1024 |
| `readOnly` | 只读缓存 | false |
| `blocking` | 阻塞获取 | false |

### 淘汰策略

| 策略 | 描述 |
|------|------|
| `LRU` | 最近最少使用（默认） |
| `FIFO` | 先进先出 |
| `SOFT` | 软引用，JVM 内存不足时回收 |
| `WEAK` | 弱引用，GC 时回收 |

### 实体类要求

使用二级缓存时，实体类必须实现 `Serializable` 接口：

```java
public class User implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private Long id;
    private String name;
    private String email;
    // ...
}
```

### 二级缓存使用示例

```java
// SqlSession 1
try (SqlSession session1 = sqlSessionFactory.openSession()) {
    UserMapper mapper1 = session1.getMapper(UserMapper.class);
    User user1 = mapper1.selectById(1L);  // 查询数据库
    session1.commit();  // 提交后，数据放入二级缓存
}

// SqlSession 2
try (SqlSession session2 = sqlSessionFactory.openSession()) {
    UserMapper mapper2 = session2.getMapper(UserMapper.class);
    User user2 = mapper2.selectById(1L);  // 从二级缓存获取
}
```

> [!IMPORTANT]
> 二级缓存在 SqlSession 提交或关闭后才会生效。

### 缓存失效

二级缓存在以下情况失效：

```xml
<!-- 默认情况下，insert/update/delete 会清空缓存 -->
<insert id="insert" flushCache="true">...</insert>
<update id="update" flushCache="true">...</update>
<delete id="delete" flushCache="true">...</delete>

<!-- select 默认使用缓存 -->
<select id="selectById" useCache="true">...</select>

<!-- 可以禁用某个查询的缓存 -->
<select id="selectRealtime" useCache="false" flushCache="true">
    SELECT * FROM user WHERE id = #{id}
</select>
```

### 只读缓存 vs 读写缓存

```xml
<!-- 只读缓存：返回缓存对象的引用，性能好但不安全 -->
<cache readOnly="true"/>

<!-- 读写缓存：返回缓存对象的副本（序列化/反序列化），安全但性能差 -->
<cache readOnly="false"/>
```

```java
// readOnly="true" 时
User user1 = mapper.selectById(1L);
User user2 = mapper.selectById(1L);
System.out.println(user1 == user2);  // true，同一个对象

// readOnly="false" 时
User user1 = mapper.selectById(1L);
User user2 = mapper.selectById(1L);
System.out.println(user1 == user2);  // false，不同对象（副本）
```

### 缓存引用

多个 Mapper 可以共享同一个缓存：

**XML 方式：**

```xml
<!-- OrderMapper.xml -->
<mapper namespace="com.example.mapper.OrderMapper">
    <!-- 引用 UserMapper 的缓存 -->
    <cache-ref namespace="com.example.mapper.UserMapper"/>
</mapper>
```

**注解方式：**

```java
@CacheNamespaceRef(UserMapper.class)
public interface OrderMapper {
    // ...
}
```

## 缓存查询顺序

```
查询请求
    │
    ▼
┌─────────────┐
│  二级缓存   │ ──命中──> 返回结果
└─────────────┘
    │ 未命中
    ▼
┌─────────────┐
│  一级缓存   │ ──命中──> 返回结果
└─────────────┘
    │ 未命中
    ▼
┌─────────────┐
│   数据库    │ ──查询──> 放入一级缓存 ──> 返回结果
└─────────────┘
                          │
                          ▼ (SqlSession 关闭/提交)
                    放入二级缓存
```

## 第三方缓存集成

### Redis 缓存

#### 1. 添加依赖

```xml
<dependency>
    <groupId>org.mybatis.caches</groupId>
    <artifactId>mybatis-redis</artifactId>
    <version>1.0.0-beta2</version>
</dependency>
```

#### 2. 配置 redis.properties

```properties
redis.host=localhost
redis.port=6379
redis.connectionTimeout=5000
redis.password=
redis.database=0
```

#### 3. 配置 Mapper

```xml
<mapper namespace="com.example.mapper.UserMapper">
    <cache type="org.mybatis.caches.redis.RedisCache"/>
</mapper>
```

### Ehcache 缓存

#### 1. 添加依赖

```xml
<dependency>
    <groupId>org.mybatis.caches</groupId>
    <artifactId>mybatis-ehcache</artifactId>
    <version>1.2.3</version>
</dependency>
```

#### 2. 配置 ehcache.xml

```xml
<?xml version="1.0" encoding="UTF-8"?>
<ehcache>
    <defaultCache
        maxElementsInMemory="10000"
        eternal="false"
        timeToIdleSeconds="120"
        timeToLiveSeconds="120"
        overflowToDisk="true"
        diskPersistent="false"
        diskExpiryThreadIntervalSeconds="120"/>
    
    <cache name="com.example.mapper.UserMapper"
        maxElementsInMemory="1000"
        eternal="false"
        timeToIdleSeconds="300"
        timeToLiveSeconds="600"
        overflowToDisk="true"/>
</ehcache>
```

#### 3. 配置 Mapper

```xml
<mapper namespace="com.example.mapper.UserMapper">
    <cache type="org.mybatis.caches.ehcache.EhcacheCache"/>
</mapper>
```

### 自定义缓存

实现 `org.apache.ibatis.cache.Cache` 接口：

```java
public class MyCustomCache implements Cache {
    
    private final String id;
    private final Map<Object, Object> cache = new ConcurrentHashMap<>();
    
    public MyCustomCache(String id) {
        this.id = id;
    }
    
    @Override
    public String getId() {
        return id;
    }
    
    @Override
    public void putObject(Object key, Object value) {
        cache.put(key, value);
    }
    
    @Override
    public Object getObject(Object key) {
        return cache.get(key);
    }
    
    @Override
    public Object removeObject(Object key) {
        return cache.remove(key);
    }
    
    @Override
    public void clear() {
        cache.clear();
    }
    
    @Override
    public int getSize() {
        return cache.size();
    }
}
```

```xml
<cache type="com.example.cache.MyCustomCache"/>
```

## 缓存使用建议

### 适合使用缓存的场景

- 查询频繁，数据变化少
- 对实时性要求不高
- 单表查询或简单关联查询

### 不适合使用缓存的场景

- 数据变化频繁
- 对实时性要求高
- 多表关联查询（可能导致脏数据）
- 财务、库存等敏感数据

### 多表关联的缓存问题

```xml
<!-- UserMapper.xml -->
<cache/>

<select id="selectWithOrders" resultMap="withOrdersMap">
    SELECT u.*, o.* FROM user u
    LEFT JOIN orders o ON u.id = o.user_id
    WHERE u.id = #{id}
</select>
```

问题：当 orders 表数据变化时，UserMapper 的缓存不会失效，导致脏数据。

解决方案：
1. 使用 `cache-ref` 共享缓存
2. 禁用该查询的缓存
3. 使用第三方缓存并设置合理的过期时间

### 最佳实践

```xml
<mapper namespace="com.example.mapper.UserMapper">
    
    <!-- 配置二级缓存 -->
    <cache
        eviction="LRU"
        flushInterval="300000"
        size="1024"
        readOnly="false"/>
    
    <!-- 普通查询使用缓存 -->
    <select id="selectById" resultType="User">
        SELECT * FROM user WHERE id = #{id}
    </select>
    
    <!-- 实时性要求高的查询禁用缓存 -->
    <select id="selectBalance" resultType="BigDecimal" useCache="false">
        SELECT balance FROM user WHERE id = #{id}
    </select>
    
    <!-- 统计查询禁用缓存 -->
    <select id="countByStatus" resultType="int" useCache="false" flushCache="true">
        SELECT COUNT(*) FROM user WHERE status = #{status}
    </select>
    
</mapper>
```

## 相关链接

- [核心概念](/docs/mybatis/core-concepts) - 了解 MyBatis 架构
- [配置详解](/docs/mybatis/configuration) - 缓存相关配置
- [Spring 集成](/docs/mybatis/spring-integration) - Spring 中的缓存配置

---

**最后更新**: 2025 年 12 月
