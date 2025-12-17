---
id: caching
title: Spring Cache 缓存抽象
sidebar_label: 缓存抽象
sidebar_position: 14
---

# Spring Cache 缓存抽象

Spring Cache 提供了“缓存的统一抽象层”，让你用注解定义缓存行为，而不强绑定某个缓存实现。

常见使用场景：

- 读多写少的数据（字典、配置、用户信息）
- 幂等查询（按 id 查询）
- 计算代价大且结果可复用的逻辑

## 1. 核心概念

- `Cache`：单个缓存（如 `users`）
- `CacheManager`：管理多个 `Cache`
- Key：缓存键（可用 SpEL 表达式）

## 2. 启用缓存

```java
@Configuration
@EnableCaching
public class CacheConfig {
}
```

## 3. 三个最常用注解

### 3.1 `@Cacheable`：查缓存，没有则执行并写入

```java
@Service
public class UserService {

    @Cacheable(cacheNames = "users", key = "#id", unless = "#result == null")
    public User findById(Long id) {
        return userRepository.findById(id).orElse(null);
    }
}
```

### 3.2 `@CachePut`：总是执行，并把结果写入缓存

```java
@CachePut(cacheNames = "users", key = "#result.id")
public User update(User user) {
    return userRepository.save(user);
}
```

### 3.3 `@CacheEvict`：删除缓存

```java
@CacheEvict(cacheNames = "users", key = "#id")
public void delete(Long id) {
    userRepository.deleteById(id);
}
```

## 4. SpEL：key、condition、unless

- `key`：决定缓存键
- `condition`：满足条件才缓存
- `unless`：满足条件则不缓存（通常用于过滤 null/空结果）

示例：

```java
@Cacheable(
    cacheNames = "products",
    key = "#id",
    condition = "#id != null",
    unless = "#result == null"
)
public Product findProduct(Long id) {
}
```

## 5. 常见坑

- **[self-invocation]**：同类内部调用带缓存注解的方法不会走代理。
- **[缓存一致性]**：写操作必须配合 `@CacheEvict/@CachePut`，否则读到旧数据。
- **[缓存穿透/击穿]**：热点 key 失效瞬间可能压垮 DB，必要时加互斥/预热/限流。
- **[TTL 与淘汰策略]**：抽象层不直接提供 TTL，通常由具体实现（如 Redis/Caffeine）配置。

---

下一步建议：

- [SpEL 表达式](/docs/spring/spel)
- [最佳实践](/docs/spring/best-practices)
