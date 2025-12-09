---
sidebar_position: 9
---

# 缓存管理

> [!TIP]
> **缓存是性能优化的利器**: Spring Cache 抽象层支持 Redis、Caffeine 等多种缓存实现。合理使用缓存可显著提升性能。

## 启用缓存

### 配置缓存

```java
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Bean;
import org.springframework.cache.CacheManager;
import org.springframework.cache.concurrent.ConcurrentMapCacheManager;

@Configuration
@EnableCaching
public class CacheConfig {
    
    @Bean
    public CacheManager cacheManager() {
        // 使用内存缓存管理器
        return new ConcurrentMapCacheManager("users", "posts", "comments");
    }
}
```

## 使用缓存注解

### @Cacheable - 读缓存

```java
import org.springframework.cache.annotation.Cacheable;

@Service
public class UserService {
    
    // 从缓存读取，如果缓存中没有则调用方法并缓存结果
    @Cacheable(value = "users", key = "#id")
    public User getUserById(Long id) {
        System.out.println("从数据库查询用户: " + id);
        return userRepository.findById(id).orElse(null);
    }
    
    // 使用条件缓存
    @Cacheable(value = "users", key = "#id", 
               condition = "#id > 0", 
               unless = "#result == null")
    public User getActiveUser(Long id) {
        return userRepository.findById(id).orElse(null);
    }
    
    // 自定义缓存 key
    @Cacheable(value = "usersByEmail", key = "'email:' + #email")
    public User getUserByEmail(String email) {
        return userRepository.findByEmail(email);
    }
}
```

### @CachePut - 写缓存

```java
import org.springframework.cache.annotation.CachePut;

@Service
public class UserService {
    
    // 每次都执行方法，并更新缓存
    @CachePut(value = "users", key = "#result.id")
    public User createUser(User user) {
        return userRepository.save(user);
    }
    
    // 更新用户并更新缓存
    @CachePut(value = "users", key = "#id")
    public User updateUser(Long id, User userDetails) {
        User user = userRepository.findById(id).orElseThrow();
        user.setName(userDetails.getName());
        user.setEmail(userDetails.getEmail());
        return userRepository.save(user);
    }
}
```

### @CacheEvict - 清除缓存

```java
import org.springframework.cache.annotation.CacheEvict;

@Service
public class UserService {
    
    // 删除单个缓存
    @CacheEvict(value = "users", key = "#id")
    public void deleteUser(Long id) {
        userRepository.deleteById(id);
    }
    
    // 清除所有缓存
    @CacheEvict(value = "users", allEntries = true)
    public void clearAllUsers() {
        // 清除 users 缓存中的所有条目
    }
    
    // 删除前清除缓存
    @CacheEvict(value = "users", key = "#id", beforeInvocation = true)
    public void deleteUserBefore(Long id) {
        userRepository.deleteById(id);
    }
}
```

### @Caching - 组合注解

```java
import org.springframework.cache.annotation.Caching;

@Service
public class UserService {
    
    // 同时更新多个缓存
    @Caching(
        cacheable = {
            @Cacheable(value = "users", key = "#id")
        },
        put = {
            @CachePut(value = "usersByEmail", key = "#result.email")
        }
    )
    public User getUserById(Long id) {
        return userRepository.findById(id).orElse(null);
    }
    
    // 删除多个缓存
    @Caching(
        evict = {
            @CacheEvict(value = "users", key = "#id"),
            @CacheEvict(value = "usersByEmail", key = "#user.email")
        }
    )
    public void deleteUser(Long id, User user) {
        userRepository.deleteById(id);
    }
}
```

## Redis 缓存

### 依赖配置

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-redis</artifactId>
</dependency>

<dependency>
    <groupId>redis.clients</groupId>
    <artifactId>jedis</artifactId>
</dependency>
```

### Redis 配置

```yaml
spring:
  redis:
    host: localhost
    port: 6379
    password: 
    database: 0
    timeout: 2000ms
    jedis:
      pool:
        max-active: 8
        max-idle: 8
        min-idle: 0
        max-wait: -1ms
  
  cache:
    type: redis
    redis:
      time-to-live: 600000  # 10 分钟
```

### 使用 Redis 作为缓存

```java
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.cache.CacheManager;
import org.springframework.data.redis.cache.RedisCacheManager;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@EnableCaching
public class RedisConfig {
    
    @Bean
    public CacheManager cacheManager(RedisConnectionFactory connectionFactory) {
        return RedisCacheManager.create(connectionFactory);
    }
}
```

### Redis 直接操作

```java
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.StringRedisTemplate;

@Service
public class RedisService {
    
    @Autowired
    private StringRedisTemplate stringRedisTemplate;
    
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;
    
    // 字符串操作
    public void setString(String key, String value) {
        stringRedisTemplate.opsForValue().set(key, value);
    }
    
    public String getString(String key) {
        return stringRedisTemplate.opsForValue().get(key);
    }
    
    // 对象操作
    public void setUser(String key, User user) {
        redisTemplate.opsForValue().set(key, user);
    }
    
    public User getUser(String key) {
        return (User) redisTemplate.opsForValue().get(key);
    }
    
    // 列表操作
    public void pushList(String key, String value) {
        redisTemplate.opsForList().rightPush(key, value);
    }
    
    public List<Object> getList(String key) {
        return redisTemplate.opsForList().range(key, 0, -1);
    }
    
    // 集合操作
    public void addSet(String key, String value) {
        redisTemplate.opsForSet().add(key, value);
    }
    
    public Set<Object> getSet(String key) {
        return redisTemplate.opsForSet().members(key);
    }
    
    // Hash 操作
    public void setHash(String key, String field, String value) {
        redisTemplate.opsForHash().put(key, field, value);
    }
    
    public Object getHash(String key, String field) {
        return redisTemplate.opsForHash().get(key, field);
    }
    
    // 过期时间设置
    public void setWithExpire(String key, String value, long timeout, TimeUnit unit) {
        stringRedisTemplate.opsForValue().set(key, value, timeout, unit);
    }
    
    // 删除
    public void delete(String key) {
        redisTemplate.delete(key);
    }
}
```

## Caffeine 缓存

### 依赖配置

```xml
<dependency>
    <groupId>com.github.ben-manes.caffeine</groupId>
    <artifactId>caffeine</artifactId>
</dependency>

<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-cache</artifactId>
</dependency>
```

### 配置 Caffeine

```yaml
spring:
  cache:
    type: caffeine
    caffeine:
      spec: maximumSize=500,expireAfterWrite=10m
```

### 使用 Caffeine

```java
import com.github.benmanes.caffeine.cache.Caffeine;
import org.springframework.cache.CacheManager;
import org.springframework.cache.caffeine.CaffeineCacheManager;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import java.util.concurrent.TimeUnit;

@Configuration
public class CaffeineConfig {
    
    @Bean
    public CacheManager cacheManager() {
        CaffeineCacheManager cacheManager = new CaffeineCacheManager("users", "posts");
        cacheManager.setCaffeine(Caffeine.newBuilder()
                .maximumSize(1000)                           // 最大缓存数
                .expireAfterWrite(10, TimeUnit.MINUTES)      // 写入后10分钟过期
                .recordStats());                             // 记录统计信息
        return cacheManager;
    }
}

@Service
public class UserService {
    
    @Cacheable(value = "users", key = "#id")
    public User getUser(Long id) {
        return userRepository.findById(id).orElse(null);
    }
}
```

## 缓存更新策略

### 旁路缓存（Cache Aside）

```java
@Service
public class UserService {
    
    @Autowired
    private UserRepository userRepository;
    
    @Autowired
    private StringRedisTemplate redisTemplate;
    
    public User getUser(Long id) {
        String key = "user:" + id;
        
        // 1. 查询缓存
        User user = (User) redisTemplate.opsForValue().get(key);
        
        // 2. 缓存未命中，查询数据库
        if (user == null) {
            user = userRepository.findById(id).orElse(null);
            
            // 3. 写入缓存
            if (user != null) {
                redisTemplate.opsForValue().set(key, user, 10, TimeUnit.MINUTES);
            }
        }
        
        return user;
    }
}
```

### 写穿（Write Through）

```java
@Service
public class UserService {
    
    @Autowired
    private UserRepository userRepository;
    
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;
    
    public User updateUser(Long id, User userDetails) {
        // 1. 更新数据库
        User user = userRepository.findById(id).orElseThrow();
        user.setName(userDetails.getName());
        user.setEmail(userDetails.getEmail());
        User savedUser = userRepository.save(user);
        
        // 2. 更新缓存
        String key = "user:" + id;
        redisTemplate.opsForValue().set(key, savedUser, 10, TimeUnit.MINUTES);
        
        return savedUser;
    }
}
```

### 写回（Write Behind）

```java
@Service
public class UserService {
    
    @Autowired
    private UserRepository userRepository;
    
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;
    
    public User updateUser(Long id, User userDetails) {
        // 1. 直接更新缓存
        String key = "user:" + id;
        User user = new User(id, userDetails.getName(), userDetails.getEmail());
        redisTemplate.opsForValue().set(key, user, 10, TimeUnit.MINUTES);
        
        // 2. 异步更新数据库
        updateDatabaseAsync(id, user);
        
        return user;
    }
    
    @Async
    public void updateDatabaseAsync(Long id, User user) {
        try {
            Thread.sleep(1000);  // 模拟异步延迟
            userRepository.save(user);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
```

## 缓存穿透和雪崩

### 缓存穿透解决方案

```java
@Service
public class UserService {
    
    private static final String CACHE_NULL_VALUE = "null";
    
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;
    
    public User getUser(Long id) {
        String key = "user:" + id;
        
        Object cached = redisTemplate.opsForValue().get(key);
        
        // 缓存中存在值（包括 null）
        if (cached != null) {
            return CACHE_NULL_VALUE.equals(cached) ? null : (User) cached;
        }
        
        User user = userRepository.findById(id).orElse(null);
        
        // 缓存 null 值，防止穿透
        Object valueToCache = user != null ? user : CACHE_NULL_VALUE;
        redisTemplate.opsForValue().set(key, valueToCache, 5, TimeUnit.MINUTES);
        
        return user;
    }
}
```

### 缓存雪崩解决方案

```java
@Service
public class UserService {
    
    public User getUser(Long id) {
        String key = "user:" + id;
        
        User user = (User) redisTemplate.opsForValue().get(key);
        
        if (user == null) {
            user = userRepository.findById(id).orElse(null);
            
            if (user != null) {
                // 随机过期时间，避免大量缓存同时过期
                long timeout = 10 + new Random().nextInt(5);  // 10-15 分钟
                redisTemplate.opsForValue().set(key, user, timeout, TimeUnit.MINUTES);
            }
        }
        
        return user;
    }
}
```

## 总结

缓存的关键点：

1. **选择合适的缓存** - 本地缓存（Caffeine）或分布式缓存（Redis）
2. **合理设置 TTL** - 避免缓存过期太快或太慢
3. **更新策略** - 旁路、写穿、写回等，根据场景选择
4. **处理边界情况** - 缓存穿透、雪崩、击穿
5. **监控指标** - 缓存命中率、大小等

下一步学习 [定时任务和异步](./scheduling)。
