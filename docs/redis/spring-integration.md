---
sidebar_position: 20
title: Spring Boot 集成
---

# Redis 与 Spring Boot 集成

本文详细介绍 Spring Boot 与 Redis 的深度集成，包括 Spring Data Redis、缓存注解、Session 管理等核心功能。

## 快速开始

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-redis</artifactId>
</dependency>

<!-- 连接池（可选，推荐使用） -->
<dependency>
    <groupId>org.apache.commons</groupId>
    <artifactId>commons-pool2</artifactId>
</dependency>
```

### 基础配置

```yaml
# application.yml
spring:
  redis:
    host: localhost
    port: 6379
    password: your_password
    database: 0
    timeout: 3000ms
    
    # Lettuce 连接池配置
    lettuce:
      pool:
        max-active: 100    # 最大连接数
        max-idle: 20       # 最大空闲连接
        min-idle: 10       # 最小空闲连接
        max-wait: 3000ms   # 获取连接最大等待时间
```

### 基本使用

```java
@Service
public class RedisService {
    
    @Autowired
    private StringRedisTemplate stringRedisTemplate;
    
    public void set(String key, String value) {
        stringRedisTemplate.opsForValue().set(key, value);
    }
    
    public String get(String key) {
        return stringRedisTemplate.opsForValue().get(key);
    }
    
    public void setWithExpire(String key, String value, long timeout) {
        stringRedisTemplate.opsForValue().set(key, value, timeout, TimeUnit.SECONDS);
    }
    
    public Boolean delete(String key) {
        return stringRedisTemplate.delete(key);
    }
}
```

## RedisTemplate 详解

### StringRedisTemplate vs RedisTemplate

| 特性 | StringRedisTemplate | RedisTemplate<Object, Object> |
|------|--------------------|-----------------------------|
| 序列化 | String | JdkSerializationRedisSerializer |
| 可读性 | 好 | 差（二进制） |
| 跨语言 | 是 | 否 |
| 适用场景 | 字符串操作 | Java 对象存储 |

### 自定义 RedisTemplate

```java
@Configuration
public class RedisConfig {
    
    @Bean
    public RedisTemplate<String, Object> redisTemplate(RedisConnectionFactory factory) {
        RedisTemplate<String, Object> template = new RedisTemplate<>();
        template.setConnectionFactory(factory);
        
        // JSON 序列化器
        Jackson2JsonRedisSerializer<Object> jsonSerializer = 
            new Jackson2JsonRedisSerializer<>(Object.class);
        
        ObjectMapper mapper = new ObjectMapper();
        mapper.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.ANY);
        mapper.activateDefaultTyping(
            LaissezFaireSubTypeValidator.instance,
            ObjectMapper.DefaultTyping.NON_FINAL
        );
        jsonSerializer.setObjectMapper(mapper);
        
        // String 序列化器
        StringRedisSerializer stringSerializer = new StringRedisSerializer();
        
        // 设置序列化器
        template.setKeySerializer(stringSerializer);
        template.setHashKeySerializer(stringSerializer);
        template.setValueSerializer(jsonSerializer);
        template.setHashValueSerializer(jsonSerializer);
        
        template.afterPropertiesSet();
        return template;
    }
}
```

### 操作不同数据类型

```java
@Service
public class RedisService {
    
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;
    
    // ===== String 操作 =====
    public void setString(String key, Object value) {
        redisTemplate.opsForValue().set(key, value);
    }
    
    public void setStringWithExpire(String key, Object value, long timeout) {
        redisTemplate.opsForValue().set(key, value, timeout, TimeUnit.SECONDS);
    }
    
    public Object getString(String key) {
        return redisTemplate.opsForValue().get(key);
    }
    
    public Long increment(String key, long delta) {
        return redisTemplate.opsForValue().increment(key, delta);
    }
    
    // ===== Hash 操作 =====
    public void hSet(String key, String field, Object value) {
        redisTemplate.opsForHash().put(key, field, value);
    }
    
    public void hSetAll(String key, Map<String, Object> map) {
        redisTemplate.opsForHash().putAll(key, map);
    }
    
    public Object hGet(String key, String field) {
        return redisTemplate.opsForHash().get(key, field);
    }
    
    public Map<Object, Object> hGetAll(String key) {
        return redisTemplate.opsForHash().entries(key);
    }
    
    public void hDelete(String key, Object... fields) {
        redisTemplate.opsForHash().delete(key, fields);
    }
    
    // ===== List 操作 =====
    public Long lPush(String key, Object... values) {
        return redisTemplate.opsForList().leftPushAll(key, values);
    }
    
    public Object lPop(String key) {
        return redisTemplate.opsForList().leftPop(key);
    }
    
    public List<Object> lRange(String key, long start, long end) {
        return redisTemplate.opsForList().range(key, start, end);
    }
    
    public Long lSize(String key) {
        return redisTemplate.opsForList().size(key);
    }
    
    // ===== Set 操作 =====
    public Long sAdd(String key, Object... values) {
        return redisTemplate.opsForSet().add(key, values);
    }
    
    public Set<Object> sMembers(String key) {
        return redisTemplate.opsForSet().members(key);
    }
    
    public Boolean sIsMember(String key, Object value) {
        return redisTemplate.opsForSet().isMember(key, value);
    }
    
    public Long sRemove(String key, Object... values) {
        return redisTemplate.opsForSet().remove(key, values);
    }
    
    // ===== ZSet (Sorted Set) 操作 =====
    public Boolean zAdd(String key, Object value, double score) {
        return redisTemplate.opsForZSet().add(key, value, score);
    }
    
    public Set<Object> zRange(String key, long start, long end) {
        return redisTemplate.opsForZSet().range(key, start, end);
    }
    
    public Set<Object> zReverseRange(String key, long start, long end) {
        return redisTemplate.opsForZSet().reverseRange(key, start, end);
    }
    
    public Long zRank(String key, Object value) {
        return redisTemplate.opsForZSet().rank(key, value);
    }
    
    // ===== 通用操作 =====
    public Boolean delete(String key) {
        return redisTemplate.delete(key);
    }
    
    public Long delete(Collection<String> keys) {
        return redisTemplate.delete(keys);
    }
    
    public Boolean expire(String key, long timeout) {
        return redisTemplate.expire(key, timeout, TimeUnit.SECONDS);
    }
    
    public Long getExpire(String key) {
        return redisTemplate.getExpire(key, TimeUnit.SECONDS);
    }
    
    public Boolean hasKey(String key) {
        return redisTemplate.hasKey(key);
    }
    
    public Set<String> keys(String pattern) {
        return redisTemplate.keys(pattern);
    }
}
```

## 缓存注解

### 启用缓存

```java
@SpringBootApplication
@EnableCaching
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}
```

### 缓存配置

```java
@Configuration
@EnableCaching
public class CacheConfig {
    
    @Bean
    public CacheManager cacheManager(RedisConnectionFactory factory) {
        // 默认配置
        RedisCacheConfiguration defaultConfig = RedisCacheConfiguration.defaultCacheConfig()
            .entryTtl(Duration.ofMinutes(30))  // 默认过期时间
            .disableCachingNullValues()        // 不缓存 null 值
            .serializeKeysWith(
                RedisSerializationContext.SerializationPair.fromSerializer(new StringRedisSerializer())
            )
            .serializeValuesWith(
                RedisSerializationContext.SerializationPair.fromSerializer(new GenericJackson2JsonRedisSerializer())
            );
        
        // 特定缓存配置
        Map<String, RedisCacheConfiguration> cacheConfigs = new HashMap<>();
        cacheConfigs.put("users", defaultConfig.entryTtl(Duration.ofHours(1)));
        cacheConfigs.put("products", defaultConfig.entryTtl(Duration.ofMinutes(10)));
        cacheConfigs.put("sessions", defaultConfig.entryTtl(Duration.ofMinutes(30)));
        
        return RedisCacheManager.builder(factory)
            .cacheDefaults(defaultConfig)
            .withInitialCacheConfigurations(cacheConfigs)
            .build();
    }
}
```

### @Cacheable - 查询缓存

```java
@Service
public class UserService {
    
    @Autowired
    private UserRepository userRepository;
    
    /**
     * 查询用户（有缓存则返回缓存，无则查询数据库并缓存）
     */
    @Cacheable(value = "users", key = "#id")
    public User findById(Long id) {
        return userRepository.findById(id).orElse(null);
    }
    
    /**
     * 条件缓存（仅缓存用户名长度大于 2 的）
     */
    @Cacheable(value = "users", key = "#username", condition = "#username.length() > 2")
    public User findByUsername(String username) {
        return userRepository.findByUsername(username);
    }
    
    /**
     * 结果条件（仅缓存非空结果）
     */
    @Cacheable(value = "users", key = "#id", unless = "#result == null")
    public User findByIdNullable(Long id) {
        return userRepository.findById(id).orElse(null);
    }
    
    /**
     * 复杂 Key
     */
    @Cacheable(value = "users", key = "'user:' + #id + ':' + #status")
    public User findByIdAndStatus(Long id, String status) {
        return userRepository.findByIdAndStatus(id, status);
    }
}
```

### @CachePut - 更新缓存

```java
/**
 * 更新用户（同时更新缓存）
 */
@CachePut(value = "users", key = "#user.id")
public User update(User user) {
    return userRepository.save(user);
}
```

### @CacheEvict - 删除缓存

```java
/**
 * 删除用户（同时删除缓存）
 */
@CacheEvict(value = "users", key = "#id")
public void deleteById(Long id) {
    userRepository.deleteById(id);
}

/**
 * 清空所有用户缓存
 */
@CacheEvict(value = "users", allEntries = true)
public void clearCache() {
    // 仅清除缓存
}

/**
 * 在方法执行前删除缓存
 */
@CacheEvict(value = "users", key = "#id", beforeInvocation = true)
public void deleteWithPreEvict(Long id) {
    userRepository.deleteById(id);
}
```

### @Caching - 组合注解

```java
/**
 * 组合多个缓存操作
 */
@Caching(
    cacheable = {
        @Cacheable(value = "users", key = "#id")
    },
    put = {
        @CachePut(value = "users", key = "#result.username")
    },
    evict = {
        @CacheEvict(value = "userList", allEntries = true)
    }
)
public User findAndCacheMultiple(Long id) {
    return userRepository.findById(id).orElse(null);
}
```

### 自定义 KeyGenerator

```java
@Configuration
public class CacheConfig {
    
    @Bean("customKeyGenerator")
    public KeyGenerator keyGenerator() {
        return (target, method, params) -> {
            StringBuilder sb = new StringBuilder();
            sb.append(target.getClass().getSimpleName());
            sb.append(":");
            sb.append(method.getName());
            sb.append(":");
            for (Object param : params) {
                sb.append(param.toString());
                sb.append("_");
            }
            return sb.toString();
        };
    }
}

// 使用
@Cacheable(value = "users", keyGenerator = "customKeyGenerator")
public User findByConditions(Long id, String status) {
    // ...
}
```

## Spring Session Redis

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.session</groupId>
    <artifactId>spring-session-data-redis</artifactId>
</dependency>
```

### 配置

```yaml
spring:
  session:
    store-type: redis
    timeout: 30m
    redis:
      namespace: spring:session
      flush-mode: on_save
```

```java
@Configuration
@EnableRedisHttpSession(maxInactiveIntervalInSeconds = 1800)
public class SessionConfig {
    // Session 过期时间 30 分钟
}
```

### 使用 Session

```java
@RestController
public class SessionController {
    
    @GetMapping("/session/set")
    public String setSession(HttpSession session) {
        session.setAttribute("user", "张三");
        session.setAttribute("loginTime", System.currentTimeMillis());
        return "Session 设置成功";
    }
    
    @GetMapping("/session/get")
    public Map<String, Object> getSession(HttpSession session) {
        Map<String, Object> result = new HashMap<>();
        result.put("user", session.getAttribute("user"));
        result.put("loginTime", session.getAttribute("loginTime"));
        result.put("sessionId", session.getId());
        return result;
    }
    
    @GetMapping("/session/invalidate")
    public String invalidateSession(HttpSession session) {
        session.invalidate();
        return "Session 已销毁";
    }
}
```

### 自定义 Session 序列化

```java
@Configuration
public class SessionConfig {
    
    @Bean
    public RedisSerializer<Object> springSessionDefaultRedisSerializer() {
        return new GenericJackson2JsonRedisSerializer();
    }
}
```

## 消息订阅

### 配置消息监听

```java
@Configuration
public class RedisMessageConfig {
    
    @Bean
    public RedisMessageListenerContainer container(
            RedisConnectionFactory factory,
            MessageListenerAdapter listenerAdapter) {
        
        RedisMessageListenerContainer container = new RedisMessageListenerContainer();
        container.setConnectionFactory(factory);
        
        // 订阅 channel
        container.addMessageListener(listenerAdapter, new ChannelTopic("channel:orders"));
        container.addMessageListener(listenerAdapter, new PatternTopic("channel:*"));
        
        return container;
    }
    
    @Bean
    public MessageListenerAdapter listenerAdapter(RedisMessageReceiver receiver) {
        return new MessageListenerAdapter(receiver, "onMessage");
    }
}
```

### 消息接收器

```java
@Component
public class RedisMessageReceiver {
    
    private static final Logger log = LoggerFactory.getLogger(RedisMessageReceiver.class);
    
    public void onMessage(String message, String channel) {
        log.info("收到消息: channel={}, message={}", channel, message);
        // 处理消息
    }
}
```

### 发送消息

```java
@Service
public class RedisPublisher {
    
    @Autowired
    private StringRedisTemplate template;
    
    public void publish(String channel, String message) {
        template.convertAndSend(channel, message);
    }
}

// 使用
redisPublisher.publish("channel:orders", "新订单: #12345");
```

## 集群与哨兵配置

### 哨兵模式

```yaml
spring:
  redis:
    sentinel:
      master: mymaster
      nodes:
        - 192.168.1.101:26379
        - 192.168.1.102:26379
        - 192.168.1.103:26379
    password: your_password
```

### 集群模式

```yaml
spring:
  redis:
    cluster:
      nodes:
        - 192.168.1.101:7001
        - 192.168.1.102:7002
        - 192.168.1.103:7003
      max-redirects: 3
    password: your_password
```

## 分布式锁

### 使用 RedisTemplate 实现

```java
@Component
public class RedisLock {
    
    @Autowired
    private StringRedisTemplate template;
    
    /**
     * 获取锁
     */
    public boolean lock(String key, String value, long expireSeconds) {
        Boolean success = template.opsForValue().setIfAbsent(
            key, value, expireSeconds, TimeUnit.SECONDS
        );
        return Boolean.TRUE.equals(success);
    }
    
    /**
     * 释放锁（Lua 脚本保证原子性）
     */
    public boolean unlock(String key, String value) {
        String script = 
            "if redis.call('get', KEYS[1]) == ARGV[1] then " +
            "    return redis.call('del', KEYS[1]) " +
            "else " +
            "    return 0 " +
            "end";
        
        Long result = template.execute(
            new DefaultRedisScript<>(script, Long.class),
            Collections.singletonList(key),
            value
        );
        return Long.valueOf(1).equals(result);
    }
}
```

### 使用示例

```java
@Service
public class OrderService {
    
    @Autowired
    private RedisLock redisLock;
    
    public void createOrder(Long productId) {
        String lockKey = "lock:product:" + productId;
        String lockValue = UUID.randomUUID().toString();
        
        if (redisLock.lock(lockKey, lockValue, 30)) {
            try {
                // 业务逻辑
                doCreateOrder(productId);
            } finally {
                redisLock.unlock(lockKey, lockValue);
            }
        } else {
            throw new RuntimeException("获取锁失败");
        }
    }
}
```

## 最佳实践

### 1. 缓存 Key 设计

```java
// 规范的 Key 命名
public class CacheKeyConstants {
    public static final String USER_PREFIX = "user:";
    public static final String ORDER_PREFIX = "order:";
    public static final String PRODUCT_PREFIX = "product:";
    
    public static String userKey(Long id) {
        return USER_PREFIX + id;
    }
    
    public static String orderKey(Long id) {
        return ORDER_PREFIX + id;
    }
}
```

### 2. 防止缓存穿透

```java
@Cacheable(value = "users", key = "#id", unless = "#result == null")
public User findById(Long id) {
    User user = userRepository.findById(id).orElse(null);
    if (user == null) {
        // 缓存空对象，设置较短过期时间
        redisTemplate.opsForValue().set("users:" + id, "null", 60, TimeUnit.SECONDS);
    }
    return user;
}
```

### 3. 防止缓存击穿

```java
public User findByIdWithLock(Long id) {
    String key = "users:" + id;
    User user = (User) redisTemplate.opsForValue().get(key);
    
    if (user == null) {
        String lockKey = "lock:users:" + id;
        if (redisLock.lock(lockKey, "1", 10)) {
            try {
                // 双重检查
                user = (User) redisTemplate.opsForValue().get(key);
                if (user == null) {
                    user = userRepository.findById(id).orElse(null);
                    if (user != null) {
                        redisTemplate.opsForValue().set(key, user, 1, TimeUnit.HOURS);
                    }
                }
            } finally {
                redisLock.unlock(lockKey, "1");
            }
        }
    }
    return user;
}
```

### 4. 健康检查

```java
@Component
public class RedisHealthIndicator implements HealthIndicator {
    
    @Autowired
    private StringRedisTemplate template;
    
    @Override
    public Health health() {
        try {
            String result = template.getConnectionFactory()
                .getConnection().ping();
            if ("PONG".equals(result)) {
                return Health.up()
                    .withDetail("ping", "PONG")
                    .build();
            }
        } catch (Exception e) {
            return Health.down()
                .withException(e)
                .build();
        }
        return Health.down().build();
    }
}
```

## 小结

| 功能 | 关键类/注解 | 使用场景 |
|------|------------|----------|
| 基础操作 | RedisTemplate | 通用 Redis 操作 |
| 缓存 | @Cacheable, @CacheEvict | 方法结果缓存 |
| Session | @EnableRedisHttpSession | 分布式 Session |
| 消息 | RedisMessageListenerContainer | 发布订阅 |
| 分布式锁 | setIfAbsent + Lua | 并发控制 |

**最佳实践**：

- ✅ 使用 JSON 序列化，提高可读性
- ✅ 配置合理的连接池参数
- ✅ 使用缓存注解简化代码
- ✅ 注意缓存穿透、击穿、雪崩问题
- ✅ 实现健康检查和监控
