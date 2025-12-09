---
sidebar_position: 17
---

# 性能优化

> [!TIP]
> **性能优化四大方向**: JVM调优、数据库优化、缓存策略、异步处理。先测量再优化,避免过早优化。

## JVM 优化

### 内存配置

```bash
# 基础配置
-Xms1024m          # 初始堆内存
-Xmx2048m          # 最大堆内存
-XX:NewRatio=2     # 新生代与老年代比例

# 推荐配置
-Xms512m \
-Xmx2048m \
-XX:+UseG1GC \
-XX:MaxGCPauseMillis=200 \
-XX:+PrintGCDetails \
-XX:+PrintGCDateStamps \
-Xloggc:gc-%t.log
```

### 垃圾回收优化

```bash
# G1 GC（推荐用于中大型应用）
-XX:+UseG1GC \
-XX:MaxGCPauseMillis=200 \
-XX:InitiatingHeapOccupancyPercent=35 \
-XX:G1NewCollectionHeuristicWeight=20

# ZGC（适合超大堆）
-XX:+UseZGC \
-XX:ConcGCThreads=4 \
-XX:-ZUncommit
```

### 类加载优化

```bash
# 应用类加载缓存
-XX:+UnlockDiagnosticVMOptions \
-XX:+TraceClassLoading \
-XX:TieredStopAtLevel=4 \
-XX:+TieredCompilation
```

## 数据库优化

### 连接池配置

```yaml
spring:
  datasource:
    hikari:
      # 连接池大小
      maximum-pool-size: 20
      minimum-idle: 5
      
      # 超时配置
      connection-timeout: 30000
      idle-timeout: 600000
      max-lifetime: 1800000
      
      # 性能优化
      auto-commit: true
      cache-prep-stmts: true
      prep-stmt-cache-size: 250
      prep-stmt-cache-sql-limit: 2048
      
      # 连接验证
      connection-test-query: "SELECT 1"
      leak-detection-threshold: 60000
```

### 查询优化

```java
@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    
    // ❌ N+1 查询问题
    @Query("SELECT u FROM User u WHERE u.status = :status")
    List<User> findByStatus(@Param("status") String status);
    // 每个 User 都会触发一次查询获取关联的 Posts
    
    // ✅ 使用 JOIN FETCH 解决
    @Query("SELECT DISTINCT u FROM User u " +
           "LEFT JOIN FETCH u.posts " +
           "WHERE u.status = :status")
    List<User> findByStatusWithPosts(@Param("status") String status);
    
    // ✅ 使用投影只查询需要的字段
    @Query("SELECT new com.example.dto.UserDTO(u.id, u.username, u.email) " +
           "FROM User u WHERE u.status = :status")
    List<UserDTO> findByStatusProjection(@Param("status") String status);
}
```

### 批量操作

```java
@Service
@Transactional
public class UserService {
    
    @Autowired
    private UserRepository userRepository;
    
    // ❌ 低效：逐个保存
    public void saveUsers(List<User> users) {
        for (User user : users) {
            userRepository.save(user);
        }
    }
    
    // ✅ 高效：批量保存
    public void saveUsersBatch(List<User> users) {
        userRepository.saveAll(users);
    }
    
    // ✅ 超大批量：分批处理
    public void saveUsersLargeBatch(List<User> users) {
        int batchSize = 100;
        for (int i = 0; i < users.size(); i += batchSize) {
            int end = Math.min(i + batchSize, users.size());
            userRepository.saveAll(users.subList(i, end));
            entityManager.flush();
            entityManager.clear();
        }
    }
}
```

### JPA 配置优化

```yaml
spring:
  jpa:
    hibernate:
      jdbc:
        batch_size: 20           # 批处理大小
        fetch_size: 50           # 一次获取的行数
        use_scrollable_resultset: true
    properties:
      hibernate:
        # 二级缓存
        cache:
          use_second_level_cache: true
          region:
            factory_class: org.hibernate.cache.jcache.JCacheRegionFactory
        
        # 查询优化
        jdbc:
          batch_versioned_data: true
```

## 缓存优化

### 多层缓存策略

```java
@Service
@RequiredArgsConstructor
@Slf4j
public class UserService {
    
    private final UserRepository userRepository;
    private final RedisTemplate<String, Object> redisTemplate;
    private final ConcurrentHashMap<String, User> localCache;
    
    public User getUser(Long id) {
        String key = "user:" + id;
        
        // 第一层：本地缓存
        User user = localCache.get(key);
        if (user != null) {
            log.debug("本地缓存命中");
            return user;
        }
        
        // 第二层：Redis 缓存
        user = (User) redisTemplate.opsForValue().get(key);
        if (user != null) {
            log.debug("Redis 缓存命中");
            localCache.put(key, user);
            return user;
        }
        
        // 第三层：数据库
        user = userRepository.findById(id).orElse(null);
        if (user != null) {
            // 存入 Redis 缓存
            redisTemplate.opsForValue().set(key, user, 24, TimeUnit.HOURS);
            // 存入本地缓存
            localCache.put(key, user);
        }
        
        return user;
    }
}
```

### 缓存预热

```java
@Component
@Slf4j
public class CacheWarmer {
    
    @Autowired
    private UserService userService;
    
    @EventListener(ApplicationReadyEvent.class)
    public void warmCache() {
        log.info("开始缓存预热");
        
        // 预热热点数据
        List<User> hotUsers = userService.getHotUsers();
        hotUsers.forEach(user -> {
            userService.cache(user);
        });
        
        log.info("缓存预热完成");
    }
}
```

## 异步处理优化

### 异步任务池配置

```java
@Configuration
public class AsyncConfig implements AsyncConfigurer {
    
    @Override
    public Executor getAsyncExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        
        // 根据 CPU 核心数配置
        int cpuCount = Runtime.getRuntime().availableProcessors();
        executor.setCorePoolSize(cpuCount);
        executor.setMaxPoolSize(cpuCount * 2);
        executor.setQueueCapacity(cpuCount * 100);
        
        executor.setThreadNamePrefix("async-");
        executor.setWaitForTasksToCompleteOnShutdown(true);
        executor.setAwaitTerminationSeconds(60);
        executor.setRejectedExecutionHandler(
            new ThreadPoolTaskExecutor.CallerRunsPolicy()
        );
        
        executor.initialize();
        return executor;
    }
}
```

### 异步 I/O 操作

```java
@Service
public class FileUploadService {
    
    @Async
    public CompletableFuture<String> uploadFile(MultipartFile file) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                // 非阻塞文件上传
                String filePath = saveFile(file);
                return filePath;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }
    
    @Async
    public CompletableFuture<String> sendEmail(String to, String subject, String body) {
        return CompletableFuture.supplyAsync(() -> {
            // 异步发送邮件，不阻塞主线程
            mailService.send(to, subject, body);
            return "Email sent";
        });
    }
}
```

## SQL 查询优化

### 索引优化

```sql
-- 查看执行计划
EXPLAIN SELECT u.* FROM users u 
WHERE u.status = 'ACTIVE' AND u.age > 20;

-- 创建复合索引
CREATE INDEX idx_status_age ON users(status, age);

-- 查看索引使用情况
SELECT * FROM performance_schema.table_io_waits_summary_by_table;
```

### 慢查询日志

```yaml
spring:
  datasource:
    url: "jdbc:mysql://localhost:3306/mydb?enableQueryTimeoutKillsConnection=true&slowQueryThresholdMs=1000&logSlowQueries=true"
```

### 查询分析示例

```java
// ❌ 低效的查询
@Query("SELECT u FROM User u LEFT JOIN u.posts p WHERE u.status = 'ACTIVE'")
List<User> findActiveUsers();

// ✅ 高效的查询
@Query("SELECT u FROM User u WHERE u.status = 'ACTIVE'")
List<User> findActiveUsers();

// 如果确实需要关联：
@Query("SELECT u FROM User u LEFT JOIN FETCH u.posts WHERE u.status = 'ACTIVE'")
List<User> findActiveUsersWithPosts();
```

## Web 层优化

### HTTP 缓存

```java
@RestController
@RequestMapping("/api/users")
public class UserController {
    
    @GetMapping("/{id}")
    public ResponseEntity<UserDTO> getUser(@PathVariable Long id) {
        UserDTO user = userService.getUser(id);
        
        return ResponseEntity.ok()
            .cacheControl(CacheControl.maxAge(1, TimeUnit.HOURS).cachePublic())
            .eTag("\"" + user.hashCode() + "\"")
            .body(user);
    }
}
```

### 分页优化

```java
@RestController
@RequestMapping("/api/users")
public class UserController {
    
    // ❌ 获取所有用户（内存溢出风险）
    @GetMapping
    public List<User> getAllUsers() {
        return userRepository.findAll();
    }
    
    // ✅ 使用分页
    @GetMapping
    public Page<User> getUsers(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        return userRepository.findAll(PageRequest.of(page, size));
    }
}
```

### 响应压缩

```yaml
server:
  compression:
    enabled: true
    min-response-size: 1024
    mime-types:
      - application/json
      - application/xml
      - text/html
      - text/xml
      - text/plain
```

## 监控和分析

### 性能指标收集

```java
@Service
public class PerformanceMonitor {
    
    private final MeterRegistry meterRegistry;
    
    @Around("@annotation(com.example.annotation.Monitor)")
    public Object monitor(ProceedingJoinPoint joinPoint) throws Throwable {
        String methodName = joinPoint.getSignature().getName();
        
        Timer.Sample sample = Timer.start(meterRegistry);
        
        try {
            return joinPoint.proceed();
        } finally {
            sample.stop(Timer.builder("method.execution")
                .tag("method", methodName)
                .register(meterRegistry));
        }
    }
}
```

### 内存泄漏检测

```bash
# 生成堆转储
jmap -dump:live,format=b,file=heap.bin <PID>

# 分析堆文件
jhat heap.bin

# 或使用 Eclipse MAT 分析
```

## 配置优化总结

### 生产环境配置模板

```yaml
spring:
  datasource:
    hikari:
      maximum-pool-size: 20
      minimum-idle: 5
      connection-timeout: 30000
      idle-timeout: 600000
      max-lifetime: 1800000
  
  jpa:
    hibernate:
      jdbc:
        batch_size: 50
        fetch_size: 100
    properties:
      hibernate:
        jdbc:
          use_scrollable_resultset: true
  
  redis:
    jedis:
      pool:
        max-active: 16
        max-idle: 8
        min-idle: 0

server:
  compression:
    enabled: true
    min-response-size: 1024
  tomcat:
    max-threads: 200
    min-spare-threads: 10
    max-connections: 10000
    accept-count: 100

logging:
  level:
    root: WARN
  file:
    name: /var/log/app.log
    max-size: 10MB
```

## 性能优化检查清单

- ✅ JVM 参数优化
- ✅ 数据库连接池配置
- ✅ SQL 查询优化
- ✅ 缓存策略实现
- ✅ 异步任务处理
- ✅ HTTP 缓存配置
- ✅ 分页查询实现
- ✅ 监控指标收集
- ✅ 定期性能测试
- ✅ 日志级别优化

---

**提示**：性能优化是一个持续的过程，需要不断地监控、分析和改进！
