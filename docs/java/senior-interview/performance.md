---
sidebar_position: 4
title: 性能调优
---

# 🎯 性能调优（专家级）

## 11. 如何排查线上 CPU 飙高问题？

**答案要点：**

**排查步骤：**

```bash
# 1. 找到 CPU 占用最高的 Java 进程
top -c

# 2. 找到进程中 CPU 占用最高的线程
top -Hp <pid>

# 3. 将线程 ID 转为 16 进制
printf "%x\n" <tid>

# 4. 导出线程堆栈
jstack <pid> > thread_dump.txt

# 5. 在堆栈中搜索对应线程
grep -A 30 "nid=0x<hex_tid>" thread_dump.txt
```

**使用 Arthas 快速定位：**

```bash
# 启动 Arthas
java -jar arthas-boot.jar

# 查看最繁忙的线程
thread -n 3

# 查看特定线程堆栈
thread <tid>

# 实时监控方法执行
watch com.example.Service method "{params, returnObj}" -x 3
```

**常见 CPU 飙高原因：**

```java
// 1. 死循环
while (true) {
    // 没有 sleep 或阻塞操作
}

// 2. 频繁 GC
// 检查 GC 日志，可能是内存泄漏导致

// 3. 正则表达式回溯
String regex = "(a+)+b";  // 灾难性回溯
"aaaaaaaaaaaaaaaaaaaaac".matches(regex);

// 4. 序列化/反序列化
// 大对象频繁序列化
```

**延伸：** 参考 [性能优化 - 问题排查](/docs/java/performance)

---

## 12. 如何排查内存泄漏问题？

**答案要点：**

**内存泄漏排查步骤：**

```bash
# 1. 查看堆内存使用情况
jmap -heap <pid>

# 2. 导出堆转储文件
jmap -dump:format=b,file=heap.hprof <pid>

# 3. 使用 MAT 或 VisualVM 分析
# 重点关注：
# - Dominator Tree（支配树）
# - Leak Suspects（泄漏嫌疑）
# - Histogram（对象直方图）
```

**使用 Arthas 在线分析：**

```bash
# 查看堆内存概况
memory

# 查看对象实例数量
heapdump --live /tmp/heap.hprof

# 查看类加载信息
classloader -l

# 搜索类实例
vmtool --action getInstances --className java.util.HashMap --limit 10
```

**常见内存泄漏场景：**

```java
// 1. 静态集合持有对象引用
public class Cache {
    private static Map<String, Object> cache = new HashMap<>();
    
    public void add(String key, Object value) {
        cache.put(key, value);  // 永远不会被 GC
    }
}

// 2. 未关闭的资源
public void readFile() {
    InputStream is = new FileInputStream("file.txt");
    // 忘记关闭，导致资源泄漏
}

// 3. 监听器未注销
public class EventManager {
    private List<EventListener> listeners = new ArrayList<>();
    
    public void addListener(EventListener listener) {
        listeners.add(listener);
    }
    // 缺少 removeListener 方法
}

// 4. ThreadLocal 未清理
private static ThreadLocal<User> userHolder = new ThreadLocal<>();

public void process() {
    userHolder.set(new User());
    // 线程池场景下，线程复用导致 ThreadLocal 不会被清理
    // 应该在 finally 中调用 userHolder.remove()
}
```

**延伸：** 参考 [性能优化 - 内存优化](/docs/java/performance)

---

## 13. Arthas 有哪些常用命令？如何使用？

**答案要点：**

**Arthas 核心命令：**

| 命令 | 功能 | 示例 |
|------|------|------|
| `dashboard` | 系统实时面板 | `dashboard` |
| `thread` | 线程信息 | `thread -n 3` |
| `jvm` | JVM 信息 | `jvm` |
| `memory` | 内存信息 | `memory` |
| `watch` | 方法监控 | `watch class method "{params}"` |
| `trace` | 方法调用链路 | `trace class method` |
| `stack` | 方法调用栈 | `stack class method` |
| `tt` | 时间隧道 | `tt -t class method` |
| `profiler` | 火焰图 | `profiler start` |

**实战示例：**

```bash
# 1. 查看方法入参和返回值
watch com.example.UserService getUser "{params, returnObj}" -x 3

# 2. 追踪方法调用耗时
trace com.example.UserService getUser '#cost > 100'

# 3. 查看方法调用栈
stack com.example.UserService getUser

# 4. 时间隧道 - 记录方法调用
tt -t com.example.UserService getUser
tt -i 1001  # 查看第1001次调用
tt -i 1001 -p  # 重放调用

# 5. 生成火焰图
profiler start
# 等待一段时间
profiler stop --format html --file /tmp/flame.html

# 6. 反编译类
jad com.example.UserService

# 7. 热更新代码
redefine /tmp/UserService.class
```

**延伸：** 参考 [性能优化 - 监控工具](/docs/java/performance)

---

## 14. 如何优化数据库查询性能？

**答案要点：**

**SQL 优化原则：**

```sql
-- 1. 避免 SELECT *
SELECT id, name, age FROM users WHERE id = 1;

-- 2. 使用覆盖索引
CREATE INDEX idx_name_age ON users(name, age);
SELECT name, age FROM users WHERE name = 'Tom';  -- 不需要回表

-- 3. 避免索引失效
-- 错误：函数操作导致索引失效
SELECT * FROM users WHERE YEAR(create_time) = 2024;
-- 正确：范围查询
SELECT * FROM users WHERE create_time >= '2024-01-01' AND create_time < '2025-01-01';

-- 4. 避免 OR 导致索引失效
-- 错误
SELECT * FROM users WHERE name = 'Tom' OR age = 20;
-- 正确：使用 UNION
SELECT * FROM users WHERE name = 'Tom'
UNION
SELECT * FROM users WHERE age = 20;

-- 5. 分页优化
-- 错误：深分页性能差
SELECT * FROM users LIMIT 1000000, 10;
-- 正确：使用游标分页
SELECT * FROM users WHERE id > 1000000 LIMIT 10;
```

**连接池配置优化：**

```yaml
# HikariCP 配置
spring:
  datasource:
    hikari:
      minimum-idle: 10
      maximum-pool-size: 50
      idle-timeout: 600000
      max-lifetime: 1800000
      connection-timeout: 30000
      connection-test-query: SELECT 1
```

**慢查询分析：**

```sql
-- 开启慢查询日志
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 1;

-- 使用 EXPLAIN 分析
EXPLAIN SELECT * FROM users WHERE name = 'Tom';

-- 关注字段：
-- type: 访问类型（ALL < index < range < ref < eq_ref < const）
-- key: 使用的索引
-- rows: 扫描行数
-- Extra: 额外信息（Using filesort, Using temporary 需要优化）
```

**延伸：** 参考 [MySQL 性能优化](/docs/mysql/performance-optimization)

---

## 15. 缓存穿透、缓存击穿、缓存雪崩如何解决？

**答案要点：**

| 问题 | 描述 | 解决方案 |
|------|------|---------|
| **缓存穿透** | 查询不存在的数据 | 布隆过滤器、空值缓存 |
| **缓存击穿** | 热点 key 过期 | 互斥锁、永不过期 |
| **缓存雪崩** | 大量 key 同时过期 | 随机过期时间、多级缓存 |

**缓存穿透解决方案：**

```java
// 方案1：布隆过滤器
public class BloomFilterDemo {
    private BloomFilter<String> bloomFilter = BloomFilter.create(
        Funnels.stringFunnel(Charset.defaultCharset()),
        1000000,  // 预期元素数量
        0.01      // 误判率
    );
    
    public User getUser(String id) {
        // 先检查布隆过滤器
        if (!bloomFilter.mightContain(id)) {
            return null;  // 一定不存在
        }
        // 查缓存和数据库
        return getUserFromCacheOrDB(id);
    }
}

// 方案2：空值缓存
public User getUser(String id) {
    String cacheKey = "user:" + id;
    User user = cache.get(cacheKey);
    
    if (user == null) {
        user = db.getUser(id);
        if (user == null) {
            // 缓存空值，设置较短过期时间
            cache.set(cacheKey, NULL_USER, 60);
        } else {
            cache.set(cacheKey, user, 3600);
        }
    }
    return user == NULL_USER ? null : user;
}
```

**缓存击穿解决方案：**

```java
// 方案：互斥锁
public User getUser(String id) {
    String cacheKey = "user:" + id;
    User user = cache.get(cacheKey);
    
    if (user == null) {
        String lockKey = "lock:user:" + id;
        // 尝试获取分布式锁
        if (redis.setnx(lockKey, "1", 10)) {
            try {
                // 双重检查
                user = cache.get(cacheKey);
                if (user == null) {
                    user = db.getUser(id);
                    cache.set(cacheKey, user, 3600);
                }
            } finally {
                redis.del(lockKey);
            }
        } else {
            // 等待后重试
            Thread.sleep(100);
            return getUser(id);
        }
    }
    return user;
}
```

**缓存雪崩解决方案：**

```java
// 方案1：随机过期时间
public void setCache(String key, Object value) {
    int baseExpire = 3600;
    int randomExpire = new Random().nextInt(600);  // 0-600秒随机
    cache.set(key, value, baseExpire + randomExpire);
}

// 方案2：多级缓存
public User getUser(String id) {
    // L1: 本地缓存（Caffeine）
    User user = localCache.get(id);
    if (user != null) return user;
    
    // L2: 分布式缓存（Redis）
    user = redisCache.get(id);
    if (user != null) {
        localCache.put(id, user);
        return user;
    }
    
    // L3: 数据库
    user = db.getUser(id);
    redisCache.set(id, user);
    localCache.put(id, user);
    return user;
}
```

**延伸：** 参考 [Redis 缓存策略](/docs/redis/cache-strategies)
