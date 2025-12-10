---
sidebar_position: 14
title: 性能优化
---

# Java 性能优化

性能优化是提升应用响应速度和吞吐量的关键。本文介绍 Java 应用性能优化的常见技巧和最佳实践。

## 代码层面优化

### 1. 字符串优化

```java
public class StringOptimization {
    // ❌ 不好：频繁字符串拼接
    public String badConcat(String[] words) {
        String result = "";
        for (String word : words) {
            result += word;  // 每次都创建新对象
        }
        return result;
    }
    
    // ✅ 好：使用 StringBuilder
    public String goodConcat(String[] words) {
        StringBuilder sb = new StringBuilder();
        for (String word : words) {
            sb.append(word);
        }
        return sb.toString();
    }
    
    // ✅ 更好：使用 String.join（Java 8+）
    public String betterConcat(String[] words) {
        return String.join("", words);
    }
    
    // 性能对比
    public static void main(String[] args) {
        String[] words = new String[10000];
        for (int i = 0; i < words.length; i++) {
            words[i] = "word" + i;
        }
        
        StringOptimization opt = new StringOptimization();
        
        long start = System.currentTimeMillis();
        opt.badConcat(words);
        System.out.println("String +: " + (System.currentTimeMillis() - start) + "ms");
        
        start = System.currentTimeMillis();
        opt.goodConcat(words);
        System.out.println("StringBuilder: " + (System.currentTimeMillis() - start) + "ms");
        
        start = System.currentTimeMillis();
        opt.betterConcat(words);
        System.out.println("String.join: " + (System.currentTimeMillis() - start) + "ms");
    }
}
```

### 2. 集合优化

```java
import java.util.*;

public class CollectionOptimization {
    // ❌ 不好：未指定初始容量
    public List<String> badList() {
        List<String> list = new ArrayList<>();  // 默认容量10
        for (int i = 0; i < 1000; i++) {
            list.add("item" + i);  // 多次扩容
        }
        return list;
    }
    
    // ✅ 好：指定初始容量
    public List<String> goodList() {
        List<String> list = new ArrayList<>(1000);  // 预分配容量
        for (int i = 0; i < 1000; i++) {
            list.add("item" + i);
        }
        return list;
    }
    
    // ✅ 选择合适的集合类型
    public void rightCollection() {
        // 频繁查询：ArrayList
        List<String> queryList = new ArrayList<>();
        
        // 频繁插入删除：LinkedList
        List<String> modifyList = new LinkedList<>();
        
        // 去重：HashSet
        Set<String> uniqueSet = new HashSet<>();
        
        // 有序去重：TreeSet
        Set<String> sortedSet = new TreeSet<>();
        
        // 键值对：HashMap
        Map<String, Integer> map = new HashMap<>();
    }
    
    // ✅ 使用 EnumSet 和 EnumMap
    enum Color { RED, GREEN, BLUE }
    
    public void enumCollections() {
        // EnumSet 比 HashSet 更高效
        Set<Color> colors = EnumSet.of(Color.RED, Color.BLUE);
        
        // EnumMap 比 HashMap 更高效
        Map<Color, String> colorNames = new EnumMap<>(Color.class);
        colorNames.put(Color.RED, "红色");
    }
}
```

### 3. 循环优化

```java
import java.util.*;

public class LoopOptimization {
    private List<String> list = new ArrayList<>();
    
    // ❌ 不好：每次都计算 size()
    public void badLoop() {
        for (int i = 0; i < list.size(); i++) {
            process(list.get(i));
        }
    }
    
    // ✅ 好：缓存 size()
    public void goodLoop() {
        int size = list.size();
        for (int i = 0; i < size; i++) {
            process(list.get(i));
        }
    }
    
    // ✅ 更好：使用增强 for 循环
    public void betterLoop() {
        for (String item : list) {
            process(item);
        }
    }
    
    // ✅ 最好：使用 forEach（Java 8+）
    public void bestLoop() {
        list.forEach(this::process);
    }
    
    private void process(String item) {
        // 处理逻辑
    }
}
```

### 4. 对象创建优化

```java
public class ObjectCreationOptimization {
    // ❌ 不好：频繁创建对象
    public void badCreation() {
        for (int i = 0; i < 10000; i++) {
            String str = new String("hello");  // 不必要的对象创建
            Integer num = new Integer(100);    // 应该用 valueOf
        }
    }
    
    // ✅ 好：复用对象
    public void goodCreation() {
        String str = "hello";  // 字符串常量池
        for (int i = 0; i < 10000; i++) {
            Integer num = Integer.valueOf(100);  // 使用缓存
        }
    }
    
    // ✅ 对象池模式
    private static final List<StringBuilder> builderPool = new ArrayList<>();
    
    public StringBuilder borrowBuilder() {
        synchronized (builderPool) {
            if (!builderPool.isEmpty()) {
                return builderPool.remove(builderPool.size() - 1);
            }
        }
        return new StringBuilder();
    }
    
    public void returnBuilder(StringBuilder builder) {
        builder.setLength(0);  // 清空
        synchronized (builderPool) {
            builderPool.add(builder);
        }
    }
    
    // ✅ 使用享元模式
    public void flyweightPattern() {
        // Integer 缓存 -128 到 127
        Integer a = 100;
        Integer b = 100;
        System.out.println(a == b);  // true（同一对象）
        
        Integer c = 1000;
        Integer d = 1000;
        System.out.println(c == d);  // false（不同对象）
    }
}
```

### 5. 异常处理优化

```java
public class ExceptionOptimization {
    // ❌ 不好：用异常控制流程
    public int badParse(String str) {
        try {
            return Integer.parseInt(str);
        } catch (NumberFormatException e) {
            return 0;  // 异常开销大
        }
    }
    
    // ✅ 好：提前检查
    public int goodParse(String str) {
        if (str == null || str.isEmpty()) {
            return 0;
        }
        try {
            return Integer.parseInt(str);
        } catch (NumberFormatException e) {
            return 0;
        }
    }
    
    // ✅ 更好：使用正则预检查
    public int betterParse(String str) {
        if (str == null || !str.matches("-?\\d+")) {
            return 0;
        }
        return Integer.parseInt(str);
    }
}
```

## 并发优化

### 1. 线程池优化

```java
import java.util.concurrent.*;

public class ThreadPoolOptimization {
    // ❌ 不好：每次创建线程
    public void badThreading() {
        for (int i = 0; i < 100; i++) {
            new Thread(() -> doWork()).start();
        }
    }
    
    // ✅ 好：使用线程池
    private final ExecutorService executor = Executors.newFixedThreadPool(10);
    
    public void goodThreading() {
        for (int i = 0; i < 100; i++) {
            executor.submit(() -> doWork());
        }
    }
    
    // ✅ 更好：自定义线程池参数
    private final ExecutorService customExecutor = new ThreadPoolExecutor(
        10,                      // 核心线程数
        20,                      // 最大线程数
        60L,                     // 空闲线程存活时间
        TimeUnit.SECONDS,
        new LinkedBlockingQueue<>(100),  // 任务队列
        new ThreadPoolExecutor.CallerRunsPolicy()  // 拒绝策略
    );
    
    // ✅ 针对不同场景选择线程池
    public void chooseRightPool() {
        // CPU密集型：线程数 = CPU核心数 + 1
        int cpuCount = Runtime.getRuntime().availableProcessors();
        ExecutorService cpuPool = Executors.newFixedThreadPool(cpuCount + 1);
        
        // IO密集型：线程数 = CPU核心数 * 2
        ExecutorService ioPool = Executors.newFixedThreadPool(cpuCount * 2);
        
        // 定时任务
        ScheduledExecutorService scheduledPool = 
            Executors.newScheduledThreadPool(5);
    }
    
    private void doWork() {
        // 业务逻辑
    }
}
```

### 2. 锁优化

```java
import java.util.concurrent.locks.*;

public class LockOptimization {
    private int count = 0;
    
    // ❌ 不好：粗粒度锁
    public synchronized void badIncrement() {
        // 大量非线程安全操作
        String temp = "temp";
        int x = 100;
        
        // 只有这里需要同步
        count++;
    }
    
    // ✅ 好：细粒度锁
    public void goodIncrement() {
        // 非线程安全操作
        String temp = "temp";
        int x = 100;
        
        // 只锁必要部分
        synchronized (this) {
            count++;
        }
    }
    
    // ✅ 使用读写锁
    private final ReadWriteLock rwLock = new ReentrantReadWriteLock();
    private Map<String, String> cache = new HashMap<>();
    
    public String read(String key) {
        rwLock.readLock().lock();  // 读锁
        try {
            return cache.get(key);
        } finally {
            rwLock.readLock().unlock();
        }
    }
    
    public void write(String key, String value) {
        rwLock.writeLock().lock();  // 写锁
        try {
            cache.put(key, value);
        } finally {
            rwLock.writeLock().unlock();
        }
    }
    
    // ✅ 使用原子类
    private final AtomicInteger atomicCount = new AtomicInteger(0);
    
    public void atomicIncrement() {
        atomicCount.incrementAndGet();  // 无锁，CAS操作
    }
}
```

### 3. 并发集合

```java
import java.util.concurrent.*;

public class ConcurrentCollectionOptimization {
    // ❌ 不好：同步集合
    private final Map<String, String> syncMap = 
        Collections.synchronizedMap(new HashMap<>());
    
    // ✅ 好：并发集合
    private final ConcurrentHashMap<String, String> concurrentMap = 
        new ConcurrentHashMap<>();
    
    // ✅ 选择合适的并发集合
    public void rightConcurrentCollection() {
        // 并发HashMap
        ConcurrentMap<String, Integer> map = new ConcurrentHashMap<>();
        
        // 并发队列
        Queue<String> queue = new ConcurrentLinkedQueue<>();
        
        // 阻塞队列
        BlockingQueue<String> blockingQueue = new LinkedBlockingQueue<>();
        
        // 跳表实现的有序Map
        ConcurrentNavigableMap<String, Integer> skipListMap = 
            new ConcurrentSkipListMap<>();
    }
}
```

## IO 优化

### 1. 缓冲流

```java
import java.io.*;

public class IOOptimization {
    // ❌ 不好：无缓冲
    public void badIO(String filename) throws IOException {
        try (FileInputStream fis = new FileInputStream(filename)) {
            int data;
            while ((data = fis.read()) != -1) {  // 每次读一个字节
                process(data);
            }
        }
    }
    
    // ✅ 好：使用缓冲流
    public void goodIO(String filename) throws IOException {
        try (BufferedInputStream bis = new BufferedInputStream(
                new FileInputStream(filename))) {
            byte[] buffer = new byte[8192];
            int length;
            while ((length = bis.read(buffer)) != -1) {
                processBuffer(buffer, length);
            }
        }
    }
    
    // ✅ 更好：使用 NIO
    public void nioIO(String filename) throws IOException {
        try (FileChannel channel = FileChannel.open(
                java.nio.file.Paths.get(filename))) {
            java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate(8192);
            while (channel.read(buffer) != -1) {
                buffer.flip();
                processBuffer(buffer);
                buffer.clear();
            }
        }
    }
    
    private void process(int data) {}
    private void processBuffer(byte[] buffer, int length) {}
    private void processBuffer(java.nio.ByteBuffer buffer) {}
}
```

### 2. 内存映射文件

```java
import java.io.*;
import java.nio.*;
import java.nio.channels.*;

public class MemoryMappedFileOptimization {
    public void processLargeFile(String filename) throws IOException {
        try (RandomAccessFile file = new RandomAccessFile(filename, "r");
             FileChannel channel = file.getChannel()) {
            
            // 内存映射文件
            MappedByteBuffer buffer = channel.map(
                FileChannel.MapMode.READ_ONLY,
                0,
                channel.size()
            );
            
            // 直接在内存中操作
            while (buffer.hasRemaining()) {
                byte b = buffer.get();
                // 处理数据
            }
        }
    }
}
```

## 数据库优化

### 1. 连接池

```java
import javax.sql.DataSource;
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;

public class DatabaseOptimization {
    // ✅ 使用连接池
    public DataSource createDataSource() {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:mysql://localhost:3306/mydb");
        config.setUsername("user");
        config.setPassword("password");
        
        // 连接池配置
        config.setMaximumPoolSize(20);
        config.setMinimumIdle(5);
        config.setConnectionTimeout(30000);
        config.setIdleTimeout(600000);
        
        return new HikariDataSource(config);
    }
}
```

### 2. 批量操作

```java
import java.sql.*;

public class BatchOperationOptimization {
    // ❌ 不好：逐条插入
    public void badBatchInsert(Connection conn, List<User> users) 
            throws SQLException {
        String sql = "INSERT INTO users (name, age) VALUES (?, ?)";
        try (PreparedStatement stmt = conn.prepareStatement(sql)) {
            for (User user : users) {
                stmt.setString(1, user.getName());
                stmt.setInt(2, user.getAge());
                stmt.executeUpdate();  // 每次都执行
            }
        }
    }
    
    // ✅ 好：批量插入
    public void goodBatchInsert(Connection conn, List<User> users) 
            throws SQLException {
        String sql = "INSERT INTO users (name, age) VALUES (?, ?)";
        try (PreparedStatement stmt = conn.prepareStatement(sql)) {
            for (User user : users) {
                stmt.setString(1, user.getName());
                stmt.setInt(2, user.getAge());
                stmt.addBatch();  // 添加到批处理
            }
            stmt.executeBatch();  // 一次性执行
        }
    }
    
    static class User {
        private String name;
        private int age;
        public String getName() { return name; }
        public int getAge() { return age; }
    }
}
```

## 缓存优化

### 1. 本地缓存

```java
import java.util.concurrent.*;

public class LocalCacheOptimization {
    // ✅ 使用 Caffeine 缓存
    private final Cache<String, Object> cache = Caffeine.newBuilder()
        .maximumSize(10000)
        .expireAfterWrite(10, TimeUnit.MINUTES)
        .build();
    
    public Object getData(String key) {
        return cache.get(key, k -> loadFromDatabase(k));
    }
    
    private Object loadFromDatabase(String key) {
        // 从数据库加载数据
        return new Object();
    }
}
```

### 2. 缓存策略

```java
public class CacheStrategy {
    // ✅ LRU 缓存
    private final Map<String, Object> lruCache = 
        new LinkedHashMap<String, Object>(100, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry eldest) {
                return size() > 100;
            }
        };
    
    // ✅ 多级缓存
    public Object getWithMultiLevelCache(String key) {
        // 1. 本地缓存
        Object value = localCache.get(key);
        if (value != null) return value;
        
        // 2. Redis 缓存
        value = redisCache.get(key);
        if (value != null) {
            localCache.put(key, value);
            return value;
        }
        
        // 3. 数据库
        value = database.get(key);
        redisCache.put(key, value);
        localCache.put(key, value);
        return value;
    }
}
```

## 性能监控和分析

### 1. 性能计时

```java
public class PerformanceTiming {
    public void measurePerformance() {
        // 简单计时
        long start = System.currentTimeMillis();
        doWork();
        long end = System.currentTimeMillis();
        System.out.println("耗时: " + (end - start) + "ms");
        
        // 纳秒级计时
        long nanoStart = System.nanoTime();
        doWork();
        long nanoEnd = System.nanoTime();
        System.out.println("耗时: " + (nanoEnd - nanoStart) + "ns");
    }
    
    // 使用 StopWatch
    public void stopWatchTiming() {
        org.springframework.util.StopWatch sw = 
            new org.springframework.util.StopWatch();
        
        sw.start("任务1");
        doWork();
        sw.stop();
        
        sw.start("任务2");
        doWork();
        sw.stop();
        
        System.out.println(sw.prettyPrint());
    }
    
    private void doWork() {
        // 业务逻辑
    }
}
```

### 2. 性能分析工具

```java
/**
 * 常用性能分析工具：
 * 
 * 1. JProfiler - 商业工具，功能强大
 * 2. YourKit - 商业工具，易用
 * 3. VisualVM - 免费，JDK自带
 * 4. JMC (Java Mission Control) - Oracle提供
 * 5. Arthas - 阿里开源的诊断工具
 * 
 * 使用示例：
 * // VisualVM
 * jvisualvm
 * 
 * // Arthas
 * java -jar arthas-boot.jar
 * 
 * // JMC
 * jmc
 */
```

## 最佳实践总结

### 代码优化清单

- ✅ 避免不必要的对象创建
- ✅ 使用 StringBuilder 拼接字符串
- ✅ 为集合指定初始容量
- ✅ 选择合适的数据结构
- ✅ 使用基本类型而非包装类
- ✅ 避免在循环中创建对象
- ✅ 及时释放资源（使用 try-with-resources）
- ✅ 使用局部变量而非成员变量

### 并发优化清单

- ✅ 使用线程池而非直接创建线程
- ✅ 选择合适的锁粒度
- ✅ 使用并发集合
- ✅ 考虑使用无锁算法（CAS）
- ✅ 避免锁竞争
- ✅ 减少锁持有时间

### 资源优化清单

- ✅ 使用连接池
- ✅ 使用缓冲流
- ✅ 批量操作数据库
- ✅ 合理使用缓存
- ✅ 避免内存泄漏
- ✅ 及时关闭流和连接

## 总结

性能优化需要：

- ✅ 先测量后优化（不要过早优化）
- ✅ 找到性能瓶颈
- ✅ 使用合适的工具
- ✅ 权衡可读性和性能
- ✅ 持续监控和改进

掌握这些优化技巧可以显著提升 Java 应用的性能。建议结合 [JVM 基础](/docs/java/jvm-basics) 和实际业务场景进行优化。
