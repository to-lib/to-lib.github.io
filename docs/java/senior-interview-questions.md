---
sidebar_position: 101
title: Java 高级面试题精选
---

# Java 高级面试题精选

> [!TIP]
> 本文精选了 30+ 道 Java 高级开发工程师面试题，涵盖 JVM 深度、高级并发、性能调优、架构设计、框架源码等核心主题。适合 3-5 年以上经验的开发者面试准备。

## 目录

- [🎯 JVM 深度（高级）](#-jvm-深度高级)
- [🎯 高级并发编程（高级）](#-高级并发编程高级)
- [🎯 性能调优（专家级）](#-性能调优专家级)
- [🎯 架构设计（专家级）](#-架构设计专家级)
- [🎯 框架源码分析（专家级）](#-框架源码分析专家级)
- [📌 总结与学习建议](#-总结与学习建议)

---

## 🎯 JVM 深度（高级）

### 1. 详细描述 JVM 内存模型，各区域的作用和特点？

**答案要点：**

**JDK 8+ 运行时数据区域：**

| 区域 | 线程共享 | 作用 | 异常 |
|------|---------|------|------|
| **堆（Heap）** | 共享 | 存储对象实例和数组 | OutOfMemoryError |
| **方法区/元空间** | 共享 | 存储类信息、常量、静态变量 | OutOfMemoryError |
| **虚拟机栈** | 私有 | 存储局部变量、操作数栈、方法出口 | StackOverflowError/OOM |
| **本地方法栈** | 私有 | 为 native 方法服务 | StackOverflowError/OOM |
| **程序计数器** | 私有 | 记录当前执行的字节码指令地址 | 无 |

**堆内存分代结构：**

```
堆内存
├── 新生代（Young Generation）- 1/3 堆
│   ├── Eden 区 - 8/10 新生代
│   ├── Survivor From (S0) - 1/10 新生代
│   └── Survivor To (S1) - 1/10 新生代
└── 老年代（Old Generation）- 2/3 堆
```

**代码示例 - 查看内存分配：**

```java
public class MemoryDemo {
    public static void main(String[] args) {
        // 堆内存信息
        Runtime runtime = Runtime.getRuntime();
        long maxMemory = runtime.maxMemory();      // 最大堆内存
        long totalMemory = runtime.totalMemory();  // 当前堆内存
        long freeMemory = runtime.freeMemory();    // 空闲堆内存
        
        System.out.println("Max: " + maxMemory / 1024 / 1024 + "MB");
        System.out.println("Total: " + totalMemory / 1024 / 1024 + "MB");
        System.out.println("Free: " + freeMemory / 1024 / 1024 + "MB");
    }
}
```

**JVM 参数配置：**

```bash
# 堆内存配置
-Xms512m          # 初始堆大小
-Xmx1024m         # 最大堆大小
-Xmn256m          # 新生代大小

# 元空间配置（JDK 8+）
-XX:MetaspaceSize=128m
-XX:MaxMetaspaceSize=256m

# 栈大小配置
-Xss256k          # 每个线程栈大小
```

**延伸：** 参考 [JVM 基础 - 内存模型](/docs/java/jvm-basics#内存模型)

---

### 2. 对比 CMS、G1、ZGC 垃圾回收器的特点和适用场景？

**答案要点：**

| 特性 | CMS | G1 | ZGC |
|------|-----|----|----|
| **算法** | 标记-清除 | 标记-整理 | 染色指针+读屏障 |
| **停顿时间** | 不可预测 | 可预测（-XX:MaxGCPauseMillis） | <10ms |
| **内存碎片** | 有 | 无 | 无 |
| **堆大小** | <32GB | 4GB-64GB | 8MB-16TB |
| **JDK版本** | JDK 5+ | JDK 7+ | JDK 11+ |
| **适用场景** | 低延迟、中小堆 | 大堆、可控停顿 | 超大堆、极低延迟 |

**G1 收集器工作原理：**

```
G1 堆内存布局（Region 化）
┌─────┬─────┬─────┬─────┬─────┬─────┐
│ Eden│ Eden│ Sur │ Old │ Old │ Hum │
├─────┼─────┼─────┼─────┼─────┼─────┤
│ Old │ Free│ Eden│ Old │ Sur │ Old │
└─────┴─────┴─────┴─────┴─────┴─────┘
每个 Region 大小：1MB-32MB（2的幂次）
```

**GC 选择建议：**

```bash
# CMS（JDK 8 默认可用，JDK 14 移除）
-XX:+UseConcMarkSweepGC

# G1（JDK 9+ 默认）
-XX:+UseG1GC
-XX:MaxGCPauseMillis=200

# ZGC（JDK 15+ 生产可用）
-XX:+UseZGC
-XX:+ZGenerational  # JDK 21+ 分代 ZGC
```

**延伸：** 参考 [JVM 基础 - 垃圾回收](/docs/java/jvm-basics#垃圾回收gc)

---

### 3. 如何进行 GC 调优？请描述一个实际的调优案例

**答案要点：**

**GC 调优步骤：**

1. **收集 GC 日志**
2. **分析 GC 行为**
3. **调整 JVM 参数**
4. **验证优化效果**

**开启 GC 日志：**

```bash
# JDK 8
-XX:+PrintGCDetails
-XX:+PrintGCDateStamps
-Xloggc:/path/to/gc.log

# JDK 9+
-Xlog:gc*:file=/path/to/gc.log:time,uptime,level,tags
```

**实际调优案例 - Full GC 频繁：**

```bash
# 问题现象：每隔几分钟发生 Full GC，停顿 2-3 秒

# 原始配置
-Xms2g -Xmx2g -Xmn512m

# 分析发现：
# 1. 新生代太小，对象过早晋升到老年代
# 2. 老年代很快被填满，触发 Full GC

# 优化后配置
-Xms4g -Xmx4g -Xmn1536m
-XX:SurvivorRatio=8
-XX:MaxTenuringThreshold=15
-XX:+UseG1GC
-XX:MaxGCPauseMillis=200
```

**GC 日志分析关键指标：**

```
[GC (Allocation Failure) [PSYoungGen: 524288K->87654K(611840K)] 
 524288K->87654K(2010112K), 0.0876543 secs]
 
关键指标：
- GC 原因：Allocation Failure
- Young GC 前后：524288K -> 87654K
- 堆总量变化：524288K -> 87654K
- GC 耗时：0.0876543 秒
```

**延伸：** 参考 [JVM 基础 - 性能调优](/docs/java/jvm-basics)

---

### 4. 解释类加载机制和双亲委派模型，如何打破双亲委派？

**答案要点：**

**类加载过程：**

```
加载 → 验证 → 准备 → 解析 → 初始化 → 使用 → 卸载
```

**双亲委派模型：**

```
                    ┌─────────────────┐
                    │ Bootstrap       │ 加载 rt.jar
                    │ ClassLoader     │ (C++ 实现)
                    └────────┬────────┘
                             │ 委派
                    ┌────────▼────────┐
                    │ Extension       │ 加载 ext/*.jar
                    │ ClassLoader     │
                    └────────┬────────┘
                             │ 委派
                    ┌────────▼────────┐
                    │ Application     │ 加载 classpath
                    │ ClassLoader     │
                    └────────┬────────┘
                             │ 委派
                    ┌────────▼────────┐
                    │ Custom          │ 自定义加载
                    │ ClassLoader     │
                    └─────────────────┘
```

**打破双亲委派的场景：**

1. **SPI 机制**（JDBC、JNDI）
2. **热部署**（Tomcat、OSGi）
3. **代码隔离**（不同版本类库）

**自定义类加载器示例：**

```java
public class HotSwapClassLoader extends ClassLoader {
    
    @Override
    protected Class<?> loadClass(String name, boolean resolve) 
            throws ClassNotFoundException {
        // 打破双亲委派：先尝试自己加载
        if (name.startsWith("com.myapp.")) {
            return findClass(name);
        }
        // 其他类仍走双亲委派
        return super.loadClass(name, resolve);
    }
    
    @Override
    protected Class<?> findClass(String name) throws ClassNotFoundException {
        byte[] classData = loadClassData(name);
        if (classData == null) {
            throw new ClassNotFoundException(name);
        }
        return defineClass(name, classData, 0, classData.length);
    }
    
    private byte[] loadClassData(String name) {
        // 从文件/网络加载类字节码
        String path = name.replace('.', '/') + ".class";
        try (InputStream is = new FileInputStream(path)) {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buffer = new byte[1024];
            int len;
            while ((len = is.read(buffer)) != -1) {
                baos.write(buffer, 0, len);
            }
            return baos.toByteArray();
        } catch (IOException e) {
            return null;
        }
    }
}
```

**延伸：** 参考 [JVM 基础 - 类加载机制](/docs/java/jvm-basics#类加载机制)

---

### 5. JIT 编译器有哪些优化技术？什么是逃逸分析？

**答案要点：**

**JIT 主要优化技术：**

| 优化技术 | 说明 | 效果 |
|---------|------|------|
| **方法内联** | 将小方法代码直接嵌入调用处 | 减少方法调用开销 |
| **逃逸分析** | 分析对象作用域 | 栈上分配、锁消除 |
| **锁消除** | 消除不必要的同步 | 提升并发性能 |
| **锁粗化** | 合并连续的加锁操作 | 减少锁开销 |
| **标量替换** | 将对象拆解为基本类型 | 减少内存分配 |

**逃逸分析详解：**

```java
// 不逃逸 - 可以栈上分配
public void noEscape() {
    Point p = new Point(1, 2);  // 对象只在方法内使用
    System.out.println(p.x + p.y);
}

// 方法逃逸 - 不能栈上分配
public Point methodEscape() {
    Point p = new Point(1, 2);
    return p;  // 对象被返回，逃逸到方法外
}

// 线程逃逸 - 不能栈上分配
public void threadEscape() {
    Point p = new Point(1, 2);
    new Thread(() -> System.out.println(p)).start();  // 被其他线程访问
}
```

**锁消除示例：**

```java
// JIT 会消除这个同步，因为 sb 不会逃逸
public String concat(String s1, String s2) {
    StringBuffer sb = new StringBuffer();  // 局部变量，不逃逸
    sb.append(s1);
    sb.append(s2);
    return sb.toString();
}
// 优化后等价于使用 StringBuilder（无同步）
```

**JIT 相关 JVM 参数：**

```bash
# 开启/关闭逃逸分析
-XX:+DoEscapeAnalysis    # 默认开启
-XX:-DoEscapeAnalysis    # 关闭

# 开启/关闭锁消除
-XX:+EliminateLocks      # 默认开启

# 开启/关闭标量替换
-XX:+EliminateAllocations

# 查看 JIT 编译日志
-XX:+PrintCompilation
-XX:+UnlockDiagnosticVMOptions
-XX:+PrintInlining
```

**延伸：** 参考 [JVM 基础 - JIT 编译](/docs/java/jvm-basics)

---

## 🎯 高级并发编程（高级）

### 6. 详细解释 AQS（AbstractQueuedSynchronizer）的原理

**答案要点：**

**AQS 核心结构：**

```java
public abstract class AbstractQueuedSynchronizer {
    // 同步状态
    private volatile int state;
    
    // CLH 队列头尾节点
    private transient volatile Node head;
    private transient volatile Node tail;
    
    // 内部节点类
    static final class Node {
        volatile int waitStatus;
        volatile Node prev;
        volatile Node next;
        volatile Thread thread;
    }
}
```

**AQS 工作原理图：**

```
获取锁失败的线程进入 CLH 队列等待

     head                                    tail
       │                                       │
       ▼                                       ▼
    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
    │ Node │◄──►│ Node │◄──►│ Node │◄──►│ Node │
    │(持锁)│    │(等待)│    │(等待)│    │(等待)│
    └──────┘    └──────┘    └──────┘    └──────┘
```

**ReentrantLock 获取锁流程：**

```java
// 非公平锁获取
final boolean nonfairTryAcquire(int acquires) {
    final Thread current = Thread.currentThread();
    int c = getState();
    if (c == 0) {
        // 状态为0，CAS尝试获取锁
        if (compareAndSetState(0, acquires)) {
            setExclusiveOwnerThread(current);
            return true;
        }
    }
    else if (current == getExclusiveOwnerThread()) {
        // 重入：当前线程已持有锁
        int nextc = c + acquires;
        setState(nextc);
        return true;
    }
    return false;
}
```

**基于 AQS 实现自定义同步器：**

```java
public class SimpleLock {
    private final Sync sync = new Sync();
    
    private static class Sync extends AbstractQueuedSynchronizer {
        @Override
        protected boolean tryAcquire(int arg) {
            if (compareAndSetState(0, 1)) {
                setExclusiveOwnerThread(Thread.currentThread());
                return true;
            }
            return false;
        }
        
        @Override
        protected boolean tryRelease(int arg) {
            setExclusiveOwnerThread(null);
            setState(0);
            return true;
        }
        
        @Override
        protected boolean isHeldExclusively() {
            return getState() == 1;
        }
    }
    
    public void lock() { sync.acquire(1); }
    public void unlock() { sync.release(1); }
}
```

**延伸：** 参考 [多线程 - JUC 工具类](/docs/java/multithreading)

---

### 7. 线程池核心参数如何配置？如何监控线程池状态？

**答案要点：**

**线程池参数配置原则：**

| 场景 | corePoolSize | maximumPoolSize | 队列 |
|------|-------------|-----------------|------|
| **CPU 密集型** | CPU 核心数 | CPU 核心数 | 小队列 |
| **IO 密集型** | 2 * CPU 核心数 | 2 * CPU 核心数 | 大队列 |
| **混合型** | 根据 IO/CPU 比例调整 | - | - |

**线程池参数计算公式：**

```
线程数 = CPU 核心数 * (1 + 等待时间/计算时间)

例如：8核CPU，IO等待时间是计算时间的2倍
线程数 = 8 * (1 + 2) = 24
```

**生产环境线程池配置示例：**

```java
@Configuration
public class ThreadPoolConfig {
    
    @Bean("businessThreadPool")
    public ThreadPoolExecutor businessThreadPool() {
        int coreSize = Runtime.getRuntime().availableProcessors();
        
        return new ThreadPoolExecutor(
            coreSize,                              // 核心线程数
            coreSize * 2,                          // 最大线程数
            60L, TimeUnit.SECONDS,                 // 空闲线程存活时间
            new LinkedBlockingQueue<>(1000),       // 任务队列
            new ThreadFactoryBuilder()
                .setNameFormat("business-pool-%d")
                .setUncaughtExceptionHandler((t, e) -> 
                    log.error("Thread {} error", t.getName(), e))
                .build(),
            new ThreadPoolExecutor.CallerRunsPolicy()  // 拒绝策略
        );
    }
}
```

**线程池监控方案：**

```java
@Scheduled(fixedRate = 60000)
public void monitorThreadPool() {
    ThreadPoolExecutor executor = businessThreadPool;
    
    // 核心指标
    int poolSize = executor.getPoolSize();           // 当前线程数
    int activeCount = executor.getActiveCount();     // 活跃线程数
    int queueSize = executor.getQueue().size();      // 队列任务数
    long completedCount = executor.getCompletedTaskCount();  // 已完成任务数
    long taskCount = executor.getTaskCount();        // 总任务数
    
    // 告警阈值
    double queueUsage = queueSize / 1000.0;
    if (queueUsage > 0.8) {
        log.warn("线程池队列使用率过高: {}%", queueUsage * 100);
    }
    
    // 上报监控指标
    Metrics.gauge("threadpool.pool.size", poolSize);
    Metrics.gauge("threadpool.active.count", activeCount);
    Metrics.gauge("threadpool.queue.size", queueSize);
}
```

**延伸：** 参考 [多线程 - 线程池](/docs/java/multithreading#线程池)

---

### 8. synchronized 锁升级过程是怎样的？

**答案要点：**

**锁状态演进：**

```
无锁 → 偏向锁 → 轻量级锁 → 重量级锁
```

**对象头 Mark Word 结构（64位）：**

```
┌────────────────────────────────────────────────────────────────┐
│                        Mark Word (64 bits)                      │
├────────────────────────────────────────────────────────────────┤
│ 无锁    │ unused:25 │ hashcode:31 │ unused:1 │ age:4 │ 0 │ 01 │
├────────────────────────────────────────────────────────────────┤
│ 偏向锁  │ thread:54 │ epoch:2 │ unused:1 │ age:4 │ 1 │ 01 │
├────────────────────────────────────────────────────────────────┤
│ 轻量级锁│ ptr_to_lock_record:62                      │ 00 │
├────────────────────────────────────────────────────────────────┤
│ 重量级锁│ ptr_to_heavyweight_monitor:62              │ 10 │
├────────────────────────────────────────────────────────────────┤
│ GC标记  │                                            │ 11 │
└────────────────────────────────────────────────────────────────┘
```

**锁升级详细过程：**

```java
public class LockEscalation {
    private Object lock = new Object();
    
    public void method() {
        synchronized (lock) {
            // 1. 首次获取：偏向锁
            //    - 检查 Mark Word 是否为可偏向状态
            //    - CAS 将线程 ID 写入 Mark Word
            //    - 后续同一线程进入无需 CAS
            
            // 2. 其他线程竞争：升级为轻量级锁
            //    - 撤销偏向锁
            //    - 在栈帧中创建 Lock Record
            //    - CAS 将 Mark Word 替换为 Lock Record 指针
            
            // 3. CAS 自旋失败：升级为重量级锁
            //    - 自旋超过阈值（默认10次）
            //    - 膨胀为 Monitor 对象
            //    - 线程进入阻塞状态
        }
    }
}
```

**JVM 锁优化参数：**

```bash
# 偏向锁（JDK 15 默认关闭）
-XX:+UseBiasedLocking
-XX:BiasedLockingStartupDelay=0

# 自旋锁
-XX:PreBlockSpin=10  # 自旋次数

# 查看锁信息
-XX:+PrintSafepointStatistics
```

**延伸：** 参考 [多线程 - 线程同步](/docs/java/multithreading#线程同步)

---

### 9. CAS 原理是什么？ABA 问题如何解决？

**答案要点：**

**CAS（Compare And Swap）原理：**

```java
// CAS 伪代码
boolean compareAndSwap(V* address, V expectedValue, V newValue) {
    if (*address == expectedValue) {
        *address = newValue;
        return true;
    }
    return false;
}
```

**Java 中的 CAS 实现：**

```java
public class CASDemo {
    private AtomicInteger count = new AtomicInteger(0);
    
    public void increment() {
        int oldValue, newValue;
        do {
            oldValue = count.get();
            newValue = oldValue + 1;
        } while (!count.compareAndSet(oldValue, newValue));
    }
}
```

**ABA 问题示例：**

```
线程1：读取值 A
线程2：将 A 改为 B
线程2：将 B 改回 A
线程1：CAS 成功（但值已被修改过）
```

**解决方案 - AtomicStampedReference：**

```java
public class ABADemo {
    // 使用版本号解决 ABA 问题
    private AtomicStampedReference<Integer> ref = 
        new AtomicStampedReference<>(100, 0);
    
    public void update() {
        int[] stampHolder = new int[1];
        Integer value = ref.get(stampHolder);
        int stamp = stampHolder[0];
        
        // CAS 同时比较值和版本号
        boolean success = ref.compareAndSet(
            value,           // 期望值
            value + 1,       // 新值
            stamp,           // 期望版本号
            stamp + 1        // 新版本号
        );
    }
}
```

**LongAdder 优化原理：**

```java
// AtomicLong：所有线程竞争同一个 value
// LongAdder：分散热点，减少竞争

public class LongAdderDemo {
    // 高并发场景推荐使用 LongAdder
    private LongAdder counter = new LongAdder();
    
    public void increment() {
        counter.increment();  // 内部分散到多个 Cell
    }
    
    public long get() {
        return counter.sum();  // 汇总所有 Cell
    }
}
```

**延伸：** 参考 [多线程 - 原子类](/docs/java/multithreading)

---

### 10. 如何实现一个高性能的生产者消费者模式？

**答案要点：**

**方案对比：**

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| wait/notify | 简单 | 性能一般 | 简单场景 |
| BlockingQueue | 易用 | 有锁开销 | 一般场景 |
| Disruptor | 极高性能 | 复杂 | 高性能场景 |

**BlockingQueue 实现：**

```java
public class ProducerConsumer {
    private final BlockingQueue<Task> queue = 
        new ArrayBlockingQueue<>(1000);
    
    // 生产者
    class Producer implements Runnable {
        @Override
        public void run() {
            while (true) {
                Task task = createTask();
                try {
                    queue.put(task);  // 队列满时阻塞
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }
    }
    
    // 消费者
    class Consumer implements Runnable {
        @Override
        public void run() {
            while (true) {
                try {
                    Task task = queue.take();  // 队列空时阻塞
                    process(task);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }
    }
}
```

**Disruptor 高性能实现：**

```java
public class DisruptorDemo {
    
    public static void main(String[] args) {
        // 创建 Disruptor
        Disruptor<OrderEvent> disruptor = new Disruptor<>(
            OrderEvent::new,
            1024 * 1024,  // RingBuffer 大小，必须是2的幂
            DaemonThreadFactory.INSTANCE,
            ProducerType.MULTI,
            new YieldingWaitStrategy()  // 等待策略
        );
        
        // 设置消费者
        disruptor.handleEventsWith(new OrderEventHandler());
        
        // 启动
        RingBuffer<OrderEvent> ringBuffer = disruptor.start();
        
        // 生产者发布事件
        long sequence = ringBuffer.next();
        try {
            OrderEvent event = ringBuffer.get(sequence);
            event.setOrderId(12345L);
        } finally {
            ringBuffer.publish(sequence);
        }
    }
}
```

**延伸：** 参考 [多线程 - 并发设计模式](/docs/java/multithreading)

---

## 🎯 性能调优（专家级）

### 11. 如何排查线上 CPU 飙高问题？

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

### 12. 如何排查内存泄漏问题？

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

### 13. Arthas 有哪些常用命令？如何使用？

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

### 14. 如何优化数据库查询性能？

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

### 15. 缓存穿透、缓存击穿、缓存雪崩如何解决？

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

---

## 🎯 架构设计（专家级）

### 16. 什么是 CAP 理论？如何在实际系统中权衡？

**答案要点：**

**CAP 三要素：**

- **C（Consistency）一致性**：所有节点同一时刻数据相同
- **A（Availability）可用性**：每个请求都能得到响应
- **P（Partition tolerance）分区容错**：网络分区时系统仍能运行

**CAP 不可能三角：**

```
        C（一致性）
           /\
          /  \
         /    \
        /  CA  \
       /________\
      A          P
   (可用性)  (分区容错)

分布式系统必须选择 P，因此只能在 CP 和 AP 之间选择
```

**实际系统选择：**

| 系统 | 选择 | 说明 |
|------|------|------|
| ZooKeeper | CP | 强一致性，可能短暂不可用 |
| Eureka | AP | 高可用，允许数据不一致 |
| Redis Cluster | AP | 异步复制，可能丢数据 |
| MySQL 主从 | CP | 同步复制保证一致性 |

**BASE 理论（AP 的延伸）：**

```
BA - Basically Available（基本可用）
S  - Soft state（软状态）
E  - Eventually consistent（最终一致性）
```

**实际权衡示例：**

```java
// 电商下单场景
// 1. 库存扣减 - 需要强一致性（CP）
// 2. 订单创建 - 需要高可用（AP）
// 3. 通知发送 - 最终一致性即可

@Transactional
public Order createOrder(OrderRequest request) {
    // CP: 同步扣减库存
    inventoryService.deduct(request.getProductId(), request.getQuantity());
    
    // AP: 创建订单
    Order order = orderRepository.save(new Order(request));
    
    // 最终一致性: 异步发送通知
    messageQueue.send(new OrderCreatedEvent(order));
    
    return order;
}
```

**延伸：** 参考 [微服务 - 分布式理论](/docs/microservices/core-concepts)

---

### 17. 分布式事务有哪些解决方案？各有什么优缺点？

**答案要点：**

**分布式事务方案对比：**

| 方案 | 一致性 | 性能 | 复杂度 | 适用场景 |
|------|--------|------|--------|---------|
| 2PC | 强一致 | 低 | 中 | 数据库分布式事务 |
| TCC | 最终一致 | 中 | 高 | 资金交易 |
| Saga | 最终一致 | 高 | 中 | 长事务 |
| 本地消息表 | 最终一致 | 高 | 低 | 异步场景 |
| MQ 事务消息 | 最终一致 | 高 | 低 | 消息驱动 |

**TCC 实现示例：**

```java
// TCC: Try-Confirm-Cancel
public interface AccountService {
    // Try: 预留资源
    @TwoPhaseBusinessAction(name = "deduct", 
        commitMethod = "confirm", rollbackMethod = "cancel")
    boolean tryDeduct(BusinessActionContext context, 
                      @BusinessActionContextParameter("accountId") String accountId,
                      @BusinessActionContextParameter("amount") BigDecimal amount);
    
    // Confirm: 确认提交
    boolean confirm(BusinessActionContext context);
    
    // Cancel: 取消回滚
    boolean cancel(BusinessActionContext context);
}

@Service
public class AccountServiceImpl implements AccountService {
    
    @Override
    public boolean tryDeduct(BusinessActionContext context, 
                             String accountId, BigDecimal amount) {
        // 冻结金额
        accountDao.freeze(accountId, amount);
        return true;
    }
    
    @Override
    public boolean confirm(BusinessActionContext context) {
        String accountId = context.getActionContext("accountId");
        BigDecimal amount = context.getActionContext("amount");
        // 扣减冻结金额
        accountDao.deductFrozen(accountId, amount);
        return true;
    }
    
    @Override
    public boolean cancel(BusinessActionContext context) {
        String accountId = context.getActionContext("accountId");
        BigDecimal amount = context.getActionContext("amount");
        // 解冻金额
        accountDao.unfreeze(accountId, amount);
        return true;
    }
}
```

**本地消息表方案：**

```java
@Transactional
public void createOrder(OrderRequest request) {
    // 1. 创建订单
    Order order = orderRepository.save(new Order(request));
    
    // 2. 写入本地消息表（同一事务）
    LocalMessage message = new LocalMessage();
    message.setMessageId(UUID.randomUUID().toString());
    message.setContent(JSON.toJSONString(new OrderCreatedEvent(order)));
    message.setStatus("PENDING");
    localMessageRepository.save(message);
}

// 定时任务发送消息
@Scheduled(fixedRate = 1000)
public void sendPendingMessages() {
    List<LocalMessage> messages = localMessageRepository.findByStatus("PENDING");
    for (LocalMessage message : messages) {
        try {
            messageQueue.send(message.getContent());
            message.setStatus("SENT");
            localMessageRepository.save(message);
        } catch (Exception e) {
            // 重试
        }
    }
}
```

**延伸：** 参考 [微服务 - 分布式事务](/docs/microservices/design-patterns)

---

### 18. 如何设计一个高可用的系统？

**答案要点：**

**高可用设计原则：**

```
可用性 = MTBF / (MTBF + MTTR)

MTBF: 平均故障间隔时间
MTTR: 平均修复时间

99.9%  = 8.76 小时/年 停机
99.99% = 52.6 分钟/年 停机
```

**高可用架构设计：**

```
                    ┌─────────────┐
                    │   DNS/CDN   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
        │   LB-1    │ │  LB-2   │ │   LB-3    │  负载均衡
        └─────┬─────┘ └────┬────┘ └─────┬─────┘
              │            │            │
    ┌─────────┼────────────┼────────────┼─────────┐
    │         │            │            │         │
┌───▼───┐ ┌───▼───┐ ┌──────▼──────┐ ┌───▼───┐ ┌───▼───┐
│ App-1 │ │ App-2 │ │    App-3    │ │ App-4 │ │ App-5 │  应用集群
└───┬───┘ └───┬───┘ └──────┬──────┘ └───┬───┘ └───┬───┘
    │         │            │            │         │
    └─────────┼────────────┼────────────┼─────────┘
              │            │            │
        ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
        │  Redis-M  │ │ Redis-S │ │  Redis-S  │  缓存集群
        └───────────┘ └─────────┘ └───────────┘
              │
        ┌─────▼─────┐ ┌─────────┐ ┌───────────┐
        │  MySQL-M  │ │ MySQL-S │ │  MySQL-S  │  数据库集群
        └───────────┘ └─────────┘ └───────────┘
```

**限流熔断实现：**

```java
// Sentinel 限流配置
@SentinelResource(value = "getUser", 
    blockHandler = "getUserBlockHandler",
    fallback = "getUserFallback")
public User getUser(String id) {
    return userService.getUser(id);
}

// 限流处理
public User getUserBlockHandler(String id, BlockException e) {
    return new User("限流中，请稍后重试");
}

// 降级处理
public User getUserFallback(String id, Throwable e) {
    return new User("服务暂时不可用");
}
```

**延伸：** 参考 [微服务 - 服务治理](/docs/microservices/service-governance)

---

### 19. 如何设计一个秒杀系统？

**答案要点：**

**秒杀系统架构：**

```
用户请求 → CDN → 网关（限流）→ 秒杀服务 → Redis → 消息队列 → 订单服务
                    ↓
              静态页面缓存
```

**核心设计要点：**

```java
// 1. 库存预热到 Redis
@PostConstruct
public void initStock() {
    List<SeckillProduct> products = productService.getSeckillProducts();
    for (SeckillProduct product : products) {
        redisTemplate.opsForValue().set(
            "seckill:stock:" + product.getId(), 
            product.getStock()
        );
    }
}

// 2. Redis 原子扣减库存
public boolean deductStock(Long productId) {
    String key = "seckill:stock:" + productId;
    Long stock = redisTemplate.opsForValue().decrement(key);
    if (stock < 0) {
        // 库存不足，恢复
        redisTemplate.opsForValue().increment(key);
        return false;
    }
    return true;
}

// 3. 异步创建订单
@Transactional
public void createOrder(SeckillRequest request) {
    // 扣减 Redis 库存
    if (!deductStock(request.getProductId())) {
        throw new SeckillException("库存不足");
    }
    
    // 发送消息异步创建订单
    OrderMessage message = new OrderMessage(
        request.getUserId(), 
        request.getProductId()
    );
    kafkaTemplate.send("seckill-order", message);
}

// 4. 消费者处理订单
@KafkaListener(topics = "seckill-order")
public void handleOrder(OrderMessage message) {
    // 创建订单
    Order order = new Order();
    order.setUserId(message.getUserId());
    order.setProductId(message.getProductId());
    orderRepository.save(order);
    
    // 扣减数据库库存
    productRepository.deductStock(message.getProductId());
}
```

**防刷策略：**

```java
// 1. 用户限流
@RateLimiter(key = "seckill:user:#userId", rate = 1, interval = 1)
public void seckill(Long userId, Long productId) { }

// 2. IP 限流
@RateLimiter(key = "seckill:ip:#ip", rate = 10, interval = 1)
public void seckill(String ip, Long productId) { }

// 3. 验证码
// 4. 隐藏秒杀接口（动态 URL）
```

**延伸：** 参考 [微服务 - 高并发设计](/docs/microservices/design-patterns)

---

### 20. 微服务拆分的原则是什么？如何确定服务边界？

**答案要点：**

**服务拆分原则：**

| 原则 | 说明 |
|------|------|
| **单一职责** | 每个服务只负责一个业务领域 |
| **高内聚低耦合** | 服务内部高度相关，服务间依赖最小 |
| **业务边界清晰** | 基于领域驱动设计（DDD）划分 |
| **数据独立** | 每个服务拥有独立的数据存储 |
| **可独立部署** | 服务可以独立开发、测试、部署 |

**DDD 领域划分：**

```
电商系统领域划分

┌─────────────────────────────────────────────────────────┐
│                      核心域                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ 商品域   │  │ 订单域   │  │ 支付域   │  │ 库存域   │    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                      支撑域                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
│  │ 用户域   │  │ 营销域   │  │ 物流域   │                 │
│  └─────────┘  └─────────┘  └─────────┘                 │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                      通用域                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
│  │ 消息通知 │  │ 文件存储 │  │ 日志监控 │                 │
│  └─────────┘  └─────────┘  └─────────┘                 │
└─────────────────────────────────────────────────────────┘
```

**服务拆分反模式：**

```java
// ❌ 错误：分布式单体
// 服务间强依赖，同步调用链过长
OrderService → InventoryService → PaymentService → LogisticsService

// ✅ 正确：事件驱动解耦
OrderService --发布事件--> EventBus
                              ↓
              InventoryService（订阅）
              PaymentService（订阅）
              LogisticsService（订阅）
```

**延伸：** 参考 [微服务 - 核心概念](/docs/microservices/core-concepts)

---

## 🎯 框架源码分析（专家级）

### 21. Spring IoC 容器启动流程是怎样的？

**答案要点：**

**核心启动流程：**

```java
// AbstractApplicationContext.refresh() 方法
public void refresh() {
    // 1. 准备刷新
    prepareRefresh();
    
    // 2. 获取 BeanFactory
    ConfigurableListableBeanFactory beanFactory = obtainFreshBeanFactory();
    
    // 3. 准备 BeanFactory
    prepareBeanFactory(beanFactory);
    
    // 4. 后置处理 BeanFactory
    postProcessBeanFactory(beanFactory);
    
    // 5. 调用 BeanFactoryPostProcessor
    invokeBeanFactoryPostProcessors(beanFactory);
    
    // 6. 注册 BeanPostProcessor
    registerBeanPostProcessors(beanFactory);
    
    // 7. 初始化消息源
    initMessageSource();
    
    // 8. 初始化事件广播器
    initApplicationEventMulticaster();
    
    // 9. 子类扩展点
    onRefresh();
    
    // 10. 注册监听器
    registerListeners();
    
    // 11. 实例化所有非懒加载的单例 Bean
    finishBeanFactoryInitialization(beanFactory);
    
    // 12. 完成刷新
    finishRefresh();
}
```

**Bean 创建流程：**

```
getBean() 
    → doGetBean()
        → getSingleton() // 从缓存获取
        → createBean()
            → resolveBeforeInstantiation() // 实例化前处理
            → doCreateBean()
                → createBeanInstance() // 实例化
                → populateBean() // 属性填充
                → initializeBean() // 初始化
                    → invokeAwareMethods()
                    → applyBeanPostProcessorsBeforeInitialization()
                    → invokeInitMethods()
                    → applyBeanPostProcessorsAfterInitialization()
```

**三级缓存解决循环依赖：**

```java
// DefaultSingletonBeanRegistry
// 一级缓存：完整的 Bean
private final Map<String, Object> singletonObjects = new ConcurrentHashMap<>();

// 二级缓存：早期暴露的 Bean（未完成属性填充）
private final Map<String, Object> earlySingletonObjects = new ConcurrentHashMap<>();

// 三级缓存：Bean 工厂
private final Map<String, ObjectFactory<?>> singletonFactories = new HashMap<>();
```

**延伸：** 参考 [Spring 核心概念](/docs/spring/core-concepts)

---

### 22. Spring AOP 是如何实现的？

**答案要点：**

**AOP 实现方式：**

| 方式 | 条件 | 特点 |
|------|------|------|
| JDK 动态代理 | 目标类实现接口 | 基于接口代理 |
| CGLIB 代理 | 目标类无接口 | 基于继承代理 |

**JDK 动态代理原理：**

```java
public class JdkProxyDemo {
    public static void main(String[] args) {
        UserService target = new UserServiceImpl();
        
        UserService proxy = (UserService) Proxy.newProxyInstance(
            target.getClass().getClassLoader(),
            target.getClass().getInterfaces(),
            new InvocationHandler() {
                @Override
                public Object invoke(Object proxy, Method method, Object[] args) 
                        throws Throwable {
                    System.out.println("Before: " + method.getName());
                    Object result = method.invoke(target, args);
                    System.out.println("After: " + method.getName());
                    return result;
                }
            }
        );
        
        proxy.getUser("1");
    }
}
```

**CGLIB 代理原理：**

```java
public class CglibProxyDemo {
    public static void main(String[] args) {
        Enhancer enhancer = new Enhancer();
        enhancer.setSuperclass(UserServiceImpl.class);
        enhancer.setCallback(new MethodInterceptor() {
            @Override
            public Object intercept(Object obj, Method method, Object[] args, 
                    MethodProxy proxy) throws Throwable {
                System.out.println("Before: " + method.getName());
                Object result = proxy.invokeSuper(obj, args);
                System.out.println("After: " + method.getName());
                return result;
            }
        });
        
        UserServiceImpl proxy = (UserServiceImpl) enhancer.create();
        proxy.getUser("1");
    }
}
```

**Spring AOP 代理创建流程：**

```
@EnableAspectJAutoProxy
    → 注册 AnnotationAwareAspectJAutoProxyCreator
        → postProcessAfterInitialization()
            → wrapIfNecessary()
                → getAdvicesAndAdvisorsForBean() // 获取切面
                → createProxy() // 创建代理
                    → ProxyFactory.getProxy()
                        → JdkDynamicAopProxy 或 CglibAopProxy
```

**延伸：** 参考 [Spring AOP 详解](/docs/spring/aop)

---

### 23. Spring Boot 自动配置原理是什么？

**答案要点：**

**自动配置核心注解：**

```java
@SpringBootApplication
    ├── @SpringBootConfiguration  // 配置类
    ├── @EnableAutoConfiguration  // 启用自动配置
    │       └── @Import(AutoConfigurationImportSelector.class)
    └── @ComponentScan            // 组件扫描
```

**自动配置加载流程：**

```
1. @EnableAutoConfiguration
    ↓
2. AutoConfigurationImportSelector.selectImports()
    ↓
3. SpringFactoriesLoader.loadFactoryNames()
    ↓
4. 读取 META-INF/spring.factories
    ↓
5. 过滤条件注解（@ConditionalOnXxx）
    ↓
6. 加载符合条件的自动配置类
```

**spring.factories 示例：**

```properties
# META-INF/spring.factories
org.springframework.boot.autoconfigure.EnableAutoConfiguration=\
org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration,\
org.springframework.boot.autoconfigure.web.servlet.WebMvcAutoConfiguration,\
org.springframework.boot.autoconfigure.data.redis.RedisAutoConfiguration
```

**条件注解原理：**

```java
@Configuration
@ConditionalOnClass(DataSource.class)  // 类路径存在 DataSource
@ConditionalOnMissingBean(DataSource.class)  // 未自定义 DataSource Bean
@EnableConfigurationProperties(DataSourceProperties.class)
public class DataSourceAutoConfiguration {
    
    @Bean
    @ConfigurationProperties(prefix = "spring.datasource")
    public DataSource dataSource() {
        return DataSourceBuilder.create().build();
    }
}
```

**自定义 Starter：**

```java
// 1. 创建自动配置类
@Configuration
@ConditionalOnClass(MyService.class)
@EnableConfigurationProperties(MyProperties.class)
public class MyAutoConfiguration {
    
    @Bean
    @ConditionalOnMissingBean
    public MyService myService(MyProperties properties) {
        return new MyService(properties);
    }
}

// 2. 创建 spring.factories
// META-INF/spring.factories
org.springframework.boot.autoconfigure.EnableAutoConfiguration=\
com.example.MyAutoConfiguration
```

**延伸：** 参考 [Spring Boot 自动配置](/docs/springboot)

---

### 24. MyBatis 的执行流程和缓存机制是怎样的？

**答案要点：**

**MyBatis 执行流程：**

```
SqlSessionFactory
    ↓ openSession()
SqlSession
    ↓ getMapper()
MapperProxy（动态代理）
    ↓ invoke()
MapperMethod
    ↓ execute()
Executor（执行器）
    ↓ query/update
StatementHandler
    ↓ prepare/parameterize/query
ResultSetHandler
    ↓ handleResultSets
返回结果
```

**核心组件：**

```java
// 1. SqlSessionFactory 创建
SqlSessionFactory factory = new SqlSessionFactoryBuilder()
    .build(Resources.getResourceAsStream("mybatis-config.xml"));

// 2. 获取 SqlSession
try (SqlSession session = factory.openSession()) {
    // 3. 获取 Mapper 代理
    UserMapper mapper = session.getMapper(UserMapper.class);
    
    // 4. 执行查询
    User user = mapper.selectById(1L);
}
```

**缓存机制：**

```
┌─────────────────────────────────────────────────────────┐
│                    二级缓存（Mapper 级别）                │
│                    namespace 范围共享                    │
└─────────────────────────────────────────────────────────┘
                           ↓ 未命中
┌─────────────────────────────────────────────────────────┐
│                    一级缓存（SqlSession 级别）           │
│                    默认开启，同一 SqlSession 共享        │
└─────────────────────────────────────────────────────────┘
                           ↓ 未命中
┌─────────────────────────────────────────────────────────┐
│                         数据库                           │
└─────────────────────────────────────────────────────────┘
```

**二级缓存配置：**

```xml
<!-- mybatis-config.xml -->
<settings>
    <setting name="cacheEnabled" value="true"/>
</settings>

<!-- UserMapper.xml -->
<mapper namespace="com.example.mapper.UserMapper">
    <cache eviction="LRU" flushInterval="60000" size="512" readOnly="true"/>
    
    <select id="selectById" resultType="User" useCache="true">
        SELECT * FROM user WHERE id = #{id}
    </select>
</mapper>
```

**延伸：** 参考 [MyBatis 核心原理](/docs/spring)

---

### 25. Netty 的线程模型和核心组件是什么？

**答案要点：**

**Netty 线程模型（主从 Reactor）：**

```
                    ┌─────────────────┐
                    │   BossGroup     │  接收连接
                    │  (1个EventLoop) │
                    └────────┬────────┘
                             │ 分发
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐  ┌─────────▼─────────┐  ┌───────▼───────┐
│  WorkerGroup  │  │   WorkerGroup     │  │  WorkerGroup  │
│  EventLoop-1  │  │   EventLoop-2     │  │  EventLoop-N  │
│  (处理IO)     │  │   (处理IO)        │  │  (处理IO)     │
└───────────────┘  └───────────────────┘  └───────────────┘
```

**核心组件：**

| 组件 | 作用 |
|------|------|
| **Channel** | 网络连接通道 |
| **EventLoop** | 事件循环，处理 IO 事件 |
| **ChannelPipeline** | 处理器链 |
| **ChannelHandler** | 事件处理器 |
| **ByteBuf** | 字节缓冲区 |

**Netty 服务端示例：**

```java
public class NettyServer {
    public static void main(String[] args) throws Exception {
        EventLoopGroup bossGroup = new NioEventLoopGroup(1);
        EventLoopGroup workerGroup = new NioEventLoopGroup();
        
        try {
            ServerBootstrap bootstrap = new ServerBootstrap();
            bootstrap.group(bossGroup, workerGroup)
                .channel(NioServerSocketChannel.class)
                .option(ChannelOption.SO_BACKLOG, 128)
                .childOption(ChannelOption.SO_KEEPALIVE, true)
                .childHandler(new ChannelInitializer<SocketChannel>() {
                    @Override
                    protected void initChannel(SocketChannel ch) {
                        ChannelPipeline pipeline = ch.pipeline();
                        pipeline.addLast(new StringDecoder());
                        pipeline.addLast(new StringEncoder());
                        pipeline.addLast(new MyServerHandler());
                    }
                });
            
            ChannelFuture future = bootstrap.bind(8080).sync();
            future.channel().closeFuture().sync();
        } finally {
            bossGroup.shutdownGracefully();
            workerGroup.shutdownGracefully();
        }
    }
}

class MyServerHandler extends SimpleChannelInboundHandler<String> {
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, String msg) {
        System.out.println("Received: " + msg);
        ctx.writeAndFlush("Server: " + msg);
    }
}
```

**延伸：** 参考 [Netty 核心组件](/docs/netty/core-components)

---

## 📌 总结与学习建议

### 难度分级

- **高级（3-5年）：** JVM 深度、高级并发、框架源码基础
- **专家级（5年+）：** 性能调优、架构设计、源码深度分析

### 学习路径

```
1. 夯实基础
   └── Java 基础 → 集合 → 多线程 → JVM

2. 深入原理
   └── 并发包源码 → Spring 源码 → MyBatis 源码

3. 架构提升
   └── 设计模式 → 分布式理论 → 微服务架构

4. 实战经验
   └── 性能调优 → 问题排查 → 系统设计
```

### 面试准备建议

1. **理解原理** > 背诵答案
2. **动手实践** > 纸上谈兵
3. **源码阅读** > 文档浏览
4. **项目经验** > 理论知识

### 相关资源

- [Java 基础面试题](/docs/java/interview-questions)
- [Spring 面试题](/docs/spring/interview-questions)
- [JVM 基础](/docs/java/jvm-basics)
- [多线程详解](/docs/java/multithreading)
- [性能优化](/docs/java/performance)
- [微服务架构](/docs/microservices)
- [设计模式](/docs/java-design-patterns)

---

**持续更新中...** 欢迎反馈和补充！
