---
sidebar_position: 3
title: 高级并发编程
---

# 🎯 高级并发编程（高级）

## 6. 详细解释 AQS（AbstractQueuedSynchronizer）的原理

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

## 7. 线程池核心参数如何配置？如何监控线程池状态？

**答案要点：**

**线程池参数配置原则：**

| 场景           | corePoolSize         | maximumPoolSize | 队列   |
| -------------- | -------------------- | --------------- | ------ |
| **CPU 密集型** | CPU 核心数           | CPU 核心数      | 小队列 |
| **IO 密集型**  | 2 \* CPU 核心数      | 2 \* CPU 核心数 | 大队列 |
| **混合型**     | 根据 IO/CPU 比例调整 | -               | -      |

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

## 8. synchronized 锁升级过程是怎样的？

**答案要点：**

**锁状态演进：**

```
无锁 → 偏向锁 → 轻量级锁 → 重量级锁
```

**对象头 Mark Word 结构（64 位）：**

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

## 9. CAS 原理是什么？ABA 问题如何解决？

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

## 10. 如何实现一个高性能的生产者消费者模式？

**答案要点：**

**方案对比：**

| 方案          | 优点     | 缺点     | 适用场景   |
| ------------- | -------- | -------- | ---------- |
| wait/notify   | 简单     | 性能一般 | 简单场景   |
| BlockingQueue | 易用     | 有锁开销 | 一般场景   |
| Disruptor     | 极高性能 | 复杂     | 高性能场景 |

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

## 11. ThreadLocal 原理及内存泄漏原因？

**答案要点：**

**核心原理：**

ThreadLocal 提供了线程局部变量，每个线程访问该变量时都有自己独立的副本。

```java
public void set(T value) {
    Thread t = Thread.currentThread();
    ThreadLocalMap map = getMap(t);
    if (map != null)
        map.set(this, value); // key 是 ThreadLocal 本身
    else
        createMap(t, value);
}
```

**内存结构：**

- 每个 Thread 维护一个 `ThreadLocalMap`。
- `ThreadLocalMap` 的 Key 是 `ThreadLocal` 实例（WeakReference）。
- Value 是真正存储的对象（StrongReference）。

**内存泄漏原因：**

```
Thread -> ThreadLocalMap -> Entry(Key(Weak), Value(Strong))
```

1.  **Key 被回收**：Key 是弱引用，下一次 GC 会被回收，Entry 的 Key 变为 null。
2.  **Value 无法回收**：Value 是强引用，且 ThreadLocalMap 生命周期与 Thread 一致。如果线程（如线程池中的线程）长时间运行，Value 就会一直存在，导致内存泄漏。

**解决方案：**

使用完 ThreadLocal 后，**必须强制调用 `remove()` 方法**。

```java
try {
    threadLocal.set("value");
    // 业务逻辑
} finally {
    threadLocal.remove(); // 防止内存泄漏
}
```
