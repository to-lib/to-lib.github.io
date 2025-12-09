---
sidebar_position: 6
title: 多线程编程
---

# 多线程编程

多线程是 Java 的重要特性，使程序能够同时执行多个任务。本文介绍线程的创建、同步、线程池等核心概念。

## 线程基础

### 什么是线程

- **进程**：操作系统分配资源的基本单位
- **线程**：CPU 调度的基本单位，一个进程可以有多个线程

### 创建线程的方式

#### 方式 1：继承 Thread 类

```java
public class MyThread extends Thread {
    @Override
    public void run() {
        for (int i = 0; i < 5; i++) {
            System.out.println(Thread.currentThread().getName() + ": " + i);
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
    
    public static void main(String[] args) {
        MyThread thread1 = new MyThread();
        MyThread thread2 = new MyThread();
        
        thread1.setName("线程1");
        thread2.setName("线程2");
        
        thread1.start();  // 启动线程
        thread2.start();
    }
}
```

#### 方式 2：实现 Runnable 接口（推荐）

```java
public class MyRunnable implements Runnable {
    @Override
    public void run() {
        for (int i = 0; i < 5; i++) {
            System.out.println(Thread.currentThread().getName() + ": " + i);
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
    
    public static void main(String[] args) {
        MyRunnable runnable = new MyRunnable();
        
        Thread thread1 = new Thread(runnable, "线程1");
        Thread thread2 = new Thread(runnable, "线程2");
        
        thread1.start();
        thread2.start();
    }
}
```

#### 方式 3：实现 Callable 接口（有返回值）

```java
import java.util.concurrent.*;

public class MyCallable implements Callable<Integer> {
    private int number;
    
    public MyCallable(int number) {
        this.number = number;
    }
    
    @Override
    public Integer call() throws Exception {
        int sum = 0;
        for (int i = 1; i <= number; i++) {
            sum += i;
            Thread.sleep(10);
        }
        return sum;
    }
    
    public static void main(String[] args) {
        // 创建线程池
        ExecutorService executor = Executors.newFixedThreadPool(2);
        
        // 提交任务
        Future<Integer> future1 = executor.submit(new MyCallable(100));
        Future<Integer> future2 = executor.submit(new MyCallable(50));
        
        try {
            // 获取结果（阻塞）
            Integer result1 = future1.get();
            Integer result2 = future2.get();
            
            System.out.println("结果1: " + result1);  // 5050
            System.out.println("结果2: " + result2);  // 1275
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        
        executor.shutdown();
    }
}
```

#### 方式 4：使用 Lambda 表达式（Java 8+）

```java
public class LambdaThreadExample {
    public static void main(String[] args) {
        // 使用 Lambda 创建线程
        Thread thread = new Thread(() -> {
            for (int i = 0; i < 5; i++) {
                System.out.println("Lambda 线程: " + i);
            }
        });
        
        thread.start();
    }
}
```

## 线程的生命周期

```mermaid
stateDiagram-v2
    [*] --> 新建: new Thread()
    新建 --> 就绪: start()
    就绪 --> 运行: 获得CPU
    运行 --> 就绪: yield()/时间片用完
    运行 --> 阻塞: sleep()/wait()/IO阻塞
    阻塞 --> 就绪: sleep时间到/notify()/IO完成
    运行 --> 终止: run()结束
    终止 --> [*]
```

### 线程状态

```java
public class ThreadStateExample {
    public static void main(String[] args) throws InterruptedException {
        Thread thread = new Thread(() -> {
            try {
                Thread.sleep(2000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
        
        System.out.println("NEW: " + thread.getState());  // NEW
        
        thread.start();
        System.out.println("RUNNABLE: " + thread.getState());  // RUNNABLE
        
        Thread.sleep(100);
        System.out.println("TIMED_WAITING: " + thread.getState());  // TIMED_WAITING
        
        thread.join();
        System.out.println("TERMINATED: " + thread.getState());  // TERMINATED
    }
}
```

## 线程常用方法

### 基本方法

```java
public class ThreadMethodsExample {
    public static void main(String[] args) throws InterruptedException {
        Thread thread = new Thread(() -> {
            System.out.println("线程运行中...");
            try {
                Thread.sleep(2000);
            } catch (InterruptedException e) {
                System.out.println("线程被中断");
            }
        });
        
        // 设置线程名称
        thread.setName("工作线程");
        
        // 设置优先级 (1-10, 默认5)
        thread.setPriority(Thread.MAX_PRIORITY);
        
        // 设置为守护线程
        thread.setDaemon(true);
        
        // 启动线程
        thread.start();
        
        // 等待线程结束
        thread.join();
        
        // 检查线程是否存活
        System.out.println("线程存活: " + thread.isAlive());
    }
}
```

### sleep() vs wait()

```java
public class SleepVsWait {
    private static final Object lock = new Object();
    
    public static void main(String[] args) {
        // sleep：不释放锁，线程休眠
        Thread sleepThread = new Thread(() -> {
            synchronized (lock) {
                System.out.println("sleep 开始");
                try {
                    Thread.sleep(1000);  // 不释放 lock
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("sleep 结束");
            }
        });
        
        // wait：释放锁，等待通知
        Thread waitThread = new Thread(() -> {
            synchronized (lock) {
                System.out.println("wait 开始");
                try {
                    lock.wait();  // 释放 lock，等待 notify
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("wait 结束");
            }
        });
        
        waitThread.start();
        sleepThread.start();
        
        // 通知等待的线程
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        
        synchronized (lock) {
            lock.notify();  // 唤醒 waitThread
        }
    }
}
```

## 线程同步

### synchronized 关键字

#### 同步方法

```java
public class SynchronizedMethodExample {
    private int count = 0;
    
    // 同步实例方法（锁是 this）
    public synchronized void increment() {
        count++;
    }
    
    // 同步静态方法（锁是 Class 对象）
    public static synchronized void staticMethod() {
        System.out.println("静态同步方法");
    }
    
    public static void main(String[] args) throws InterruptedException {
        SynchronizedMethodExample example = new SynchronizedMethodExample();
        
        // 创建10个线程，每个线程增加1000次
        Thread[] threads = new Thread[10];
        for (int i = 0; i < 10; i++) {
            threads[i] = new Thread(() -> {
                for (int j = 0; j < 1000; j++) {
                    example.increment();
                }
            });
            threads[i].start();
        }
        
        // 等待所有线程结束
        for (Thread thread : threads) {
            thread.join();
        }
        
        System.out.println("最终计数: " + example.count);  // 10000
    }
}
```

#### 同步代码块

```java
public class SynchronizedBlockExample {
    private int count = 0;
    private final Object lock = new Object();
    
    public void increment() {
        // 同步代码块（锁是 lock 对象）
        synchronized (lock) {
            count++;
        }
    }
    
    // 也可以使用 this 作为锁
    public void increment2() {
        synchronized (this) {
            count++;
        }
    }
}
```

### Lock 接口

```java
import java.util.concurrent.locks.*;

public class ReentrantLockExample {
    private int count = 0;
    private final Lock lock = new ReentrantLock();
    
    public void increment() {
        lock.lock();  // 获取锁
        try {
            count++;
        } finally {
            lock.unlock();  // 释放锁（必须在 finally 中）
        }
    }
    
    // 尝试获取锁
    public void tryIncrement() {
        if (lock.tryLock()) {
            try {
                count++;
            } finally {
                lock.unlock();
            }
        } else {
            System.out.println("无法获取锁");
        }
    }
    
    // 读写锁
    private final ReadWriteLock rwLock = new ReentrantReadWriteLock();
    private String data = "";
    
    public String read() {
        rwLock.readLock().lock();
        try {
            return data;
        } finally {
            rwLock.readLock().unlock();
        }
    }
    
    public void write(String newData) {
        rwLock.writeLock().lock();
        try {
            data = newData;
        } finally {
            rwLock.writeLock().unlock();
        }
    }
}
```

## 线程通信

### wait/notify 机制

```java
public class ProducerConsumer {
    private static final int MAX_SIZE = 10;
    private LinkedList<Integer> queue = new LinkedList<>();
    
    // 生产者
    public void produce() throws InterruptedException {
        int value = 0;
        while (true) {
            synchronized (queue) {
                while (queue.size() == MAX_SIZE) {
                    queue.wait();  // 队列满，等待
                }
                
                queue.add(value);
                System.out.println("生产: " + value);
                value++;
                
                queue.notifyAll();  // 通知消费者
                Thread.sleep(100);
            }
        }
    }
    
    // 消费者
    public void consume() throws InterruptedException {
        while (true) {
            synchronized (queue) {
                while (queue.isEmpty()) {
                    queue.wait();  // 队列空，等待
                }
                
                int value = queue.removeFirst();
                System.out.println("消费: " + value);
                
                queue.notifyAll();  // 通知生产者
                Thread.sleep(200);
            }
        }
    }
    
    public static void main(String[] args) {
        ProducerConsumer pc = new ProducerConsumer();
        
        new Thread(() -> {
            try {
                pc.produce();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }).start();
        
        new Thread(() -> {
            try {
                pc.consume();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }).start();
    }
}
```

### Condition 接口

```java
import java.util.concurrent.locks.*;

public class ConditionExample {
    private final Lock lock = new ReentrantLock();
    private final Condition notFull = lock.newCondition();
    private final Condition notEmpty = lock.newCondition();
    private LinkedList<Integer> queue = new LinkedList<>();
    private final int MAX_SIZE = 10;
    
    public void produce(int value) throws InterruptedException {
        lock.lock();
        try {
            while (queue.size() == MAX_SIZE) {
                notFull.await();  // 等待队列不满
            }
            queue.add(value);
            notEmpty.signal();  // 通知消费者
        } finally {
            lock.unlock();
        }
    }
    
    public int consume() throws InterruptedException {
        lock.lock();
        try {
            while (queue.isEmpty()) {
                notEmpty.await();  // 等待队列不空
            }
            int value = queue.removeFirst();
            notFull.signal();  // 通知生产者
            return value;
        } finally {
            lock.unlock();
        }
    }
}
```

## 线程池

### 为什么使用线程池

- 减少线程创建和销毁的开销
- 提高响应速度
- 便于线程管理

### 创建线程池

```java
import java.util.concurrent.*;

public class ThreadPoolExample {
    public static void main(String[] args) {
        // 1. 固定大小线程池
        ExecutorService fixedPool = Executors.newFixedThreadPool(5);
        
        // 2. 缓存线程池
        ExecutorService cachedPool = Executors.newCachedThreadPool();
        
        // 3. 单线程线程池
        ExecutorService singlePool = Executors.newSingleThreadExecutor();
        
        // 4. 定时任务线程池
        ScheduledExecutorService scheduledPool = Executors.newScheduledThreadPool(3);
        
        // 5. 自定义线程池（推荐）
        ThreadPoolExecutor customPool = new ThreadPoolExecutor(
            5,                      // 核心线程数
            10,                     // 最大线程数
            60L,                    // 空闲线程存活时间
            TimeUnit.SECONDS,       // 时间单位
            new LinkedBlockingQueue<>(100),  // 任务队列
            Executors.defaultThreadFactory(),  // 线程工厂
            new ThreadPoolExecutor.AbortPolicy()  // 拒绝策略
        );
        
        // 提交任务
        for (int i = 0; i < 10; i++) {
            final int taskId = i;
            fixedPool.execute(() -> {
                System.out.println(Thread.currentThread().getName() + 
                    " 执行任务 " + taskId);
            });
        }
        
        // 关闭线程池
        fixedPool.shutdown();
    }
}
```

### 定时任务

```java
import java.util.concurrent.*;

public class ScheduledTaskExample {
    public static void main(String[] args) {
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(2);
        
        // 延迟 3 秒执行一次
        scheduler.schedule(() -> {
            System.out.println("延迟任务执行");
        }, 3, TimeUnit.SECONDS);
        
        // 延迟 1 秒后，每隔 2 秒执行一次（固定速率）
        scheduler.scheduleAtFixedRate(() -> {
            System.out.println("固定速率任务: " + System.currentTimeMillis());
        }, 1, 2, TimeUnit.SECONDS);
        
        // 延迟 1 秒后，上次执行完成后间隔 2 秒再执行（固定延迟）
        scheduler.scheduleWithFixedDelay(() -> {
            System.out.println("固定延迟任务: " + System.currentTimeMillis());
            try {
                Thread.sleep(1000);  // 模拟任务耗时
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }, 1, 2, TimeUnit.SECONDS);
    }
}
```

## 并发工具类

### CountDownLatch

等待多个线程完成。

```java
import java.util.concurrent.CountDownLatch;

public class CountDownLatchExample {
    public static void main(String[] args) throws InterruptedException {
        int threadCount = 5;
        CountDownLatch latch = new CountDownLatch(threadCount);
        
        for (int i = 0; i < threadCount; i++) {
            final int taskId = i;
            new Thread(() -> {
                System.out.println("任务 " + taskId + " 开始");
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("任务 " + taskId + " 完成");
                latch.countDown();  // 计数减 1
            }).start();
        }
        
        latch.await();  // 等待所有任务完成
        System.out.println("所有任务完成");
    }
}
```

### CyclicBarrier

让一组线程互相等待。

```java
import java.util.concurrent.*;

public class CyclicBarrierExample {
    public static void main(String[] args) {
        int threadCount = 3;
        CyclicBarrier barrier = new CyclicBarrier(threadCount, () -> {
            System.out.println("所有线程到达屏障，继续执行");
        });
        
        for (int i = 0; i < threadCount; i++) {
            final int taskId = i;
            new Thread(() -> {
                System.out.println("线程 " + taskId + " 到达屏障");
                try {
                    barrier.await();  // 等待其他线程
                } catch (InterruptedException | BrokenBarrierException e) {
                    e.printStackTrace();
                }
                System.out.println("线程 " + taskId + " 继续执行");
            }).start();
        }
    }
}
```

### Semaphore

控制同时访问资源的线程数。

```java
import java.util.concurrent.Semaphore;

public class SemaphoreExample {
    public static void main(String[] args) {
        // 最多允许 3 个线程同时访问
        Semaphore semaphore = new Semaphore(3);
        
        for (int i = 0; i < 10; i++) {
            final int taskId = i;
            new Thread(() -> {
                try {
                    semaphore.acquire();  // 获取许可
                    System.out.println("线程 " + taskId + " 获得许可");
                    Thread.sleep(2000);
                    System.out.println("线程 " + taskId + " 释放许可");
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    semaphore.release();  // 释放许可
                }
            }).start();
        }
    }
}
```

## 线程安全的集合

```java
import java.util.concurrent.*;

public class ConcurrentCollections {
    public static void main(String[] args) {
        // 线程安全的 HashMap
        ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();
        map.put("key", 1);
        
        // 线程安全的队列
        BlockingQueue<String> queue = new LinkedBlockingQueue<>();
        
        // 线程安全的 List
        CopyOnWriteArrayList<String> list = new CopyOnWriteArrayList<>();
        
        // 线程安全的 Set
        CopyOnWriteArraySet<String> set = new CopyOnWriteArraySet<>();
    }
}
```

## 最佳实践

### 1. 避免死锁

```java
// 不好：可能死锁
synchronized (lock1) {
    synchronized (lock2) {
        // ...
    }
}

// 好：统一加锁顺序
if (System.identityHashCode(lock1) < System.identityHashCode(lock2)) {
    synchronized (lock1) {
        synchronized (lock2) {
            // ...
        }
    }
} else {
    synchronized (lock2) {
        synchronized (lock1) {
            // ...
        }
    }
}
```

### 2. 使用线程池

```java
// 不好：频繁创建线程
for (int i = 0; i < 1000; i++) {
    new Thread(() -> {
        // 任务
    }).start();
}

// 好：使用线程池
ExecutorService executor = Executors.newFixedThreadPool(10);
for (int i = 0; i < 1000; i++) {
    executor.execute(() -> {
        // 任务
    });
}
executor.shutdown();
```

### 3. 使用并发集合

```java
// 不好：使用 Collections.synchronizedMap
Map<String, Integer> map = Collections.synchronizedMap(new HashMap<>());

// 好：使用 ConcurrentHashMap
Map<String, Integer> map = new ConcurrentHashMap<>();
```

## 总结

本文介绍了 Java 多线程编程的核心内容：

- ✅ 线程创建的多种方式
- ✅ 线程的生命周期和状态
- ✅ 线程同步：synchronized、Lock
- ✅ 线程通信：wait/notify、Condition
- ✅ 线程池的使用
- ✅ 并发工具类：CountDownLatch、CyclicBarrier、Semaphore
- ✅ 线程安全的集合

掌握多线程后，继续学习 [IO 流](./io-streams) 和 [函数式编程](./functional-programming)。
