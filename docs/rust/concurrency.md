---
sidebar_position: 7
title: 并发编程
---

# 并发编程

Rust 的并发编程无惧数据竞争,编译器在编译时就能捕获并发错误。本文涵盖线程、消息传递、共享状态和异步编程。

## 并发 vs 并行

### 概念区分

- **并发(Concurrency)** - 处理多个任务的能力,任务可能交替执行
- **并行(Parallelism)** - 同时执行多个任务,需要多核支持

```rust
// 并发:任务交替执行
// 时间 ->
// 任务1: ---|  wait  |---
// 任务2:    |---| wait |---

// 并行:任务同时执行
// 时间 ->
// 任务1: ---|---|---|
// 任务2: ---|---|---|
```

## 使用线程

### 创建线程

```rust
use std::thread;
use std::time::Duration;

fn main() {
    // 创建新线程
    thread::spawn(|| {
        for i in 1..10 {
            println!("新线程中的数字 {}!", i);
            thread::sleep(Duration::from_millis(1));
        }
    });
    
    // 主线程继续执行
    for i in 1..5 {
        println!("主线程中的数字 {}!", i);
        thread::sleep(Duration::from_millis(1));
    }
    
    // 主线程结束时,所有线程都会被终止
}
```

### 等待线程结束

```rust
use std::thread;
use std::time::Duration;

fn main() {
    let handle = thread::spawn(|| {
        for i in 1..10 {
            println!("新线程中的数字 {}!", i);
            thread::sleep(Duration::from_millis(1));
        }
    });
    
    for i in 1..5 {
        println!("主线程中的数字 {}!", i);
        thread::sleep(Duration::from_millis(1));
    }
    
    // 等待线程结束
    handle.join().unwrap();
    println!("所有线程执行完毕");
}
```

### 使用 move 闭包

```rust
use std::thread;

fn main() {
    let v = vec![1, 2, 3];
    
    // move 关键字转移所有权
    let handle = thread::spawn(move || {
        println!("向量: {:?}", v);
    });
    
    // println!("{:?}", v);  // 错误:v 已被移动
    
    handle.join().unwrap();
}
```

### 线程返回值

```rust
use std::thread;

fn main() {
    let handle = thread::spawn(|| {
        // 计算并返回结果
        let mut sum = 0;
        for i in 1..=100 {
            sum += i;
        }
        sum
    });
    
    // 获取线程返回值
    let result = handle.join().unwrap();
    println!("结果: {}", result);  // 5050
}
```

### 线程命名

```rust
use std::thread;

fn main() {
    let builder = thread::Builder::new()
        .name("worker-1".to_string())
        .stack_size(4 * 1024 * 1024);  // 4MB 栈
    
    let handle = builder.spawn(|| {
        println!("线程名: {:?}", thread::current().name());
    }).unwrap();
    
    handle.join().unwrap();
}
```

## 消息传递

Rust 哲学:**不要通过共享内存来通信,要通过通信来共享内存**。

### 创建通道

```rust
use std::sync::mpsc;  // multi-producer, single-consumer
use std::thread;

fn main() {
    // 创建通道
    let (tx, rx) = mpsc::channel();
    
    thread::spawn(move || {
        let val = String::from("hi");
        tx.send(val).unwrap();
        // println!("{}", val);  // 错误:val 已被发送
    });
    
    // 接收消息(阻塞)
    let received = rx.recv().unwrap();
    println!("收到: {}", received);
}
```

### recv vs try_recv

```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::channel();
    
    thread::spawn(move || {
        tx.send("hello").unwrap();
    });
    
    // recv():阻塞直到收到消息
    let msg = rx.recv().unwrap();
    println!("收到: {}", msg);
    
    // try_recv():非阻塞,立即返回
    match rx.try_recv() {
        Ok(msg) => println!("收到: {}", msg),
        Err(_) => println!("没有消息"),
    }
}
```

### 发送多个值

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    let (tx, rx) = mpsc::channel();
    
    thread::spawn(move || {
        let vals = vec![
            String::from("hi"),
            String::from("from"),
            String::from("the"),
            String::from("thread"),
        ];
        
        for val in vals {
            tx.send(val).unwrap();
            thread::sleep(Duration::from_secs(1));
        }
    });
    
    // 将 rx 当作迭代器
    for received in rx {
        println!("收到: {}", received);
    }
}
```

### 多个生产者

```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::channel();
    
    // 克隆发送者
    let tx1 = tx.clone();
    
    thread::spawn(move || {
        tx.send(String::from("线程1: hi")).unwrap();
    });
    
    thread::spawn(move || {
        tx1.send(String::from("线程2: hello")).unwrap();
    });
    
    for received in rx {
        println!("收到: {}", received);
    }
}
```

### 有界通道

```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    // 创建容量为3的有界通道
    let (tx, rx) = mpsc::sync_channel(3);
    
    thread::spawn(move || {
        for i in 1..=5 {
            println!("发送: {}", i);
            tx.send(i).unwrap();
            println!("已发送: {}", i);
        }
    });
    
    thread::sleep(std::time::Duration::from_secs(2));
    
    for received in rx {
        println!("收到: {}", received);
    }
}
```

## 共享状态并发

### 互斥器 Mutex

```rust
use std::sync::Mutex;

fn main() {
    let m = Mutex::new(5);
    
    {
        // 获取锁
        let mut num = m.lock().unwrap();
        *num = 6;
    }  // 锁在这里自动释放
    
    println!("m = {:?}", m);
}
```

### 多线程共享 Mutex

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    // Arc: 原子引用计数
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];
    
    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("结果: {}", *counter.lock().unwrap());  // 10
}
```

### Mutex 的 RAII 模式

```rust
use std::sync::Mutex;

fn main() {
    let data = Mutex::new(vec![1, 2, 3]);
    
    {
        let mut v = data.lock().unwrap();
        v.push(4);
        // 锁会在作用域结束时自动释放
    }
    
    println!("{:?}", data);
}
```

### RwLock - 读写锁

```rust
use std::sync::{Arc, RwLock};
use std::thread;

fn main() {
    let data = Arc::new(RwLock::new(vec![1, 2, 3]));
    let mut handles = vec![];
    
    // 多个读线程可以同时访问
    for i in 0..5 {
        let data = Arc::clone(&data);
        let handle = thread::spawn(move || {
            let r = data.read().unwrap();
            println!("线程 {} 读取: {:?}", i, *r);
        });
        handles.push(handle);
    }
    
    // 写线程需要独占访问
    let data_clone = Arc::clone(&data);
    let handle = thread::spawn(move || {
        let mut w = data_clone.write().unwrap();
        w.push(4);
        println!("写入完成");
    });
    handles.push(handle);
    
    for handle in handles {
        handle.join().unwrap();
    }
}
```

### Atomic 原子类型

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

fn main() {
    let counter = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];
    
    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            for _ in 0..1000 {
                // 无锁原子操作
                counter.fetch_add(1, Ordering::SeqCst);
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("结果: {}", counter.load(Ordering::SeqCst));
}
```

## Send 和 Sync Trait

### Send Trait

```rust
// Send: 允许在线程间转移所有权
// 几乎所有类型都实现了 Send
// 例外: Rc<T>, 裸指针

use std::rc::Rc;
use std::thread;

fn main() {
    let rc = Rc::new(5);
    
    // 错误: Rc<T> 没有实现 Send
    // thread::spawn(move || {
    //     println!("{}", rc);
    // });
    
    // 使用 Arc<T> 代替
    use std::sync::Arc;
    let arc = Arc::new(5);
    
    thread::spawn(move || {
        println!("{}", arc);
    });
}
```

### Sync Trait

```rust
// Sync: 允许多个线程同时访问
// 如果 &T 是 Send,那么 T 就是 Sync

// 实现了 Sync:
// - 基本类型(i32, bool等)
// - Arc<T>
// - Mutex<T>

// 未实现 Sync:
// - Rc<T>
// - RefCell<T>
// - Cell<T>
```

## 异步编程基础

### async/await 语法

```rust
// 异步函数
async fn say_hello() {
    println!("Hello from async!");
}

// 异步块
fn main() {
    let fut = async {
        println!("异步块");
    };
    
    // 注意:异步代码需要运行时执行
}
```

### Future Trait

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

// Future 的定义
trait MyFuture {
    type Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}

// Poll 枚举
enum MyPoll<T> {
    Ready(T),
    Pending,
}
```

### Tokio 运行时

需要添加依赖:

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
```

#### 基本使用

```rust
#[tokio::main]
async fn main() {
    println!("Hello");
    say_hello().await;
    println!("World");
}

async fn say_hello() {
    println!("异步 Hello!");
}
```

#### 异步任务

```rust
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    // 并发执行多个任务
    let task1 = tokio::spawn(async {
        sleep(Duration::from_secs(1)).await;
        println!("任务1完成");
    });
    
    let task2 = tokio::spawn(async {
        sleep(Duration::from_secs(1)).await;
        println!("任务2完成");
    });
    
    // 等待所有任务
    let _ = tokio::join!(task1, task2);
}
```

#### 异步 HTTP 示例

```rust
use tokio::net::TcpListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;
    println!("服务器运行在 http://127.0.0.1:8080");
    
    loop {
        let (mut socket, _) = listener.accept().await?;
        
        tokio::spawn(async move {
            let mut buffer = [0; 1024];
            
            match socket.read(&mut buffer).await {
                Ok(_) => {
                    let response = "HTTP/1.1 200 OK\r\n\r\nHello from Tokio!";
                    let _ = socket.write_all(response.as_bytes()).await;
                }
                Err(e) => eprintln!("读取错误: {}", e),
            }
        });
    }
}
```

### 异步并发模式

#### select! 宏

```rust
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    let result = tokio::select! {
        _ = sleep(Duration::from_secs(1)) => {
            "第一个完成"
        }
        _ = sleep(Duration::from_secs(2)) => {
            "第二个完成"
        }
    };
    
    println!("结果: {}", result);
}
```

#### join! 和 try_join

```rust
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    // join!: 等待所有任务完成
    let (a, b) = tokio::join!(
        async { sleep(Duration::from_secs(1)).await; "A" },
        async { sleep(Duration::from_secs(1)).await; "B" }
    );
    println!("结果: {}, {}", a, b);
    
    // try_join!: 任何一个失败则返回
    let result = tokio::try_join!(
        async { Ok::<_, &str>("成功1") },
        async { Ok::<_, &str>("成功2") }
    );
    
    match result {
        Ok((a, b)) => println!("都成功: {}, {}", a, b),
        Err(e) => println!("失败: {}", e),
    }
}
```

#### 超时和取消

```rust
use tokio::time::{sleep, Duration, timeout};

#[tokio::main]
async fn main() {
    // 设置超时
    let result = timeout(
        Duration::from_secs(1),
        async {
            sleep(Duration::from_secs(2)).await;
            "完成"
        }
    ).await;
    
    match result {
        Ok(msg) => println!("及时完成: {}", msg),
        Err(_) => println!("超时!"),
    }
}
```

## 线程池

### 手动实现线程池

```rust
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: mpsc::Sender<Job>,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

impl ThreadPool {
    pub fn new(size: usize) -> ThreadPool {
        assert!(size > 0);
        
        let (sender, receiver) = mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));
        
        let mut workers = Vec::with_capacity(size);
        
        for id in 0..size {
            workers.push(Worker::new(id, Arc::clone(&receiver)));
        }
        
        ThreadPool { workers, sender }
    }
    
    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.sender.send(job).unwrap();
    }
}

struct Worker {
    id: usize,
    thread: thread::JoinHandle<()>,
}

impl Worker {
    fn new(id: usize, receiver: Arc<Mutex<mpsc::Receiver<Job>>>) -> Worker {
        let thread = thread::spawn(move || loop {
            let job = receiver.lock().unwrap().recv().unwrap();
            println!("Worker {} 执行任务", id);
            job();
        });
        
        Worker { id, thread }
    }
}

fn main() {
    let pool = ThreadPool::new(4);
    
    for i in 0..10 {
        pool.execute(move || {
            println!("执行任务 {}", i);
        });
    }
    
    thread::sleep(std::time::Duration::from_secs(2));
}
```

### Rayon 数据并行

```toml
[dependencies]
rayon = "1.7"
```

```rust
use rayon::prelude::*;

fn main() {
    let numbers: Vec<i32> = (1..=1000).collect();
    
    // 并行迭代
    let sum: i32 = numbers.par_iter().sum();
    println!("和: {}", sum);
    
    // 并行过滤和映射
    let result: Vec<_> = numbers
        .par_iter()
        .filter(|&&x| x % 2 == 0)
        .map(|&x| x * x)
        .collect();
    
    println!("前10个结果: {:?}", &result[..10]);
}
```

## 并发模式和最佳实践

### 1. 优先使用消息传递

```rust
use std::sync::mpsc;
use std::thread;

// 好: 使用通道通信
fn good_pattern() {
    let (tx, rx) = mpsc::channel();
    
    thread::spawn(move || {
        tx.send("数据".to_string()).unwrap();
    });
    
    let data = rx.recv().unwrap();
    println!("{}", data);
}
```

### 2. 必要时使用共享状态

```rust
use std::sync::{Arc, Mutex};
use std::thread;

// 当确实需要共享状态时
fn shared_state_pattern() {
    let data = Arc::new(Mutex::new(Vec::new()));
    let data_clone = Arc::clone(&data);
    
    thread::spawn(move || {
        data_clone.lock().unwrap().push(1);
    });
}
```

### 3. 避免死锁

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn avoid_deadlock() {
    let lock1 = Arc::new(Mutex::new(1));
    let lock2 = Arc::new(Mutex::new(2));
    
    // 好: 统一的锁获取顺序
    let lock1_clone = Arc::clone(&lock1);
    let lock2_clone = Arc::clone(&lock2);
    
    thread::spawn(move || {
        let _l1 = lock1_clone.lock().unwrap();
        let _l2 = lock2_clone.lock().unwrap();
    });
    
    // 使用相同的顺序
    let _l1 = lock1.lock().unwrap();
    let _l2 = lock2.lock().unwrap();
}
```

### 4. 使用作用域线程

```rust
use std::thread;

fn main() {
    let mut data = vec![1, 2, 3];
    
    thread::scope(|s| {
        s.spawn(|| {
            println!("数据: {:?}", data);
        });
        
        s.spawn(|| {
            data.push(4);
        });
    });
    
    // 作用域结束后可以继续使用 data
    println!("最终数据: {:?}", data);
}
```

### 5. 选择合适的同步原语

| 场景 | 推荐方案 |
|------|----------|
| 简单计数器 | `AtomicUsize` |
| 小数据保护 | `Mutex<T>` |
| 读多写少 | `RwLock<T>` |
| 线程间通信 | `mpsc::channel` |
| 异步任务 | `tokio::spawn` |

## 性能调优

### 1. 减少锁竞争

```rust
// 不好: 长时间持有锁
fn bad_locking() {
    use std::sync::Mutex;
    let data = Mutex::new(vec![1, 2, 3]);
    
    let mut v = data.lock().unwrap();
    // 执行大量工作...
    v.push(4);
}

// 好: 最小化锁持有时间
fn good_locking() {
    use std::sync::Mutex;
    let data = Mutex::new(vec![1, 2, 3]);
    
    // 执行大量工作...
    
    // 只在必要时获取锁
    data.lock().unwrap().push(4);
}
```

### 2. 使用无锁数据结构

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

fn main() {
    // 原子操作比 Mutex 更快
    let counter = AtomicUsize::new(0);
    
    counter.fetch_add(1, Ordering::Relaxed);
    println!("计数: {}", counter.load(Ordering::Relaxed));
}
```

### 3. 批处理

```rust
use std::sync::mpsc;

fn batch_processing() {
    let (tx, rx) = mpsc::channel();
    
    // 批量发送
    for i in 0..1000 {
        tx.send(i).unwrap();
    }
    
    // 批量处理
    let batch: Vec<_> = rx.try_iter().collect();
    println!("处理 {} 个项目", batch.len());
}
```

## 常见并发问题

### 1. 数据竞争

Rust 的类型系统在编译时防止数据竞争:

```rust
// Rust 编译器会拒绝这段代码
// let mut data = vec![1, 2, 3];
// 
// thread::spawn(|| {
//     data.push(4);  // 错误: 数据竞争
// });
// 
// data.push(5);  // 错误: 数据竞争
```

### 2. 死锁

```rust
// 死锁示例(避免)
use std::sync::Mutex;
use std::thread;

fn potential_deadlock() {
    let lock1 = Mutex::new(1);
    let lock2 = Mutex::new(2);
    
    // 线程1: lock1 -> lock2
    // 线程2: lock2 -> lock1
    // 可能导致死锁
}
```

### 3. 优先级反转

使用优先级继承或优先级天花板协议解决。

## 总结

本文全面介绍了 Rust 的并发编程:

- ✅ 线程创建和管理
- ✅ 消息传递(mpsc 通道)
- ✅ 共享状态(Mutex、RwLock)
- ✅ Send 和 Sync Trait
- ✅ 异步编程基础(async/await)
- ✅ Tokio 异步运行时
- ✅ 线程池和 Rayon
- ✅ 并发模式和最佳实践
- ✅ 性能调优技巧

**关键要点:**

1. **优先使用消息传递**而非共享状态
2. **类型系统保证线程安全**,编译时防止数据竞争
3. **Send 和 Sync** 控制跨线程安全性
4. **异步编程**适用于 I/O 密集型任务
5. **选择合适的同步原语**优化性能

掌握并发编程后,继续学习 [异步编程详解](/docs/rust/async-programming) 和 [生命周期](/docs/rust/lifetimes)。
