---
sidebar_position: 12
title: 异步编程
---

# 异步编程

Rust 的异步编程提供了高性能的并发能力，无需操作系统线程的开销。

## async/await 基础

### 异步函数

```rust
// 异步函数返回 Future
async fn hello_async() {
    println!("Hello from async!");
}

async fn get_data() -> String {
    "data".to_string()
}

// 使用 async 块
fn main() {
    let future = async {
        let data = get_data().await;
        println!("{}", data);
    };

    // future 需要被执行器运行
}
```

### .await 关键字

```rust
async fn learn_song() -> String {
    "song".to_string()
}

async fn sing_song(song: String) {
    println!("Singing: {}", song);
}

async fn learn_and_sing() {
    // .await 等待 Future 完成
    let song = learn_song().await;
    sing_song(song).await;
}
```

### 异步块

```rust
async fn example() {
    let future = async {
        println!("异步块");
        42
    };

    let result = future.await;
    println!("{}", result);
}
```

## Future Trait

```rust
use std::pin::Pin;
use std::task::{Context, Poll};
use std::future::Future;

struct MyFuture {
    count: u32,
}

impl Future for MyFuture {
    type Output = u32;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.count < 5 {
            self.count += 1;
            cx.waker().wake_by_ref();  // 请求再次轮询
            Poll::Pending
        } else {
            Poll::Ready(self.count)
        }
    }
}
```

### Poll 状态

```rust
enum Poll<T> {
    Ready(T),    // Future 已完成
    Pending,     // Future 未完成，需要再次轮询
}
```

## Tokio 运行时

Tokio 是 Rust 最流行的异步运行时。

### 安装依赖

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
```

### 基本使用

```rust
#[tokio::main]
async fn main() {
    println!("Hello");
    say_world().await;
}

async fn say_world() {
    println!("World");
}
```

### 手动创建运行时

```rust
use tokio::runtime::Runtime;

fn main() {
    let rt = Runtime::new().unwrap();

    rt.block_on(async {
        println!("异步任务");
    });
}
```

## 异步任务

### 创建任务

```rust
use tokio::task;

#[tokio::main]
async fn main() {
    let handle = task::spawn(async {
        println!("异步任务");
        42
    });

    let result = handle.await.unwrap();
    println!("结果: {}", result);
}
```

### 多任务并发

```rust
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    let task1 = task::spawn(async {
        sleep(Duration::from_secs(1)).await;
        println!("任务 1 完成");
    });

    let task2 = task::spawn(async {
        sleep(Duration::from_secs(2)).await;
        println!("任务 2 完成");
    });

    task1.await.unwrap();
    task2.await.unwrap();
}
```

### join! 宏

```rust
use tokio::join;

async fn task1() -> u32 {
    // ...
    1
}

async fn task2() -> u32 {
    // ...
    2
}

#[tokio::main]
async fn main() {
    // 并发执行，等待所有完成
    let (result1, result2) = join!(task1(), task2());
    println!("{}, {}", result1, result2);
}
```

### select! 宏

```rust
use tokio::select;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    let result = select! {
        _ = sleep(Duration::from_secs(1)) => {
            "第一个完成"
        }
        _ = sleep(Duration::from_secs(2)) => {
            "第二个完成"
        }
    };

    println!("{}", result);  // "第一个完成"
}
```

## 超时和取消

### 超时控制

```rust
use tokio::time::{timeout, Duration};

#[tokio::main]
async fn main() {
    let result = timeout(
        Duration::from_secs(1),
        async {
            sleep(Duration::from_secs(2)).await;
            "完成"
        }
    ).await;

    match result {
        Ok(v) => println!("及时完成: {}", v),
        Err(_) => println!("超时!"),
    }
}
```

### 任务取消

```rust
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    let handle = task::spawn(async {
        loop {
            println!("工作中...");
            sleep(Duration::from_millis(100)).await;
        }
    });

    sleep(Duration::from_secs(1)).await;

    // 取消任务
    handle.abort();
}
```

## 异步 I/O

### 异步文件读取

```rust
use tokio::fs::File;
use tokio::io::AsyncReadExt;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let mut file = File::open("test.txt").await?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).await?;
    println!("{}", contents);
    Ok(())
}
```

### 异步文件写入

```rust
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let mut file = File::create("output.txt").await?;
    file.write_all(b"Hello, async world!").await?;
    Ok(())
}
```

### 异步网络编程

```rust
use tokio::net::TcpListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;

    loop {
        let (mut socket, _) = listener.accept().await?;

        tokio::spawn(async move {
            let mut buf = [0; 1024];

            match socket.read(&mut buf).await {
                Ok(n) if n == 0 => return,
                Ok(n) => {
                    if socket.write_all(&buf[0..n]).await.is_err() {
                        return;
                    }
                }
                Err(_) => return,
            }
        });
    }
}
```

## 异步通道

### mpsc 通道

```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel(32);

    tokio::spawn(async move {
        for i in 0..10 {
            if tx.send(i).await.is_err() {
                return;
            }
        }
    });

    while let Some(i) = rx.recv().await {
        println!("收到: {}", i);
    }
}
```

### oneshot 通道

```rust
use tokio::sync::oneshot;

#[tokio::main]
async fn main() {
    let (tx, rx) = oneshot::channel();

    tokio::spawn(async move {
        let _ = tx.send("Hello");
    });

    match rx.await {
        Ok(v) => println!("收到: {}", v),
        Err(_) => println!("发送者已丢弃"),
    }
}
```

## 异步流 (Stream)

```rust
use tokio_stream::{self as stream, StreamExt};

#[tokio::main]
async fn main() {
    let mut stream = stream::iter(vec![1, 2, 3]);

    while let Some(v) = stream.next().await {
        println!("{}", v);
    }
}
```

### 流适配器

```rust
use tokio_stream::{self as stream, StreamExt};

#[tokio::main]
async fn main() {
    let mut stream = stream::iter(vec![1, 2, 3, 4, 5])
        .filter(|x| x % 2 == 0)
        .map(|x| x * 2);

    while let Some(v) = stream.next().await {
        println!("{}", v);
    }
}
```

## async-std

async-std 是另一个流行的异步运行时。

### 安装依赖

```toml
[dependencies]
async-std = "1"
```

### 基本使用

```rust
use async_std::task;

fn main() {
    task::block_on(async {
        println!("Hello from async-std");
    });
}
```

### 异步任务

```rust
use async_std::task;
use std::time::Duration;

fn main() {
    task::block_on(async {
        let handle = task::spawn(async {
            task::sleep(Duration::from_secs(1)).await;
            42
        });

        let result = handle.await;
        println!("{}", result);
    });
}
```

## 最佳实践

### 1. 避免阻塞操作

```rust
// 不好：阻塞线程
#[tokio::main]
async fn main() {
    std::thread::sleep(std::time::Duration::from_secs(1));  // 阻塞！
}

// 好：使用异步睡眠
#[tokio::main]
async fn main() {
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
}
```

### 2. 使用 spawn_blocking 处理 CPU 密集任务

```rust
use tokio::task;

#[tokio::main]
async fn main() {
    let result = task::spawn_blocking(|| {
        // CPU 密集计算
        let mut sum = 0;
        for i in 0..1000000 {
            sum += i;
        }
        sum
    }).await.unwrap();

    println!("{}", result);
}
```

### 3. 使用 Arc 共享数据

```rust
use std::sync::Arc;
use tokio::task;

#[tokio::main]
async fn main() {
    let data = Arc::new(vec![1, 2, 3]);

    let data_clone = Arc::clone(&data);
    let handle = task::spawn(async move {
        println!("{:?}", data_clone);
    });

    handle.await.unwrap();
}
```

### 4. 合理使用超时

```rust
use tokio::time::{timeout, Duration};

async fn fetch_data() -> Result<String, &'static str> {
    // 模拟网络请求
    tokio::time::sleep(Duration::from_secs(2)).await;
    Ok("data".to_string())
}

#[tokio::main]
async fn main() {
    match timeout(Duration::from_secs(1), fetch_data()).await {
        Ok(Ok(data)) => println!("成功: {}", data),
        Ok(Err(e)) => println!("错误: {}", e),
        Err(_) => println!("超时"),
    }
}
```

### 5. 错误处理

```rust
use tokio::fs::File;
use tokio::io::AsyncReadExt;

async fn read_file(path: &str) -> Result<String, std::io::Error> {
    let mut file = File::open(path).await?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).await?;
    Ok(contents)
}

#[tokio::main]
async fn main() {
    match read_file("test.txt").await {
        Ok(contents) => println!("{}", contents),
        Err(e) => eprintln!("错误: {}", e),
    }
}
```

## 常见陷阱

### 1. 忘记 .await

```rust
async fn get_data() -> String {
    "data".to_string()
}

// 错误：future 未被执行
let data = get_data();  // 这只是创建了 Future

// 正确
let data = get_data().await;
```

### 2. 死锁

```rust
use tokio::sync::Mutex;

// 可能死锁
async fn bad_example() {
    let mutex = Mutex::new(0);

    let guard = mutex.lock().await;
    // 长时间持有锁
    some_async_operation().await;  // 可能导致死锁
    drop(guard);
}

// 好的做法
async fn good_example() {
    let mutex = Mutex::new(0);

    {
        let mut guard = mutex.lock().await;
        *guard += 1;
    }  // 尽快释放锁

    some_async_operation().await;
}
```

### 3. 过度使用 spawn

```rust
// 不好：创建太多任务
for i in 0..10000 {
    tokio::spawn(async move {
        // 简单操作
    });
}

// 好：使用 join_all 或流
use futures::future::join_all;

let futures: Vec<_> = (0..10000)
    .map(|i| async move {
        // 操作
    })
    .collect();

join_all(futures).await;
```

## 性能优化

### 1. 选择合适的 buffer 大小

```rust
use tokio::sync::mpsc;

// 太小：频繁阻塞
let (tx, rx) = mpsc::channel(1);

// 太大：内存浪费
let (tx, rx) = mpsc::channel(10000);

// 合适：根据实际情况调整
let (tx, rx) = mpsc::channel(100);
```

### 2. 使用 futures::stream::FuturesUnordered

```rust
use futures::stream::{FuturesUnordered, StreamExt};

#[tokio::main]
async fn main() {
    let mut futures = FuturesUnordered::new();

    for i in 0..10 {
        futures.push(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            i
        });
    }

    while let Some(result) = futures.next().await {
        println!("{}", result);
    }
}
```

### 3. 避免不必要的克隆

```rust
use std::sync::Arc;

// 不好
async fn bad(data: Vec<u8>) {
    tokio::spawn(async move {
        // data 被移动，需要克隆
    });
}

// 好
async fn good(data: Arc<Vec<u8>>) {
    let data_clone = Arc::clone(&data);
    tokio::spawn(async move {
        // 只克隆了 Arc，不是数据
    });
}
```

## 总结

- ✅ async/await 提供简洁的异步编程语法
- ✅ Future 是异步计算的抽象
- ✅ Tokio 是成熟的异步运行时
- ✅ 避免在异步代码中使用阻塞操作
- ✅ 使用超时机制防止任务 hang 住
- ✅ 合理使用并发原语（通道、锁等）
- ✅ 注意异步代码的错误处理

掌握异步编程，能让你的 Rust 应用实现高并发、高性能！
