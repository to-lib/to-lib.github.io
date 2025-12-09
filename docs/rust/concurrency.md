---
sidebar_position: 7
title: 并发编程
---

# 并发编程

Rust 的并发编程无惧数据竞争，编译器在编译时就能捕获并发错误。

## 使用线程

### 创建线程

```rust
use std::thread;
use std::time::Duration;

fn main() {
    thread::spawn(|| {
        for i in 1..10 {
            println!("新线程中的数字 {}!", i);
            thread::sleep(Duration::from_millis(1));
        }
    });
    
    for i in 1..5 {
        println!("主线程中的数字 {}!", i);
        thread::sleep(Duration::from_millis(1));
    }
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
    
    handle.join().unwrap();  // 等待线程结束
}
```

### 使用 move 闭包

```rust
use std::thread;

fn main() {
    let v = vec![1, 2, 3];
    
    let handle = thread::spawn(move || {
        println!("向量: {:?}", v);
    });
    
    // println!("{:?}", v);  // 错误：v 已被移动
    
    handle.join().unwrap();
}
```

## 消息传递

### 创建通道

```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::channel();
    
    thread::spawn(move || {
        let val = String::from("hi");
        tx.send(val).unwrap();
    });
    
    let received = rx.recv().unwrap();
    println!("收到: {}", received);
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
    
    for received in rx {
        println!("收到: {}", received);
    }
}
```

### 克隆发送者

```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::channel();
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

## 共享状态并发

### 互斥器 Mutex

```rust
use std::sync::Mutex;

fn main() {
    let m = Mutex::new(5);
    
    {
        let mut num = m.lock().unwrap();
        *num = 6;
    }  // 锁在这里被释放
    
    println!("m = {:?}", m);
}
```

### 多线程共享 Mutex

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
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
    
    println!("结果: {}", *counter.lock().unwrap());
}
```

### RwLock

```rust
use std::sync::{Arc, RwLock};
use std::thread;

fn main() {
    let data = Arc::new(RwLock::new(vec![1, 2, 3]));
    let mut handles = vec![];
    
    // 读线程
    for _ in 0..5 {
        let data = Arc::clone(&data);
        let handle = thread::spawn(move || {
            let r = data.read().unwrap();
            println!("读取: {:?}", *r);
        });
        handles.push(handle);
    }
    
    // 写线程
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

## Send 和 Sync Trait

### Send

```rust
// Send: 允许在线程间转移所有权
// 几乎所有类型都实现了 Send
// 例外: Rc<T>

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

### Sync

```rust
// Sync: 允许多个线程同时访问
// 如果 &T 是 Send，那么 T 就是 Sync
```

## 线程池

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
```

## 最佳实践

### 1. 优先使用消息传递

```rust
// 好：使用通道
use std::sync::mpsc;

let (tx, rx) = mpsc::channel();
// 通过消息传递数据
```

### 2. 必要时使用共享状态

```rust
// 当确实需要共享状态时，使用 Arc<Mutex<T>>
use std::sync::{Arc, Mutex};

let data = Arc::new(Mutex::new(Vec::new()));
```

### 3. 避免死锁

```rust
// 统一的锁获取顺序
// 使用超时机制
// 使用 try_lock
```

## 总结

本文介绍了 Rust 的并发编程：

- ✅ 创建和管理线程
- ✅ 消息传递（通道）
- ✅ 共享状态（Mutex、Arc）
- ✅ Send 和 Sync Trait
- ✅ 并发最佳实践

掌握并发编程后，继续学习 [智能指针](./smart-pointers)。
