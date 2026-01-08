---
sidebar_position: 18
title: 多线程编程
---

# C++ 多线程编程

C++11 引入了标准线程库，支持跨平台多线程编程。

## 🎯 std::thread

```cpp
#include <thread>
#include <iostream>

void hello() {
    std::cout << "Hello from thread!" << std::endl;
}

int main() {
    std::thread t(hello);
    t.join();  // 等待线程完成

    // Lambda
    std::thread t2([]() {
        std::cout << "Lambda thread" << std::endl;
    });
    t2.join();

    // 带参数
    std::thread t3([](int x) {
        std::cout << "Value: " << x << std::endl;
    }, 42);
    t3.join();

    return 0;
}
```

## 🔒 互斥锁

```cpp
#include <mutex>

std::mutex mtx;
int counter = 0;

void increment() {
    for (int i = 0; i < 1000; i++) {
        std::lock_guard<std::mutex> lock(mtx);
        counter++;
    }
}

// unique_lock (更灵活)
void flexible() {
    std::unique_lock<std::mutex> lock(mtx);
    // 可以手动解锁
    lock.unlock();
    // 再加锁
    lock.lock();
}

// scoped_lock (C++17) - 同时锁多个
std::mutex m1, m2;
void multiLock() {
    std::scoped_lock lock(m1, m2);
}
```

## 🔔 条件变量

```cpp
#include <condition_variable>

std::mutex mtx;
std::condition_variable cv;
bool ready = false;

void worker() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [] { return ready; });
    std::cout << "Worker running" << std::endl;
}

void signal() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        ready = true;
    }
    cv.notify_one();  // 或 notify_all()
}
```

## ⚛️ 原子操作

```cpp
#include <atomic>

std::atomic<int> counter{0};

void increment() {
    for (int i = 0; i < 1000; i++) {
        counter++;  // 原子操作
    }
}

// 原子标志
std::atomic_flag flag = ATOMIC_FLAG_INIT;
```

## 📋 async 和 future

```cpp
#include <future>

int compute() {
    return 42;
}

int main() {
    // 异步执行
    std::future<int> result = std::async(std::launch::async, compute);

    // 获取结果（阻塞）
    int value = result.get();

    // promise
    std::promise<int> prom;
    std::future<int> fut = prom.get_future();

    std::thread t([&prom]() {
        prom.set_value(100);
    });

    std::cout << fut.get() << std::endl;
    t.join();

    return 0;
}
```

## ⚡ 最佳实践

1. **使用 lock_guard/scoped_lock** - 自动管理锁
2. **避免死锁** - 统一锁顺序
3. **使用原子操作** - 简单计数器
4. **使用 async** - 简化异步编程
5. **最小化临界区** - 提高并发性能
