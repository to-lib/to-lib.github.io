---
sidebar_position: 26
title: 实战项目
---

# C++ 实战项目

通过实际项目巩固所学知识。

## 🛠️ 项目一：简易计算器

```cpp
#include <iostream>
#include <string>
#include <sstream>

class Calculator {
public:
    double calculate(const std::string& expr) {
        std::istringstream iss(expr);
        double a, b;
        char op;
        iss >> a >> op >> b;

        switch (op) {
            case '+': return a + b;
            case '-': return a - b;
            case '*': return a * b;
            case '/': return b != 0 ? a / b : 0;
            default: return 0;
        }
    }
};

int main() {
    Calculator calc;
    std::cout << calc.calculate("10 + 5") << std::endl;   // 15
    std::cout << calc.calculate("20 / 4") << std::endl;   // 5
    return 0;
}
```

## 📦 项目二：简易任务管理器

```cpp
#include <vector>
#include <string>
#include <algorithm>

struct Task {
    int id;
    std::string title;
    bool completed = false;
};

class TaskManager {
    std::vector<Task> tasks;
    int nextId = 1;

public:
    void addTask(const std::string& title) {
        tasks.push_back({nextId++, title, false});
    }

    void completeTask(int id) {
        auto it = std::find_if(tasks.begin(), tasks.end(),
            [id](const Task& t) { return t.id == id; });
        if (it != tasks.end()) it->completed = true;
    }

    void listTasks() const {
        for (const auto& t : tasks) {
            std::cout << "[" << (t.completed ? "x" : " ") << "] "
                      << t.id << ": " << t.title << std::endl;
        }
    }
};
```

## 🔗 项目三：线程池

```cpp
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>

class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    bool stop = false;

public:
    ThreadPool(size_t n) {
        for (size_t i = 0; i < n; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        cv.wait(lock, [this] {
                            return stop || !tasks.empty();
                        });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            tasks.emplace(std::forward<F>(f));
        }
        cv.notify_one();
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();
        for (auto& w : workers) w.join();
    }
};
```

## 📋 项目建议

1. **文件管理器** - 使用 `<filesystem>` 操作文件
2. **HTTP 客户端** - 学习网络编程
3. **数据库封装** - 使用 SQLite
4. **游戏开发** - 实现简单游戏逻辑
5. **日志库** - 多线程日志系统

## ⚡ 项目实践建议

- 使用 Git 管理代码
- 使用 CMake 构建项目
- 编写单元测试
- 使用 Clang-Format 格式化代码
- 持续重构和改进
