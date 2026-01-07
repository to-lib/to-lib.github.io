---
sidebar_position: 23
title: 信号处理
---

# 信号处理

信号 (Signal) 是 Unix 类系统中进程间通信的一种机制，用于通知进程发生了某种事件。C 语言通过 `<signal.h>` 提供了处理信号的标准接口。

## 常见信号

| 信号      | 值 (示例) | 说明                  | 默认动作                 |
| :-------- | :-------- | :-------------------- | :----------------------- |
| `SIGINT`  | 2         | 终端中断 (Ctrl+C)     | 终止                     |
| `SIGQUIT` | 3         | 终端退出 (Ctrl+\)     | 终止并生成 Core          |
| `SIGTERM` | 15        | 终止信号 (kill 默认)  | 终止                     |
| `SIGSEGV` | 11        | 无效内存访问 (段错误) | 终止并生成 Core          |
| `SIGFPE`  | 8         | 算术异常 (除零)       | 终止并生成 Core          |
| `SIGKILL` | 9         | 强制终止              | 终止 (**不可捕获/忽略**) |
| `SIGSTOP` | 19        | 暂停执行              | 暂停 (**不可捕获/忽略**) |

## signal() 函数

这是最简单的注册信号处理函数的方法，但其行为在不同系统间可能不一致（C 标准对其定义较宽泛）。

```c
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

// 信号处理函数
void handle_sigint(int sig) {
    printf("\n捕获到信号 %d (SIGINT)，准备退出...\n", sig);
    // 执行清理工作...
    exit(0);
}

int main(void) {
    // 注册信号处理函数
    if (signal(SIGINT, handle_sigint) == SIG_ERR) {
        perror("无法注册 SIGINT");
        return 1;
    }

    printf("按 Ctrl+C 试一下...\n");

    while (1) {
        printf("运行中...\n");
        sleep(1);
    }

    return 0;
}
```

## sigaction() 函数

POSIX 标准推荐使用 `sigaction`，它比 `signal` 更可靠、更强大。

```c
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <string.h>

void handler(int sig) {
    // 注意：信号处理函数中只应调用异步信号安全 (Async-Signal-Safe) 的函数
    // printf 不一定是安全的，这里仅作演示
    write(STDOUT_FILENO, "Signal received\n", 16);
}

int main(void) {
    struct sigaction sa;

    // 清零结构体
    memset(&sa, 0, sizeof(sa));

    // 设置处理函数
    sa.sa_handler = handler;

    // 设置标志:
    // SA_RESTART: 系统调用被信号中断后自动重启
    sa.sa_flags = SA_RESTART;

    // 初始化信号集（在处理信号时屏蔽所有其他信号）
    sigfillset(&sa.sa_mask);

    if (sigaction(SIGINT, &sa, NULL) == -1) {
        perror("sigaction");
        return 1;
    }

    while (1) {
        sleep(1);
    }

    return 0;
}
```

## 异步信号安全

在信号处理函数中，能做的事情是非常有限的。你不能：

- 分配内存 (`malloc`, `free`)
- 进行标准 I/O (`printf`, `scanf`, `fopen`)
- 使用任何使用全局锁的函数

**只能调用 "异步信号安全" 的函数**，如 `write`, `read`, `exit` (是 `_exit`), `signal`, `sigaction` 等。

## 最佳实践

通常，信号处理函数的最佳做法是：只设置一个全局标志位（类型为 `volatile sig_atomic_t`），然后立即返回。主循环检测到该标志位后，在安全的环境下进行处理。

```c
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

// sig_atomic_t 保证读写是原子的
volatile sig_atomic_t stop_flag = 0;

void handler(int sig) {
    stop_flag = 1;
}

int main(void) {
    signal(SIGINT, handler);

    while (!stop_flag) {
        printf("Working...\n");
        sleep(1);
    }

    printf("优雅退出，清理资源...\n");
    return 0;
}
```

## 总结

信号是系统编程中不可或缺的一部分。正确处理信号（特别是优雅退出）是编写健壮 C 程序的重要技能。记住：在 handler 里做得越少越好。
