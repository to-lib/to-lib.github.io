---
sidebar_position: 18
title: C11/C17 新特性
---

# C11/C17 新特性

C 语言标准持续演进，C11 和 C17 引入了许多现代化特性。

## 编译器支持

```bash
# 使用 C11 标准
gcc -std=c11 program.c -o program

# 使用 C17 标准
gcc -std=c17 program.c -o program

# 查看当前标准
gcc -dM -E - < /dev/null | grep __STDC_VERSION__
```

## \_Static_assert - 静态断言

```c
#include <stdint.h>

// 编译时检查，失败则编译报错
_Static_assert(sizeof(int) >= 4, "int must be at least 4 bytes");
_Static_assert(sizeof(void*) == 8, "This code requires 64-bit system");

typedef struct {
    int id;
    char name[32];
} Record;

_Static_assert(sizeof(Record) == 36, "Record size mismatch");

// C23 简化语法
// static_assert(sizeof(int) == 4);
```

## \_Generic - 泛型选择

```c
#include <stdio.h>
#include <math.h>

// 根据类型自动选择函数
#define abs_val(x) _Generic((x), \
    int: abs, \
    long: labs, \
    float: fabsf, \
    double: fabs, \
    default: fabs \
)(x)

#define print_val(x) _Generic((x), \
    int: printf("%d\n", x), \
    double: printf("%f\n", x), \
    char*: printf("%s\n", x), \
    default: printf("unknown type\n") \
)

#define type_name(x) _Generic((x), \
    int: "int", \
    float: "float", \
    double: "double", \
    char*: "string", \
    default: "unknown" \
)

int main(void) {
    printf("|%d| = %d\n", -5, abs_val(-5));
    printf("|%f| = %f\n", -3.14, abs_val(-3.14));

    printf("Type: %s\n", type_name(42));
    printf("Type: %s\n", type_name(3.14));
    printf("Type: %s\n", type_name("hello"));

    return 0;
}
```

## \_Noreturn - 不返回函数

```c
#include <stdlib.h>
#include <stdnoreturn.h>

// 标记函数永不返回
noreturn void fatal_error(const char *msg) {
    fprintf(stderr, "Fatal: %s\n", msg);
    exit(1);
}

// 或使用属性
_Noreturn void panic(void) {
    abort();
}

int main(void) {
    if (0) {
        fatal_error("Something went wrong");
    }
    return 0;
}
```

## \_Alignof 和 \_Alignas - 对齐

```c
#include <stdio.h>
#include <stdalign.h>

int main(void) {
    // 获取类型对齐要求
    printf("int 对齐: %zu\n", alignof(int));
    printf("double 对齐: %zu\n", alignof(double));

    // 指定对齐
    alignas(16) int aligned_arr[4];
    alignas(64) char cache_line[64];

    printf("aligned_arr 地址: %p\n", (void*)aligned_arr);
    printf("cache_line 地址: %p\n", (void*)cache_line);

    // 结构体对齐
    struct alignas(32) AlignedStruct {
        int x;
        int y;
    };

    printf("AlignedStruct 对齐: %zu\n", alignof(struct AlignedStruct));

    return 0;
}
```

## \_Atomic - 原子操作

```c
#include <stdio.h>
#include <stdatomic.h>
#include <threads.h>

// 原子变量
atomic_int counter = 0;
atomic_flag lock = ATOMIC_FLAG_INIT;

void increment(void) {
    // 原子自增
    atomic_fetch_add(&counter, 1);
}

void atomic_example(void) {
    atomic_int val = ATOMIC_VAR_INIT(0);

    // 原子操作
    atomic_store(&val, 10);
    int v = atomic_load(&val);
    atomic_fetch_add(&val, 5);
    atomic_fetch_sub(&val, 3);

    // 比较并交换
    int expected = 12;
    atomic_compare_exchange_strong(&val, &expected, 100);

    printf("val = %d\n", atomic_load(&val));
}

// 自旋锁
void spinlock_acquire(void) {
    while (atomic_flag_test_and_set(&lock)) {
        // 自旋等待
    }
}

void spinlock_release(void) {
    atomic_flag_clear(&lock);
}
```

## \_Thread_local - 线程局部存储

```c
#include <stdio.h>
#include <threads.h>

// 每个线程有独立副本
_Thread_local int thread_id = 0;

int thread_func(void *arg) {
    thread_id = *(int*)arg;
    printf("Thread %d: thread_id = %d\n", thread_id, thread_id);
    return 0;
}
```

## 匿名结构体和联合体

```c
#include <stdio.h>

struct Point3D {
    union {
        struct { float x, y, z; };  // 匿名结构体
        float coords[3];             // 数组访问
    };
};

int main(void) {
    struct Point3D p = {.x = 1.0, .y = 2.0, .z = 3.0};

    printf("Point: (%.1f, %.1f, %.1f)\n", p.x, p.y, p.z);
    printf("Array: [%.1f, %.1f, %.1f]\n",
           p.coords[0], p.coords[1], p.coords[2]);

    return 0;
}
```

## 快速退出函数

```c
#include <stdlib.h>

void cleanup(void) {
    printf("Normal cleanup\n");
}

void quick_cleanup(void) {
    printf("Quick cleanup\n");
}

int main(void) {
    atexit(cleanup);           // 正常退出时调用
    at_quick_exit(quick_cleanup);  // 快速退出时调用

    // exit(0);       // 调用 cleanup
    // quick_exit(0); // 调用 quick_cleanup

    return 0;
}
```

## Unicode 支持

```c
#include <stdio.h>
#include <uchar.h>

int main(void) {
    // UTF-8 字符串 (C11)
    const char *utf8 = u8"你好，世界！";

    // UTF-16 字符串
    const char16_t *utf16 = u"Hello";

    // UTF-32 字符串
    const char32_t *utf32 = U"Hello";

    printf("%s\n", utf8);

    return 0;
}
```

## 边界检查函数 (可选)

```c
#define __STDC_WANT_LIB_EXT1__ 1
#include <string.h>
#include <stdio.h>

// 安全版本函数 (如果支持)
#ifdef __STDC_LIB_EXT1__
void safe_copy_example(void) {
    char dest[10];

    // 安全复制，防止缓冲区溢出
    strcpy_s(dest, sizeof(dest), "Hello");

    // 安全格式化
    char buf[50];
    sprintf_s(buf, sizeof(buf), "Value: %d", 42);
}
#endif
```

## C17 改进

C17 (C18) 主要是 C11 的 bug 修复版本，没有新增主要特性：

- 修复了 `__STDC_VERSION__` 宏的值
- 澄清了一些未定义行为
- 改进了标准文档

```c
#include <stdio.h>

int main(void) {
    #if __STDC_VERSION__ >= 201710L
        printf("C17 或更高版本\n");
    #elif __STDC_VERSION__ >= 201112L
        printf("C11\n");
    #elif __STDC_VERSION__ >= 199901L
        printf("C99\n");
    #else
        printf("C89/C90\n");
    #endif

    return 0;
}
```

了解现代 C 标准，编写更安全、更高效的代码！
