---
sidebar_position: 21
title: 调试技巧
---

# C 语言调试技巧

掌握调试技巧，快速定位和修复问题。

## 常见错误类型

### 段错误 (Segmentation Fault)

```c
// 原因1: 空指针解引用
int *p = NULL;
*p = 10;  // 段错误！

// 原因2: 数组越界
int arr[5];
arr[10] = 100;  // 段错误（可能）

// 原因3: 栈溢出
void infinite_recursion(void) {
    infinite_recursion();  // 栈溢出
}

// 原因4: 释放后使用
int *p = malloc(sizeof(int));
free(p);
*p = 10;  // 段错误（可能）
```

### 内存泄漏

```c
void memory_leak(void) {
    int *p = malloc(100 * sizeof(int));
    // 忘记 free(p)
}  // 内存泄漏！

// 修复
void no_leak(void) {
    int *p = malloc(100 * sizeof(int));
    // 使用 p
    free(p);
}
```

### 未初始化变量

```c
int x;  // 未初始化，值是随机的
if (x > 0) {  // 未定义行为
    // ...
}

// 修复
int x = 0;
```

### 缓冲区溢出

```c
char buf[10];
strcpy(buf, "This string is too long");  // 溢出！

// 修复
char buf[10];
strncpy(buf, "This string is too long", sizeof(buf) - 1);
buf[sizeof(buf) - 1] = '\0';
```

## GDB 调试器

### 基本命令

```bash
# 编译时加调试信息
gcc -g program.c -o program

# 启动 GDB
gdb ./program

# 常用命令
(gdb) run                 # 运行程序
(gdb) run arg1 arg2       # 带参数运行
(gdb) break main          # 在 main 设断点
(gdb) break file.c:20     # 在第 20 行设断点
(gdb) break func if x>10  # 条件断点
(gdb) info breakpoints    # 查看断点
(gdb) delete 1            # 删除断点 1
(gdb) next                # 单步（不进入函数）
(gdb) step                # 单步（进入函数）
(gdb) continue            # 继续执行
(gdb) print x             # 打印变量
(gdb) print *arr@10       # 打印数组前 10 个元素
(gdb) backtrace           # 查看调用栈
(gdb) frame 2             # 切换到栈帧 2
(gdb) list                # 显示源码
(gdb) watch x             # 监视变量
(gdb) quit                # 退出
```

### 调试崩溃

```bash
# 生成 core dump
ulimit -c unlimited
./program  # 崩溃后生成 core 文件

# 分析 core dump
gdb ./program core
(gdb) backtrace
```

## Valgrind 内存检测

### 检测内存泄漏

```bash
valgrind --leak-check=full ./program

# 输出示例
==12345== LEAK SUMMARY:
==12345==    definitely lost: 40 bytes in 1 blocks
==12345==    indirectly lost: 0 bytes in 0 blocks
```

### 检测无效访问

```bash
valgrind --track-origins=yes ./program

# 检测未初始化内存使用
# 检测越界访问
# 检测释放后使用
```

### 常见 Valgrind 错误

| 错误类型                                        | 含义               |
| ----------------------------------------------- | ------------------ |
| Invalid read/write                              | 无效内存访问       |
| Use of uninitialised value                      | 使用未初始化变量   |
| Conditional jump depends on uninitialised value | 条件依赖未初始化值 |
| Invalid free                                    | 无效的 free 调用   |
| Mismatched free                                 | malloc/free 不匹配 |

## 打印调试

### 调试宏

```c
#include <stdio.h>

#define DEBUG 1

#if DEBUG
    #define LOG(fmt, ...) \
        fprintf(stderr, "[%s:%d] " fmt "\n", \
                __FILE__, __LINE__, ##__VA_ARGS__)
    #define TRACE() \
        fprintf(stderr, "[TRACE] %s:%d %s()\n", \
                __FILE__, __LINE__, __func__)
#else
    #define LOG(fmt, ...)
    #define TRACE()
#endif

void process(int x) {
    TRACE();
    LOG("x = %d", x);
}

int main(void) {
    LOG("程序启动");
    process(42);
    LOG("程序结束");
    return 0;
}
```

### 十六进制转储

```c
void hex_dump(const void *data, size_t size) {
    const unsigned char *p = data;
    for (size_t i = 0; i < size; i++) {
        printf("%02X ", p[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n");
}

// 使用
int arr[] = {1, 2, 3};
hex_dump(arr, sizeof(arr));
```

## AddressSanitizer

```bash
# 编译时启用
gcc -fsanitize=address -g program.c -o program

# 运行时自动检测:
# - 堆缓冲区溢出
# - 栈缓冲区溢出
# - 全局缓冲区溢出
# - 释放后使用
# - 重复释放
```

## 静态分析

```bash
# GCC 警告
gcc -Wall -Wextra -Werror program.c

# Clang 静态分析
clang --analyze program.c

# Cppcheck
cppcheck --enable=all program.c
```

## 调试检查清单

```
□ 编译时开启所有警告 (-Wall -Wextra)
□ 检查所有 malloc 返回值
□ 确保每个 malloc 对应一个 free
□ 数组访问检查边界
□ 指针使用前检查 NULL
□ 字符串操作使用安全函数 (strncpy, snprintf)
□ 初始化所有变量
□ 使用 Valgrind 检测内存问题
□ 使用 AddressSanitizer 进行测试
```

## 实用技巧

```c
// 1. 断言检查
#include <assert.h>
assert(ptr != NULL);
assert(index < size);

// 2. 安全的内存分配
void *safe_malloc(size_t size) {
    void *p = malloc(size);
    if (p == NULL) {
        fprintf(stderr, "内存分配失败\n");
        exit(1);
    }
    return p;
}

// 3. 边界检查的数组访问
int safe_get(int *arr, int size, int index) {
    assert(index >= 0 && index < size);
    return arr[index];
}

// 4. 释放后置空
#define SAFE_FREE(p) do { free(p); p = NULL; } while(0)
```

熟练掌握调试技巧，让 bug 无处可藏！🔍
