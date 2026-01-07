---
sidebar_position: 19
title: 可变参数函数
---

# 可变参数函数

C 语言支持接受不定数量参数的函数，最著名的例子就是 `printf`。这一特性通过 `<stdarg.h>` 头文件实现。

## 基本概念

可变参数函数至少需要一个固定参数，后面跟省略号 `...`。

```c
// 原型示例
int printf(const char *format, ...);
```

## <stdarg.h> 宏

| 宏                         | 描述                                 |
| :------------------------- | :----------------------------------- |
| `va_list`                  | 存储参数信息的类型                   |
| `va_start(ap, last_fixed)` | 初始化 `va_list`，指向第一个可变参数 |
| `va_arg(ap, type)`         | 获取下一个参数，并移动指针           |
| `va_end(ap)`               | 清理 `va_list`                       |
| `va_copy(dest, src)`       | 复制 `va_list` (C99)                 |

## 示例实现

### 简单的求和函数

这个函数计算任意数量整数的和。注意我们需要某种方式知道参数的数量或结束位置。这里我们把第一个参数作为计数器。

```c
#include <stdio.h>
#include <stdarg.h>

int sum(int count, ...) {
    va_list ap;
    int total = 0;

    // 1. 初始化，count 是最后一个固定参数
    va_start(ap, count);

    for (int i = 0; i < count; i++) {
        // 2. 获取下一个 int 类型的参数
        total += va_arg(ap, int);
    }

    // 3. 清理
    va_end(ap);

    return total;
}

int main(void) {
    printf("Sum: %d\n", sum(3, 10, 20, 30)); // 输出 60
    return 0;
}
```

### 自定义打印函数 (vprintf)

如果你想包装 `printf`，你需要使用 `vprintf` 系列函数，它们接受 `va_list` 而不是 `...`。

```c
#include <stdio.h>
#include <stdarg.h>

void my_log(const char *level, const char *format, ...) {
    va_list ap;

    printf("[%s] ", level);

    va_start(ap, format);

    // vprintf 接受 va_list
    vprintf(format, ap);

    va_end(ap);

    printf("\n");
}

int main(void) {
    my_log("INFO", "System initialized at %d%%", 100);
    my_log("ERROR", "File not found: %s", "config.ini");
    return 0;
}
```

## 注意事项

1.  **类型提升**：在可变参数中，`char` 和 `short` 会被提升为 `int`，`float` 会被提升为 `double`。

    - 使用 `va_arg(ap, int)` 来获取 `char` 或 `short`。
    - 使用 `va_arg(ap, double)` 来获取 `float`。

2.  **结束标志**：函数本身不知道参数有多少个，必须通过某种约定的方式告知：

    - **固定参数计数**：如上面的 `sum(int count, ...)`。
    - **特定结束值 (Sentinel)**：如指针列表以 `NULL` 结尾。
    - **格式字符串**：如 `printf` 通过分析 `%d` 等占位符个数来判断。

3.  **类型安全**：编译器通常无法检查可变参数的类型是否匹配，这很容易导致运行时错误。现代编译器可以通过 `__attribute__((format(printf, x, y)))` 来检查类 `printf` 函数的格式字符串。

4.  **多次遍历**：如果你需要遍历参数列表多次，必须在每次使用 `va_start` 之前调用 `va_end`，或者使用 `va_copy` 保存副本。

## 总结

可变参数函数提供了极大的灵活性，是构建通用接口（如日志库、格式化工具）的利器，但使用时需倍加小心，确保类型匹配和参数边界的安全。
