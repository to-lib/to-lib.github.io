---
sidebar_position: 14
title: 多文件编程
---

# 多文件编程

大型 C 项目需要将代码拆分为多个文件，实现模块化设计。

## 项目结构

```
project/
├── Makefile
├── include/
│   ├── math_utils.h
│   └── string_utils.h
├── src/
│   ├── main.c
│   ├── math_utils.c
│   └── string_utils.c
└── build/
```

## 头文件

### math_utils.h

```c
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

// 函数声明
int add(int a, int b);
int subtract(int a, int b);
int multiply(int a, int b);
double divide(int a, int b);

// 常量
#define PI 3.14159

// 类型定义
typedef struct {
    double x, y;
} Point;

double distance(Point a, Point b);

#endif // MATH_UTILS_H
```

### string_utils.h

```c
#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include <stddef.h>

// 字符串工具函数
char* str_reverse(char *str);
char* str_to_upper(char *str);
char* str_to_lower(char *str);
int str_count_char(const char *str, char c);

#endif // STRING_UTILS_H
```

## 源文件

### math_utils.c

```c
#include "math_utils.h"
#include <math.h>

int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }
int multiply(int a, int b) { return a * b; }

double divide(int a, int b) {
    if (b == 0) return 0;
    return (double)a / b;
}

double distance(Point a, Point b) {
    double dx = b.x - a.x;
    double dy = b.y - a.y;
    return sqrt(dx*dx + dy*dy);
}
```

### string_utils.c

```c
#include "string_utils.h"
#include <string.h>
#include <ctype.h>

char* str_reverse(char *str) {
    int len = strlen(str);
    for (int i = 0; i < len / 2; i++) {
        char temp = str[i];
        str[i] = str[len - 1 - i];
        str[len - 1 - i] = temp;
    }
    return str;
}

char* str_to_upper(char *str) {
    for (char *p = str; *p; p++) {
        *p = toupper(*p);
    }
    return str;
}

char* str_to_lower(char *str) {
    for (char *p = str; *p; p++) {
        *p = tolower(*p);
    }
    return str;
}

int str_count_char(const char *str, char c) {
    int count = 0;
    while (*str) {
        if (*str++ == c) count++;
    }
    return count;
}
```

### main.c

```c
#include <stdio.h>
#include "math_utils.h"
#include "string_utils.h"

int main(void) {
    // 使用 math_utils
    printf("10 + 5 = %d\n", add(10, 5));
    printf("10 / 3 = %.2f\n", divide(10, 3));

    Point a = {0, 0};
    Point b = {3, 4};
    printf("距离: %.2f\n", distance(a, b));

    // 使用 string_utils
    char str[] = "Hello World";
    printf("原始: %s\n", str);
    printf("大写: %s\n", str_to_upper(str));

    char str2[] = "Hello World";
    printf("反转: %s\n", str_reverse(str2));

    return 0;
}
```

## Makefile

```makefile
CC = gcc
CFLAGS = -Wall -Wextra -g -I./include
LDFLAGS = -lm

SRC_DIR = src
BUILD_DIR = build
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
TARGET = $(BUILD_DIR)/program

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
```

## extern 关键字

```c
// config.c - 定义全局变量
int global_count = 0;
const char *app_name = "MyApp";

// main.c - 使用 extern 声明
extern int global_count;
extern const char *app_name;

int main(void) {
    printf("%s: count = %d\n", app_name, global_count);
    global_count++;
    return 0;
}
```

## static 关键字

```c
// file1.c
static int private_var = 10;  // 只在本文件可见

static void helper(void) {    // 只在本文件可见
    printf("helper\n");
}

void public_func(void) {      // 可被其他文件调用
    helper();
}
```

## 编译命令

```bash
# 分步编译
gcc -c -I./include src/math_utils.c -o build/math_utils.o
gcc -c -I./include src/string_utils.c -o build/string_utils.o
gcc -c -I./include src/main.c -o build/main.o

# 链接
gcc build/*.o -o build/program -lm

# 或一步完成
gcc -I./include src/*.c -o program -lm

# 使用 Makefile
make
./build/program
make clean
```

多文件编程是构建大型项目的基础！
