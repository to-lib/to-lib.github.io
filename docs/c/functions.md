---
sidebar_position: 4
title: 函数
---

# C 语言函数

函数是 C 语言程序的基本组成单元，用于封装可重用的代码块。

## 函数基础

### 函数的定义和调用

```c
#include <stdio.h>

// 函数声明（原型）
int add(int a, int b);
void greet(void);

int main(void) {
    greet();
    int sum = add(5, 3);
    printf("5 + 3 = %d\n", sum);
    return 0;
}

// 函数定义
void greet(void) {
    printf("Hello, World!\n");
}

int add(int a, int b) {
    return a + b;
}
```

## 参数传递

### 值传递

```c
#include <stdio.h>

void modify(int x) {
    x = 100;  // 只修改了副本
}

int main(void) {
    int a = 10;
    modify(a);
    printf("a = %d\n", a);  // a 仍然是 10
    return 0;
}
```

### 指针传递

```c
#include <stdio.h>

void modify(int *x) {
    *x = 100;
}

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int main(void) {
    int a = 10;
    modify(&a);
    printf("a = %d\n", a);  // a 变成 100

    int x = 5, y = 10;
    swap(&x, &y);
    printf("x = %d, y = %d\n", x, y);  // 交换了
    return 0;
}
```

### 数组参数

```c
#include <stdio.h>

void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int sumArray(const int arr[], int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

int main(void) {
    int numbers[] = {1, 2, 3, 4, 5};
    int size = sizeof(numbers) / sizeof(numbers[0]);

    printArray(numbers, size);
    printf("Sum: %d\n", sumArray(numbers, size));
    return 0;
}
```

## 递归函数

```c
#include <stdio.h>

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main(void) {
    printf("5! = %d\n", factorial(5));
    printf("fib(10) = %d\n", fibonacci(10));
    return 0;
}
```

## 函数指针

```c
#include <stdio.h>

int add(int a, int b) { return a + b; }
int multiply(int a, int b) { return a * b; }

int main(void) {
    int (*operation)(int, int);

    operation = add;
    printf("10 + 5 = %d\n", operation(10, 5));

    operation = multiply;
    printf("10 * 5 = %d\n", operation(10, 5));

    // 函数指针数组
    int (*ops[])(int, int) = {add, multiply};
    printf("ops[0](3,4) = %d\n", ops[0](3, 4));

    return 0;
}
```

## 可变参数函数

```c
#include <stdio.h>
#include <stdarg.h>

int sum(int count, ...) {
    va_list args;
    va_start(args, count);

    int total = 0;
    for (int i = 0; i < count; i++) {
        total += va_arg(args, int);
    }

    va_end(args);
    return total;
}

int main(void) {
    printf("sum(1,2,3) = %d\n", sum(3, 1, 2, 3));
    printf("sum(1,2,3,4,5) = %d\n", sum(5, 1, 2, 3, 4, 5));
    return 0;
}
```

## static 和 extern

```c
#include <stdio.h>

// 静态局部变量 - 保持值
int counter(void) {
    static int count = 0;
    return ++count;
}

// 静态函数 - 只在本文件可见
static void helper(void) {
    printf("Helper function\n");
}

int main(void) {
    printf("counter: %d\n", counter());  // 1
    printf("counter: %d\n", counter());  // 2
    printf("counter: %d\n", counter());  // 3
    return 0;
}
```

掌握函数后，就可以继续学习数组和字符串了！
