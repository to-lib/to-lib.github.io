---
sidebar_position: 3
title: 基础语法
---

# C 语言基础语法

掌握 C 语言的基础语法是学习的第一步。本文涵盖变量、数据类型、运算符和控制流等核心概念。

## 程序结构

### 第一个 C 程序

```c
#include <stdio.h>  // 预处理指令，包含标准输入输出库

// main 函数是程序入口
int main(void) {
    printf("Hello, World!\n");
    return 0;  // 返回 0 表示程序成功执行
}
```

### 程序组成部分

```c
// 1. 预处理指令
#include <stdio.h>
#include <stdlib.h>
#define PI 3.14159

// 2. 全局变量声明
int globalVar = 100;

// 3. 函数声明（原型）
void greet(const char *name);
int add(int a, int b);

// 4. main 函数
int main(void) {
    // 局部变量
    int x = 10;

    greet("张三");
    printf("10 + 20 = %d\n", add(10, 20));

    return 0;
}

// 5. 函数定义
void greet(const char *name) {
    printf("你好，%s！\n", name);
}

int add(int a, int b) {
    return a + b;
}
```

## 变量和常量

### 变量声明与初始化

```c
#include <stdio.h>

int main(void) {
    // 声明变量
    int age;

    // 声明并初始化
    int count = 0;
    float temperature = 36.5f;
    char grade = 'A';

    // 多个变量声明
    int x, y, z;
    int a = 1, b = 2, c = 3;

    // 赋值
    age = 25;
    x = y = z = 0;  // 链式赋值

    printf("年龄: %d\n", age);
    printf("计数: %d\n", count);
    printf("温度: %.1f\n", temperature);
    printf("等级: %c\n", grade);

    return 0;
}
```

### 常量

```c
#include <stdio.h>

// 使用 #define 定义常量（预处理宏）
#define PI 3.14159
#define MAX_SIZE 100
#define GREETING "Hello"

int main(void) {
    // 使用 const 关键字
    const int DAYS_IN_WEEK = 7;
    const float GRAVITY = 9.8f;
    const char NEWLINE = '\n';

    printf("PI = %f\n", PI);
    printf("一周有 %d 天\n", DAYS_IN_WEEK);
    printf("重力加速度: %.1f m/s²%c", GRAVITY, NEWLINE);

    // const 变量不能修改
    // DAYS_IN_WEEK = 8;  // 错误！

    return 0;
}
```

## 数据类型

### 基本数据类型

```c
#include <stdio.h>
#include <limits.h>
#include <float.h>

int main(void) {
    // 整数类型
    char c = 'A';           // 1 字节
    short s = 1000;         // 至少 2 字节
    int i = 100000;         // 至少 2 字节，通常 4 字节
    long l = 1000000L;      // 至少 4 字节
    long long ll = 1000000000LL;  // 至少 8 字节

    // 无符号类型
    unsigned char uc = 255;
    unsigned int ui = 4000000000U;

    // 浮点类型
    float f = 3.14f;        // 4 字节
    double d = 3.14159265;  // 8 字节
    long double ld = 3.14159265358979L;  // 至少 8 字节

    // 打印类型大小
    printf("char:        %zu 字节, 范围: %d ~ %d\n",
           sizeof(char), CHAR_MIN, CHAR_MAX);
    printf("int:         %zu 字节, 范围: %d ~ %d\n",
           sizeof(int), INT_MIN, INT_MAX);
    printf("long:        %zu 字节\n", sizeof(long));
    printf("long long:   %zu 字节\n", sizeof(long long));
    printf("float:       %zu 字节, 精度: %d 位\n",
           sizeof(float), FLT_DIG);
    printf("double:      %zu 字节, 精度: %d 位\n",
           sizeof(double), DBL_DIG);

    return 0;
}
```

### 类型转换

```c
#include <stdio.h>

int main(void) {
    // 隐式转换（自动类型提升）
    int i = 10;
    float f = 3.5f;
    double result = i + f;  // i 被提升为 float，结果是 double

    printf("10 + 3.5 = %f\n", result);

    // 显式转换（强制类型转换）
    double pi = 3.14159;
    int intPart = (int)pi;  // 截断小数部分
    printf("PI 的整数部分: %d\n", intPart);

    // 整数除法 vs 浮点除法
    int a = 7, b = 3;
    printf("7 / 3 = %d (整数除法)\n", a / b);
    printf("7 / 3 = %.2f (浮点除法)\n", (double)a / b);

    // 字符和整数
    char ch = 'A';
    printf("'A' 的 ASCII 码: %d\n", ch);
    printf("ASCII 66 是: %c\n", (char)66);

    return 0;
}
```

## 运算符

### 算术运算符

```c
#include <stdio.h>

int main(void) {
    int a = 17, b = 5;

    printf("a = %d, b = %d\n", a, b);
    printf("a + b = %d\n", a + b);   // 加法: 22
    printf("a - b = %d\n", a - b);   // 减法: 12
    printf("a * b = %d\n", a * b);   // 乘法: 85
    printf("a / b = %d\n", a / b);   // 整数除法: 3
    printf("a %% b = %d\n", a % b);  // 取模: 2

    // 自增和自减
    int x = 5;
    printf("\n自增自减运算:\n");
    printf("x = %d\n", x);
    printf("x++ = %d (先用后加)\n", x++);
    printf("x = %d\n", x);
    printf("++x = %d (先加后用)\n", ++x);
    printf("x = %d\n", x);

    // 复合赋值运算符
    int n = 10;
    n += 5;   // n = n + 5
    printf("\nn += 5: %d\n", n);
    n -= 3;   // n = n - 3
    printf("n -= 3: %d\n", n);
    n *= 2;   // n = n * 2
    printf("n *= 2: %d\n", n);
    n /= 4;   // n = n / 4
    printf("n /= 4: %d\n", n);
    n %= 3;   // n = n % 3
    printf("n %%= 3: %d\n", n);

    return 0;
}
```

### 关系运算符

```c
#include <stdio.h>

int main(void) {
    int a = 10, b = 20;

    printf("a = %d, b = %d\n\n", a, b);
    printf("a == b: %d\n", a == b);  // 等于: 0 (false)
    printf("a != b: %d\n", a != b);  // 不等于: 1 (true)
    printf("a > b:  %d\n", a > b);   // 大于: 0
    printf("a < b:  %d\n", a < b);   // 小于: 1
    printf("a >= b: %d\n", a >= b);  // 大于等于: 0
    printf("a <= b: %d\n", a <= b);  // 小于等于: 1

    return 0;
}
```

### 逻辑运算符

```c
#include <stdio.h>

int main(void) {
    int a = 1, b = 0;  // 非零为真，零为假

    printf("a = %d (真), b = %d (假)\n\n", a, b);
    printf("a && b (与): %d\n", a && b);  // 0
    printf("a || b (或): %d\n", a || b);  // 1
    printf("!a (非):     %d\n", !a);      // 0
    printf("!b (非):     %d\n", !b);      // 1

    // 短路求值
    printf("\n短路求值:\n");
    int x = 5;
    // && 短路：第一个为假时，不计算第二个
    if (0 && (x = 10)) {
        // 不会执行
    }
    printf("x = %d (第二个表达式未执行)\n", x);

    // || 短路：第一个为真时，不计算第二个
    if (1 || (x = 20)) {
        // x 不会变成 20
    }
    printf("x = %d (第二个表达式未执行)\n", x);

    return 0;
}
```

### 位运算符

```c
#include <stdio.h>

int main(void) {
    unsigned char a = 0b00111100;  // 60
    unsigned char b = 0b00001101;  // 13

    printf("a = %d (0b00111100)\n", a);
    printf("b = %d (0b00001101)\n\n", b);

    printf("a & b  = %d  (按位与)\n", a & b);   // 12 = 0b00001100
    printf("a | b  = %d  (按位或)\n", a | b);   // 61 = 0b00111101
    printf("a ^ b  = %d  (按位异或)\n", a ^ b); // 49 = 0b00110001
    printf("~a     = %d  (按位取反)\n", (unsigned char)~a);  // 195
    printf("a << 2 = %d  (左移2位)\n", a << 2); // 240
    printf("a >> 2 = %d  (右移2位)\n", a >> 2); // 15

    return 0;
}
```

### 其他运算符

```c
#include <stdio.h>

int main(void) {
    // sizeof 运算符
    printf("sizeof(int) = %zu\n", sizeof(int));
    printf("sizeof(double) = %zu\n", sizeof(double));

    int arr[10];
    printf("sizeof(arr) = %zu\n", sizeof(arr));
    printf("数组元素个数: %zu\n", sizeof(arr) / sizeof(arr[0]));

    // 条件（三元）运算符
    int a = 10, b = 20;
    int max = (a > b) ? a : b;
    printf("\nmax(%d, %d) = %d\n", a, b, max);

    // 逗号运算符
    int x = (1, 2, 3);  // x = 3（最后一个表达式的值）
    printf("x = %d\n", x);

    return 0;
}
```

## 控制流

### if-else 语句

```c
#include <stdio.h>

int main(void) {
    int score = 85;

    // 简单 if
    if (score >= 60) {
        printf("及格了！\n");
    }

    // if-else
    if (score >= 90) {
        printf("优秀\n");
    } else {
        printf("继续努力\n");
    }

    // if-else if-else
    if (score >= 90) {
        printf("等级: A\n");
    } else if (score >= 80) {
        printf("等级: B\n");
    } else if (score >= 70) {
        printf("等级: C\n");
    } else if (score >= 60) {
        printf("等级: D\n");
    } else {
        printf("等级: F\n");
    }

    // 嵌套 if
    int age = 20;
    if (score >= 60) {
        if (age >= 18) {
            printf("成年人且及格\n");
        }
    }

    return 0;
}
```

### switch 语句

```c
#include <stdio.h>

int main(void) {
    char grade = 'B';

    switch (grade) {
        case 'A':
            printf("优秀！继续保持\n");
            break;
        case 'B':
            printf("良好！还有进步空间\n");
            break;
        case 'C':
            printf("中等，需要加油\n");
            break;
        case 'D':
            printf("及格，但需努力\n");
            break;
        case 'F':
            printf("不及格，需要补考\n");
            break;
        default:
            printf("无效的等级\n");
            break;
    }

    // 多个 case 共享代码
    int day = 6;
    switch (day) {
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
            printf("工作日\n");
            break;
        case 6:
        case 7:
            printf("周末\n");
            break;
        default:
            printf("无效的日期\n");
    }

    return 0;
}
```

### while 循环

```c
#include <stdio.h>

int main(void) {
    // while 循环
    int i = 1;
    while (i <= 5) {
        printf("%d ", i);
        i++;
    }
    printf("\n");

    // 计算 1+2+...+100
    int sum = 0;
    int n = 1;
    while (n <= 100) {
        sum += n;
        n++;
    }
    printf("1+2+...+100 = %d\n", sum);

    return 0;
}
```

### do-while 循环

```c
#include <stdio.h>

int main(void) {
    // do-while 至少执行一次
    int i = 1;
    do {
        printf("%d ", i);
        i++;
    } while (i <= 5);
    printf("\n");

    // 输入验证示例
    int num;
    do {
        printf("请输入一个正数: ");
        scanf("%d", &num);
    } while (num <= 0);
    printf("你输入了: %d\n", num);

    return 0;
}
```

### for 循环

```c
#include <stdio.h>

int main(void) {
    // 基本 for 循环
    for (int i = 1; i <= 5; i++) {
        printf("%d ", i);
    }
    printf("\n");

    // 倒序
    for (int i = 5; i >= 1; i--) {
        printf("%d ", i);
    }
    printf("\n");

    // 步长为 2
    for (int i = 0; i <= 10; i += 2) {
        printf("%d ", i);
    }
    printf("\n");

    // 嵌套循环 - 打印乘法表
    printf("\n乘法表:\n");
    for (int i = 1; i <= 9; i++) {
        for (int j = 1; j <= i; j++) {
            printf("%d×%d=%-2d ", j, i, i * j);
        }
        printf("\n");
    }

    return 0;
}
```

### break 和 continue

```c
#include <stdio.h>

int main(void) {
    // break - 跳出循环
    printf("break 示例:\n");
    for (int i = 1; i <= 10; i++) {
        if (i == 5) {
            break;  // 遇到 5 时跳出
        }
        printf("%d ", i);
    }
    printf("\n");  // 输出: 1 2 3 4

    // continue - 跳过本次迭代
    printf("\ncontinue 示例:\n");
    for (int i = 1; i <= 10; i++) {
        if (i % 2 == 0) {
            continue;  // 跳过偶数
        }
        printf("%d ", i);
    }
    printf("\n");  // 输出: 1 3 5 7 9

    // 在嵌套循环中使用 break
    printf("\n嵌套循环中的 break:\n");
    for (int i = 1; i <= 3; i++) {
        for (int j = 1; j <= 3; j++) {
            if (j == 2) {
                break;  // 只跳出内层循环
            }
            printf("(%d, %d) ", i, j);
        }
        printf("\n");
    }

    return 0;
}
```

### goto 语句（慎用）

```c
#include <stdio.h>

int main(void) {
    // goto 通常不推荐使用，但在某些情况下可以简化代码
    // 例如：跳出多层嵌套循环

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            if (i == 5 && j == 5) {
                goto end;  // 跳出所有循环
            }
            printf("(%d,%d) ", i, j);
        }
        printf("\n");
    }

end:
    printf("\n循环结束于 (5,5)\n");

    return 0;
}
```

## 输入输出

### printf 格式化输出

```c
#include <stdio.h>

int main(void) {
    // 基本格式说明符
    printf("整数: %d\n", 42);
    printf("无符号整数: %u\n", 42U);
    printf("长整数: %ld\n", 42L);
    printf("浮点数: %f\n", 3.14);
    printf("双精度: %lf\n", 3.14159265);
    printf("字符: %c\n", 'A');
    printf("字符串: %s\n", "Hello");
    printf("指针地址: %p\n", (void*)&main);

    // 进制输出
    int n = 255;
    printf("\n进制输出:\n");
    printf("十进制: %d\n", n);
    printf("八进制: %o\n", n);
    printf("十六进制: %x (小写)\n", n);
    printf("十六进制: %X (大写)\n", n);

    // 宽度和精度
    printf("\n宽度和精度:\n");
    printf("[%10d]\n", 42);      // 右对齐，宽度 10
    printf("[%-10d]\n", 42);     // 左对齐，宽度 10
    printf("[%010d]\n", 42);     // 零填充
    printf("[%.2f]\n", 3.14159); // 2 位小数
    printf("[%10.2f]\n", 3.14159);  // 宽度 10，2 位小数

    // 科学计数法
    printf("\n科学计数法:\n");
    printf("%e\n", 12345.6789);  // 1.234568e+04
    printf("%E\n", 12345.6789);  // 1.234568E+04
    printf("%g\n", 0.000123);    // 自动选择格式

    return 0;
}
```

### scanf 格式化输入

```c
#include <stdio.h>

int main(void) {
    int age;
    float height;
    char name[50];
    char initial;

    // 读取整数
    printf("请输入年龄: ");
    scanf("%d", &age);

    // 读取浮点数
    printf("请输入身高(米): ");
    scanf("%f", &height);

    // 清除输入缓冲区中的换行符
    while (getchar() != '\n');

    // 读取字符串（不含空格）
    printf("请输入姓名: ");
    scanf("%49s", name);  // 限制长度防止溢出

    // 读取单个字符
    printf("请输入姓名首字母: ");
    scanf(" %c", &initial);  // 注意空格，跳过空白字符

    printf("\n--- 您的信息 ---\n");
    printf("姓名: %s\n", name);
    printf("首字母: %c\n", initial);
    printf("年龄: %d\n", age);
    printf("身高: %.2f 米\n", height);

    return 0;
}
```

### 其他输入输出函数

```c
#include <stdio.h>

int main(void) {
    // putchar 和 getchar - 单字符 I/O
    printf("输入一个字符: ");
    int ch = getchar();
    printf("你输入的是: ");
    putchar(ch);
    putchar('\n');

    // 清除缓冲区
    while (getchar() != '\n');

    // puts 和 gets（gets 已弃用，使用 fgets）
    puts("使用 puts 输出字符串");

    // fgets - 安全读取一行
    char line[100];
    printf("\n输入一行文字: ");
    fgets(line, sizeof(line), stdin);
    printf("你输入的是: %s", line);

    // fputs - 输出字符串
    fputs("使用 fputs 输出\n", stdout);

    return 0;
}
```

## 注释

```c
#include <stdio.h>

// 这是单行注释

/*
 * 这是多行注释
 * 可以跨越多行
 */

/**
 * @brief 计算两个数的和
 * @param a 第一个加数
 * @param b 第二个加数
 * @return 两数之和
 *
 * 这是文档注释风格（Doxygen）
 */
int add(int a, int b) {
    return a + b;  // 行尾注释
}

int main(void) {
    int result = add(3, 5);
    printf("3 + 5 = %d\n", result);
    return 0;
}
```

## 命名规范

```c
// 推荐的 C 语言命名规范

// 变量：小写字母，下划线分隔
int student_count;
float average_score;

// 常量：全大写，下划线分隔
#define MAX_SIZE 100
const int BUFFER_LENGTH = 256;

// 函数：小写字母，下划线分隔
void calculate_average(void);
int get_student_count(void);

// 类型别名：首字母大写或全大写
typedef struct {
    int x;
    int y;
} Point;

typedef unsigned long DWORD;

// 枚举：全大写
enum Color {
    COLOR_RED,
    COLOR_GREEN,
    COLOR_BLUE
};

// 宏：全大写
#define IS_VALID(x) ((x) > 0)
```

## 最佳实践

```c
#include <stdio.h>
#include <stdbool.h>

int main(void) {
    // 1. 始终初始化变量
    int count = 0;
    float average = 0.0f;
    char buffer[100] = {0};

    // 2. 使用 stdbool.h 的 bool 类型
    bool isValid = true;
    bool hasError = false;

    // 3. 常量使用 const 而非 #define（当适用时）
    const int MAX_RETRIES = 3;

    // 4. 避免魔术数字
    // 不好：if (status == 1)
    // 好：
    const int STATUS_SUCCESS = 1;
    const int STATUS_FAILURE = 0;
    int status = STATUS_SUCCESS;
    if (status == STATUS_SUCCESS) {
        printf("成功\n");
    }

    // 5. 适当使用括号提高可读性
    int a = 5, b = 3, c = 2;
    int result = ((a + b) * c);  // 更清晰

    // 6. 每个 case 都有 break（除非故意 fall through）
    // 7. 检查所有函数返回值
    // 8. 限制行宽（通常 80 或 120 字符）

    printf("计数: %d\n", count);
    printf("是否有效: %s\n", isValid ? "是" : "否");

    return 0;
}
```

熟练掌握这些基础语法后，就可以开始学习函数的使用了！
