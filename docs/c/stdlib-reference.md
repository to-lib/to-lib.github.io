---
sidebar_position: 17
title: 标准库速查
---

# C 标准库速查

C 标准库常用函数快速参考。

## stdio.h - 输入输出

### 文件操作

| 函数      | 说明         | 示例                                 |
| --------- | ------------ | ------------------------------------ |
| `fopen`   | 打开文件     | `FILE *fp = fopen("file.txt", "r");` |
| `fclose`  | 关闭文件     | `fclose(fp);`                        |
| `fread`   | 二进制读取   | `fread(buf, size, count, fp);`       |
| `fwrite`  | 二进制写入   | `fwrite(buf, size, count, fp);`      |
| `fgets`   | 读取一行     | `fgets(buf, sizeof(buf), fp);`       |
| `fputs`   | 写入字符串   | `fputs("hello", fp);`                |
| `fprintf` | 格式化写入   | `fprintf(fp, "%d", num);`            |
| `fscanf`  | 格式化读取   | `fscanf(fp, "%d", &num);`            |
| `fseek`   | 移动文件指针 | `fseek(fp, 0, SEEK_SET);`            |
| `ftell`   | 获取位置     | `long pos = ftell(fp);`              |
| `rewind`  | 回到开头     | `rewind(fp);`                        |
| `feof`    | 检查文件结束 | `while (!feof(fp))`                  |
| `ferror`  | 检查错误     | `if (ferror(fp))`                    |

### 标准输入输出

```c
printf("格式化输出: %d, %s, %.2f\n", 42, "hello", 3.14);
scanf("%d %s", &num, str);
puts("输出字符串并换行");
gets(str);  // 已弃用，使用 fgets
putchar('A');
int c = getchar();
sprintf(buf, "格式化到字符串: %d", num);
snprintf(buf, sizeof(buf), "安全格式化: %d", num);
sscanf("123 hello", "%d %s", &num, str);
```

### 格式说明符

| 说明符 | 类型         | 示例       |
| ------ | ------------ | ---------- |
| `%d`   | int          | `-42`      |
| `%u`   | unsigned int | `42`       |
| `%ld`  | long         | `123456L`  |
| `%lld` | long long    | `123456LL` |
| `%f`   | float/double | `3.14`     |
| `%e`   | 科学计数法   | `1.23e+10` |
| `%c`   | char         | `'A'`      |
| `%s`   | 字符串       | `"hello"`  |
| `%p`   | 指针         | `0x7fff`   |
| `%x`   | 十六进制     | `ff`       |
| `%o`   | 八进制       | `77`       |
| `%%`   | 百分号       | `%`        |

## stdlib.h - 通用工具

### 内存管理

```c
void *malloc(size_t size);           // 分配内存
void *calloc(size_t n, size_t size); // 分配并清零
void *realloc(void *ptr, size_t size); // 重新分配
void free(void *ptr);                // 释放内存
```

### 类型转换

```c
int atoi(const char *str);           // 字符串转 int
long atol(const char *str);          // 字符串转 long
double atof(const char *str);        // 字符串转 double
long strtol(str, &endptr, base);     // 更安全的转换
double strtod(str, &endptr);         // 更安全的转换
```

### 随机数

```c
srand(time(NULL));        // 设置种子
int r = rand();           // 0 到 RAND_MAX
int r = rand() % 100;     // 0 到 99
```

### 排序和查找

```c
qsort(arr, count, sizeof(arr[0]), compare);  // 快速排序
void *bsearch(&key, arr, count, sizeof(arr[0]), compare); // 二分查找

int compare(const void *a, const void *b) {
    return *(int*)a - *(int*)b;
}
```

### 程序控制

```c
exit(0);           // 正常退出
exit(1);           // 异常退出
abort();           // 异常终止
atexit(cleanup);   // 注册退出函数
system("ls -la");  // 执行系统命令
getenv("PATH");    // 获取环境变量
```

## string.h - 字符串操作

### 字符串函数

| 函数      | 说明       | 示例                          |
| --------- | ---------- | ----------------------------- |
| `strlen`  | 长度       | `size_t len = strlen(s);`     |
| `strcpy`  | 复制       | `strcpy(dest, src);`          |
| `strncpy` | 安全复制   | `strncpy(dest, src, n);`      |
| `strcat`  | 连接       | `strcat(dest, src);`          |
| `strncat` | 安全连接   | `strncat(dest, src, n);`      |
| `strcmp`  | 比较       | `if (strcmp(a, b) == 0)`      |
| `strncmp` | 部分比较   | `strncmp(a, b, n);`           |
| `strchr`  | 查找字符   | `char *p = strchr(s, 'a');`   |
| `strrchr` | 反向查找   | `char *p = strrchr(s, 'a');`  |
| `strstr`  | 查找子串   | `char *p = strstr(s, "sub");` |
| `strtok`  | 分割       | `char *tok = strtok(s, ",");` |
| `strdup`  | 复制(分配) | `char *dup = strdup(s);`      |

### 内存函数

```c
memset(buf, 0, sizeof(buf));       // 填充
memcpy(dest, src, n);              // 复制
memmove(dest, src, n);             // 安全复制(可重叠)
memcmp(buf1, buf2, n);             // 比较
memchr(buf, 'a', n);               // 查找
```

## ctype.h - 字符处理

```c
isalpha(c)   // 字母
isdigit(c)   // 数字
isalnum(c)   // 字母或数字
isspace(c)   // 空白字符
isupper(c)   // 大写
islower(c)   // 小写
ispunct(c)   // 标点
isprint(c)   // 可打印
toupper(c)   // 转大写
tolower(c)   // 转小写
```

## math.h - 数学函数

```c
// 编译时需链接: gcc -lm

// 基本运算
fabs(x)       // 绝对值
sqrt(x)       // 平方根
pow(x, y)     // x 的 y 次方
exp(x)        // e 的 x 次方
log(x)        // 自然对数
log10(x)      // 常用对数

// 取整
ceil(x)       // 向上取整
floor(x)      // 向下取整
round(x)      // 四舍五入
trunc(x)      // 截断

// 三角函数
sin(x), cos(x), tan(x)
asin(x), acos(x), atan(x)
atan2(y, x)   // 计算 y/x 的反正切

// 其他
fmod(x, y)    // 浮点取余
hypot(x, y)   // sqrt(x² + y²)
```

## time.h - 时间日期

```c
time_t now = time(NULL);           // 当前时间戳
struct tm *t = localtime(&now);    // 转本地时间
struct tm *t = gmtime(&now);       // 转 UTC 时间

// struct tm 成员
t->tm_year + 1900  // 年份
t->tm_mon + 1      // 月份 (0-11)
t->tm_mday         // 日 (1-31)
t->tm_hour         // 时 (0-23)
t->tm_min          // 分 (0-59)
t->tm_sec          // 秒 (0-59)
t->tm_wday         // 星期 (0-6)

// 格式化
char buf[64];
strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", t);

// 时间差
clock_t start = clock();
// ... 代码 ...
double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
```

## stdint.h - 固定宽度整数

```c
int8_t, int16_t, int32_t, int64_t     // 有符号
uint8_t, uint16_t, uint32_t, uint64_t // 无符号
intptr_t, uintptr_t                   // 指针大小的整数

INT8_MIN, INT8_MAX, UINT8_MAX         // 范围常量
INT32_MIN, INT32_MAX, UINT32_MAX
```

## stdbool.h - 布尔类型

```c
#include <stdbool.h>

bool flag = true;
bool done = false;

if (flag) { ... }
```

## assert.h - 断言

```c
#include <assert.h>

assert(ptr != NULL);      // 失败时终止程序
assert(n > 0 && "n must be positive");

// 禁用断言
#define NDEBUG
#include <assert.h>
```

## errno.h - 错误处理

```c
#include <errno.h>
#include <string.h>

FILE *fp = fopen("nofile.txt", "r");
if (fp == NULL) {
    printf("错误码: %d\n", errno);
    printf("错误信息: %s\n", strerror(errno));
    perror("fopen");  // 自动打印错误
}
```

这份速查表涵盖了 C 标准库最常用的函数！
