---
sidebar_position: 9
title: 文件操作
---

# 文件操作

C 语言提供了强大的文件操作功能，支持文本文件和二进制文件的读写。

## 文件基础

```mermaid
graph LR
    App[Application] -->|fwrite| UBuf[User Buffer]
    UBuf -->|Flush| KBuf[Kernel Buffer]
    KBuf -->|Sync| Disk[Hard Disk]

    style App fill:#ff9999
    style UBuf fill:#99ccff
    style KBuf fill:#f9f
    style Disk fill:#e0e0e0
```

### 打开和关闭文件

```c
#include <stdio.h>

int main(void) {
    FILE *fp;

    // 打开文件（写模式）
    fp = fopen("test.txt", "w");
    if (fp == NULL) {
        perror("无法打开文件");
        return 1;
    }

    fprintf(fp, "Hello, File!\n");

    // 关闭文件
    fclose(fp);

    printf("文件操作完成\n");
    return 0;
}
```

### 文件打开模式

| 模式 | 说明                   |
| ---- | ---------------------- |
| `r`  | 只读（文件必须存在）   |
| `w`  | 写入（创建或清空文件） |
| `a`  | 追加（末尾写入）       |
| `r+` | 读写（文件必须存在）   |
| `w+` | 读写（创建或清空）     |
| `a+` | 读写追加               |
| `rb` | 二进制读               |
| `wb` | 二进制写               |

## 文本文件读写

### 写入文件

```c
#include <stdio.h>

int main(void) {
    FILE *fp = fopen("output.txt", "w");
    if (fp == NULL) return 1;

    // fprintf - 格式化写入
    fprintf(fp, "姓名: %s\n", "张三");
    fprintf(fp, "年龄: %d\n", 25);

    // fputs - 写入字符串
    fputs("这是一行文字\n", fp);

    // fputc - 写入字符
    fputc('A', fp);
    fputc('\n', fp);

    fclose(fp);
    printf("写入完成\n");
    return 0;
}
```

### 读取文件

```c
#include <stdio.h>

int main(void) {
    FILE *fp = fopen("output.txt", "r");
    if (fp == NULL) {
        perror("打开文件失败");
        return 1;
    }

    char line[256];

    // fgets - 读取一行
    printf("--- fgets 读取 ---\n");
    while (fgets(line, sizeof(line), fp) != NULL) {
        printf("%s", line);
    }

    // 重置文件指针到开头
    rewind(fp);

    // fgetc - 逐字符读取
    printf("\n--- fgetc 读取（前10个字符）---\n");
    for (int i = 0; i < 10; i++) {
        int c = fgetc(fp);
        if (c == EOF) break;
        putchar(c);
    }
    printf("\n");

    fclose(fp);
    return 0;
}
```

### fscanf 读取格式化数据

```c
#include <stdio.h>

int main(void) {
    // 先创建数据文件
    FILE *fp = fopen("data.txt", "w");
    fprintf(fp, "Alice 25 95.5\n");
    fprintf(fp, "Bob 22 88.0\n");
    fclose(fp);

    // 读取数据
    fp = fopen("data.txt", "r");
    if (fp == NULL) return 1;

    char name[50];
    int age;
    float score;

    while (fscanf(fp, "%s %d %f", name, &age, &score) == 3) {
        printf("姓名: %s, 年龄: %d, 成绩: %.1f\n",
               name, age, score);
    }

    fclose(fp);
    return 0;
}
```

## 二进制文件

```c
#include <stdio.h>

typedef struct {
    int id;
    char name[20];
    double salary;
} Employee;

int main(void) {
    Employee emps[] = {
        {1, "张三", 5000.0},
        {2, "李四", 6000.0},
        {3, "王五", 5500.0}
    };
    int n = sizeof(emps) / sizeof(emps[0]);

    // 写入二进制文件
    FILE *fp = fopen("employees.bin", "wb");
    fwrite(emps, sizeof(Employee), n, fp);
    fclose(fp);

    // 读取二进制文件
    Employee readEmps[3];
    fp = fopen("employees.bin", "rb");
    fread(readEmps, sizeof(Employee), n, fp);
    fclose(fp);

    // 显示
    for (int i = 0; i < n; i++) {
        printf("ID: %d, 姓名: %s, 工资: %.2f\n",
               readEmps[i].id, readEmps[i].name,
               readEmps[i].salary);
    }

    return 0;
}
```

## 文件定位

```c
#include <stdio.h>

int main(void) {
    FILE *fp = fopen("test.txt", "w+");
    fprintf(fp, "ABCDEFGHIJ");

    // ftell - 获取当前位置
    long pos = ftell(fp);
    printf("当前位置: %ld\n", pos);

    // fseek - 移动文件指针
    // SEEK_SET: 文件开头
    // SEEK_CUR: 当前位置
    // SEEK_END: 文件末尾

    fseek(fp, 0, SEEK_SET);  // 回到开头
    printf("读取第1个字符: %c\n", fgetc(fp));

    fseek(fp, 5, SEEK_SET);  // 移到第6个字符
    printf("读取第6个字符: %c\n", fgetc(fp));

    fseek(fp, -2, SEEK_END);  // 从末尾倒数第2个
    printf("读取倒数第2个: %c\n", fgetc(fp));

    fclose(fp);
    return 0;
}
```

## 错误处理

```c
#include <stdio.h>
#include <errno.h>
#include <string.h>

int main(void) {
    FILE *fp = fopen("nonexistent.txt", "r");

    if (fp == NULL) {
        // 方法1: perror
        perror("打开文件失败");

        // 方法2: strerror
        printf("错误: %s\n", strerror(errno));

        // 方法3: errno
        printf("错误码: %d\n", errno);

        return 1;
    }

    fclose(fp);
    return 0;
}
```

## 实用示例

### 复制文件

```c
#include <stdio.h>

int copyFile(const char *src, const char *dest) {
    FILE *in = fopen(src, "rb");
    if (in == NULL) return -1;

    FILE *out = fopen(dest, "wb");
    if (out == NULL) {
        fclose(in);
        return -1;
    }

    char buffer[4096];
    size_t bytes;

    while ((bytes = fread(buffer, 1, sizeof(buffer), in)) > 0) {
        fwrite(buffer, 1, bytes, out);
    }

    fclose(in);
    fclose(out);
    return 0;
}

int main(void) {
    if (copyFile("source.txt", "copy.txt") == 0) {
        printf("复制成功\n");
    } else {
        printf("复制失败\n");
    }
    return 0;
}
```

### 统计文件行数

```c
#include <stdio.h>

int countLines(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) return -1;

    int lines = 0;
    int c;

    while ((c = fgetc(fp)) != EOF) {
        if (c == '\n') lines++;
    }

    fclose(fp);
    return lines;
}

int main(void) {
    int lines = countLines("test.txt");
    if (lines >= 0) {
        printf("文件行数: %d\n", lines);
    }
    return 0;
}
```

掌握文件操作后，就可以继续学习预处理器了！
