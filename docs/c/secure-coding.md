---
sidebar_position: 22
title: 安全编程
---

# C 语言安全编程

C 语言给予程序员极大的控制权，但也容易导致严重的安全漏洞。编写安全的代码是每个 C 程序员的责任。

## 常见漏洞与防范

### 缓冲区溢出 (Buffer Overflow)

最常见的漏洞，通常发生在向固定大小的缓冲区写入过多数据时。

❌ **危险代码：**

```c
char buf[10];
gets(buf); // 永远不要使用 gets！
strcpy(buf, user_input); // 如果 user_input > 10 字节，崩溃或被攻击
```

✅ **安全代码：**

```c
char buf[10];
fgets(buf, sizeof(buf), stdin); // 限制读取长度

// 使用 strncpy 时要注意手动添加终止符
strncpy(buf, user_input, sizeof(buf) - 1);
buf[sizeof(buf) - 1] = '\0';

// 或者使用 snprintf (推荐)
snprintf(buf, sizeof(buf), "%s", user_input);
```

### 格式化字符串漏洞

当用户输入被直接用作格式化字符串时发生。

❌ **危险代码：**

```c
printf(user_input); // 如果输入包含 %s %n 等，可能泄漏内存或崩溃
```

✅ **安全代码：**

```c
printf("%s", user_input); // 始终指定格式字符串
```

### 整数溢出 (Integer Overflow)

整数运算结果超出其类型范围，导致回绕。这可能导致缓冲区分配过小。

❌ **危险代码：**

```c
size_t len = len1 + len2; // 如果溢出，len 可能变得很小
char *buf = malloc(len);  // 分配了过小的内存
memcpy(buf, str1, len1);  // 溢出！
```

✅ **安全代码：**

```c
if (SIZE_MAX - len1 < len2) {
    // 处理溢出错误
    return ERROR;
}
size_t len = len1 + len2;
```

### 释放后使用 (Use After Free)

使用已被释放的内存指针。

❌ **危险代码：**

```c
char *p = malloc(10);
free(p);
*p = 'A'; // 未定义行为，可能导致严重漏洞
```

✅ **安全代码：**

```c
char *p = malloc(10);
free(p);
p = NULL; // 释放后立即置空
```

## 输入验证

永远不要信任外部输入（命令行参数、环境变量、文件、网络数据）。

- **检查长度**：确保输入不超过缓冲区大小。
- **检查内容**：确保输入只包含允许的字符（例如白名单验证）。
- **检查边界**：确保数值在合理范围内。

```c
int process_age(const char *input) {
    char *endptr;
    long val = strtol(input, &endptr, 10);

    // 检查是否包含非数字字符
    if (*endptr != '\0') return -1;

    // 检查数值范围
    if (val < 0 || val > 150) return -1;

    return (int)val;
}
```

## 最小权限原则

- 尽量避免以 root 权限运行程序。
- 如果必须使用 root，在完成特权操作后立即降级 (setuid/setgid)。
- 使用 `const` 关键字保护不应修改的数据。
- 限制变量的作用域。

## 安全函数

优先使用边界检查版本的函数。虽然 C11 定义了 `_s` 后缀的安全函数（如 `strcpy_s`），但并非所有编译器都支持。通常建议使用标准替代品：

| 危险函数   | 安全替代                       |
| :--------- | :----------------------------- |
| `gets`     | `fgets`                        |
| `strcpy`   | `strncpy` (需小心), `snprintf` |
| `strcat`   | `strncat` (需小心), `snprintf` |
| `sprintf`  | `snprintf`                     |
| `vsprintf` | `vsnprintf`                    |

## 工具链检查

在编译时开启保护机制：

```bash
# 栈保护 (Stack Canary)
gcc -fstack-protector-all ...

# 地址随机化 (ASLR) - 通常由操作系统开启，编译需支持的位置无关代码
gcc -fPIC ...

# 数据执行保护 (NX/DEP)
gcc -z noexecstack ...

# 强化源码检查
gcc -D_FORTIFY_SOURCE=2 -O2 ...
```

## 总结

安全不是一种附加功能，而是编程习惯。时刻警惕内存边界，假设所有输入都是恶意的，是 C 语言开发者的生存法则。
