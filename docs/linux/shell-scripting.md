---
sidebar_position: 7
title: Shell 脚本
---

# Shell 脚本编程

Shell 脚本是 Linux 系统自动化的强大工具，可以将多个命令组合成可重复执行的程序。

## Shell 脚本基础

### 第一个脚本

```bash
#!/bin/bash
# 这是注释
# shebang 行指定解释器

echo "Hello, World!"
```

```bash
# 保存为 hello.sh，添加执行权限
chmod +x hello.sh

# 运行脚本
./hello.sh
bash hello.sh
```

### 变量

```bash
#!/bin/bash

# 定义变量（等号两边不能有空格）
name="Linux"
count=10

# 使用变量
echo "名称: $name"
echo "数量: ${count}"

# 只读变量
readonly PI=3.14159

# 删除变量
unset name
```

### 特殊变量

```bash
#!/bin/bash

echo "脚本名: $0"
echo "第一个参数: $1"
echo "第二个参数: $2"
echo "所有参数: $@"
echo "参数个数: $#"
echo "上个命令退出状态: $?"
echo "当前进程ID: $$"
```

## 条件判断

### if 语句

```bash
#!/bin/bash

if [ "$1" = "hello" ]; then
    echo "你说了 hello"
elif [ "$1" = "bye" ]; then
    echo "你说了 bye"
else
    echo "未知命令"
fi
```

### 测试运算符

```bash
# 数值比较
if [ $a -eq $b ]; then echo "相等"; fi
if [ $a -ne $b ]; then echo "不相等"; fi
if [ $a -gt $b ]; then echo "大于"; fi
if [ $a -lt $b ]; then echo "小于"; fi
if [ $a -ge $b ]; then echo "大于等于"; fi
if [ $a -le $b ]; then echo "小于等于"; fi

# 字符串比较
if [ "$a" = "$b" ]; then echo "相等"; fi
if [ "$a" != "$b" ]; then echo "不相等"; fi
if [ -z "$a" ]; then echo "长度为0"; fi
if [ -n "$a" ]; then echo "长度不为0"; fi

# 文件测试
if [ -e file.txt ]; then echo "文件存在"; fi
if [ -f file.txt ]; then echo "是普通文件"; fi
if [ -d directory ]; then echo "是目录"; fi
if [ -r file.txt ]; then echo "可读"; fi
if [ -w file.txt ]; then echo "可写"; fi
if [ -x file.sh ]; then echo "可执行"; fi
```

## 循环

### for 循环

```bash
#!/bin/bash

# 基本for循环
for i in 1 2 3 4 5; do
    echo "数字: $i"
done

# 范围
for i in {1..10}; do
    echo $i
done

# C风格for循环
for ((i=1; i<=10; i++)); do
    echo $i
done

# 遍历文件
for file in *.txt; do
    echo "处理: $file"
done

# 遍历数组
files=(file1.txt file2.txt file3.txt)
for file in "${files[@]}"; do
    echo $file
done
```

### while 循环

```bash
#!/bin/bash

count=1
while [ $count -le 5 ]; do
    echo "计数: $count"
    ((count++))
done

# 读取文件
while IFS= read -r line; do
    echo "行内容: $line"
done < file.txt
```

### until 循环

```bash
#!/bin/bash

count=1
until [ $count -gt 5 ]; do
    echo "计数: $count"
    ((count++))
done
```

## 函数

```bash
#!/bin/bash

# 定义函数
function greet() {
    echo "Hello, $1!"
}

# 另一种定义方式
say_bye() {
    echo "Goodbye, $1!"
}

# 调用函数
greet "Alice"
say_bye "Bob"

# 返回值
add() {
    local result=$(($1 + $2))
    echo $result
}

sum=$(add 5 3)
echo "Sum: $sum"
```

## 数组

```bash
#!/bin/bash

# 定义数组
fruits=("apple" "banana" "orange")

# 访问元素
echo ${fruits[0]}
echo ${fruits[1]}

# 所有元素
echo ${fruits[@]}

# 数组长度
echo ${#fruits[@]}

# 添加元素
fruits+=("grape")

# 遍历数组
for fruit in "${fruits[@]}"; do
    echo $fruit
done
```

## 输入输出

### 读取输入

```bash
#!/bin/bash

echo "请输入你的名字:"
read name
echo "你好, $name!"

# 不显示输入（密码）
read -s -p "请输入密码: " password
echo

# 超时
read -t 5 -p "5秒内输入: " input
```

### 重定向

```bash
#!/bin/bash

# 输出重定向
echo "Hello" > output.txt      # 覆盖
echo "World" >> output.txt     # 追加

# 输入重定向
while read line; do
    echo $line
done < input.txt

# 错误重定向
command 2> error.log
command &> all.log
```

## 实用脚本示例

### 备份脚本

```bash
#!/bin/bash

# 备份目录
SOURCE="/home/user/documents"
DEST="/backup"
DATE=$(date +%Y%m%d)
BACKUP_FILE="backup_$DATE.tar.gz"

# 创建备份
tar -czf "$DEST/$BACKUP_FILE" "$SOURCE"

# 检查是否成功
if [ $? -eq 0 ]; then
    echo "备份成功: $BACKUP_FILE"
else
    echo "备份失败" >&2
    exit 1
fi

# 删除旧备份（保留7天）
find "$DEST" -name "backup_*.tar.gz" -mtime +7 -delete
```

### 系统监控脚本

```bash
#!/bin/bash

# CPU使用率
CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)

# 内存使用率
MEM=$(free | grep Mem | awk '{print ($3/$2) * 100.0}')

# 磁盘使用率
DISK=$(df -h / | tail -1 | awk '{print $5}' | tr -d '%')

echo "=== 系统状态 ==="
echo "CPU: $CPU%"
echo "内存: ${MEM}%"
echo "磁盘: $DISK%"

# 发送告警
if [ $(echo "$CPU > 80" | bc) -eq 1 ]; then
    echo "警告: CPU使用率过高！"
fi
```

### 批量重命名

```bash
#!/bin/bash

# 批量添加前缀
for file in *.jpg; do
    mv "$file" "photo_$file"
done

# 批量修改扩展名
for file in *.jpeg; do
    mv "$file" "${file%.jpeg}.jpg"
done
```

## 调试技巧

```bash
# 显示执行的命令
bash -x script.sh

# 在脚本中启用调试
#!/bin/bash
set -x  # 开启
set +x  # 关闭

# 遇到错误立即退出
set -e

# 变量未定义时报错
set -u

# 管道错误时退出
set -o pipefail

# 组合使用
set -euo pipefail
```

## 最佳实践

### 1. 脚本模板

```bash
#!/bin/bash
set -euo pipefail

# 脚本说明
# 用途：...
# 作者：...
# 日期：...

# 变量定义
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/myscript.log"

# 函数定义
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

cleanup() {
    log "清理中..."
}

trap cleanup EXIT

# 主逻辑
main() {
    log "脚本开始"
    # ...
    log "脚本结束"
}

main "$@"
```

### 2. 错误处理

```bash
#!/bin/bash

command || {
    echo "命令失败" >&2
    exit 1
}

# 检查命令是否存在
if ! command -v curl &> /dev/null; then
    echo "curl 未安装" >&2
    exit 1
fi
```

### 3. 参数解析

```bash
#!/bin/bash

usage() {
    echo "用法: $0 [-h] [-f file] [-v]"
    exit 1
}

while getopts "hf:v" opt; do
    case $opt in
        h) usage ;;
        f) FILE="$OPTARG" ;;
        v) VERBOSE=1 ;;
        *) usage ;;
    esac
done
```

## 总结

本文介绍了 Shell 脚本编程：

- ✅ 变量和参数
- ✅ 条件判断和循环
- ✅ 函数和数组
- ✅ 实用脚本示例
- ✅ 调试和最佳实践

这些是 Linux 的核心知识，掌握后可以高效地使用和管理 Linux 系统！
