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

## 文本处理三剑客

### grep 高级用法

```bash
#!/bin/bash

# 基本搜索
grep "pattern" file.txt
grep -i "pattern" file.txt      # 忽略大小写
grep -r "pattern" /path/        # 递归搜索
grep -n "pattern" file.txt      # 显示行号
grep -v "pattern" file.txt      # 反向匹配

# 使用正则表达式
grep -E "pattern1|pattern2" file.txt    # 扩展正则
grep -P "pattern" file.txt               # Perl正则

# 常用示例
grep -E "^[0-9]+" file.txt               # 以数字开头的行
grep -E "error|warning" /var/log/syslog  # 搜索错误或警告
grep -o "[0-9]\+\.[0-9]\+\.[0-9]\+\.[0-9]\+" access.log  # 提取IP地址

# 上下文显示
grep -A 3 "error" file.txt    # 显示匹配行后3行
grep -B 3 "error" file.txt    # 显示匹配行前3行
grep -C 3 "error" file.txt    # 显示匹配行前后各3行
```

### sed 流编辑器

```bash
#!/bin/bash

# 基本替换
sed 's/old/new/' file.txt           # 替换第一个匹配
sed 's/old/new/g' file.txt          # 替换所有匹配
sed -i 's/old/new/g' file.txt       # 直接修改文件

# 删除操作
sed '/pattern/d' file.txt           # 删除匹配行
sed '1d' file.txt                   # 删除第一行
sed '$d' file.txt                   # 删除最后一行
sed '1,5d' file.txt                 # 删除1-5行

# 插入和追加
sed '3i\插入的文本' file.txt         # 在第3行前插入
sed '3a\追加的文本' file.txt         # 在第3行后追加

# 打印操作
sed -n '5p' file.txt                # 打印第5行
sed -n '5,10p' file.txt             # 打印5-10行
sed -n '/pattern/p' file.txt        # 打印匹配行

# 实用示例
# 删除空行
sed '/^$/d' file.txt

# 删除注释行
sed '/^#/d' config.conf

# 在每行末尾添加内容
sed 's/$/;/' file.txt

# 替换多个模式
sed -e 's/foo/bar/g' -e 's/baz/qux/g' file.txt
```

### awk 文本处理

```bash
#!/bin/bash

# 基本语法：awk 'pattern { action }' file

# 打印特定列
awk '{print $1}' file.txt           # 第一列
awk '{print $1, $3}' file.txt       # 第一和第三列
awk '{print $NF}' file.txt          # 最后一列

# 字段分隔符
awk -F':' '{print $1}' /etc/passwd  # 用冒号分隔
awk -F',' '{print $2}' data.csv     # 用逗号分隔

# 条件过滤
awk '$3 > 100' data.txt             # 第三列大于100
awk '/pattern/' file.txt            # 包含pattern的行
awk '$1 == "value"' file.txt        # 第一列等于value

# 内置变量
awk '{print NR, $0}' file.txt       # NR: 行号
awk '{print NF}' file.txt           # NF: 字段数
awk 'END{print NR}' file.txt        # 总行数

# 计算统计
awk '{sum+=$1} END{print sum}' numbers.txt              # 求和
awk '{sum+=$1} END{print sum/NR}' numbers.txt           # 平均值
awk 'NR==1{max=$1} $1>max{max=$1} END{print max}' file  # 最大值

# 格式化输出
awk '{printf "%-10s %5d\n", $1, $2}' file.txt

# 实用示例
# 统计每个用户的进程数
ps aux | awk '{count[$1]++} END{for(user in count) print user, count[user]}'

# 提取日志中的IP地址并统计访问次数
awk '{print $1}' access.log | sort | uniq -c | sort -rn | head -10

# 计算文件大小总和
ls -l | awk '{sum+=$5} END{print sum}'
```

## 正则表达式

### 基本正则表达式

```bash
# 字符匹配
.       # 任意单个字符
*       # 前一个字符0次或多次
^       # 行首
$       # 行尾
[]      # 字符集合
[^]     # 不在集合中的字符

# 示例
^hello      # 以hello开头
world$      # 以world结尾
h.llo       # h后跟任意字符再跟llo
colou*r     # color 或 colour
[aeiou]     # 任意元音字母
[0-9]       # 任意数字
[^0-9]      # 非数字字符
```

### 扩展正则表达式

```bash
# 使用 grep -E 或 egrep

+       # 前一个字符1次或多次
?       # 前一个字符0次或1次
{n}     # 恰好n次
{n,}    # 至少n次
{n,m}   # n到m次
|       # 或
()      # 分组

# 示例
[0-9]+          # 一个或多个数字
https?          # http 或 https
colou?r         # color 或 colour
[0-9]{3}        # 恰好3个数字
[0-9]{1,3}      # 1到3个数字
cat|dog         # cat 或 dog
```

### 实用正则表达式

```bash
# IP地址
[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}

# 邮箱地址（简化版）
[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}

# 手机号（中国）
1[3-9][0-9]{9}

# 日期格式 YYYY-MM-DD
[0-9]{4}-[0-9]{2}-[0-9]{2}

# URL
https?://[a-zA-Z0-9.-]+(/[a-zA-Z0-9./?&=_-]*)?
```

## 实用脚本案例

### 日志分析脚本

```bash
#!/bin/bash
# 分析 Nginx 访问日志

LOG_FILE="/var/log/nginx/access.log"

echo "=== 访问统计 ==="

# 总请求数
echo "总请求数: $(wc -l < "$LOG_FILE")"

# Top 10 IP
echo -e "\n=== Top 10 访问IP ==="
awk '{print $1}' "$LOG_FILE" | sort | uniq -c | sort -rn | head -10

# HTTP状态码统计
echo -e "\n=== HTTP状态码统计 ==="
awk '{print $9}' "$LOG_FILE" | sort | uniq -c | sort -rn

# 最常访问的URL
echo -e "\n=== Top 10 访问URL ==="
awk '{print $7}' "$LOG_FILE" | sort | uniq -c | sort -rn | head -10

# 每小时请求数
echo -e "\n=== 每小时请求分布 ==="
awk '{print substr($4,14,2)}' "$LOG_FILE" | sort | uniq -c
```

### 批量文件处理

```bash
#!/bin/bash
# 批量重命名和处理文件

# 批量添加日期前缀
for file in *.log; do
    date_prefix=$(date +%Y%m%d)
    mv "$file" "${date_prefix}_${file}"
done

# 批量转换文件编码
for file in *.txt; do
    iconv -f GBK -t UTF-8 "$file" -o "${file%.txt}_utf8.txt"
done

# 批量压缩旧文件
find /var/log -name "*.log" -mtime +7 -exec gzip {} \;

# 批量替换文件内容
find . -name "*.conf" -exec sed -i 's/old_value/new_value/g' {} \;
```

### 系统监控脚本

```bash
#!/bin/bash
# 系统资源监控

THRESHOLD_CPU=80
THRESHOLD_MEM=80
THRESHOLD_DISK=90
ALERT_EMAIL="admin@example.com"

check_cpu() {
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print int($2)}')
    if [ "$cpu_usage" -gt "$THRESHOLD_CPU" ]; then
        echo "警告: CPU使用率过高 (${cpu_usage}%)"
    fi
}

check_memory() {
    mem_usage=$(free | awk '/Mem/{printf("%.0f", $3/$2*100)}')
    if [ "$mem_usage" -gt "$THRESHOLD_MEM" ]; then
        echo "警告: 内存使用率过高 (${mem_usage}%)"
    fi
}

check_disk() {
    df -h | awk 'NR>1{gsub(/%/,"",$5); if($5>'"$THRESHOLD_DISK"') print "警告: 磁盘"$6"使用率过高 ("$5"%)"}'
}

echo "=== 系统监控报告 $(date) ==="
check_cpu
check_memory
check_disk
```

## 总结

本文介绍了 Shell 脚本编程：

- ✅ 变量和参数
- ✅ 条件判断和循环
- ✅ 函数和数组
- ✅ 文本处理三剑客（grep/sed/awk）
- ✅ 正则表达式
- ✅ 实用脚本示例
- ✅ 调试和最佳实践

这些是 Linux 的核心知识，掌握后可以高效地使用和管理 Linux 系统！

继续学习 [定时任务](/docs/linux/cron-scheduling) 和 [服务管理](/docs/linux/service-management)。
