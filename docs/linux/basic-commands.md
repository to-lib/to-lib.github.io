---
sidebar_position: 2
title: 基础命令
---

# Linux 基础命令

掌握 Linux 基础命令是学习 Linux 的第一步。本文介绍最常用的命令和操作。

## 文件和目录操作

### 目录导航

```bash
# 显示当前目录
pwd

# 切换目录
cd /home/user          # 绝对路径
cd Documents           # 相对路径
cd ..                  # 上级目录
cd ~                   # 用户主目录
cd -                   # 返回上一个目录

# 列出文件
ls                     # 基本列表
ls -l                  # 详细信息
ls -a                  # 显示隐藏文件
ls -lh                 # 人类可读的文件大小
ls -lt                 # 按时间排序
ls -lS                 # 按大小排序
```

### 创建和删除

```bash
# 创建目录
mkdir directory
mkdir -p path/to/directory    # 创建多级目录

# 创建文件
touch file.txt
touch file1.txt file2.txt     # 创建多个文件

# 删除文件
rm file.txt
rm -i file.txt                # 交互式删除（确认）
rm -f file.txt                # 强制删除

# 删除目录
rmdir empty_dir               # 删除空目录
rm -r directory               # 递归删除目录
rm -rf directory              # 强制递归删除（危险！）
```

### 复制和移动

```bash
# 复制文件
cp source.txt dest.txt
cp file.txt /path/to/destination/

# 复制目录
cp -r source_dir dest_dir

# 保留属性复制
cp -p file.txt dest.txt       # 保留权限、时间戳

# 移动/重命名
mv old_name.txt new_name.txt
mv file.txt /path/to/destination/
mv *.txt documents/           # 移动多个文件
```

## 文件查看和编辑

### 查看文件内容

```bash
# 查看整个文件
cat file.txt
cat file1.txt file2.txt       # 查看多个文件

# 分页查看
less file.txt                 # 可前后翻页
more file.txt                 # 只能向前翻页

# 查看文件开头
head file.txt                 # 默认前10行
head -n 20 file.txt           # 前20行

# 查看文件结尾
tail file.txt                 # 默认后10行
tail -n 20 file.txt           # 后20行
tail -f /var/log/syslog       # 实时查看日志
```

### 文本编辑器

```bash
# Vim
vim file.txt
# 基本操作：
# i - 插入模式
# Esc - 命令模式
# :w - 保存
# :q - 退出
# :wq - 保存并退出
# :q! - 强制退出不保存

# Nano（更简单）
nano file.txt
# Ctrl+O - 保存
# Ctrl+X - 退出
```

## 文件搜索

### find 命令

```bash
# 按名称查找
find /path -name "*.txt"
find . -name "file.txt"
find / -name "config*"

# 按类型查找
find /path -type f            # 文件
find /path -type d            # 目录
find /path -type l            # 符号链接

# 按大小查找
find /path -size +100M        # 大于100MB
find /path -size -1k          # 小于1KB

# 按时间查找
find /path -mtime -7          # 7天内修改
find /path -atime +30         # 30天前访问

# 组合条件
find /path -name "*.log" -size +10M -mtime -7
```

### locate 命令

```bash
# 快速查找（使用数据库）
locate filename
locate "*.txt"

# 更新数据库
sudo updatedb
```

## 文本处理

### grep 搜索

```bash
# 基本搜索
grep "pattern" file.txt
grep "error" /var/log/syslog

# 递归搜索
grep -r "pattern" /path/to/search

# 忽略大小写
grep -i "pattern" file.txt

# 显示行号
grep -n "pattern" file.txt

# 反向匹配
grep -v "pattern" file.txt

# 正则表达式
grep -E "pattern1|pattern2" file.txt

# 统计匹配行数
grep -c "pattern" file.txt
```

### sed 文本替换

```bash
# 替换（不修改原文件）
sed 's/old/new/' file.txt

# 替换所有匹配
sed 's/old/new/g' file.txt

# 修改原文件
sed -i 's/old/new/g' file.txt

# 删除行
sed '/pattern/d' file.txt

# 指定行范围
sed '1,5s/old/new/g' file.txt
```

### awk 文本分析

```bash
# 打印列
awk '{print $1}' file.txt     # 第一列
awk '{print $1, $3}' file.txt # 第一和第三列

# 设置分隔符
awk -F: '{print $1}' /etc/passwd

# 条件过滤
awk '$3 > 100 {print $1}' file.txt

# 统计
awk '{sum += $1} END {print sum}' file.txt
```

## 文件压缩和解压

### tar 归档

```bash
# 创建归档
tar -cf archive.tar files/
tar -czf archive.tar.gz files/    # gzip 压缩
tar -cjf archive.tar.bz2 files/   # bzip2 压缩

# 解压归档
tar -xf archive.tar
tar -xzf archive.tar.gz
tar -xjf archive.tar.bz2

# 查看归档内容
tar -tf archive.tar

# 解压到指定目录
tar -xzf archive.tar.gz -C /path/to/destination
```

### gzip/gunzip

```bash
# 压缩
gzip file.txt                 # 生成 file.txt.gz
gzip -k file.txt              # 保留原文件

# 解压
gunzip file.txt.gz
gzip -d file.txt.gz
```

### zip/unzip

```bash
# 压缩
zip archive.zip file1 file2
zip -r archive.zip directory/

# 解压
unzip archive.zip
unzip archive.zip -d /path/to/destination
```

## 系统信息

### 系统查看

```bash
# 系统信息
uname -a                      # 所有信息
uname -r                      # 内核版本
hostnamectl                   # 主机信息

# 发行版信息
cat /etc/os-release
lsb_release -a

# CPU 信息
lscpu
cat /proc/cpuinfo

# 内存信息
free -h
cat /proc/meminfo

# 磁盘信息
df -h                         # 磁盘使用情况
du -sh /path                  # 目录大小
du -h --max-depth=1           # 查看一级目录大小
```

### 进程查看

```bash
# 查看进程
ps aux                        # 所有进程
ps aux | grep nginx           # 查找特定进程

# 实时监控
top                           # 基本监控
htop                          # 增强版（需安装）

# 进程树
pstree
```

## 网络命令

### 基本网络操作

```bash
# 查看 IP 地址
ip addr
ifconfig                      # 旧命令

# 测试连接
ping google.com
ping -c 4 google.com          # 发送4个包

# 追踪路由
traceroute google.com

# DNS 查询
nslookup google.com
dig google.com

# 下载文件
wget https://example.com/file.zip
curl -O https://example.com/file.zip
```

### 网络连接

```bash
# 查看端口
netstat -tuln                 # 监听的端口
ss -tuln                      # 更现代的命令

# SSH 连接
ssh user@hostname
ssh -p 2222 user@hostname     # 指定端口

# 文件传输
scp file.txt user@remote:/path/
scp -r directory/ user@remote:/path/
```

## 权限和所有者

```bash
# 查看权限
ls -l

# 修改权限
chmod 755 file.txt
chmod u+x script.sh           # 所有者添加执行权限
chmod g-w file.txt            # 组去除写权限

# 修改所有者
chown user:group file.txt
chown -R user:group directory/

# 修改组
chgrp groupname file.txt
```

## 常用技巧

### 命令历史

```bash
# 查看历史
history

# 执行历史命令
!100                          # 执行第100条命令
!!                            # 执行上一条命令
!ping                         # 执行最近的ping命令

# 搜索历史
Ctrl+R                        # 反向搜索
```

### 命令别名

```bash
# 创建别名
alias ll='ls -lah'
alias ..='cd ..'

# 查看别名
alias

# 永久保存（添加到 ~/.bashrc）
echo "alias ll='ls -lah'" >> ~/.bashrc
source ~/.bashrc
```

### 管道和重定向

```bash
# 管道
ls -l | grep ".txt"
ps aux | grep nginx | awk '{print $2}'

# 输出重定向
command > output.txt          # 覆盖
command >> output.txt         # 追加
command 2> error.log          # 错误输出
command &> all.log            # 所有输出

# 输入重定向
sort < input.txt
```

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| Tab | 自动补全 |
| Ctrl+C | 终止当前命令 |
| Ctrl+D | 退出当前 Shell |
| Ctrl+L | 清屏 |
| Ctrl+A | 移到行首 |
| Ctrl+E | 移到行尾 |
| Ctrl+U | 删除到行首 |
| Ctrl+K | 删除到行尾 |
| Ctrl+R | 搜索历史命令 |

## 最佳实践

1. **使用 Tab 补全** - 提高效率，减少错误
2. **善用历史命令** - Ctrl+R 快速查找
3. **小心使用 rm** - 特别是 `rm -rf`
4. **定期备份** - 重要数据要备份
5. **阅读手册** - `man command` 了解详细用法

## 总结

本文介绍了 Linux 基础命令：

- ✅ 文件和目录操作
- ✅ 文件查看和编辑
- ✅ 文本处理工具
- ✅ 文件压缩解压
- ✅ 系统信息查看
- ✅ 网络基本操作

继续学习 [文件系统](/docs/linux/file-system) 和 [权限管理](/docs/linux/permissions)。
