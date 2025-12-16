---
sidebar_position: 8
title: 进程管理
---

# Linux 进程管理

进程是 Linux 系统中运行的程序实例。掌握进程管理对系统运维和问题排查至关重要。

## 进程基础

### 什么是进程

- **进程（Process）** - 运行中的程序实例
- **PID（Process ID）** - 进程唯一标识符
- **PPID（Parent PID）** - 父进程 ID
- **UID** - 进程所有者的用户 ID

### 进程类型

- **前台进程** - 直接与终端交互
- **后台进程** - 在后台运行
- **守护进程（Daemon）** - 系统服务进程

## 查看进程

### ps 命令

```bash
# 查看当前用户的进程
ps

# 查看所有进程（BSD 风格）
ps aux

# 查看所有进程（System V 风格）
ps -ef

# 自定义输出
ps -eo pid,ppid,cmd,% mem,%cpu

# 查看进程树
ps auxf
pstree

# 查找特定进程
ps aux | grep nginx
pgrep nginx
```

### top 实时监控

```bash
# 启动 top
top

# 常用快捷键：
# h - 帮助
# q - 退出
# M - 按内存排序
# P - 按 CPU 排序
# k - 终止进程
# r - 重新设置优先级
# 1 - 显示所有 CPU 核心
```

### htop（增强版）

```bash
# 安装 htop
sudo apt install htop

# 运行
htop

# 特点：
# - 彩色显示
# - 鼠标支持
# - 易于使用
# - 进程树视图
```

## 进程控制

### 启动进程

```bash
# 前台运行
command

# 后台运行
command &

# nohup（忽略挂断信号）
nohup command &
nohup command > output.log 2>&1 &
```

### 进程状态转换

```bash
# 前台任务转后台
Ctrl+Z          # 暂停当前任务
bg              # 在后台继续运行

# 后台任务转前台
fg              # 最近的后台任务
fg %1           # 指定任务号

# 查看后台任务
jobs
jobs -l         # 显示 PID
```

### 终止进程

```bash
# 正常终止
kill PID
kill -15 PID
kill -TERM PID

# 强制终止
kill -9 PID
kill -KILL PID

# 终止进程组
killall process_name
pkill process_name

# 交互式终止
killall -i nginx

# 按用户终止
pkill -u username
```

### 信号类型

| 信号 | 编号 | 说明 | 用途 |
|------|------|------|------|
| SIGHUP | 1 | 挂断 | 重新加载配置 |
| SIGINT | 2 | 中断 (Ctrl+C) | 正常中断 |
| SIGQUIT | 3 | 退出 (Ctrl+\) | 生成 core dump |
| SIGKILL | 9 | 强制终止 | 无法捕获 |
| SIGTERM | 15 | 终止 | 正常终止（默认） |
| SIGCONT | 18 | 继续 | 恢复暂停的进程 |
| SIGSTOP | 19 | 停止 | 暂停进程 |

```bash
# 发送信号
kill -HUP PID          # 重新加载配置
kill -USR1 PID         # 用户自定义信号
```

## 进程优先级

### nice 和 renice

```bash
# nice 值范围：-20 到 19
# -20：最高优先级
# 0：默认优先级
# 19：最低优先级

# 以特定优先级启动
nice -n 10 command
nice --10 command

# 修改运行中进程的优先级
renice -n 5 -p PID
renice 5 PID

# 按用户修改
renice -n 5 -u username
```

## 进程监控

### 系统负载

```bash
# 查看负载平均值
uptime
# 输出：10:00:00 up 5 days,  3:30,  2 users,  load average: 0.50, 0.40, 0.30
#                                                            1分钟  5分钟  15分钟

# /proc/loadavg
cat /proc/loadavg
```

### CPU 使用

```bash
# 查看 CPU 信息
lscpu
cat /proc/cpuinfo

# 实时 CPU 使用
top
htop
mpstat 1         # 每秒更新
```

### 内存使用

```bash
# 查看内存
free -h
cat /proc/meminfo

# 查看进程内存
ps aux --sort=-%mem | head
pmap PID

# 内存详情
vmstat 1
```

## /proc 文件系统

```bash
# 进程信息目录
/proc/[PID]/

# 常用文件
/proc/[PID]/cmdline    # 启动命令
/proc/[PID]/cwd        # 当前工作目录
/proc/[PID]/environ    # 环境变量
/proc/[PID]/exe        # 可执行文件链接
/proc/[PID]/fd/        # 文件描述符
/proc/[PID]/status     # 进程状态
/proc/[PID]/limits     # 资源限制

# 示例
cat /proc/1/cmdline    # init/systemd 命令
ls -l /proc/self/fd    # 当前shell的文件描述符
```

## 系统服务管理

### systemd（现代 Linux）

```bash
# 启动服务
sudo systemctl start nginx

# 停止服务
sudo systemctl stop nginx

# 重启服务
sudo systemctl restart nginx

# 重新加载配置
sudo systemctl reload nginx

# 查看状态
sudo systemctl status nginx

# 开机自启
sudo systemctl enable nginx
sudo systemctl disable nginx

# 查看所有服务
systemctl list-units --type=service

# 查看失败的服务
systemctl --failed
```

### 服务配置

```bash
# 服务单元文件位置
/etc/systemd/system/
/lib/systemd/system/

# 查看服务配置
systemctl cat nginx

# 编辑服务
sudo systemctl edit nginx

# 重新加载 systemd
sudo systemctl daemon-reload
```

## 计划任务

### cron 定时任务

```bash
# 编辑 crontab
crontab -e

# 查看 crontab
crontab -l

# 删除 crontab
crontab -r

# crontab 语法
# 分 时 日 月 周 命令
# *  *  *  *  *  command

# 示例
0 2 * * * /backup.sh          # 每天 2:00
30 */6 * * * /script.sh        # 每 6 小时的第 30 分钟
0 0 * * 0 /weekly.sh           # 每周日 0:00
0 0 1 * * /monthly.sh          # 每月 1 号 0:00
*/15 * * * * /check.sh         # 每 15 分钟
```

### at 一次性任务

```bash
# 安装 at
sudo apt install at

# 设置任务
at 10:00
at 'now + 1 hour'
at '2:30 PM tomorrow'

# 输入命令后 Ctrl+D 保存

# 查看任务队列
atq

# 删除任务
atrm job_number
```

## 进程间通信

### 管道

```bash
# 命名管道（FIFO）
mkfifo mypipe

# 写入管道
echo "data" > mypipe &

# 从管道读取
cat < mypipe
```

### 信号量和共享内存

```bash
# 查看 IPC 资源
ipcs

# 查看共享内存
ipcs -m

# 查看信号量
ipcs -s

# 查看消息队列
ipcs -q

# 删除资源
ipcrm -m shmid
```

## 性能分析

### strace 系统调用追踪

```bash
# 追踪进程
strace command
strace -p PID

# 统计系统调用
strace -c command

# 追踪特定系统调用
strace -e open,read command

# 输出到文件
strace -o output.txt command
```

### lsof 打开文件

```bash
# 列出所有打开的文件
lsof

# 查看进程打开的文件
lsof -p PID

# 查看文件被哪些进程打开
lsof /path/to/file

# 查看端口占用
lsof -i :80
lsof -i TCP:22

# 查看用户打开的文件
lsof -u username
```

## 最佳实践

### 1. 安全终止进程

```bash
# 先尝试 TERM
kill PID

# 等待几秒
sleep 5

# 如果还在运行，使用 KILL
kill -9 PID
```

### 2. 后台任务管理

```bash
# 使用 screen 或 tmux
screen -S session_name
# Ctrl+A+D 分离
screen -r session_name  # 重新连接

# 使用 systemd
# 创建服务单元文件而不是直接 nohup
```

### 3. 资源限制

```bash
# 查看限制
ulimit -a

# 设置限制
ulimit -n 4096    # 最大打开文件数
ulimit -u 2048    # 最大进程数

# 永久设置（/etc/security/limits.conf）
username soft nofile 4096
username hard nofile 8192
```

## 总结

本文介绍了 Linux 进程管理：

- ✅ 进程查看（ps、top、htop）
- ✅ 进程控制（启动、终止、优先级）
- ✅ 系统服务管理（systemd）
- ✅ 计划任务（cron、at）
- ✅ 进程监控和性能分析

继续学习 [用户和组管理](/docs/linux/users-groups) 和 [网络配置](/docs/linux/networking)。
