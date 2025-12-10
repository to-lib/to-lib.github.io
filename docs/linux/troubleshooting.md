---
sidebar_position: 16
title: 故障排查
---

# Linux 故障排查

故障排查是系统运维的核心技能，本文介绍常见问题的诊断和解决方法。

## 排查方法论

### 故障排查流程

```mermaid
graph TD
    A[发现问题] --> B[收集信息]
    B --> C[分析原因]
    C --> D[制定方案]
    D --> E[实施修复]
    E --> F[验证结果]
    F --> G[记录文档]
```

### 基本原则

- **重现问题** - 确保能稳定重现
- **收集日志** - 查看所有相关日志
- **逐步排查** - 从简单到复杂
- **做好备份** - 修改前备份
- **记录过程** - 文档化所有步骤

## 系统无法启动

### GRUB 问题

```bash
# 症状：启动时卡在 GRUB

# 解决方案1：进入 GRUB 命令行
# 按 'c' 进入命令行模式
grub> ls                      # 列出分区
grub> ls (hd0,1)/             # 查看分区内容
grub> set root=(hd0,1)        # 设置根分区
grub> linux /vmlinuz root=/dev/sda1
grub> initrd /initrd.img
grub> boot

# 解决方案2：从 Live CD 修复
# 启动 Live CD
sudo mount /dev/sda1 /mnt
sudo mount --bind /dev /mnt/dev
sudo mount --bind /proc /mnt/proc
sudo mount --bind /sys /mnt/sys
sudo chroot /mnt
grub-install /dev/sda
update-grub
exit
sudo reboot
```

### 文件系统错误

```bash
# 症状：提示文件系统只读或损坏

# 单用户模式检查
# 启动时按 'e'，在 linux 行末添加：
single

# 或
systemd.unit=rescue.target

# 检查文件系统（需先卸载）
sudo fsck -y /dev/sda1

# 强制检查
sudo fsck -f /dev/sda1

# 恢复超级块
sudo e2fsck -b 32768 /dev/sda1
```

### 内核 Panic

```bash
# 症状：系统崩溃，显示 kernel panic

# 查看崩溃日志
dmesg | grep -i panic
journalctl -k

# 启动时选择旧内核
# 在 GRUB 菜单选择 "Advanced options"

# 移除问题内核
sudo apt remove linux-image-x.x.x
```

## 无法登录

### 忘记密码

```bash
# 单用户模式重置密码
# 1. 启动时按 'e' 编辑 GRUB
# 2. 找到 linux 开头的行，在末尾添加：
init=/bin/bash

# 3. 按 Ctrl+X 或 F10 启动

# 4. 重新挂载根分区为读写
mount -o remount,rw /

# 5. 重置密码
passwd username

# 6. 重启
exec /sbin/init
```

### SSH 连接失败

```bash
# 检查 SSH 服务
sudo systemctl status sshd

# 查看 SSH 日志
sudo tail -f /var/log/auth.log
journalctl -u sshd -f

# 检查端口
sudo ss -tuln | grep 22

# 检查防火墙
sudo ufw status
sudo iptables -L -n | grep 22

# 测试网络连接
ping server
telnet server 22

# 权限问题
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

### 账户锁定

```bash
# 查看账户状态
sudo passwd -S username

# 解锁账户
sudo passwd -u username
sudo usermod -U username

# 查看失败登录
sudo lastb

# 检查 PAM 配置
sudo cat /etc/pam.d/common-auth
```

## 性能问题

### CPU 100%

```bash
# 查找 CPU 占用高的进程
top
htop
ps aux --sort=-%cpu | head

# 查看进程详情
ps -p PID -o %cpu,%mem,cmd

# 查看线程
top -H -p PID
ps -Lf -p PID

# 追踪系统调用
strace -p PID

# 生成进程堆栈
sudo gdb -p PID
(gdb) thread apply all bt
(gdb) quit

# kill 占用 CPU 的进程
kill -15 PID
kill -9 PID    # 强制终止
```

### 内存不足

```bash
# 查看内存使用
free -h
top
htop

# 查看内存占用进程
ps aux --sort=-%mem | head

# 查看详细内存信息
cat /proc/meminfo

# 查看 OOM（Out of Memory）日志
dmesg | grep -i oom
journalctl | grep -i oom

# 清理缓存
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches

# 增加 swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 磁盘空间满

```bash
# 查看磁盘使用
df -h
df -i          # inode 使用情况

# 查找大文件
sudo find / -type f -size +100M 2>/dev/null
sudo du -sh /* | sort -rh | head

# 查找大目录
sudo du -h --max-depth=1 / | sort -rh | head

# 清理空间
# 1. 清理包缓存
sudo apt clean
sudo apt autoclean

# 2. 清理日志
sudo journalctl --vacuum-time=7d
sudo find /var/log -type f -name "*.log" -mtime +30 -delete

# 3. 清理临时文件
sudo rm -rf /tmp/*
sudo rm -rf /var/tmp/*

# 4. 删除旧内核
dpkg --list | grep linux-image
sudo apt autoremove

# 5. 查找删除的但仍占用空间的文件
sudo lsof | grep deleted
```

### 高负载

```bash
# 查看负载
uptime
w
top

# 查看等待 IO 的进程
top
# 查看 wa（IO wait）列

# IO 统计
iostat -x 1
iotop

# 查找导致高负载的进程
ps aux | awk '$3 > 50'    # CPU > 50%
ps aux --sort=-%cpu | head
```

## 网络问题

### 网络不通

```bash
# 检查网络接口
ip addr
ifconfig

# 检查网络接口状态
ip link
ip link set eth0 up

# 检查路由
ip route
route -n

# 测试网关
ping 192.168.1.1

# 测试 DNS
ping 8.8.8.8
nslookup google.com
dig google.com

# 检查 DNS 配置
cat /etc/resolv.conf

# 追踪路由
traceroute google.com
mtr google.com

# 检查防火墙
sudo iptables -L -n
sudo ufw status
```

### 端口问题

```bash
# 检查端口监听
sudo ss -tuln
sudo netstat -tuln

# 检查端口占用
sudo lsof -i :80
sudo fuser 80/tcp

# 测试端口连接
telnet server 80
nc -zv server 80

# 扫描端口
nmap server
```

### DNS 问题

```bash
# 测试 DNS 解析
nslookup google.com
dig google.com
host google.com

# 指定 DNS 服务器测试
nslookup google.com 8.8.8.8
dig @8.8.8.8 google.com

# 清除 DNS 缓存
sudo systemd-resolve --flush-caches
sudo /etc/init.d/nscd restart

# 检查 DNS 配置
cat /etc/resolv.conf
cat /etc/nsswitch.conf
```

## 服务问题

### 服务无法启动

```bash
# 查看服务状态
sudo systemctl status service_name

# 查看详细日志
journalctl -u service_name -n 50
journalctl -u service_name -f

# 查看服务配置
systemctl cat service_name

# 检查配置文件语法
# Nginx
sudo nginx -t

# Apache
sudo apache2ctl configtest

# 查看依赖
systemctl list-dependencies service_name

# 重新加载配置
sudo systemctl daemon-reload
```

### 服务频繁重启

```bash
# 查看重启日志
journalctl -u service_name | grep -i restart

# 查看服务配置
systemctl cat service_name

# 检查重启策略
Restart=on-failure
RestartSec=5s

# 禁用自动重启进行调试
sudo systemctl edit service_name
[Service]
Restart=no

# 手动运行服务查看错误
/path/to/service --debug
```

## 应用问题

### 应用崩溃

```bash
# 查看崩溃日志
dmesg | tail
journalctl -xe

# 查看应用日志
tail -f /var/log/app.log

# 生成 core dump
ulimit -c unlimited
echo "/tmp/core.%e.%p" | sudo tee /proc/sys/kernel/core_pattern

# 分析 core dump
gdb /path/to/binary /tmp/core.xxx

# 追踪进程
strace -p PID
strace -f -e trace=open,read,write command
```

### 数据库问题

```bash
# MySQL 无法启动
sudo systemctl status mysql
sudo tail -f /var/log/mysql/error.log

# 检查配置文件
sudo mysqld --help --verbose | grep my.cnf

# 检查磁盘空间
df -h /var/lib/mysql

# 修复表
mysqlcheck -u root -p --auto-repair --all-databases

# 慢查询
sudo tail -f /var/log/mysql/mysql-slow.log
```

## 文件系统问题

### 文件损坏

```bash
# 检查文件系统
sudo umount /dev/sda1
sudo fsck -y /dev/sda1

# 恢复删除的文件
sudo apt install extundelete
sudo extundelete /dev/sda1 --restore-all
```

### inode 耗尽

```bash
# 查看 inode 使用
df -i

# 查找大量小文件
sudo find / -xdev -printf '%h\n' | sort | uniq -c | sort -k 1 -n

# 删除不需要的小文件
sudo find /tmp -type f -delete
```

## 日志分析

### 查找错误

```bash
# 系统日志
sudo grep -i error /var/log/syslog
sudo grep -i fail /var/log/syslog

# 认证日志
sudo grep "Failed password" /var/log/auth.log
sudo grep "authentication failure" /var/log/auth.log

# 内核日志
dmesg | grep -i error
dmesg | grep -i fail

# journalctl
journalctl -p err
journalctl -p crit
```

## 总结

本文介绍了 Linux 故障排查：

- ✅ 启动问题（GRUB、文件系统）
- ✅ 登录问题（密码、SSH、账户锁定）
- ✅ 性能问题（CPU、内存、磁盘、负载）
- ✅ 网络问题（连接、端口、DNS）
- ✅ 服务问题（启动失败、重启）
- ✅ 日志分析和调试

继续学习 [最佳实践](./best-practices) 和 [快速参考](./quick-reference)。
