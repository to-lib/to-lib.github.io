---
sidebar_position: 20
title: 面试题集
---

# Linux 面试题集

本文整理 Linux 系统管理岗位常见的面试题和参考答案。

## 基础知识

### 1. Linux 和 Unix 的区别是什么？

**答案：**

- **Unix**：商业操作系统，源代码不公开，需要授权费用
- **Linux**：开源操作系统，基于 Unix 设计，免费使用

**主要区别：**

- Linux 是开源的，Unix 是专有的
- Linux 可免费使用，Unix 需要付费
- Linux 支持更多硬件平台
- Linux 社区活跃，更新快

### 2. 什么是发行版？常见的 Linux 发行版有哪些？

**答案：**

发行版是 Linux 内核加上软件包、配置工具形成的完整操作系统。

**常见发行版：**

- **Debian 系**：Ubuntu、Debian、Linux Mint
- **Red Hat 系**：RHEL、CentOS、Fedora
- **独立发行版**：Arch Linux、Gentoo、SUSE

### 3. Linux 目录结构中各个目录的作用

**答案：**

```bash
/bin        # 基本命令二进制文件
/boot       # 启动文件（内核、Initrd）
/dev        # 设备文件
/etc        # 系统配置文件
/home       # 用户主目录
/lib        # 共享库文件
/opt        # 第三方软件
/proc       # 进程和内核信息（虚拟）
/root       # root 用户主目录
/sbin       # 系统管理命令
/tmp        # 临时文件
/usr        # 用户程序和数据
/var        # 可变数据（日志、缓存）
```

## 文件系统和权限

### 4. Linux 文件权限的含义？

**答案：**

```bash
-rwxr-xr-x
│││││││││
│││││││└└─ 其他人权限：r-x（读、执行）
││││└└└─── 组权限：r-x（读、执行）
│└└└────── 所有者权限：rwx（读、写、执行）
└───────── 文件类型（-=文件，d=目录，l=链接）
```

**数字表示：**

- r=4, w=2, x=1
- 755 = rwxr-xr-x
- 644 = rw-r--r--

### 5. 硬链接和软链接的区别？

**答案：**

| 特性       | 硬链接         | 软链接（符号链接） |
| ---------- | -------------- | ------------------ |
| inode      | 相同           | 不同               |
| 跨文件系统 | 不能           | 可以               |
| 链接目录   | 不能           | 可以               |
| 删除源文件 | 链接仍有效     | 链接失效           |
| 创建命令   | ln source link | ln -s source link  |

### 6. 如何查找大于 100MB 的文件？

**答案：**

```bash
find / -type f -size +100M 2>/dev/null
find /home -type f -size +100M -exec ls -lh {} \;

# 按大小排序
find / -type f -size +100M -exec ls -lh {} \; | sort -rh
```

## 进程管理

### 7. 如何查看系统负载？负载值的含义？

**答案：**

```bash
# 查看负载
uptime
w
top

# 输出示例：
load average: 0.50, 0.40, 0.30
#             1分钟 5分钟 15分钟
```

**含义：**

- 负载表示等待 CPU 执行的进程数
- 负载 < CPU 核心数：系统正常
- 负载 = CPU 核心数：系统满载
- 负载 > CPU 核心数：系统过载

### 8. 如何查看占用 CPU 最高的进程？

**答案：**

```bash
# 方法1：top
top
# 然后按 P（按CPU排序）

# 方法2：ps
ps aux --sort=-%cpu | head -10

# 方法3：htop
htop
# 按F6选择排序方式
```

### 9. kill、kill -9 的区别？

**答案：**

```bash
kill PID        # 发送 SIGTERM(15)，允许进程清理资源
kill -9 PID     # 发送 SIGKILL(9)，强制立即终止

kill -15 PID    # 同 kill PID
kill -SIGTERM PID  # 同 kill PID
kill -SIGKILL PID  # 同 kill -9 PID
```

**建议使用顺序：**

1. 先用 `kill PID`
2. 等待几秒
3. 如果进程仍在运行，使用 `kill -9 PID`

### 10. 如何让进程在后台运行？

**答案：**

```bash
# 方法1：&
command &

# 方法2：nohup
nohup command &

# 方法3：screen
screen -S session_name
command
# Ctrl+A+D 分离

# 方法4：tmux
tmux new -s session_name
command
# Ctrl+B+D 分离

# 方法5：systemd
# 创建服务单元文件
```

## 网络管理

### 11. 如何查看网络连接状态？

**答案：**

```bash
# 查看所有连接
ss -tuln
netstat -tuln

# 查看监听端口
ss -tuln | grep LISTEN
netstat -tuln | grep LISTEN

# 查看established连接
ss -tun | grep ESTAB
netstat -tun | grep ESTABLISHED

# 查看端口占用
lsof -i :80
fuser 80/tcp
```

### 12. 如何测试网络连通性？

**答案：**

```bash
# 测试连通性
ping host

# 测试端口
telnet host port
nc -zv host port

# 追踪路由
traceroute host
mtr host

# 测试 DNS
nslookup domain
dig domain

# 测试 HTTP
curl -I http://example.com
wget --spider http://example.com
```

### 13. 如何配置静态 IP？

**答案：**

**Ubuntu 18.04+ (netplan)：**

```yaml
sudo vim /etc/netplan/01-netcfg.yaml

network:
  version: 2
  ethernets:
    eth0:
      addresses: [192.168.1.100/24]
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]

sudo netplan apply
```

**RHEL/CentOS：**

```bash
sudo vim /etc/sysconfig/network-scripts/ifcfg-eth0

BOOTPROTO=static
IPADDR=192.168.1.100
NETMASK=255.255.255.0
GATEWAY=192.168.1.1
DNS1=8.8.8.8

sudo systemctl restart network
```

## 系统管理

### 14. 如何查看系统 CPU、内存、磁盘信息？

**答案：**

```bash
# CPU
lscpu
cat /proc/cpuinfo
nproc  # CPU核心数

# 内存
free -h
cat /proc/meminfo

# 磁盘
df -h          # 磁盘空间
du -sh /path   # 目录大小
lsblk          # 块设备
fdisk -l       # 分区信息
```

### 15. 如何查看系统启动时间和运行时间？

**答案：**

```bash
# 系统运行时间
uptime

# 系统启动时间
who -b
last reboot

# 详细启动信息
systemd-analyze
systemd-analyze blame  # 启动耗时分析
```

### 16. systemd 和 init 的区别？

**答案：**

| 特性     | systemd      | init (sysvinit) |
| -------- | ------------ | --------------- |
| 启动速度 | 并行启动，快 | 串行启动，慢    |
| 服务管理 | systemctl    | service         |
| 配置文件 | Unit 文件    | Shell 脚本      |
| 依赖管理 | 自动处理     | 需手动配置      |
| 日志     | journald     | 传统日志文件    |

**systemd 优势：**

- 更快的启动速度
- 按需启动服务
- 更好的依赖管理
- 统一的日志系统

## 日志和监控

### 17. 如何查看系统日志？

**答案：**

```bash
# journalctl（systemd）
journalctl                # 所有日志
journalctl -f             # 实时查看
journalctl -u service     # 特定服务
journalctl --since today  # 今天的日志
journalctl -p err         # 错误级别

# 传统日志
tail -f /var/log/syslog   # 系统日志
tail -f /var/log/auth.log # 认证日志
dmesg                     # 内核日志
```

### 18. 如何查看登录失败的记录？

**答案：**

```bash
# 查看失败登录
lastb
sudo lastb

# 从auth.log查看
grep "Failed password" /var/log/auth.log
grep "authentication failure" /var/log/auth.log

# 统计失败次数
grep "Failed password" /var/log/auth.log | wc -l

# 按IP统计
grep "Failed password" /var/log/auth.log | awk '{print $11}' | sort | uniq -c | sort -rn
```

## 磁盘管理

### 19. 如何创建和挂载分区？

**答案：**

```bash
# 1. 创建分区
sudo fdisk /dev/sdb
# n - 新建分区
# p - 主分区
# w - 写入

# 2. 格式化
sudo mkfs.ext4 /dev/sdb1

# 3. 创建挂载点
sudo mkdir /mnt/data

# 4. 挂载
sudo mount /dev/sdb1 /mnt/data

# 5. 永久挂载（/etc/fstab）
echo '/dev/sdb1 /mnt/data ext4 defaults 0 2' | sudo tee -a /etc/fstab
```

### 20. LVM 的优势是什么？如何创建 LVM？

**答案：**

**LVM 优势：**

- 灵活调整分区大小
- 跨多个磁盘创建逻辑卷
- 支持快照功能
- 方便数据迁移

**创建步骤：**

```bash
# 1. 创建物理卷
sudo pvcreate /dev/sdb1

# 2. 创建卷组
sudo vgcreate vg_data /dev/sdb1

# 3. 创建逻辑卷
sudo lvcreate -L 10G -n lv_data vg_data

# 4. 格式化
sudo mkfs.ext4 /dev/vg_data/lv_data

# 5. 挂载
sudo mount /dev/vg_data/lv_data /mnt/data
```

## 安全

### 21. 如何加固 SSH 安全？

**答案：**

```bash
sudo vim /etc/ssh/sshd_config

# 1. 修改默认端口
Port 2222

# 2. 禁止root登录
PermitRootLogin no

# 3. 禁用密码认证
PasswordAuthentication no
PubkeyAuthentication yes

# 4. 限制登录尝试
MaxAuthTries 3

# 5. 设置超时
LoginGraceTime 60
ClientAliveInterval 300
ClientAliveCountMax 2

# 6. 限制用户
AllowUsers user1 user2

# 7. 禁用空密码
PermitEmptyPasswords no

# 重启SSH
sudo systemctl restart sshd
```

### 22. iptables 和 firewalld 的区别？

**答案：**

| 特性       | iptables   | firewalld        |
| ---------- | ---------- | ---------------- |
| 管理方式   | 规则表     | 区域（zone）     |
| 动态修改   | 不支持     | 支持             |
| 默认系统   | 传统 Linux | RHEL 7+          |
| 配置复杂度 | 较高       | 较低             |
| 底层       | netfilter  | iptables（封装） |

## Shell 脚本

### 23. Shell 脚本中 $0, $1, $#, $@ 的含义？

**答案：**

```bash
$0    # 脚本名称
$1    # 第一个参数
$2    # 第二个参数
$#    # 参数个数
$@    # 所有参数（保留空格）
$*    # 所有参数（作为一个字符串）
$?    # 上一个命令的退出状态
$$    # 当前进程ID
```

### 24. 如何调试 Shell 脚本？

**答案：**

```bash
# 方法1：bash -x
bash -x script.sh

# 方法2：set -x
#!/bin/bash
set -x    # 开启调试
command
set +x    # 关闭调试

# 方法3：使用 echo
echo "Debug: variable = $variable"

# 方法4：检查语法
bash -n script.sh

# 最佳实践
#!/bin/bash
set -euo pipefail  # 严格模式
```

## 性能优化

### 25. 如何优化系统性能？

**答案：**

**CPU 优化：**

```bash
# 查看CPU使用
top, htop, mpstat

# 调整进程优先级
nice -n 10 command
renice -n 5 -p PID
```

**内存优化：**

```bash
# 查看内存
free -h

# 调整swappiness
echo 10 | sudo tee /proc/sys/vm/swappiness

# 清理缓存
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

**磁盘 IO 优化：**

```bash
# 查看IO
iostat, iotop

# 文件系统挂载优化
# noatime,nodiratime 选项

# 调整IO调度器
echo noop | sudo tee /sys/block/sda/queue/scheduler
```

## 总结

本文整理了 Linux 常见面试题：

- ✅ 基础知识（系统架构、文件系统）
- ✅ 权限管理和用户管理
- ✅ 进程和网络管理
- ✅ 系统管理和监控
- ✅ 磁盘和 LVM
- ✅ 安全加固
- ✅ Shell 脚本
- ✅ 性能优化

掌握这些知识点，可以应对大多数 Linux 系统管理岗位的面试。建议结合实践操作加深理解。
