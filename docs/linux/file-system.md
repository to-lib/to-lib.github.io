---
sidebar_position: 3
title: 文件系统
---

# Linux 文件系统

Linux 采用层次化的文件系统结构，所有文件和目录都从根目录 `/` 开始。

## 文件系统层次结构

### 标准目录结构

```
/
├── bin/       → 基本用户命令（如 ls, cp, mv）
├── boot/      → 启动加载器文件（内核、initrd）
├── dev/       → 设备文件
├── etc/       → 系统配置文件
├── home/      → 用户主目录
├── lib/       → 共享库文件
├── media/     → 可移动媒体挂载点
├── mnt/       → 临时挂载点
├── opt/       → 第三方软件
├── proc/      → 进程和内核信息（虚拟）
├── root/      → root 用户主目录
├── run/       → 运行时数据
├── sbin/      → 系统管理命令
├── srv/       → 服务数据
├── sys/       → 设备和驱动信息（虚拟）
├── tmp/       → 临时文件
├── usr/       → 用户程序和数据
└── var/       → 可变数据（日志、缓存等）
```

### 重要目录详解

#### /etc - 配置文件

```bash
/etc/passwd          # 用户账户信息
/etc/shadow          # 用户密码
/etc/group           # 组信息
/etc/fstab           # 文件系统挂载表
/etc/hosts           # 主机名解析
/etc/hostname        # 主机名
/etc/network/        # 网络配置
/etc/ssh/            # SSH 配置
```

#### /var - 可变数据

```bash
/var/log/            # 系统日志
/var/cache/          # 应用程序缓存
/var/tmp/            # 临时文件（持久）
/var/spool/          # 任务队列
/var/www/            # Web 服务器文件
```

#### /usr - 用户程序

```bash
/usr/bin/            # 用户命令
/usr/sbin/           # 系统管理命令
/usr/lib/            # 库文件
/usr/local/          # 本地安装的软件
/usr/share/          # 共享数据
```

## 文件类型

### 七种文件类型

| 类型 | 符号 | 说明 | 示例 |
|------|------|------|------|
| 普通文件 | `-` | 文本、二进制文件 | `-rw-r--r--` |
| 目录 | `d` | 文件夹 | `drwxr-xr-x` |
| 符号链接 | `l` | 软链接 | `lrwxrwxrwx` |
| 字符设备 | `c` | 字符设备文件 | `crw-rw-rw-` |
| 块设备 | `b` | 块设备文件 | `brw-rw----` |
| 套接字 | `s` | 网络套接字 | `srwxr-xr-x` |
| 管道 | `p` | 命名管道 | `prw-r--r--` |

### 查看文件类型

```bash
# 使用 ls -l
ls -l file.txt
# -rw-r--r-- 1 user group 1234 Dec 9 10:00 file.txt

# 使用 file 命令
file file.txt
file /dev/sda
file directory/
```

## 链接文件

### 硬链接

```bash
# 创建硬链接
ln source.txt hardlink.txt

# 特点：
# - 指向相同的 inode
# - 删除源文件不影响硬链接
# - 不能跨文件系统
# - 不能链接目录
```

### 软链接（符号链接）

```bash
# 创建软链接
ln -s source.txt symlink.txt
ln -s /path/to/source /path/to/link

# 特点：
# - 类似 Windows 快捷方式
# - 可以跨文件系统
# - 可以链接目录
# - 删除源文件会导致链接失效
```

### 链接对比

```bash
# 创建测试文件
echo "Hello" > original.txt

# 创建硬链接
ln original.txt hard.txt

# 创建软链接
ln -s original.txt soft.txt

# 查看 inode
ls -li
# 输出示例：
# 12345 -rw-r--r-- 2 user group 6 Dec 9 10:00 original.txt
# 12345 -rw-r--r-- 2 user group 6 Dec 9 10:00 hard.txt
# 67890 lrwxrwxrwx 1 user group 12 Dec 9 10:00 soft.txt -> original.txt
```

## 挂载文件系统

### 挂载和卸载

```bash
# 查看已挂载的文件系统
mount
df -h

# 挂载设备
sudo mount /dev/sdb1 /mnt/usb
sudo mount -t ext4 /dev/sdb1 /mnt/disk

# 挂载 ISO 文件
sudo mount -o loop image.iso /mnt/iso

# 卸载
sudo umount /mnt/usb
sudo umount /dev/sdb1
```

### /etc/fstab 配置

```bash
# 查看 fstab
cat /etc/fstab

# fstab 格式：
# <设备> <挂载点> <文件系统类型> <选项> <dump> <fsck>

# 示例：
UUID=xxx-xxx /               ext4    defaults        0 1
UUID=xxx-xxx /home           ext4    defaults        0 2
/dev/sdb1    /mnt/data       ext4    defaults        0 0
//server/share /mnt/cifs     cifs    credentials=/etc/samba/creds 0 0
```

### 常用挂载选项

```bash
# 只读挂载
sudo mount -o ro /dev/sdb1 /mnt/usb

# 读写挂载
sudo mount -o rw /dev/sdb1 /mnt/usb

# 设置权限
sudo mount -o uid=1000,gid=1000 /dev/sdb1 /mnt/usb

# 重新挂载
sudo mount -o remount,rw /
```

## 磁盘管理

### 查看磁盘信息

```bash
# 磁盘使用情况
df -h                         # 人类可读
df -i                         # inode 使用情况

# 目录大小
du -sh /path                  # 目录总大小
du -h --max-depth=1           # 一级子目录大小
du -sh * | sort -rh           # 按大小排序

# 磁盘分区信息
lsblk                         # 块设备列表
sudo fdisk -l                 # 分区详情
```

### 分区管理

```bash
# 使用 fdisk（传统）
sudo fdisk /dev/sdb
# 命令：
# n - 新建分区
# d - 删除分区
# p - 显示分区表
# w - 写入并退出

# 使用 parted（推荐）
sudo parted /dev/sdb
# (parted) print           # 显示分区
# (parted) mklabel gpt     # 创建 GPT 分区表
# (parted) mkpart primary ext4 0% 100%
```

### 格式化文件系统

```bash
# 格式化为 ext4
sudo mkfs.ext4 /dev/sdb1

# 格式化为 xfs
sudo mkfs.xfs /dev/sdb1

# 格式化为 FAT32
sudo mkfs.vfat -F 32 /dev/sdb1

# 格式化为 NTFS
sudo mkfs.ntfs /dev/sdb1
```

## 文件系统检查和修复

### fsck 文件系统检查

```bash
# 检查文件系统（需先卸载）
sudo umount /dev/sdb1
sudo fsck /dev/sdb1

# 自动修复
sudo fsck -y /dev/sdb1

# 强制检查
sudo fsck -f /dev/sdb1

# 检查 ext4
sudo e2fsck /dev/sdb1
```

### 检查磁盘健康

```bash
# SMART 信息
sudo smartctl -a /dev/sda

# 坏块检查
sudo badblocks -v /dev/sdb1
```

## inode

### 什么是 inode

inode（索引节点）存储文件的元数据：

- 文件大小
- 文件所有者
- 权限
- 时间戳
- 数据块位置

```bash
# 查看 inode 信息
ls -i file.txt
stat file.txt

# 查看 inode 使用情况
df -i
```

### inode 耗尽问题

```bash
# 症状：磁盘空间足够，但无法创建文件
df -h        # 显示空间充足
df -i        # 显示 inode 100% 使用

# 查找大量小文件
find / -xdev -printf '%h\n' | sort | uniq -c | sort -k 1 -n
```

## 特殊文件系统

### /proc 虚拟文件系统

```bash
# 进程信息
/proc/[pid]/               # 进程目录
/proc/[pid]/cmdline        # 命令行
/proc/[pid]/status         # 状态信息

# 系统信息
/proc/cpuinfo              # CPU 信息
/proc/meminfo              # 内存信息
/proc/version              # 内核版本
```

### /sys 设备文件系统

```bash
# 设备信息
/sys/block/               # 块设备
/sys/class/net/           # 网络接口
/sys/devices/             # 设备树
```

### tmpfs 临时文件系统

```bash
# 查看 tmpfs
df -h | grep tmpfs

# /tmp 通常使用 tmpfs（内存）
# /dev/shm 共享内存

# 创建 tmpfs
sudo mount -t tmpfs -o size=1G tmpfs /mnt/ramdisk
```

## 最佳实践

### 1. 目录结构规范

```bash
# 用户数据放 /home
/home/user/documents/

# 应用程序数据放 /opt 或 /usr/local
/opt/myapp/

# 日志放 /var/log
/var/log/myapp/
```

### 2. 磁盘空间管理

```bash
# 定期清理
sudo apt clean                    # 清理包缓存
sudo journalctl --vacuum-time=7d  # 清理旧日志

# 查找大文件
find / -type f -size +100M 2>/dev/null
```

### 3. 备份重要数据

```bash
# 使用 rsync
rsync -av --delete /source/ /backup/

# 使用 tar
tar -czf backup.tar.gz /important/data/
```

## 总结

本文介绍了 Linux 文件系统：

- ✅ 文件系统层次结构
- ✅ 文件类型和链接
- ✅ 挂载和 fstab 配置
- ✅ 磁盘管理和分区
- ✅ 文件系统检查修复
- ✅ inode 和特殊文件系统

继续学习 [权限管理](./permissions) 和 [进程管理](./process-management)。
