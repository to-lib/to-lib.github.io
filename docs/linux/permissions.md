---
sidebar_position: 4
title: 权限管理
---

# Linux 权限管理

Linux 的权限系统是多用户系统安全的基础。理解和正确使用权限是系统管理的重要技能。

## 权限基础

### 权限表示

```bash
# ls -l 输出示例
-rw-r--r-- 1 user group 1234 Dec 9 10:00 file.txt
│││││││││  │ │    │     │    │          │
│││││││││  │ │    │     │    │          └─ 文件名
│││││││││  │ │    │     │    └──────────── 修改时间
│││││││││  │ │    │     └───────────────── 文件大小
│││││││││  │ │    └─────────────────────── 所属组
│││││││││  │ └──────────────────────────── 所有者
│││││││││  └─────────────────────────────── 链接数
││││││││└─ 其他人权限（r--）
│││││└──── 组权限（r--）
││└────── 所有者权限（rw-）
└──────── 文件类型（- 普通文件）
```

### 三种权限

| 权限 | 符号 | 数字 | 对文件的含义 | 对目录的含义 |
|------|------|------|--------------|--------------|
| 读 | r | 4 | 可读取文件内容 | 可列出目录内容 |
| 写 | w | 2 | 可修改文件内容 | 可创建/删除文件 |
| 执行 | x | 1 | 可执行文件 | 可进入目录 |

### 三类用户

- **所有者（Owner）** - 文件的创建者
- **组（Group）** - 文件所属的组
- **其他人（Others）** - 除所有者和组之外的所有用户

## chmod 修改权限

### 数字方式

```bash
# 权限计算
# rwx = 4+2+1 = 7
# rw- = 4+2+0 = 6
# r-x = 4+0+1 = 5
# r-- = 4+0+0 = 4

# 常用权限
chmod 755 file.sh     # rwxr-xr-x（脚本）
chmod 644 file.txt    # rw-r--r--（文档）
chmod 600 secret.txt  # rw-------（私密文件）
chmod 777 file        # rwxrwxrwx（所有人可读写执行）

# 递归修改
chmod -R 755 directory/
```

### 符号方式

```bash
# 基本语法：chmod [ugoa][+-=][rwx] file

# 用户类别
# u - 所有者（user）
# g - 组（group）
# o - 其他人（others）
# a - 所有人（all）

# 操作符
# + 添加权限
# - 移除权限
# = 设置权限

# 示例
chmod u+x file.sh          # 所有者添加执行权限
chmod g-w file.txt         # 组移除写权限
chmod o=r file.txt         # 其他人只读
chmod a+r file.txt         # 所有人可读
chmod ug+rw file.txt       # 所有者和组可读写
chmod a-x file             # 所有人移除执行权限
```

### 特殊权限

#### SUID（Set User ID）

```bash
# 数字：4xxx
chmod 4755 program
chmod u+s program

# 效果：执行时以文件所有者身份运行
# 示例：/usr/bin/passwd
ls -l /usr/bin/passwd
# -rwsr-xr-x root root /usr/bin/passwd
```

#### SGID（Set Group ID）

```bash
# 数字：2xxx  
chmod 2755 directory
chmod g+s directory

# 对文件：执行时以文件所属组身份运行
# 对目录：新建文件继承目录的组
```

#### Sticky Bit

```bash
# 数字：1xxx
chmod 1777 /tmp
chmod +t directory

# 效果：只有文件所有者和root可以删除文件
# 典型应用：/tmp 目录
ls -ld /tmp
# drwxrwxrwt root root /tmp
```

## chown 修改所有者

```bash
# 修改所有者
chown user file.txt
chown user:group file.txt

# 只修改组
chown :group file.txt
chgrp group file.txt

# 递归修改
chown -R user:group directory/

# 示例
sudo chown root:root /etc/passwd
sudo chown www-data:www-data /var/www/html/
```

## 默认权限 umask

```bash
# 查看当前 umask
umask
# 输出：0022

# umask 计算
# 文件默认：666 - umask = 644
# 目录默认：777 - umask = 755

# 设置 umask
umask 027
# 文件：640 (rw-r-----)
# 目录：750 (rwxr-x---)

# 永久设置（添加到 ~/.bashrc）
echo "umask 027" >> ~/.bashrc
```

## ACL 访问控制列表

ACL 提供更细粒度的权限控制。

### 查看 ACL

```bash
# 查看 ACL
getfacl file.txt

# 示例输出
# file: file.txt
# owner: user
# group: group
# user::rw-
# user:alice:rw-
# group::r--
# mask::rw-
# other::r--
```

### 设置 ACL

```bash
# 给特定用户添加权限
setfacl -m u:alice:rw file.txt

# 给特定组添加权限
setfacl -m g:developers:rwx directory/

# 递归设置
setfacl -R -m u:alice:rx directory/

# 设置默认 ACL（新文件继承）
setfacl -d -m u:alice:rw directory/

# 删除 ACL
setfacl -x u:alice file.txt

# 删除所有 ACL
setfacl -b file.txt
```

## sudo 权限提升

### 使用 sudo

```bash
# 以 root 身份执行命令
sudo command

# 切换到 root
sudo -i
sudo su -

# 以其他用户身份执行
sudo -u username command

# 列出 sudo 权限
sudo -l
```

### 配置 sudoers

```bash
# 编辑 sudoers 文件（安全方式）
sudo visudo

# sudoers 语法
# 用户 主机=(身份) 命令
user    ALL=(ALL:ALL) ALL
%group  ALL=(ALL:ALL) ALL

# 示例
# alice 可以执行所有命令
alice ALL=(ALL:ALL) ALL

# bob 可以无密码重启系统
bob ALL=(ALL) NOPASSWD: /sbin/reboot

# developers 组可以执行所有命令
%developers ALL=(ALL:ALL) ALL

# 特定命令权限
john ALL=(ALL) /usr/bin/apt, /usr/bin/systemctl
```

## 常见权限问题

### 问题 1：Permission denied

```bash
# 原因：
# 1. 文件权限不足
# 2. 目录权限不足
# 3. 需要 root 权限

# 解决：
chmod +x script.sh     # 添加执行权限
sudo command           # 使用 sudo
```

### 问题 2：无法删除文件

```bash
# 检查权限
ls -ld directory/
ls -l directory/file.txt

# 需要目录的写权限
chmod u+w directory/

# 检查 sticky bit
# 在有 sticky bit 的目录，只能删除自己的文件
```

### 问题 3：Web 服务器权限

```bash
# 设置 Web 文件权限
sudo chown -R www-data:www-data /var/www/html
sudo find /var/www/html -type d -exec chmod 755 {} \;
sudo find /var/www/html -type f -exec chmod 644 {} \;
```

## 最佳实践

### 1. 最小权限原则

```bash
# 不好：过于宽松
chmod 777 file.txt

# 好：只给必要的权限
chmod 644 file.txt        # 文件
chmod 755 directory/      # 目录
chmod 700 ~/.ssh/         # 敏感目录
chmod 600 ~/.ssh/id_rsa   # 私钥
```

### 2. 使用组管理权限

```bash
# 创建项目组
sudo groupadd developers

# 添加用户到组
sudo usermod -aG developers alice

# 设置目录权限
sudo chown :developers  /var/project
sudo chmod 2775 /var/project
```

### 3. 定期审查权限

```bash
# 查找 SUID 文件
find / -perm -4000 -type f 2>/dev/null

# 查找所有人可写的文件
find / -perm -002 -type f 2>/dev/null

# 查找无主文件
find / -nouser -o -nogroup 2>/dev/null
```

## 权限速查表

### 常用数字权限

| 权限 | 数字 | 说明 | 适用 |
|------|------|------|------|
| rw------- | 600 | 只有所有者可读写 | 私密文件 |
| rw-r--r-- | 644 | 所有者可读写，其他人只读 | 普通文件 |
| rw-rw-r-- | 664 | 所有者和组可读写 | 共享文件 |
| rwx------ | 700 | 只有所有者可以访问 | 私密目录 |
| rwxr-xr-x | 755 | 所有者全权，其他人可读执行 | 目录/脚本 |
| rwxrwxr-x | 775 | 所有者和组全权 | 共享目录 |

### 特殊权limited

```bash
# SUID：4000
chmod 4755 file

# SGID：2000
chmod 2755 directory

# Sticky Bit：1000
chmod 1777 /tmp

# 组合使用：6755（SUID+SGID）
chmod 6755 file
```

## 总结

本文介绍了 Linux 权限管理：

- ✅ 基本权限（rwx）和数字表示
- ✅ chmod、chown 命令
- ✅ 特殊权限（SUID、SGID、Sticky Bit）
- ✅ ACL 高级权限控制
- ✅ sudo 权限提升
- ✅ 权限最佳实践

继续学习 [进程管理](./process-management) 和 [用户和组管理](./users-groups)。
