---
sidebar_position: 8
title: 用户和组管理
---

# Linux 用户和组管理

Linux 是一个多用户操作系统，理解用户和组管理对系统安全和权限控制至关重要。

## 用户基础

### 用户类型

- **root（超级用户）** - UID 0，拥有所有权限
- **系统用户** - UID 1-999，用于系统服务
- **普通用户** - UID 1000+，日常使用账户

### 用户信息文件

```bash
# /etc/passwd - 用户账户信息
# 格式：用户名:密码占位符:UID:GID:描述:主目录:Shell
cat /etc/passwd
# 示例：
# john:x:1000:1000:John Doe:/home/john:/bin/bash

# /etc/shadow - 用户密码信息（加密）
sudo cat /etc/shadow
# 格式：用户名:加密密码:最后修改日期:最小间隔:最大间隔:警告期:禁用期:过期日期

# /etc/group - 组信息
cat /etc/group
# 格式：组名:密码:GID:成员列表
```

## 用户管理

### 创建用户

```bash
# 基本创建
sudo useradd username

# 创建并指定选项
sudo useradd -m -s /bin/bash -c "Full Name" username
# -m: 创建主目录
# -s: 指定 Shell
# -c: 添加注释

# 创建用户并设置主目录
sudo useradd -m -d /home/customdir username

# 指定 UID
sudo useradd -u 1500 username

# 指定主组和附加组
sudo useradd -g users -G sudo,developers username

# 设置密码
sudo passwd username

# 交互式创建（推荐）
sudo adduser username
```

### 修改用户

```bash
# 修改用户名
sudo usermod -l newname oldname

# 修改主目录
sudo usermod -d /new/home -m username

# 修改 Shell
sudo usermod -s /bin/zsh username

# 修改 UID
sudo usermod -u 1600 username

# 添加到附加组
sudo usermod -aG groupname username

# 锁定用户
sudo usermod -L username
sudo passwd -l username

# 解锁用户
sudo usermod -U username
sudo passwd -u username

# 设置账户过期
sudo usermod -e 2024-12-31 username
```

### 删除用户

```bash
# 删除用户（保留主目录）
sudo userdel username

# 删除用户及主目录
sudo userdel -r username

# 强制删除（即使用户已登录）
sudo userdel -f username
```

### 查看用户信息

```bash
# 查看当前用户
whoami
id

# 查看用户详细信息
id username
finger username

# 查看已登录用户
who
w
users

# 查看用户最后登录
last
lastlog

# 查看用户所属组
groups username
```

## 组管理

### 创建组

```bash
# 创建组
sudo groupadd groupname

# 指定 GID
sudo groupadd -g 5000 groupname

# 创建系统组
sudo groupadd -r groupname
```

### 修改组

```bash
# 修改组名
sudo groupmod -n newname oldname

# 修改 GID
sudo groupmod -g 5001 groupname
```

### 删除组

```bash
# 删除组
sudo groupdel groupname

# 注意：不能删除用户的主组，需先删除或修改用户
```

### 组成员管理

```bash
# 添加用户到组
sudo usermod -aG groupname username
sudo gpasswd -a username groupname

# 从组中删除用户
sudo gpasswd -d username groupname

# 设置组管理员
sudo gpasswd -A admin_user groupname

# 查看组成员
getent group groupname
members groupname
```

## sudo 配置

### sudo 基础

```bash
# 使用 sudo
sudo command

# 切换到 root
sudo -i
sudo su -

# 以其他用户身份执行
sudo -u username command

# 编辑需要 root 权限的文件
sudo vim /etc/hosts

# 列出 sudo 权限
sudo -l

# 更新 sudo 时间戳
sudo -v

# 清除 sudo 时间戳
sudo -k
```

### 配置 sudoers

```bash
# 安全编辑 sudoers 文件
sudo visudo

# sudoers 文件位置
/etc/sudoers
/etc/sudoers.d/

# 基本语法
# 用户 主机=(运行身份) 命令
# User Host=(RunAs) Commands
```

### sudoers 配置示例

```bash
# 允许用户执行所有命令
username ALL=(ALL:ALL) ALL

# 允许组执行所有命令
%groupname ALL=(ALL:ALL) ALL

# 无密码执行所有命令
username ALL=(ALL) NOPASSWD: ALL

# 无密码执行特定命令
username ALL=(ALL) NOPASSWD: /sbin/shutdown, /sbin/reboot

# 限制特定命令
username ALL=(ALL) /usr/bin/apt, /usr/bin/systemctl

# 命令别名
Cmnd_Alias NETWORKING = /sbin/route, /sbin/ifconfig, /bin/ping
username ALL=NETWORKING

# 用户别名
User_Alias ADMINS = alice, bob, charlie
ADMINS ALL=(ALL) ALL

# 主机别名
Host_Alias SERVERS = server1, server2
username SERVERS=(ALL) ALL
```

### sudo 最佳实践

```bash
# 1. 不要直接修改 /etc/sudoers
# 使用 visudo 或在 /etc/sudoers.d/ 创建文件

# 2. 创建独立的 sudoers 文件
sudo visudo -f /etc/sudoers.d/username

# 3. 文件权限必须是 0440
sudo chmod 0440 /etc/sudoers.d/username

# 4. 使用组管理权限
%sudo ALL=(ALL:ALL) ALL
%wheel ALL=(ALL:ALL) ALL
```

## 用户环境

### 配置文件

```bash
# 全局配置
/etc/profile              # 所有用户登录时执行
/etc/bash.bashrc          # 所有用户的 bash 配置
/etc/environment          # 环境变量

# 用户配置
~/.profile                # 用户登录时执行
~/.bashrc                 # 每次打开 bash 时执行
~/.bash_profile           # 登录 shell 执行
~/.bash_logout            # 退出时执行
~/.bash_history           # 命令历史
```

### 环境变量

```bash
# 查看环境变量
env
printenv
echo $PATH

# 设置临时环境变量
export VAR=value

# 永久设置（当前用户）
echo 'export VAR=value' >> ~/.bashrc
source ~/.bashrc

# 永久设置（所有用户）
sudo vim /etc/environment
# 添加：VAR=value
```

## 密码策略

### 密码设置

```bash
# 修改自己的密码
passwd

# 修改其他用户密码（需 root）
sudo passwd username

# 强制用户下次登录修改密码
sudo passwd -e username

# 生成随机密码
pwgen 16 1
openssl rand -base64 16
```

### 密码策略配置

```bash
# 安装 PAM 密码质量模块
sudo apt install libpam-pwquality

# 配置密码策略
sudo vim /etc/security/pwquality.conf

# 常用配置：
minlen = 12              # 最小长度
dcredit = -1             # 至少1个数字
ucredit = -1             # 至少1个大写字母
lcredit = -1             # 至少1个小写字母
ocredit = -1             # 至少1个特殊字符
```

### 密码有效期

```bash
# 设置密码最长有效期
sudo chage -M 90 username

# 设置密码最短有效期
sudo chage -m 7 username

# 设置密码过期警告
sudo chage -W 14 username

# 查看密码状态
sudo chage -l username

# 配置默认策略
sudo vim /etc/login.defs
# PASS_MAX_DAYS   90
# PASS_MIN_DAYS   7
# PASS_WARN_AGE   14
```

## 用户配额

### 磁盘配额

```bash
# 安装配额工具
sudo apt install quota

# 启用配额（编辑 /etc/fstab）
/dev/sda1 /home ext4 defaults,usrquota,grpquota 0 2

# 重新挂载
sudo mount -o remount /home

# 创建配额文件
sudo quotacheck -cugm /home

# 启用配额
sudo quotaon /home

# 编辑用户配额
sudo edquota username

# 查看配额
quota -u username
sudo repquota /home
```

## 切换用户

### su 命令

```bash
# 切换到其他用户
su username

# 切换到 root
su -
su - root

# 以其他用户身份执行命令
su -c "command" username

# su 和 su - 的区别
# su: 切换用户但保留当前环境
# su -: 切换用户并加载目标用户环境
```

## 批量管理

### 批量创建用户

```bash
#!/bin/bash
# 从文件批量创建用户

while IFS=: read -r username password; do
    sudo useradd -m "$username"
    echo "$username:$password" | sudo chpasswd
    echo "创建用户: $username"
done < users.txt
```

### 批量修改

```bash
# 批量修改密码有效期
for user in $(cut -d: -f1 /etc/passwd); do
    sudo chage -M 90 "$user"
done

# 批量添加到组
for user in alice bob charlie; do
    sudo usermod -aG developers "$user"
done
```

## 用户审计

### 查看用户活动

```bash
# 查看登录历史
last
last -n 10           # 最近10次
last username        # 特定用户

# 查看失败的登录
lastb
sudo lastb

# 查看当前登录
w
who
who -a

# 查看用户进程
ps -u username
```

### 账户安全审计

```bash
# 查找没有密码的账户
sudo awk -F: '($2 == "" ) {print $1}' /etc/shadow

# 查找 UID 为 0 的账户（除了 root）
sudo awk -F: '($3 == 0) {print $1}' /etc/passwd

# 查找空 GID
sudo awk -F: '($4 == "" ) {print $1}' /etc/passwd

# 检查可疑的 shell
grep -v '/nologin\|/false' /etc/passwd
```

## 最佳实践

### 1. 用户管理

```bash
# ✅ 使用普通用户日常工作
# ✅ 需要时使用 sudo
# ✅ 禁用 root 远程登录
# ✅ 定期审查用户账户
# ✅ 删除不需要的账户

# ❌ 不要以 root 身份日常工作
# ❌ 不要共享账户
# ❌ 不要使用简单密码
```

### 2. 组管理

```bash
# 按职能创建组
sudo groupadd developers
sudo groupadd admins
sudo groupadd readonly

# 使用组管理权限
sudo chown :developers /var/www/project
sudo chmod 2775 /var/www/project
```

### 3. sudo 配置

```bash
# 使用最小权限原则
# 只授予必要的命令权限
username ALL=(ALL) NOPASSWD: /sbin/reboot, /sbin/poweroff

# 记录 sudo 操作
Defaults logfile=/var/log/sudo.log
Defaults log_year, log_host, log_input, log_output
```

### 4. 安全建议

```bash
# 1. 强制使用强密码
# 配置 pwquality

# 2. 实施密码有效期
# 使用 chage 设置

# 3. 禁用不必要的账户
sudo usermod -L username

# 4. 监控可疑活动
# 定期检查 /var/log/auth.log

# 5. 使用 SSH 密钥替代密码
# 配置公钥认证
```

## 故障排查

### 无法登录

```bash
# 检查账户是否锁定
sudo passwd -S username

# 检查密码有效期
sudo chage -l username

# 检查 Shell 是否有效
grep username /etc/passwd

# 重置密码（单用户模式）
passwd username
```

### sudo 问题

```bash
# 检查 sudo 权限
sudo -l

# 检查 sudoers 语法
sudo visudo -c

# 查看 sudo 日志
sudo cat /var/log/auth.log | grep sudo
```

## 总结

本文介绍了 Linux 用户和组管理：

- ✅ 用户创建、修改、删除
- ✅ 组管理和成员控制
- ✅ sudo 配置和最佳实践
- ✅ 密码策略和安全
- ✅ 用户审计和故障排查

继续学习 [软件包管理](./package-management) 和 [系统管理](./system-admin)。
