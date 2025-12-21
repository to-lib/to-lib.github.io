---
sidebar_position: 19
title: 常见问题
---

# Linux 常见问题 FAQ

本文整理 Linux 使用过程中的常见问题和解决方案。

## 系统安装

### Q: 如何选择 Linux 发行版？

**A:** 根据使用场景选择：

- **初学者**：Ubuntu、Linux Mint - 界面友好，社区活跃
- **服务器**：Ubuntu Server、CentOS、Debian - 稳定可靠
- **开发者**：Fedora、Arch Linux - 新技术，滚动更新
- **企业级**：RHEL、SUSE Enterprise - 商业支持

### Q: 双系统好还是虚拟机好？

**A:** 各有优缺点：

**双系统优势：**

- 完整硬件性能
- 真实环境

**虚拟机优势：**

- 随时切换
- 安全隔离
- 可以快照恢复

推荐：初学者使用虚拟机，深度使用者安装双系统

## 命令行

### Q: 如何在终端中复制粘贴？

**A:**

```bash
# 复制：Ctrl+Shift+C 或 鼠标选中自动复制
# 粘贴：Ctrl+Shift+V 或 鼠标中键

# Vim 中：
# 复制：y（yank）
# 粘贴：p（paste）
```

### Q: 如何查看命令帮助？

**A:**

```bash
man command           # 详细手册
command --help        # 快速帮助
info command          # Info文档
whatis command        # 简短说明
```

### Q: 如何查看命令历史？

**A:**

```bash
history               # 查看历史
history | grep keyword  # 搜索历史
!n                    # 执行第n条命令
!!                    # 执行上一条命令
!string               # 执行最近以string开头的命令
Ctrl+R                # 反向搜索历史
```

## 文件管理

### Q: 如何显示隐藏文件？

**A:**

```bash
ls -a                 # 显示所有文件（包括隐藏）
ls -la                # 详细列表包括隐藏文件

# Linux中以.开头的文件是隐藏文件
```

### Q: 如何处理文件名中的空格？

**A:**

```bash
# 使用引号
cp "file name.txt" destination/

# 使用反斜杠转义
cp file\ name.txt destination/

# 使用Tab自动补全
cp file[Tab]
```

### Q: 如何恢复删除的文件？

**A:**

```bash
# rm删除的文件很难恢复，建议：
# 1. 使用trash-cli代替rm
sudo apt install trash-cli
alias rm='trash-put'

# 2. 定期备份重要文件
rsync -av --delete /source/ /backup/

# 3. 如果必须恢复，尽快使用：
sudo apt install extundelete
sudo extundelete /dev/sda1 --restore-all
```

## 权限问题

### Q: Permission denied 怎么办？

**A:**

```bash
# 1. 检查文件权限
ls -l file

# 2. 添加执行权限
chmod +x script.sh

# 3. 修改文件所有者
sudo chown $USER file

# 4. 使用sudo
sudo command

# 5. 检查父目录权限
ls -ld /path/to/directory
```

### Q: sudo 提示用户不在 sudoers 文件中？

**A:**

```bash
# 切换到root用户
su -

# 添加用户到sudo组
usermod -aG sudo username    # Ubuntu/Debian
usermod -aG wheel username   # RHEL/CentOS

# 或编辑sudoers文件
visudo
# 添加：
username ALL=(ALL:ALL) ALL
```

## 网络问题

### Q: 网络连接正常但无法上网？

**A:**

```bash
# 1. 检查网络接口
ip addr

# 2. 检查网关
ip route
ping 192.168.1.1

# 3. 检查DNS
cat /etc/resolv.conf
nslookup google.com
ping 8.8.8.8

# 4. 重启网络
sudo systemctl restart NetworkManager
```

### Q: SSH 连接被拒绝？

**A:**

```bash
# 1. 检查SSH服务
sudo systemctl status sshd

# 2. 检查端口
sudo ss -tuln | grep 22

# 3. 检查防火墙
sudo ufw status
sudo ufw allow 22

# 4. 检查SSH配置
sudo vim /etc/ssh/sshd_config
# 确保：
# Port 22
# PermitRootLogin yes/no
# PasswordAuthentication yes

sudo systemctl restart sshd
```

## 软件安装

### Q: apt-get 和 apt 有什么区别？

**A:**

- `apt` 是新命令，更用户友好，有进度条和彩色输出
- `apt-get` 是传统命令，更稳定，适合脚本
- 日常使用推荐 `apt`

### Q: 如何解决软件包依赖问题？

**A:**

```bash
# Ubuntu/Debian
sudo apt --fix-broken install
sudo dpkg --configure -a
sudo apt install -f

# RHEL/CentOS
sudo yum clean all
sudo yum update
```

### Q: PPA 是什么？

**A:**
PPA (Personal Package Archive) 是 Ubuntu 的第三方软件源

```bash
# 添加PPA
sudo add-apt-repository ppa:user/ppa-name
sudo apt update

# 删除PPA
sudo add-apt-repository --remove ppa:user/ppa-name

# 查看已添加的PPA
ls /etc/apt/sources.list.d/
```

## 磁盘问题

### Q: 磁盘空间不足怎么办？

**A:**

```bash
# 1. 查找大文件
sudo find / -type f -size +100M 2>/dev/null
sudo du -sh /* | sort -rh | head

# 2. 清理包缓存
sudo apt clean
sudo apt autoclean

# 3. 清理日志
sudo journalctl --vacuum-time=7d

# 4. 删除旧内核
sudo apt autoremove

# 5. 清理临时文件
sudo rm -rf /tmp/*
```

### Q: inode 耗尽怎么办？

**A:**

```bash
# 查看inode使用
df -i

# 查找小文件多的目录
sudo find / -xdev -printf '%h\n' | sort | uniq -c | sort -rn | head

# 删除不需要的小文件
sudo find /tmp -type f -delete
```

## 性能问题

### Q: 系统很慢，如何诊断？

**A:**

```bash
# 1. 查看负载
uptime
top

# 2. 查看CPU
top
mpstat 1

# 3. 查看内存
free -h
vmstat 1

# 4. 查看磁盘IO
iostat -x 1
iotop

# 5. 查看网络
iftop
nethogs
```

### Q: 如何限制进程资源使用？

**A:**

```bash
# 使用nice设置优先级
nice -n 10 command

# 使用cpulimit限制CPU
cpulimit -l 50 -p PID

# 使用systemd限制资源
# 在service文件中：
[Service]
CPUQuota=50%
MemoryLimit=512M
```

## 用户管理

### Q: 忘记 root 密码怎么办？

**A:**

```bash
# 1. 重启进入GRUB
# 2. 按'e'编辑启动项
# 3. 在linux行末添加：init=/bin/bash
# 4. 按Ctrl+X启动
# 5. 重新挂载根分区：
mount -o remount,rw /
# 6. 重置密码：
passwd root
# 7. 重启：
exec /sbin/init
```

### Q: 如何禁用用户账户？

**A:**

```bash
# 锁定用户
sudo usermod -L username
sudo passwd -l username

# 设置过期
sudo chage -E 0 username

# 禁用shell
sudo usermod -s /sbin/nologin username
```

## Shell 使用

### Q: 如何让命令在后台运行？

**A:**

```bash
# 后台运行
command &

# 忽略挂断信号
nohup command &

# 使用screen
screen -S session_name
# 执行命令
# Ctrl+A+D分离
screen -r session_name  # 重新连接

# 使用tmux
tmux new -s session_name
# 执行命令
# Ctrl+B+D分离
tmux attach -t session_name
```

### Q: 如何设置命令别名？

**A:**

```bash
# 临时别名
alias ll='ls -lah'
alias update='sudo apt update && sudo apt upgrade'

# 永久别名
echo "alias ll='ls -lah'" >> ~/.bashrc
source ~/.bashrc

# 查看别名
alias

# 删除别名
unalias ll
```

## 日志查看

### Q: 如何查看系统日志？

**A:**

```bash
# 使用journalctl
journalctl                  # 所有日志
journalctl -f               # 实时查看
journalctl -u service       # 特定服务
journalctl --since today    # 今天的日志
journalctl -p err           # 错误级别

# 传统日志文件
tail -f /var/log/syslog     # 系统日志
tail -f /var/log/auth.log   # 认证日志
dmesg                       # 内核日志
```

## 安全相关

### Q: 如何检查系统是否被入侵？

**A:**

```bash
# 1. 检查异常登录
last
lastb

# 2. 检查运行进程
ps aux
top

# 3. 检查网络连接
ss -tuln
netstat -tunlp

# 4. 检查定时任务
crontab -l
ls /etc/cron.*

# 5. 检查启动项
systemctl list-units --type=service

# 6. 查看sudo使用
sudo grep sudo /var/log/auth.log

# 7. 使用安全扫描工具
sudo apt install rkhunter chkrootkit
sudo rkhunter --check
sudo chkrootkit
```

### Q: 如何加固系统安全？

**A:**

```bash
# 1. 定期更新
sudo apt update && sudo apt upgrade

# 2. 配置防火墙
sudo ufw enable
sudo ufw default deny incoming
sudo ufw allow ssh

# 3. SSH安全加固
# 修改默认端口
# 禁用root登录
# 使用密钥认证

# 4. 安装fail2ban
sudo apt install fail2ban

# 5. 最小化安装
# 只安装必要的软件

# 6. 定期审计
sudo lynis audit system
```

## 总结

本文整理了 Linux 常见问题：

- ✅ 系统安装和使用
- ✅ 命令行技巧
- ✅ 文件和权限管理
- ✅ 网络和软件安装
- ✅ 性能优化和安全

继续学习 [面试题集](/docs/interview/linux-interview-questions)。
