---
sidebar_position: 18
title: 快速参考
---

# Linux 快速参考

本文提供 Linux 常用命令和操作的快速参考。

## 文件操作

### 基本操作

```bash
# 目录导航
pwd                     # 当前目录
cd /path               # 切换目录
cd ~                   # 主目录
cd -                   # 上一个目录

# 列出文件
ls                     # 列出文件
ls -l                  # 详细信息
ls -a                  # 显示隐藏文件
ls -lh                 # 人类可读大小

# 创建/删除
mkdir dir              # 创建目录
mkdir -p a/b/c         # 递归创建
touch file             # 创建文件
rm file                # 删除文件
rm -r dir              # 递归删除
rm -f file             # 强制删除

# 复制/移动
cp src dest            # 复制文件
cp -r src dest         # 递归复制
mv src dest            # 移动/重命名

# 查看文件
cat file               # 显示全部内容
less file              # 分页查看
head -n 10 file        # 前10行
tail -n 10 file        # 后10行
tail -f file           # 实时查看
```

### 文件搜索

```bash
# 查找文件
find /path -name "*.txt"
find /path -type f -size +100M
find /path -mtime -7

# 快速查找
locate filename
which command

# 文件内容搜索
grep "pattern" file
grep -r "pattern" /path
grep -i "pattern" file     # 忽略大小写
```

## 权限管理

```bash
# 查看权限
ls -l file

# 修改权限
chmod 755 file         # rwxr-xr-x
chmod u+x file         # 所有者添加执行
chmod g-w file         # 组移除写
chmod o=r file         # 其他人只读

# 修改所有者
chown user file
chown user:group file
chown -R user:group dir

# 常用权限
644 file               # rw-r--r-- 普通文件
755 dir                # rwxr-xr-x 目录
600 secret             # rw------- 私密文件
700 bin                # rwx------ 可执行文件
```

## 用户管理

```bash
# 用户操作
whoami                 # 当前用户
id                     # 用户信息
sudo command           # 以root执行
su - user              # 切换用户

# 用户管理
sudo useradd username
sudo passwd username
sudo usermod -aG group user
sudo userdel username

# 组管理
groups                 # 查看组
sudo groupadd group
sudo groupdel group
```

## 进程管理

```bash
# 查看进程
ps aux                 # 所有进程
ps aux | grep name     # 搜索进程
top                    # 实时监控
htop                   # 增强版top
pgrep name             # 查找进程ID

# 进程控制
kill PID               # 终止进程
kill -9 PID            # 强制终止
killall name           # 按名称终止
pkill name             # 按名称终止

# 后台任务
command &              # 后台运行
jobs                   # 查看后台任务
fg %1                  # 转前台
bg %1                  # 转后台
nohup command &        # 忽略挂断
```

## 系统信息

```bash
# 系统
uname -a               # 系统信息
hostname               # 主机名
uptime                 # 运行时间
date                   # 日期时间

# 硬件
lscpu                  # CPU信息
free -h                # 内存信息
df -h                  # 磁盘空间
lsblk                  # 块设备

# 网络
ip addr                # IP地址
ip route               # 路由表
ping host              # 测试连接
ss -tuln               # 端口监听
```

## 软件包管理

### Debian/Ubuntu (APT)

```bash
sudo apt update                  # 更新列表
sudo apt upgrade                 # 升级软件
sudo apt install package         # 安装
sudo apt remove package          # 卸载
sudo apt search keyword          # 搜索
apt list --installed             # 已安装
```

### RHEL/CentOS (YUM/DNF)

```bash
sudo yum update                  # 更新
sudo yum install package         # 安装
sudo yum remove package          # 卸载
sudo yum search keyword          # 搜索
yum list installed               # 已安装
```

## 网络操作

```bash
# 网络信息
ip addr                # IP地址
ip link                # 网络接口
ip route               # 路由

# 连接测试
ping host              # ICMP测试
traceroute host        # 路由追踪
nslookup domain        # DNS查询
dig domain             # DNS详细

# 端口操作
ss -tuln               # 监听端口
lsof -i :80            # 端口占用
nc -zv host port       # 端口测试

# 下载
wget URL               # 下载文件
curl URL               # HTTP请求
curl -O URL            # 下载并保存
```

## 文件压缩

```bash
# tar
tar -czf file.tar.gz dir/        # 压缩
tar -xzf file.tar.gz             # 解压
tar -tzf file.tar.gz             # 查看

# gzip
gzip file                        # 压缩
gunzip file.gz                   # 解压

# zip
zip -r file.zip dir/             # 压缩
unzip file.zip                   # 解压
```

## 文本处理

```bash
# 查看
cat file               # 显示全部
less file              # 分页查看
head file              # 前10行
tail file              # 后10行

# 编辑
vim file               # Vim编辑器
nano file              # Nano编辑器

# 搜索
grep "pattern" file
grep -r "pattern" dir/
grep -i "pattern" file # 忽略大小写

# 处理
sort file              # 排序
uniq file              # 去重
wc file                # 统计
cut -d: -f1 file       # 分割
tr ' ' '\n' < file      # 替换
```

## 磁盘管理

```bash
# 磁盘信息
df -h                  # 磁盘空间
du -sh /path           # 目录大小
du -h --max-depth=1    # 一级目录大小
lsblk                  # 块设备
fdisk -l               # 分区列表

# 挂载
mount /dev/sdb1 /mnt
umount /mnt
cat /etc/fstab         # 自动挂载配置
```

## 服务管理

```bash
# systemd
sudo systemctl start service
sudo systemctl stop service
sudo systemctl restart service
sudo systemctl status service
sudo systemctl enable service
sudo systemctl disable service

# 查看
systemctl list-units --type=service
systemctl --failed

# 日志
journalctl -u service
journalctl -f
```

## 防火墙

### UFW (Ubuntu)

```bash
sudo ufw enable
sudo ufw disable
sudo ufw status
sudo ufw allow 22
sudo ufw deny 80
sudo ufw delete allow 80
```

### firewalld (CentOS)

```bash
sudo firewall-cmd --state
sudo firewall-cmd --list-all
sudo firewall-cmd --add-port=80/tcp --permanent
sudo firewall-cmd --reload
```

## SSH 操作

```bash
# 连接
ssh user@host
ssh -p port user@host
ssh -i key user@host

# 密钥
ssh-keygen -t ed25519
ssh-copy-id user@host

# 传输文件
scp file user@host:/path
scp user@host:/path file
scp -r dir user@host:/path

# rsync
rsync -av source/ dest/
rsync -av user@host:/remote/ /local/
```

## 日志查看

```bash
# 系统日志
tail -f /var/log/syslog
tail -f /var/log/messages

# 认证日志
tail -f /var/log/auth.log

# journalctl
journalctl -f
journalctl -u service
journalctl --since today
journalctl -p err

# dmesg
dmesg
dmesg | grep -i error
```

## 性能监控

```bash
# CPU
top                    # 实时监控
htop                   # 增强版
mpstat 1               # CPU统计

# 内存
free -h                # 内存使用
vmstat 1               # 虚拟内存统计

# 磁盘IO
iostat                 # IO统计
iotop                  # IO监控

# 网络
iftop                  # 网络流量
nethogs                # 进程网络使用
```

## 环境变量

```bash
# 查看
env                    # 所有环境变量
echo $PATH             # PATH变量
printenv HOME          # 特定变量

# 设置
export VAR=value       # 临时设置
echo 'export VAR=value' >> ~/.bashrc # 永久设置
source ~/.bashrc       # 重新加载
```

## Shell 技巧

### 快捷键

```bash
Ctrl+A                 # 行首
Ctrl+E                 # 行尾
Ctrl+U                 # 删除到行首
Ctrl+K                 # 删除到行尾
Ctrl+W                 # 删除前一个单词
Ctrl+L                 # 清屏
Ctrl+R                 # 搜索历史
Ctrl+C                 # 中断命令
Ctrl+Z                 # 暂停命令
Ctrl+D                 # 退出
```

### 管道和重定向

```bash
command1 | command2    # 管道
command > file         # 重定向输出（覆盖）
command >> file        # 重定向输出（追加）
command < file         # 重定向输入
command 2> file        # 错误输出
command &> file        # 所有输出
```

## 常用正则

```bash
.                      # 任意字符
*                      # 0或多次
+                      # 1或多次
?                      # 0或1次
^                      # 行首
$                      # 行尾
[abc]                  # 字符集
[^abc]                 # 非字符集
\d                     # 数字
\w                     # 字母数字下划线
\s                     # 空白字符
```

## 时间日期

```bash
date                   # 当前日期时间
date +%Y%m%d           # 格式化日期
date -d "yesterday"    # 昨天
date -d "1 week ago"   # 一周前
timedatectl            # 时间设置
```

## 定时任务

```bash
# crontab
crontab -e             # 编辑
crontab -l             # 列出
crontab -r             # 删除

# 格式: 分 时 日 月 周 命令
0 2 * * * /backup.sh   # 每天2点
*/15 * * * * script    # 每15分钟
0 0 * * 0 script       # 每周日0点
```

## 符号说明

```bash
/                      # 根目录
~                      # 主目录
.                      # 当前目录
..                     # 上级目录
-                      # 上一个目录
*                      # 通配符
?                      # 单字符通配
|                      # 管道
>                      # 重定向
>>                     # 追加重定向
&                      # 后台运行
&&                     # 逻辑与
||                     # 逻辑或
;                      # 命令分隔
$                      # 变量
#                      # 注释
```

## 总结

本文提供了 Linux 常用命令的快速参考，涵盖：

- ✅ 文件和目录操作
- ✅ 权限和用户管理
- ✅ 进程和服务管理
- ✅ 网络操作
- ✅ 系统监控
- ✅ Shell 技巧

继续学习 [常见问题](/docs/linux/faq) 和 [面试题集](/docs/linux/interview-questions)。
