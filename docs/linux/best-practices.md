---
sidebar_position: 17
title: 最佳实践
---

# Linux 最佳实践

本文总结 Linux 系统管理和使用的最佳实践，帮助你构建稳定、安全、高效的系统。

## 系统管理

### 1. 最小安装原则

```bash
# ✅ 只安装必要的软件包
# ✅ 禁用不需要的服务
# ✅ 移除默认安装但不使用的软件

# 查看运行的服务
systemctl list-units --type=service --state=running

# 禁用不需要的服务
sudo systemctl disable bluetooth
sudo systemctl stop bluetooth
```

### 2. 定期更新

```bash
# 启用自动安全更新
sudo apt install unattended-upgrades
sudo dpkg-reconfigure unattended-upgrades

# 手动更新
sudo apt update && sudo apt upgrade -y

# 重启后更新
sudo apt dist-upgrade
sudo reboot
```

### 3. 配置管理

```bash
# ✅ 使用版本控制管理配置
cd /etc
sudo git init
sudo git add nginx/ apache2/
sudo git commit -m "Initial config"

# ✅ 修改前备份
sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup.$(date +%Y%m%d)

# ✅ 使用配置管理工具
# - Ansible
# - Puppet
# - Chef
```

## 安全实践

### 1. 用户管理

```bash
# ✅ 使用普通用户日常工作
# ✅ 需要时使用 sudo
# ✅ 禁用 root 远程登录

sudo vim /etc/ssh/sshd_config
PermitRootLogin no

# ✅ 使用强密码策略
sudo vim /etc/security/pwquality.conf
minlen = 12
dcredit = -1
ucredit = -1
lcredit = -1
```

### 2. SSH 安全

```bash
# ✅ 修改默认端口
Port 2222

# ✅ 禁用密码认证，使用密钥
PasswordAuthentication no
PubkeyAuthentication yes

# ✅ 限制登录尝试
MaxAuthTries 3

# ✅ 使用 Fail2ban
sudo apt install fail2ban
```

### 3. 防火墙

```bash
# ✅ 启用防火墙
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# ✅ 只开放必要的端口
sudo ufw allow 22/tcp    # 或你的 SSH 端口
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

### 4. 系统加固

```bash
# ✅ 禁用不需要的服务
sudo systemctl disable bluetooth
sudo systemctl disable cups

# ✅使用 SELinux 或 AppArmor
sudo aa-status

# ✅ 定期审计
sudo apt install lynis
sudo lynis audit system
```

## 文件系统

### 1. 目录组织

```bash
# ✅ 遵循 FHS（文件系统层次标准）
/home/          # 用户数据
/opt/           # 第三方应用
/usr/local/     # 本地安装软件
/var/log/       # 日志文件
/etc/           # 配置文件

# ✅ 分离分区
/boot           # 引导分区
/home           # 用户数据（单独分区便于重装系统）
/var            # 变化数据（防止日志填满根分区）
/tmp            # 临时文件（可使用 tmpfs）
```

### 2. 权限管理

```bash
# ✅ 最小权限原则
chmod 755 /path/to/directory     # 目录
chmod 644 /path/to/file          # 普通文件
chmod 600 ~/.ssh/id_rsa          # 私钥
chmod 700 ~/.ssh                 # SSH 目录

# ✅ 定期检查权限
find / -perm -4000 2>/dev/null   # SUID 文件
find / -perm -002 -type f 2>/dev/null  # 所有人可写
```

### 3. 备份策略

```bash
# ✅ 3-2-1 备份原则
# - 3 份副本
# - 2 种不同介质
# - 1 份异地备份

# ✅ 定期备份
# - 每日增量备份
# - 每周完整备份
# - 每月异地备份

# 使用 rsync
rsync -av --delete /source/ /backup/

# 使用 tar
tar -czf backup-$(date +%Y%m%d).tar.gz /important/data
```

## 性能优化

### 1. 系统调优

```bash
# ✅ 调整 swappiness
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf

# ✅ 文件系统挂载优化
# /etc/fstab
/dev/sda1 / ext4 defaults,noatime,nodiratime 0 1

# ✅ 增加文件描述符限制
sudo vim /etc/security/limits.conf
*  soft  nofile  65536
*  hard  nofile  65536
```

### 2. 监控

```bash
# ✅ 设置监控系统
# - CPU 使用率 > 80%
# - 内存使用率 > 90%
# - 磁盘使用率 > 85%
# - 负载 > CPU 核心数

# ✅ 定期查看日志
sudo journalctl -p err
sudo grep -i error /var/log/syslog
```

## 网络配置

### 1. DNS 配置

```bash
# ✅ 配置可靠的 DNS
sudo vim /etc/resolv.conf
nameserver 8.8.8.8
nameserver 8.8.4.4
nameserver 1.1.1.1

# ✅ 使用 systemd-resolved
sudo vim /etc/systemd/resolved.conf
DNS=8.8.8.8 1.1.1.1
```

### 2. 网络安全

```bash
# ✅ 禁用不需要的网络服务
sudo systemctl disable avahi-daemon

# ✅ 配置防火墙规则
sudo ufw default deny incoming

# ✅ 使用 fail2ban 防止暴力破解
```

## 日志管理

### 1. 日志轮转

```bash
# ✅ 配置 logrotate
sudo vim /etc/logrotate.d/myapp

/var/log/myapp/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data adm
    sharedscripts
    postrotate
        systemctl reload myapp
    endscript
}
```

### 2. 日志监控

```bash
# ✅ 定期检查重要日志
# - /var/log/auth.log - 认证日志
# - /var/log/syslog - 系统日志
# - /var/log/nginx/ - Web 服务器日志

# ✅ 设置日志告警
sudo apt install logwatch
```

## Shell 脚本

### 1. 脚本规范

```bash
#!/bin/bash
set -euo pipefail

# 脚本说明
# 用途：系统备份
# 作者：Your Name
# 日期：2024-12-10

# 全局变量使用大写
BACKUP_DIR="/backup"
LOG_FILE="/var/log/backup.log"

# 函数使用小写
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# 错误处理
error_exit() {
    log "ERROR: $1"
    exit 1
}

# 主函数
main() {
    log "Backup started"
    # 备份逻辑
    log "Backup completed"
}

main "$@"
```

### 2. 脚本最佳实践

```bash
# ✅ 使用 set 选项
set -e              # 遇到错误退出
set -u              # 使用未定义变量退出
set -o pipefail     # 管道错误退出

# ✅ 使用引号
echo "$variable"    # 而不是 echo $variable

# ✅ 检查命令是否存在
if ! command -v rsync &> /dev/null; then
    echo "rsync not found"
    exit 1
fi

# ✅ 使用函数
function backup() {
    local source=$1
    local dest=$2
    # 备份逻辑
}

# ✅ 记录日志
log() {
    echo "[$(date)] $*" >> /var/log/script.log
}
```

## 服务管理

### 1. Systemd 服务

```bash
# ✅ 使用 systemd 管理服务
sudo vim /etc/systemd/system/myapp.service

[Unit]
Description=My Application
After=network.target

[Service]
Type=simple
User=myapp
WorkingDirectory=/opt/myapp
ExecStart=/opt/myapp/bin/myapp
Restart=on-failure
RestartSec=5s

# 资源限制
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

### 2. 服务监控

```bash
# ✅ 监控服务状态
systemctl status myapp

# ✅ 查看服务日志
journalctl -u myapp -f

# ✅ 设置开机自启
sudo systemctl enable myapp
```

## 数据库

### 1. MySQL/MariaDB

```bash
# ✅ 定期备份
mysqldump -u root -p --all-databases > backup.sql

# ✅ 优化配置
[mysqld]
innodb_buffer_pool_size = 2G    # 70% 可用内存
max_connections = 500
query_cache_size = 64M

# ✅ 慢查询日志
slow_query_log = 1
slow_query_log_file = /var/log/mysql/slow.log
long_query_time = 2
```

### 2. Redis

```bash
# ✅ 持久化配置
appendonly yes
appendfsync everysec

# ✅ 内存管理
maxmemory 2gb
maxmemory-policy allkeys-lru

# ✅ 定期备份 RDB
save 900 1
save 300 10
```

## 容器化

### 1. Docker 最佳实践

```bash
# ✅ 使用官方镜像
FROM node:18-alpine

# ✅ 最小化镜像层
RUN apt-get update && apt-get install -y \
    package1 \
    package2 \
    && rm -rf /var/lib/apt/lists/*

# ✅ 使用 .dockerignore
.git
node_modules
*.log

# ✅ 设置资源限制
docker run --memory="512m" --cpus="1.0" myapp
```

## 文档化

### 1. 系统文档

```bash
# ✅ 记录系统配置
# - 服务器规格
# - 安装的软件
# - 配置更改
# - 网络拓扑

# ✅ 维护运维手册
# - 常见操作流程
# - 故障排查步骤
# - 应急响应计划

# ✅ 记录变更
# - 使用 Git 管理配置文件
# - 维护变更日志
```

### 2. 注释规范

```bash
# ✅ Shell 脚本注释
# 脚本开头说明用途
# 复杂逻辑添加注释
# 函数添加说明

# ✅ 配置文件注释
# 说明每个配置项的作用
# 记录修改原因和时间
```

## 团队协作

### 1. 权限分离

```bash
# ✅ 按职能分配权限
# - 开发人员：只读访问生产环境
# - 运维人员：完整权限
# - 审计人员：日志访问权限

# ✅ 使用组管理
sudo groupadd developers
sudo groupadd operators
sudo usermod -aG developers alice
```

### 2. 审计追踪

```bash
# ✅ 启用命令审计
sudo apt install auditd

# ✅ 记录 sudo 使用
Defaults logfile=/var/log/sudo.log

# ✅ 定期审查日志
sudo grep sudo /var/log/auth.log
```

## 灾难恢复

### 1. 备份验证

```bash
# ✅ 定期测试备份恢复
# ✅ 文档化恢复流程
# ✅ 演练灾难恢复

# 验证备份完整性
tar -tzf backup.tar.gz > /dev/null
```

### 2. 应急预案

```bash
# ✅ 制定应急响应计划
# - 问题分类和优先级
# - 联系人列表
# - 处理流程

# ✅ 准备应急工具
# - Live CD/USB
# - 备份配置文件
# - 常用脚本
```

## 自动化

### 1. 使用自动化工具

```bash
# ✅ Ansible for 配置管理
# ✅ Cron for 定时任务
# ✅ Systemd timers for 服务定时

# Ansible playbook 示例
- hosts: servers
  tasks:
    - name: Update packages
      apt:
        update_cache: yes
        upgrade: dist
```

### 2. 脚本自动化

```bash
# ✅ 自动化重复任务
# - 系统更新
# - 日志清理
# - 备份

# ✅ 使用版本控制管理脚本
```

## 总结

本文总结了 Linux 最佳实践：

- ✅ 系统管理（最小安装、定期更新）
- ✅ 安全实践（用户管理、SSH、防火墙）
- ✅ 性能优化（系统调优、监控）
- ✅ 备份和恢复
- ✅ 文档化和团队协作
- ✅ 自动化运维

继续学习 [快速参考](/docs/linux/quick-reference) 和 [常见问题](/docs/linux/faq)。
