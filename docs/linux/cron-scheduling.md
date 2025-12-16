---
sidebar_position: 10
title: 定时任务
---

# Linux 定时任务

定时任务是 Linux 系统自动化的核心功能，可以在指定时间自动执行脚本和命令。

## Cron 定时任务

### Cron 语法

```
┌───────── 分钟 (0-59)
│ ┌───────── 小时 (0-23)
│ │ ┌───────── 日期 (1-31)
│ │ │ ┌───────── 月份 (1-12)
│ │ │ │ ┌───────── 星期 (0-7, 0和7都表示周日)
│ │ │ │ │
│ │ │ │ │
* * * * * command
```

### 特殊符号

| 符号 | 含义   | 示例                        |
| ---- | ------ | --------------------------- |
| `*`  | 任意值 | `* * * * *` 每分钟          |
| `,`  | 列举   | `0,30 * * * *` 0 分和 30 分 |
| `-`  | 范围   | `0-30 * * * *` 0-30 分      |
| `/`  | 步长   | `*/5 * * * *` 每 5 分钟     |

### 常用时间表达式

```bash
# 每分钟
* * * * * command

# 每小时（整点）
0 * * * * command

# 每天凌晨3点
0 3 * * * command

# 每周一凌晨3点
0 3 * * 1 command

# 每月1号凌晨3点
0 3 1 * * command

# 每年1月1日凌晨3点
0 3 1 1 * command

# 每5分钟
*/5 * * * * command

# 每小时的第30分钟
30 * * * * command

# 工作日每天9点
0 9 * * 1-5 command

# 每天9点和18点
0 9,18 * * * command

# 每隔2小时
0 */2 * * * command
```

### 预定义时间表达式

```bash
@reboot     # 系统启动时执行
@yearly     # 每年执行一次 (等同于 0 0 1 1 *)
@annually   # 同 @yearly
@monthly    # 每月执行一次 (等同于 0 0 1 * *)
@weekly     # 每周执行一次 (等同于 0 0 * * 0)
@daily      # 每天执行一次 (等同于 0 0 * * *)
@midnight   # 同 @daily
@hourly     # 每小时执行一次 (等同于 0 * * * *)
```

## crontab 命令

### 基本操作

```bash
# 编辑当前用户的 crontab
crontab -e

# 列出当前用户的 crontab
crontab -l

# 删除当前用户的 crontab
crontab -r

# 编辑指定用户的 crontab（需要 root 权限）
sudo crontab -u username -e

# 列出指定用户的 crontab
sudo crontab -u username -l
```

### crontab 文件位置

```bash
# 用户 crontab 文件
/var/spool/cron/crontabs/username   # Debian/Ubuntu
/var/spool/cron/username            # RHEL/CentOS

# 系统 crontab 文件
/etc/crontab

# 系统 cron 目录
/etc/cron.d/          # 自定义系统任务
/etc/cron.hourly/     # 每小时执行
/etc/cron.daily/      # 每天执行
/etc/cron.weekly/     # 每周执行
/etc/cron.monthly/    # 每月执行
```

### 系统 crontab 格式

```bash
# /etc/crontab 格式（多一个用户字段）
# 分 时 日 月 周 用户 命令
0 3 * * * root /usr/local/bin/backup.sh
```

## 定时任务示例

### 备份任务

```bash
# 每天凌晨2点备份数据库
0 2 * * * /usr/bin/mysqldump -u root -ppassword database > /backup/db_$(date +\%Y\%m\%d).sql

# 每周日凌晨3点备份目录
0 3 * * 0 tar -czf /backup/home_$(date +\%Y\%m\%d).tar.gz /home

# 每天清理7天前的备份
0 4 * * * find /backup -name "*.sql" -mtime +7 -delete
```

### 系统维护

```bash
# 每天凌晨5点清理临时文件
0 5 * * * find /tmp -type f -atime +7 -delete

# 每小时同步时间
0 * * * * /usr/sbin/ntpdate pool.ntp.org > /dev/null 2>&1

# 每天清理日志
0 0 * * * journalctl --vacuum-time=7d

# 每周日凌晨更新系统（Debian/Ubuntu）
0 3 * * 0 apt update && apt upgrade -y
```

### 监控任务

```bash
# 每5分钟检查服务状态
*/5 * * * * /usr/local/bin/check_services.sh

# 每分钟检查磁盘空间
* * * * * df -h | awk '$5 > 80 {print}' | mail -s "磁盘告警" admin@example.com

# 每30分钟检查网站可用性
*/30 * * * * curl -s -o /dev/null -w "%{http_code}" http://example.com | grep -q 200 || echo "网站异常" | mail -s "网站告警" admin@example.com
```

## 日志和调试

### 查看 cron 日志

```bash
# Debian/Ubuntu
grep CRON /var/log/syslog
tail -f /var/log/syslog | grep CRON

# RHEL/CentOS
cat /var/log/cron
tail -f /var/log/cron

# 使用 journalctl
journalctl -u cron
journalctl -u crond
```

### 输出重定向

```bash
# 丢弃所有输出
* * * * * command > /dev/null 2>&1

# 记录输出到文件
* * * * * command >> /var/log/mycron.log 2>&1

# 只记录错误
* * * * * command > /dev/null 2>> /var/log/mycron_error.log

# 邮件通知（需要配置邮件服务）
MAILTO="admin@example.com"
* * * * * command
```

### 调试技巧

```bash
# 在脚本中添加调试信息
#!/bin/bash
echo "$(date): 脚本开始执行" >> /var/log/script_debug.log
# ... 脚本内容
echo "$(date): 脚本执行完成" >> /var/log/script_debug.log

# 测试 cron 环境
* * * * * env > /tmp/cron_env.txt
```

## systemd 定时器

### Timer 单元文件

systemd timer 是现代 Linux 系统的替代方案，提供更强大的功能。

```ini
# /etc/systemd/system/backup.timer
[Unit]
Description=每日备份定时器

[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

```ini
# /etc/systemd/system/backup.service
[Unit]
Description=执行备份任务

[Service]
Type=oneshot
ExecStart=/usr/local/bin/backup.sh
```

### 常用时间表达式

```bash
# OnCalendar 时间格式
OnCalendar=*-*-* 02:00:00           # 每天凌晨2点
OnCalendar=Mon *-*-* 02:00:00       # 每周一凌晨2点
OnCalendar=*-*-01 02:00:00          # 每月1号凌晨2点
OnCalendar=hourly                    # 每小时
OnCalendar=daily                     # 每天
OnCalendar=weekly                    # 每周
OnCalendar=monthly                   # 每月

# 其他触发方式
OnBootSec=5min                       # 启动后5分钟
OnUnitActiveSec=1h                   # 上次执行后1小时
OnUnitInactiveSec=1h                 # 单元变为非活动状态后1小时
```

### Timer 管理命令

```bash
# 启用和启动 timer
sudo systemctl enable backup.timer
sudo systemctl start backup.timer

# 查看 timer 状态
systemctl status backup.timer
systemctl list-timers
systemctl list-timers --all

# 立即执行关联的服务
sudo systemctl start backup.service

# 查看日志
journalctl -u backup.service
```

### Timer vs Cron 对比

| 特性       | Cron   | systemd Timer |
| ---------- | ------ | ------------- |
| 配置复杂度 | 简单   | 稍复杂        |
| 依赖管理   | 无     | 支持          |
| 日志       | syslog | journald      |
| 错过执行   | 丢失   | 可配置持久化  |
| 精确度     | 分钟级 | 秒级          |
| 资源控制   | 无     | 支持 cgroup   |

## at 一次性任务

### 基本用法

```bash
# 在指定时间执行命令
at 10:00
> command
> Ctrl+D

# 使用 echo 输入
echo "command" | at 10:00

# 使用时间表达式
at now + 5 minutes
at now + 1 hour
at now + 2 days
at midnight
at noon
at teatime (16:00)

# 指定日期
at 10:00 AM Dec 25
at 10:00 AM 12/25/2024
```

### 管理 at 任务

```bash
# 列出待执行的任务
atq

# 查看任务内容
at -c job_id

# 删除任务
atrm job_id

# 批处理模式（系统负载低时执行）
batch
> command
> Ctrl+D
```

## 最佳实践

### 1. 使用绝对路径

```bash
# 错误：可能找不到命令
* * * * * backup.sh

# 正确：使用绝对路径
* * * * * /usr/local/bin/backup.sh
```

### 2. 设置环境变量

```bash
# 在 crontab 中定义
PATH=/usr/local/bin:/usr/bin:/bin
SHELL=/bin/bash
MAILTO=admin@example.com

# 或在脚本开头
#!/bin/bash
export PATH=/usr/local/bin:/usr/bin:/bin
. /etc/profile
```

### 3. 使用锁文件防止重复执行

```bash
#!/bin/bash
LOCKFILE="/tmp/myscript.lock"

if [ -f "$LOCKFILE" ]; then
    echo "脚本正在运行中"
    exit 1
fi

trap "rm -f $LOCKFILE" EXIT
touch "$LOCKFILE"

# 脚本主体
# ...
```

### 4. 添加执行日志

```bash
#!/bin/bash
LOG_FILE="/var/log/mycron.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

log "开始执行"
# 主要逻辑
log "执行完成"
```

### 5. 错误处理

```bash
#!/bin/bash
set -e  # 遇到错误立即退出

# 或手动处理
command1 || { echo "command1 失败" | mail -s "任务失败" admin@example.com; exit 1; }
```

## 总结

本文介绍了 Linux 定时任务：

- ✅ Cron 语法和表达式
- ✅ crontab 命令管理
- ✅ 定时任务实际案例
- ✅ 日志和调试技巧
- ✅ systemd 定时器
- ✅ at 一次性任务
- ✅ 最佳实践

继续学习 [服务管理](/docs/linux/service-management) 和 [Shell 脚本](/docs/linux/shell-scripting)。
