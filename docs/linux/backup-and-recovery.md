---
sidebar_position: 24
title: 备份与恢复
---

# Linux 备份与恢复

备份不是“做了就行”，关键是：

- **能恢复**（可验证）
- **恢复得够快**（满足 RTO）
- **丢数据在可控范围**（满足 RPO）

## 设计原则

### 3-2-1 原则

- 3 份副本
- 2 种介质
- 1 份异地

### RPO / RTO

- **RPO**：最多允许丢失的数据时间窗口
- **RTO**：从故障到恢复可用的最长耗时

建议把关键系统的 RPO/RTO 写进运维规范，并定期演练。

## 备份范围清单（常见）

- `/etc/`（系统与服务配置）
- `/etc/systemd/system/`（自定义 systemd 单元）
- `/var/lib/`（数据库、状态数据）
- `/opt/`、`/usr/local/`、`/var/www/`（按部署方式选择）
- 证书与密钥（注意权限与合规）

## 文件级备份：rsync

适合：配置目录、应用目录、增量同步。

```bash
# 本地同步
rsync -av /etc/ /backup/etc/

# 同步到远端（SSH）
rsync -av -e "ssh -p 22" /etc/ user@backup-host:/backup/etc/

# 谨慎使用：让目标端与源端保持一致
rsync -av --delete /etc/ /backup/etc/
```

建议：

- 源端与目标端都使用绝对路径
- 备份目录分层（按主机名/业务/日期）
- 备份前后输出日志，失败要告警

## 归档备份：tar

适合：离线打包、传输、短期留存。

```bash
# 打包
sudo tar -czf /backup/etc-$(date +%Y%m%d).tar.gz /etc

# 快速校验（能列出文件即基本可用）
tar -tzf /backup/etc-$(date +%Y%m%d).tar.gz > /dev/null
```

## 快照备份：LVM（可选）

适合：对变化中的数据目录做一致性快照（仍建议结合应用层一致性策略）。

```bash
# 创建快照
sudo lvcreate -L 5G -s -n lv_data_snap /dev/vg_data/lv_data

# 挂载只读导出
sudo mkdir -p /mnt/snap
sudo mount -o ro /dev/vg_data/lv_data_snap /mnt/snap

# 导出
sudo rsync -av /mnt/snap/ /backup/lv_data/

# 清理
sudo umount /mnt/snap
sudo lvremove -f /dev/vg_data/lv_data_snap
```

## 数据库备份（示例）

### MySQL（逻辑备份）

```bash
mysqldump -u root -p --single-transaction --routines --events --databases dbname \
  | gzip > /backup/mysql-dbname-$(date +%Y%m%d%H%M).sql.gz
```

### PostgreSQL（逻辑备份）

```bash
pg_dump -U postgres -d dbname \
  | gzip > /backup/pg-dbname-$(date +%Y%m%d%H%M).sql.gz
```

## 恢复要点

- **先在隔离环境演练**（测试机/临时实例）
- **恢复顺序**：基础设施/配置 -> 数据 -> 服务 -> 业务验证
- **恢复后校验**：端口、核心接口、关键数据抽样

## 最佳实践

- 定期做恢复演练并记录耗时（RTO）
- 备份结果做校验（例如：能解压、校验和、抽样比对）
- 备份数据加密与访问控制（权限、密钥托管）

继续学习：

- [定时任务](/docs/linux/cron-scheduling)
- [磁盘和存储管理](/docs/linux/disk-management)
- [系统安全](/docs/linux/security)
