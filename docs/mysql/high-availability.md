---
sidebar_position: 14
title: 高可用架构
---

# MySQL 高可用架构

> [!TIP] > **业务连续性保障**: 高可用架构是保证 MySQL 服务稳定运行的关键。本文介绍常见的高可用方案和最佳实践。

## 高可用概述

### 可用性指标

| 可用性  | 年停机时间 | 描述   |
| ------- | ---------- | ------ |
| 99%     | 3.65 天    | 1 个 9 |
| 99.9%   | 8.76 小时  | 3 个 9 |
| 99.99%  | 52.6 分钟  | 4 个 9 |
| 99.999% | 5.26 分钟  | 5 个 9 |

### 高可用方案对比

| 方案     | 数据一致性 | 自动切换 | 复杂度 | 适用场景 |
| -------- | ---------- | -------- | ------ | -------- |
| 主从复制 | 最终一致   | 否       | 低     | 读写分离 |
| MHA      | 较好       | 是       | 中     | 中小型   |
| MGR      | 强一致     | 是       | 高     | 金融级   |
| Galera   | 强一致     | 是       | 高     | 多主写入 |

## 主从复制架构

### 基本架构

```
    [主库]
      │
      ├── [从库1] - 读
      ├── [从库2] - 读
      └── [从库3] - 备份
```

### 配置示例

```sql
-- 主库配置 (my.cnf)
[mysqld]
server-id = 1
log-bin = mysql-bin
gtid-mode = ON
enforce-gtid-consistency = ON

-- 从库配置
[mysqld]
server-id = 2
relay-log = mysql-relay-bin
log-slave-updates = 1
read-only = 1
gtid-mode = ON
enforce-gtid-consistency = ON
```

### 优缺点

**优点**：

- ✅ 实现简单
- ✅ 读写分离
- ✅ 数据备份

**缺点**：

- ❌ 不能自动故障切换
- ❌ 主库单点故障
- ❌ 存在复制延迟

## MHA（Master High Availability）

### 架构概述

```
         [MHA Manager]
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
[Master]  [Slave1]  [Slave2]
    │         ▲         ▲
    └─────────┴─────────┘
         复制关系
```

### 核心组件

- **MHA Manager** - 管理节点，监控和故障切换
- **MHA Node** - 数据节点，安装在所有 MySQL 服务器

### 安装配置

```bash
# 1. 安装 MHA Node（所有 MySQL 服务器）
yum install perl-DBD-MySQL
rpm -ivh mha4mysql-node-*.rpm

# 2. 安装 MHA Manager（管理节点）
rpm -ivh mha4mysql-manager-*.rpm

# 3. 配置文件 /etc/mha/app1.cnf
[server default]
manager_log=/var/log/mha/app1/manager.log
manager_workdir=/var/log/mha/app1
master_binlog_dir=/var/lib/mysql
password=password
ping_interval=1
repl_password=repl_password
repl_user=repl
ssh_user=root
user=root

[server1]
candidate_master=1
hostname=192.168.1.100

[server2]
candidate_master=1
hostname=192.168.1.101

[server3]
hostname=192.168.1.102
no_master=1
```

### 故障切换

```bash
# 手动切换
masterha_master_switch --conf=/etc/mha/app1.cnf \
    --master_state=alive \
    --new_master_host=192.168.1.101

# 自动故障检测和切换
masterha_manager --conf=/etc/mha/app1.cnf &
```

## MySQL Group Replication（MGR）

### 架构概述

基于 Paxos 协议的组复制，支持多主或单主模式。

```
    ┌─────────────────────┐
    │   Group Replication │
    │     (Paxos)         │
    └─────────────────────┘
        │     │     │
    [Node1] [Node2] [Node3]
      RW      RW      RW
```

### 配置单主模式

```ini
# my.cnf
[mysqld]
server-id = 1
gtid-mode = ON
enforce-gtid-consistency = ON
binlog-checksum = NONE
log-bin = mysql-bin
log-slave-updates = ON
binlog-format = ROW
master-info-repository = TABLE
relay-log-info-repository = TABLE

# Group Replication 配置
plugin-load-add = group_replication.so
group_replication_group_name = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
group_replication_start_on_boot = OFF
group_replication_local_address = "192.168.1.100:33061"
group_replication_group_seeds = "192.168.1.100:33061,192.168.1.101:33061,192.168.1.102:33061"
group_replication_bootstrap_group = OFF
group_replication_single_primary_mode = ON
```

### 启动 MGR

```sql
-- 创建复制用户
CREATE USER 'repl'@'%' IDENTIFIED BY 'password';
GRANT REPLICATION SLAVE ON *.* TO 'repl'@'%';
GRANT BACKUP_ADMIN ON *.* TO 'repl'@'%';
FLUSH PRIVILEGES;

-- 配置复制通道
CHANGE REPLICATION SOURCE TO SOURCE_USER='repl', SOURCE_PASSWORD='password'
FOR CHANNEL 'group_replication_recovery';

-- 引导第一个节点
SET GLOBAL group_replication_bootstrap_group = ON;
START GROUP_REPLICATION;
SET GLOBAL group_replication_bootstrap_group = OFF;

-- 其他节点加入
START GROUP_REPLICATION;

-- 查看成员
SELECT * FROM performance_schema.replication_group_members;
```

### MGR 优缺点

**优点**：

- ✅ 数据强一致性
- ✅ 自动故障切换
- ✅ 弹性扩缩容

**缺点**：

- ❌ 网络要求高
- ❌ 配置复杂
- ❌ 并发写性能受限

## 读写分离

### 架构设计

```
         [应用层]
            │
     ┌──────┴──────┐
     ▼             ▼
  [写入]        [读取]
     │             │
     ▼             ▼
  [主库]       [负载均衡]
     │         ┌───┴───┐
     │         ▼       ▼
     └────> [从库1] [从库2]
```

### ProxySQL 配置

```sql
-- 安装 ProxySQL
yum install proxysql

-- 配置后端服务器
INSERT INTO mysql_servers(hostgroup_id, hostname, port) VALUES (10, '192.168.1.100', 3306);  -- 主库
INSERT INTO mysql_servers(hostgroup_id, hostname, port) VALUES (20, '192.168.1.101', 3306);  -- 从库
INSERT INTO mysql_servers(hostgroup_id, hostname, port) VALUES (20, '192.168.1.102', 3306);  -- 从库

-- 配置读写分离规则
INSERT INTO mysql_query_rules(rule_id, match_pattern, destination_hostgroup, apply)
VALUES (1, '^SELECT', 20, 1);

INSERT INTO mysql_query_rules(rule_id, match_pattern, destination_hostgroup, apply)
VALUES (2, '.*', 10, 1);

-- 加载配置
LOAD MYSQL SERVERS TO RUNTIME;
LOAD MYSQL QUERY RULES TO RUNTIME;
SAVE MYSQL SERVERS TO DISK;
SAVE MYSQL QUERY RULES TO DISK;
```

### 应用层实现

```java
// Spring Boot 读写分离配置
@Configuration
public class DataSourceConfig {

    @Bean
    @ConfigurationProperties("spring.datasource.master")
    public DataSource masterDataSource() {
        return DataSourceBuilder.create().build();
    }

    @Bean
    @ConfigurationProperties("spring.datasource.slave")
    public DataSource slaveDataSource() {
        return DataSourceBuilder.create().build();
    }

    @Bean
    public DataSource routingDataSource() {
        Map<Object, Object> targetDataSources = new HashMap<>();
        targetDataSources.put("master", masterDataSource());
        targetDataSources.put("slave", slaveDataSource());

        RoutingDataSource routingDataSource = new RoutingDataSource();
        routingDataSource.setTargetDataSources(targetDataSources);
        routingDataSource.setDefaultTargetDataSource(masterDataSource());
        return routingDataSource;
    }
}

// 路由数据源
public class RoutingDataSource extends AbstractRoutingDataSource {
    @Override
    protected Object determineCurrentLookupKey() {
        return TransactionSynchronizationManager.isCurrentTransactionReadOnly()
            ? "slave" : "master";
    }
}
```

## VIP 漂移

### Keepalived 配置

```conf
# /etc/keepalived/keepalived.conf（主库）
vrrp_script check_mysql {
    script "/etc/keepalived/check_mysql.sh"
    interval 2
    weight -20
}

vrrp_instance VI_1 {
    state MASTER
    interface eth0
    virtual_router_id 51
    priority 100
    advert_int 1

    authentication {
        auth_type PASS
        auth_pass 1111
    }

    virtual_ipaddress {
        192.168.1.200
    }

    track_script {
        check_mysql
    }
}
```

### 健康检查脚本

```bash
#!/bin/bash
# /etc/keepalived/check_mysql.sh

MYSQL_HOST="127.0.0.1"
MYSQL_USER="monitor"
MYSQL_PASSWORD="password"

mysql -h $MYSQL_HOST -u $MYSQL_USER -p$MYSQL_PASSWORD -e "SELECT 1" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    exit 0
else
    exit 1
fi
```

## Kubernetes 部署

### StatefulSet 配置

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql
spec:
  serviceName: mysql
  replicas: 3
  selector:
    matchLabels:
      app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
        - name: mysql
          image: mysql:8.0
          env:
            - name: MYSQL_ROOT_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: mysql-secret
                  key: password
          ports:
            - containerPort: 3306
          volumeMounts:
            - name: data
              mountPath: /var/lib/mysql
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 100Gi
```

### MySQL Operator

```yaml
# 使用 MySQL Operator 简化管理
apiVersion: mysql.oracle.com/v2
kind: InnoDBCluster
metadata:
  name: mycluster
spec:
  secretName: mysql-secret
  tlsUseSelfSigned: true
  instances: 3
  router:
    instances: 2
```

## 最佳实践

> [!IMPORTANT] > **高可用最佳实践**:
>
> 1. ✅ 根据业务需求选择合适的方案
> 2. ✅ 使用 GTID 简化故障切换
> 3. ✅ 实现读写分离分担主库压力
> 4. ✅ 监控复制延迟和数据一致性
> 5. ✅ 定期进行故障演练
> 6. ✅ 配置自动故障告警
> 7. ✅ 准备故障切换手册
> 8. ❌ 避免跨机房同步复制

## 方案选型建议

| 场景     | 推荐方案              | 说明             |
| -------- | --------------------- | ---------------- |
| 小型应用 | 主从复制 + Keepalived | 简单易维护       |
| 中型应用 | MHA + 读写分离        | 平衡成本和可用性 |
| 大型应用 | MGR + ProxySQL        | 高可用、自动切换 |
| 金融级   | MGR 强一致模式        | 数据零丢失       |
| 云原生   | MySQL Operator        | 容器化部署       |

## 总结

本文介绍了 MySQL 高可用架构：

- ✅ 可用性指标和方案对比
- ✅ 主从复制架构
- ✅ MHA 高可用方案
- ✅ MGR 组复制
- ✅ 读写分离实现
- ✅ VIP 漂移配置
- ✅ Kubernetes 部署

继续学习 [主从复制](/docs/mysql/replication) 和 [备份恢复](/docs/mysql/backup-recovery)！
