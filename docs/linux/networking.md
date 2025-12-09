---
sidebar_position: 6
title: 网络配置
---

# Linux 网络配置

Linux 网络配置是系统管理的重要组成部分，包括网络接口配置、路由设置、防火墙管理等。

## 网络基础

### 查看网络信息

```bash
# 查看 IP 地址
ip addr
ip a

# 查看特定接口
ip addr show eth0

# 旧命令（仍然可用）
ifconfig
ifconfig eth0

# 查看网络接口
ip link
ip link show
```

### 网络连接测试

```bash
# Ping 测试
ping google.com
ping -c 4 8.8.8.8        # 发送4个包

# 追踪路由
traceroute google.com
tracepath google.com

# DNS 查询
nslookup google.com
dig google.com
host google.com
```

## 网络接口配置

### 临时配置（重启后失效）

```bash
# 启用/禁用接口
sudo ip link set eth0 up
sudo ip link set eth0 down

# 设置 IP 地址
sudo ip addr add 192.168.1.100/24 dev eth0

# 删除 IP 地址
sudo ip addr del 192.168.1.100/24 dev eth0

# 设置默认网关
sudo ip route add default via 192.168.1.1
```

### 永久配置（Ubuntu/Debian）

```bash
# 编辑netplan配置（Ubuntu 18.04+）
sudo vim /etc/netplan/01-netcfg.yaml

# 示例配置：
network:
  version: 2
  ethernets:
    eth0:
      addresses: [192.168.1.100/24]
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]

# 应用配置
sudo netplan apply
```

### 永久配置（CentOS/RHEL）

```bash
# 编辑网络配置
sudo vim /etc/sysconfig/network-scripts/ifcfg-eth0

# 示例配置：
TYPE=Ethernet
BOOTPROTO=static
NAME=eth0
DEVICE=eth0
ONBOOT=yes
IPADDR=192.168.1.100
NETMASK=255.255.255.0
GATEWAY=192.168.1.1
DNS1=8.8.8.8
DNS2=8.8.4.4

# 重启网络
sudo systemctl restart network
```

## 路由配置

```bash
# 查看路由表
ip route
ip route show
route -n

# 添加路由
sudo ip route add 10.0.0.0/8 via 192.168.1.1

# 删除路由
sudo ip route del 10.0.0.0/8

# 修改默认网关
sudo ip route change default via 192.168.1.254
```

## DNS 配置

```bash
# 编辑 DNS 服务器
sudo vim /etc/resolv.conf

# 示例：
nameserver 8.8.8.8
nameserver 8.8.4.4

# 查看 DNS 解析顺序
cat /etc/nsswitch.conf | grep hosts

# 刷新 DNS 缓存
sudo systemctl restart systemd-resolved
```

## 防火墙

### firewalld（CentOS/RHEL）

```bash
# 查看状态
sudo firewall-cmd --state

# 查看默认区域
sudo firewall-cmd --get-default-zone

# 查看活动区域
sudo firewall-cmd --get-active-zones

# 开放端口
sudo firewall-cmd --add-port=80/tcp --permanent
sudo firewall-cmd --add-port=443/tcp --permanent

# 开放服务
sudo firewall-cmd --add-service=http --permanent
sudo firewall-cmd --add-service=https --permanent

# 重新加载
sudo firewall-cmd --reload

# 查看规则
sudo firewall-cmd --list-all
```

### ufw（Ubuntu）

```bash
# 启用防火墙
sudo ufw enable

# 禁用防火墙
sudo ufw disable

# 查看状态
sudo ufw status
sudo ufw status verbose

# 允许端口
sudo ufw allow 22
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# 允许服务
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https

# 拒绝
sudo ufw deny 23

# 删除规则
sudo ufw delete allow 80

# 允许特定IP
sudo ufw allow from 192.168.1.100

# 重置
sudo ufw reset
```

### iptables（底层工具）

```bash
# 查看规则
sudo iptables -L -n -v

# 允许入站SSH
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# 允许已建立的连接
sudo iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# 阻止所有其他入站
sudo iptables -A INPUT -j DROP

# 保存规则
sudo iptables-save > /etc/iptables/rules.v4

# 恢复规则
sudo iptables-restore < /etc/iptables/rules.v4
```

## SSH 配置

### SSH 服务器

```bash
# 安装SSH服务器
sudo apt install openssh-server

# 启动SSH
sudo systemctl start sshd
sudo systemctl enable sshd

# 配置SSH
sudo vim /etc/ssh/sshd_config

# 常用配置：
Port 22
PermitRootLogin no
PasswordAuthentication yes
PubkeyAuthentication yes

# 重启SSH
sudo systemctl restart sshd
```

### SSH 密钥认证

```bash
# 生成密钥对
ssh-keygen -t rsa -b 4096
ssh-keygen -t ed25519

# 复制公钥到服务器
ssh-copy-id user@server
# 或手动复制
cat ~/.ssh/id_rsa.pub | ssh user@server "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"

# 连接
ssh user@server
ssh -i ~/.ssh/id_rsa user@server

# SSH 配置（~/.ssh/config）
Host myserver
    HostName 192.168.1.100
    User username
    Port 22
    IdentityFile ~/.ssh/id_rsa
```

## 网络监控

```bash
# 实时网络流量
sudo iftop
sudo nload
sudo nethogs

# 网络统计
netstat -tuln          # 监听端口
netstat -tunap         # 所有连接
ss -tuln               # 更现代的命令

# 查看连接
ss -tun
ss -o state established '( dport = :ssh or sport = :ssh )'

# 带宽测试
iperf3 -s              # 服务器端
iperf3 -c server_ip    # 客户端
```

## 网络故障排查

### 基本步骤

```bash
# 1. 检查物理连接
ip link

# 2. 检查IP配置
ip addr

# 3. 检查路由
ip route

# 4. 测试网关
ping 192.168.1.1

# 5. 测试DNS
ping 8.8.8.8
nslookup google.com

# 6. 测试外网
ping google.com

# 7. 检查防火墙
sudo iptables -L -n

# 8. 检查服务
sudo systemctl status networking
```

### 常用工具

```bash
# tcpdump 抓包
sudo tcpdump -i eth0
sudo tcpdump -i eth0 port 80
sudo tcpdump -i eth0 -w capture.pcap

# nc（netcat）测试
nc -zv server 80       # 测试端口
nc -l 1234             # 监听端口

# curl 测试HTTP
curl http://example.com
curl -I http://example.com
```

## 总结

本文介绍了 Linux 网络配置：

- ✅ 网络接口配置
- ✅ 路由和 DNS
- ✅ 防火墙管理
- ✅ SSH 配置
- ✅ 网络监控和故障排查

继续学习 [Shell 脚本](./shell-scripting)。
