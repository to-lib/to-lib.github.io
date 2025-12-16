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

## 高级网络工具

### curl 高级用法

```bash
# 基本请求
curl -X GET http://api.example.com
curl -X POST http://api.example.com -d "data=value"
curl -X PUT http://api.example.com -d '{"key":"value"}'
curl -X DELETE http://api.example.com/id

# 请求头和认证
curl -H "Content-Type: application/json" http://api.example.com
curl -H "Authorization: Bearer token" http://api.example.com
curl -u username:password http://api.example.com

# JSON 数据
curl -X POST http://api.example.com \
  -H "Content-Type: application/json" \
  -d '{"name":"test","value":123}'

# 文件上传/下载
curl -O http://example.com/file.zip          # 下载文件
curl -o newname.zip http://example.com/file  # 下载并重命名
curl -F "file=@/path/to/file" http://upload  # 上传文件
curl -C - -O http://example.com/largefile    # 断点续传

# 调试选项
curl -v http://example.com                    # 详细输出
curl -s http://example.com                    # 静默模式
curl -w "%{http_code}" http://example.com     # 只输出状态码
curl --connect-timeout 5 http://example.com   # 连接超时
```

### wget 高级用法

```bash
# 基本下载
wget http://example.com/file.zip

# 断点续传
wget -c http://example.com/largefile.zip

# 后台下载
wget -b http://example.com/file.zip

# 下载整个网站
wget -r -np -nd http://example.com/directory/

# 限速下载
wget --limit-rate=1m http://example.com/file.zip

# 使用代理
wget -e "http_proxy=http://proxy:8080" http://example.com

# 镜像网站
wget -m -k -p http://example.com
```

### 网络诊断工具

```bash
# mtr - 结合 ping 和 traceroute
mtr google.com
mtr -r -c 10 google.com    # 报告模式

# nmap - 端口扫描
nmap -sT localhost                # TCP 扫描
nmap -sU localhost                # UDP 扫描
nmap -p 1-1000 target             # 指定端口范围
nmap -sV target                   # 版本检测
nmap -O target                    # 操作系统检测

# lsof - 查看网络连接
lsof -i :80                       # 查看80端口
lsof -i -P -n                     # 所有网络连接
lsof -i @192.168.1.100            # 指定IP连接

# 带宽测试
speedtest-cli                     # 网络速度测试
iperf3 -s                         # 服务器端
iperf3 -c server_ip               # 客户端测试
```

## 网络代理配置

### 系统代理

```bash
# 临时设置代理
export http_proxy="http://proxy:8080"
export https_proxy="http://proxy:8080"
export no_proxy="localhost,127.0.0.1,.local"

# 永久设置（添加到 ~/.bashrc）
echo 'export http_proxy="http://proxy:8080"' >> ~/.bashrc
echo 'export https_proxy="http://proxy:8080"' >> ~/.bashrc

# 取消代理
unset http_proxy https_proxy
```

### apt 代理配置

```bash
# 配置 apt 代理
sudo vim /etc/apt/apt.conf.d/proxy.conf

# 添加内容
Acquire::http::Proxy "http://proxy:8080";
Acquire::https::Proxy "http://proxy:8080";
```

### Git 代理配置

```bash
# HTTP 代理
git config --global http.proxy http://proxy:8080
git config --global https.proxy http://proxy:8080

# SOCKS5 代理
git config --global http.proxy socks5://127.0.0.1:1080

# 取消代理
git config --global --unset http.proxy
git config --global --unset https.proxy
```

### SSH 代理

```bash
# SSH 配置（~/.ssh/config）
Host github.com
    HostName github.com
    User git
    ProxyCommand nc -X 5 -x 127.0.0.1:1080 %h %p
```

## 端口转发

### SSH 端口转发

```bash
# 本地端口转发（访问远程服务）
# 将本地8080端口转发到远程服务器的3306端口
ssh -L 8080:localhost:3306 user@server

# 远程端口转发（暴露本地服务）
# 将远程服务器的8080端口转发到本地的80端口
ssh -R 8080:localhost:80 user@server

# 动态端口转发（SOCKS代理）
ssh -D 1080 user@server

# 后台运行
ssh -fN -L 8080:localhost:3306 user@server
ssh -fN -D 1080 user@server

# 保持连接
ssh -L 8080:localhost:3306 -o ServerAliveInterval=60 user@server
```

### iptables 端口转发

```bash
# 启用 IP 转发
echo 1 > /proc/sys/net/ipv4/ip_forward
# 永久启用
echo "net.ipv4.ip_forward = 1" >> /etc/sysctl.conf
sysctl -p

# 端口转发规则
# 将本机的8080端口转发到192.168.1.100的80端口
iptables -t nat -A PREROUTING -p tcp --dport 8080 -j DNAT --to-destination 192.168.1.100:80
iptables -t nat -A POSTROUTING -j MASQUERADE

# 查看 NAT 规则
iptables -t nat -L -n -v

# 删除规则
iptables -t nat -D PREROUTING 1
```

### socat 端口转发

```bash
# TCP 端口转发
socat TCP-LISTEN:8080,fork TCP:192.168.1.100:80

# UDP 端口转发
socat UDP-LISTEN:8080,fork UDP:192.168.1.100:80

# 后台运行
socat -d -d TCP-LISTEN:8080,fork TCP:192.168.1.100:80 &
```

## 虚拟网络

### 网桥配置

```bash
# 创建网桥
sudo ip link add name br0 type bridge
sudo ip link set br0 up

# 添加接口到网桥
sudo ip link set eth0 master br0

# 配置网桥 IP
sudo ip addr add 192.168.1.100/24 dev br0
```

### VLAN 配置

```bash
# 加载 8021q 模块
sudo modprobe 8021q

# 创建 VLAN 接口
sudo ip link add link eth0 name eth0.100 type vlan id 100
sudo ip link set eth0.100 up
sudo ip addr add 192.168.100.1/24 dev eth0.100
```

## 总结

本文介绍了 Linux 网络配置：

- ✅ 网络接口配置
- ✅ 路由和 DNS
- ✅ 防火墙管理
- ✅ SSH 配置
- ✅ 网络监控和故障排查
- ✅ 高级网络工具（curl/wget/nmap）
- ✅ 网络代理配置
- ✅ 端口转发

继续学习 [系统安全](/docs/linux/security) 和 [性能调优](/docs/linux/performance-tuning)。
