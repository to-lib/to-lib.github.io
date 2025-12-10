---
id: nacos
title: Nacos 服务注册与配置中心
sidebar_label: Nacos
sidebar_position: 3
---

# Nacos 服务注册与配置中心

> [!TIP] > **一站式解决方案**: Nacos 同时提供服务注册发现和配置管理功能，是 Spring Cloud Alibaba 的核心组件，可以替代 Eureka + Config。

## 1. Nacos 简介

### 什么是 Nacos？

**Nacos** (Dynamic Naming and Configuration Service) 是阿里巴巴开源的服务注册发现和配置管理平台。

### 核心功能

- **服务注册与发现** - 替代 Eureka
- **动态配置管理** - 替代 Spring Cloud Config
- **动态 DNS 服务** - 支持 DNS 协议
- **服务健康监测** - 实时健康检查

##2. Nacos Server 安装

### 下载启动

```bash
# 下载
wget https://github.com/alibaba/nacos/releases/download/2.3.0/nacos-server-2.3.0.zip

# 解压
unzip nacos-server-2.3.0.zip

# 启动（单机模式）
cd nacos/bin
sh startup.sh -m standalone

# Windows
startup.cmd -m standalone
```

### 访问控制台

访问：`http://localhost:8848/nacos`

默认账号密码：`nacos/nacos`

## 3. 服务注册与发现

### 添加依赖

```xml
<dependency>
    <groupId>com.alibaba.cloud</groupId>
    <artifactId>spring-cloud-starter-alibaba-nacos-discovery</artifactId>
</dependency>
```

### 配置

```yaml
server:
  port: 8081

spring:
  application:
    name: user-service
  cloud:
    nacos:
      discovery:
        # Nacos 服务地址
        server-addr: localhost:8848
        # 命名空间
        namespace: dev
        # 分组
        group: DEFAULT_GROUP
        # 权重（0-1）
        weight: 1
        # 集群名称
        cluster-name: DEFAULT
        # 元数据
        metadata:
          version: 1.0
          region: cn-hangzhou
```

### 启用服务注册

```java
@SpringBootApplication
@EnableDiscoveryClient
public class UserServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(UserServiceApplication.class, args);
    }
}
```

### 服务调用

```java
@RestController
public class OrderController {

    @Autowired
    private RestTemplate restTemplate;

    @GetMapping("/orders/{id}")
    public Order getOrder(@PathVariable Long id) {
        // 通过服务名调用
        User user = restTemplate.getForObject(
            "http://user-service/users/1",
            User.class
        );
        return new Order(id, user);
    }
}

@Configuration
public class RestTemplateConfig {

    @Bean
    @LoadBalanced
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}
```

## 4. 配置管理

### 添加依赖

```xml
<dependency>
    <groupId>com.alibaba.cloud</groupId>
    <artifactId>spring-cloud-starter-alibaba-nacos-config</artifactId>
</dependency>
```

### bootstrap.yml 配置

```yaml
spring:
  application:
    name: user-service
  cloud:
    nacos:
      config:
        # Nacos 配置中心地址
        server-addr: localhost:8848
        # 配置文件格式
        file-extension: yaml
        # 命名空间
        namespace: dev
        # 分组
        group: DEFAULT_GROUP
        # 共享配置
        shared-configs:
          - data-id: common.yaml
            group: COMMON_GROUP
            refresh: true
        # 扩展配置
        extension-configs:
          - data-id: redis.yaml
            group: MIDDLEWARE_GROUP
            refresh: true
```

### 配置文件命名规则

```
${spring.application.name}-${spring.profiles.active}.${file-extension}
```

例如：`user-service-dev.yaml`

### 动态刷新

```java
@RestController
@RefreshScope  // 支持动态刷新
public class ConfigController {

    @Value("${app.message:默认值}")
    private String message;

    @GetMapping("/config")
    public String getConfig() {
        return message;
    }
}
```

## 5. 命名空间与分组

### 命名空间（Namespace）

用于环境隔离：

```
Namespace: dev (开发环境 - 9d5e7c6a...)
Namespace: test (测试环境 - 3f2a8b1c...)
Namespace: prod (生产环境 - 7a4e9d2b...)
```

```yaml
spring:
  cloud:
    nacos:
      discovery:
        namespace: 9d5e7c6a-xxxx-xxxx-xxxx-xxxxxxxxxxxx
      config:
        namespace: 9d5e7c6a-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

### 分组（Group）

用于业务隔离：

```yaml
spring:
  cloud:
    nacos:
      discovery:
        group: ORDER_GROUP
      config:
        group: ORDER_GROUP
```

## 6. 配置优先级

加载顺序（优先级从高到低）：

```
1. ${spring.application.name}-${spring.profiles.active}.${file-extension}
2. ${spring.application.name}.${file-extension}
3. shared-configs（从下到上）
4. extension-configs（从下到上）
```

## 7. 配置监听

```java
@Component
public class ConfigListener {

    @Autowired
    private NacosConfigManager nacosConfigManager;

    @PostConstruct
    public void init() throws NacosException {
        String dataId = "user-service.yaml";
        String group = "DEFAULT_GROUP";

        nacosConfigManager.getConfigService().addListener(
            dataId,
            group,
            new Listener() {
                @Override
                public void receiveConfigInfo(String configInfo) {
                    System.out.println("配置更新: " + configInfo);
                }

                @Override
                public Executor getExecutor() {
                    return null;
                }
            }
        );
    }
}
```

## 8. 临时实例 vs 持久化实例

### 临时实例（默认）

- 通过心跳保持注册
- 心跳停止后自动注销
- 适合云原生应用

### 持久化实例

- 主动注销才会移除
- Nacos 主动探测健康状态
- 适合传统应用

```yaml
spring:
  cloud:
    nacos:
      discovery:
        # 是否是临时实例
        ephemeral: false
```

## 9. 权重与保护阈值

### 权重

```yaml
spring:
  cloud:
    nacos:
      discovery:
        # 权重 0-1，默认 1
        weight: 0.5
```

### 保护阈值

当健康实例比例低于阈值时，返回所有实例（包括不健康的）：

```
保护阈值: 0.3
健康实例: 2
总实例: 10
健康比例: 2/10 = 0.2 < 0.3 → 返回所有实例
```

## 10. 最佳实践

### 环境隔离

```yaml
# dev
spring.cloud.nacos.discovery.namespace=dev
spring.cloud.nacos.config.namespace=dev

# prod
spring.cloud.nacos.discovery.namespace=prod
spring.cloud.nacos.config.namespace=prod
```

### 配置分层

```
Level 1: 环境 (Namespace)
Level 2: 应用 (Group)
Level 3: 配置 (DataId)
```

### 配置加密

在 Nacos 控制台中：

1. 创建配置时选择"加密配置"
2. 输入密钥
3. 在应用中配置密钥

## 11. 集群部署

### 集群配置

```
# cluster.conf
192.168.1.101:8848
192.168.1.102:8848
192.168.1.103:8848
```

### 客户端配置

```yaml
spring:
  cloud:
    nacos:
      discovery:
        server-addr: 192.168.1.101:8848,192.168.1.102:8848,192.168.1.103:8848
```

## 12. 常见问题

### 服务注册失败

**检查项**:

- Nacos Server 是否正常运行
- 网络是否连通
- namespace 和 group 是否正确

### 配置不生效

**检查项**:

- DataId 是否正确
- namespace 和 group 是否匹配
- 是否加了 `@RefreshScope`

## 13. 总结

| 功能         | 说明        |
| ------------ | ----------- |
| 服务注册发现 | 替代 Eureka |
| 配置管理     | 替代 Config |
| 动态刷新     | 无需重启    |
| 多环境隔离   | Namespace   |

---

**关键要点**：

- Nacos 同时提供服务注册和配置管理
- 使用 Namespace 隔离环境，Group 隔离业务
- 支持配置动态刷新
- 生产环境建议集群部署

**下一步**：学习 [Sentinel 流量控制](/docs/springcloud-alibaba/sentinel)
