---
id: migration-guide
title: 版本升级指南
sidebar_label: 版本升级
sidebar_position: 16
---

# Spring Cloud Alibaba 版本升级指南

> [!WARNING]
> **升级前必读**: 版本升级可能带来不兼容的变更,请仔细阅读本指南并做好充分测试。

## 1. 版本对应关系

### 官方推荐版本

| Spring Cloud Alibaba | Spring Cloud | Spring Boot | JDK    |
| -------------------- | ------------ | ----------- | ------ |
| 2023.0.0.0           | 2023.0.x     | 3.2.x       | 17+    |
| 2022.0.0.0           | 2022.0.x     | 3.0.x       | 17+    |
| 2021.0.5.0           | 2021.0.x     | 2.6.x       | 8/11   |
| 2021.0.1.0           | 2021.0.x     | 2.6.x       | 8/11   |
| 2.2.9.RELEASE        | Hoxton.SR12  | 2.3.x       | 8/11   |

### 组件版本对应

**2023.0.0.0**:

- Nacos: 2.3.0
- Sentinel: 1.8.6
- Seata: 1.7.0
- RocketMQ: 5.1.4
- Dubbo: 3.2.0

## 2. 从 2.2.x 升级到 2021.0.x

### 重大变更

**1. 版本号格式变更**:

```xml
<!-- 旧版本 -->
<version>2.2.9.RELEASE</version>

<!-- 新版本 -->
<version>2021.0.5.0</version>
```

**2. Spring Boot 升级到 2.6.x**:

```xml
<parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>2.6.13</version>
</parent>
```

**3. Bootstrap 配置变更**:

Spring Boot 2.4+ 默认不加载 `bootstrap.yml`,需要添加依赖:

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-bootstrap</artifactId>
</dependency>
```

或使用新的配置方式:

```yaml
# application.yml
spring:
  config:
    import: optional:nacos:user-service.yaml
```

**4. Nacos 配置变更**:

```yaml
# 旧版本
spring:
  cloud:
    nacos:
      config:
        prefix: ${spring.application.name}

# 新版本 (prefix 被移除)
# 直接使用 spring.application.name
```

### 升级步骤

**Step 1: 更新依赖**

```xml
<properties>
    <spring-boot.version>2.6.13</spring-boot.version>
    <spring-cloud.version>2021.0.5</spring-cloud.version>
    <spring-cloud-alibaba.version>2021.0.5.0</spring-cloud-alibaba.version>
</properties>
```

**Step 2: 处理配置**

添加 bootstrap 支持:

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-bootstrap</artifactId>
</dependency>
```

**Step 3: 测试验证**

```bash
# 编译
mvn clean compile

# 单元测试
mvn test

# 本地运行
mvn spring-boot:run
```

## 3. 从 2021.0.x 升级到 2022.0.x

### 重大变更

**1. Spring Boot 3.0 (Java 17 必需)**:

```xml
<properties>
    <java.version>17</java.version>
    <spring-boot.version>3.0.2</spring-boot.version>
</properties>
```

**2. Jakarta EE 迁移**:

所有 `javax.*` 包名改为 `jakarta.*`:

```java
// 旧版本
import javax.servlet.http.HttpServletRequest;
import javax.validation.Valid;

// 新版本
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
```

**3. Dubbo 升级到 3.x**:

```java
// 旧版本
import com.alibaba.dubbo.config.annotation.Service;
import com.alibaba.dubbo.config.annotation.Reference;

// 新版本
import org.apache.dubbo.config.annotation.DubboService;
import org.apache.dubbo.config.annotation.DubboReference;
```

### 升级步骤

**Step 1: 升级 JDK**

```bash
# 安装 JDK 17
# 配置 JAVA_HOME

# 验证
java -version
```

**Step 2: 批量替换包名**

使用 IDE 全局替换:

- `javax.servlet` → `jakarta.servlet`
- `javax.validation` → `jakarta.validation`
- `javax.persistence` → `jakarta.persistence`

**Step 3: 更新依赖**

```xml
<properties>
    <spring-boot.version>3.0.2</spring-boot.version>
    <spring-cloud.version>2022.0.0</spring-cloud.version>
    <spring-cloud-alibaba.version>2022.0.0.0</spring-cloud-alibaba.version>
</properties>
```

**Step 4: 处理不兼容的依赖**

某些第三方库可能还不支持 Spring Boot 3.0,需要:

- 查找新版本
- 寻找替代方案
- 暂时移除

## 4. 从 2022.0.x 升级到 2023.0.x

### 主要变更

**1. Spring Boot 3.2**:

```xml
<parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>3.2.0</version>
</parent>
```

**2. Nacos 2.3.0**:

- 性能优化
- 新的健康检查机制
- 改进的集群同步

**3. Sentinel 1.8.6**:

- 新增集群流控功能
- 改进的控制台

### 升级步骤

**Step 1: 更新版本**

```xml
<properties>
    <spring-cloud-alibaba.version>2023.0.0.0</spring-cloud-alibaba.version>
</properties>
```

**Step 2: 测试兼容性**

重点测试:

- 服务注册发现
- 配置刷新
- 限流熔断
- 分布式事务

## 5. 升级checklist

### 升级前

- [ ] 阅读官方发布说明
- [ ] 检查版本兼容性
- [ ] 备份现有配置
- [ ] 准备回滚方案

### 升级中

- [ ] 更新依赖版本
- [ ] 处理包名变更
- [ ] 修复编译错误
- [ ] 运行单元测试
- [ ] 本地验证功能

### 升级后

- [ ] 开发环境验证
- [ ] 测试环境部署
- [ ] 灰度发布
- [ ] 监控观察
- [ ] 全量发布

## 6. 常见问题

### JDK 版本不兼容

**问题**: 升级后无法编译

**解决**:

```bash
# 检查 JDK 版本
java -version

# Maven 配置
mvn -version
```

```xml
<!-- pom.xml -->
<properties>
    <maven.compiler.source>17</maven.compiler.source>
    <maven.compiler.target>17</maven.compiler.target>
</properties>
```

### 依赖冲突

**问题**: 类找不到或方法不存在

**解决**:

```bash
# 查看依赖树
mvn dependency:tree

# 排除冲突依赖
```

### 配置不生效

**问题**: 升级后配置中心配置不生效

**检查**:

- [ ] bootstrap 依赖是否添加
- [ ] Data ID格式是否正确
- [ ] namespace 和 group 是否匹配

## 7. 回滚方案

### 回滚步骤

**1. 代码回滚**:

```bash
git revert <commit-hash>
# 或
git reset --hard <commit-hash>
```

**2. 重新部署旧版本**:

```bash
# 使用旧版本镜像
kubectl set image deployment/user-service user-service=user-service:v1.0.0
```

**3. 验证服务**:

```bash
# 健康检查
curl http://localhost:8081/actuator/health

# 验证功能
```

## 8. 最佳实践

### 渐进式升级

```
1. 本地开发环境
2. 测试环境
3. 预发布环境
4. 生产环境(灰度)
5. 生产环境(全量)
```

### 自动化测试

```java
@SpringBootTest
public class UpgradeCompatibilityTest {

    @Test
    public void testServiceDiscovery() {
        // 测试服务注册发现
    }

    @Test
    public void testConfigRefresh() {
        // 测试配置刷新
    }

    @Test
    public void testCircuitBreaker() {
        // 测试熔断降级
    }
}
```

---

**关键要点**:

- 仔细阅读发布说明
- 充分测试
- 准备回滚方案
- 渐进式升级

**参考资料**:

- [Spring Cloud Alibaba 版本说明](https://github.com/alibaba/spring-cloud-alibaba/wiki/%E7%89%88%E6%9C%AC%E8%AF%B4%E6%98%8E)
- [Spring Boot 3.0 迁移指南](https://github.com/spring-projects/spring-boot/wiki/Spring-Boot-3.0-Migration-Guide)
