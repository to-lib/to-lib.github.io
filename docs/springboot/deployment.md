---
sidebar_position: 18
---

# 部署上线

## 构建与打包

```bash
# 清理并打包
mvn clean package

# 跳过测试打包
mvn clean package -DskipTests

# 打包 WAR 文件（用于应用服务器）
mvn clean package -P war
```

### Gradle 打包

```bash
# 编译并构建
gradle build

# 跳过测试
gradle build -x test

# 生成 JAR
gradle bootJar
```

### 检查打包结果

```bash
# 查看 JAR 内容
jar tf target/myapp.jar

# 查看 MANIFEST 文件
jar xf target/myapp.jar META-INF/MANIFEST.MF
```

## JAR 运行

### 基本运行

```bash
# 默认参数运行
java -jar myapp.jar

# 指定环境
java -jar myapp.jar --spring.profiles.active=prod

# 指定端口
java -jar myapp.jar --server.port=9000

# 指定多个参数
java -jar myapp.jar \
  --spring.profiles.active=prod \
  --server.port=8443 \
  --spring.datasource.url=jdbc:mysql://db:3306/mydb
```

### JVM 参数调优

```bash
# 内存配置
java -Xms512m -Xmx1024m -jar myapp.jar

# 垃圾回收配置
java -XX:+UseG1GC \
     -XX:MaxGCPauseMillis=200 \
     -jar myapp.jar

# 详细 GC 日志
java -Xms512m -Xmx1024m \
     -XX:+PrintGCDetails \
     -XX:+PrintGCTimeStamps \
     -Xloggc:gc.log \
     -jar myapp.jar
```

### 后台运行

```bash
# 使用 nohup
nohup java -jar myapp.jar > app.log 2>&1 &

# 使用 systemd (Linux)
systemctl start myapp

# 使用 screen
screen -dmS myapp java -jar myapp.jar
```

## Docker 部署

### Dockerfile 创建

```dockerfile
# 多阶段构建
FROM maven:3.8-openjdk-17 AS builder
WORKDIR /app
COPY pom.xml .
RUN mvn dependency:go-offline
COPY src ./src
RUN mvn clean package -DskipTests

# 最终镜像
FROM openjdk:17-slim
WORKDIR /app
COPY --from=builder /app/target/*.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "/app/app.jar"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - SPRING_PROFILES_ACTIVE=prod
      - SPRING_DATASOURCE_URL=jdbc:mysql://mysql:3306/mydb
      - SPRING_DATASOURCE_USERNAME=root
      - SPRING_DATASOURCE_PASSWORD=password
    depends_on:
      - mysql
      - redis
    networks:
      - myapp-network

  mysql:
    image: mysql:8.0
    environment:
      - MYSQL_ROOT_PASSWORD=password
      - MYSQL_DATABASE=mydb
    volumes:
      - mysql-data:/var/lib/mysql
    networks:
      - myapp-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - myapp-network

volumes:
  mysql-data:

networks:
  myapp-network:
```

构建和运行：

```bash
docker-compose up -d
```

## Kubernetes 部署

### Deployment 配置

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myrepo/myapp:1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: SPRING_PROFILES_ACTIVE
          value: prod
        - name: SPRING_DATASOURCE_URL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1024Mi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /actuator/health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /actuator/health/readiness
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
```

### Service 配置

```yaml
apiVersion: v1
kind: Service
metadata:
  name: myapp-service
spec:
  type: LoadBalancer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  selector:
    app: myapp
```

部署：

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl get pods
kubectl logs -f deployment/myapp
```

## 配置管理

### 环境变量配置

```bash
# 编辑 .env 文件
export SPRING_PROFILES_ACTIVE=prod
export SPRING_DATASOURCE_URL=jdbc:mysql://db:3306/mydb
export SPRING_DATASOURCE_USERNAME=root
export SPRING_DATASOURCE_PASSWORD=password
export SERVER_PORT=8080
export LOGGING_LEVEL_ROOT=WARN
```

### 配置文件方式

```bash
java -jar myapp.jar --spring.config.location=file:/etc/config/application.yml
```

### 外部配置中心（Spring Cloud Config）

```yaml
spring:
  cloud:
    config:
      uri: http://config-server:8888
      fail-fast: true
      retry:
        initial-interval: 1000
        max-interval: 2000
        multiplier: 1.1
        max-attempts: 6
```

## 监控和日志

### Actuator 端点配置

```yaml
management:
  endpoints:
    web:
      exposure:
        include: health,metrics,env,configprops,prometheus
      base-path: /actuator
  
  metrics:
    export:
      prometheus:
        enabled: true
  
  endpoint:
    health:
      show-details: always
      show-components: always
```

### 访问监控端点

```bash
# 健康检查
curl http://localhost:8080/actuator/health

# 应用信息
curl http://localhost:8080/actuator/info

# 获取指标
curl http://localhost:8080/actuator/metrics

# Prometheus 格式指标
curl http://localhost:8080/actuator/prometheus
```

### 日志配置

```yaml
logging:
  level:
    root: WARN
    com.example: INFO
  
  file:
    name: /var/log/myapp/application.log
    max-size: 10MB
    max-history: 30
    total-size-cap: 1GB
  
  pattern:
    console: "%d{yyyy-MM-dd HH:mm:ss} %-5level %logger{36} - %msg%n"
    file: "%d{yyyy-MM-dd HH:mm:ss} [%thread] %-5level %logger{36} - %msg%n"

# Logback 配置（可选）
  config: classpath:logback-spring.xml
```

## 健康检查

### 自定义健康检查

```java
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.stereotype.Component;

@Component
public class CustomHealthIndicator implements HealthIndicator {
    
    @Override
    public Health health() {
        try {
            // 检查自定义条件
            boolean serviceRunning = checkServiceStatus();
            
            if (serviceRunning) {
                return Health.up()
                    .withDetail("status", "All systems operational")
                    .build();
            } else {
                return Health.down()
                    .withDetail("status", "Service is not responding")
                    .build();
            }
        } catch (Exception e) {
            return Health.unknown()
                .withDetail("error", e.getMessage())
                .build();
        }
    }
    
    private boolean checkServiceStatus() {
        // 自定义检查逻辑
        return true;
    }
}
```

### 就绪性和存活性探针

```yaml
management:
  health:
    livenessState:
      enabled: true
    readinessState:
      enabled: true
```

```bash
# 存活性探针（应用是否还在运行）
curl http://localhost:8080/actuator/health/liveness

# 就绪性探针（应用是否准备好处理请求）
curl http://localhost:8080/actuator/health/readiness
```

## 性能优化建议

### JVM 参数

```bash
# G1 垃圾回收
-XX:+UseG1GC \
-XX:MaxGCPauseMillis=200 \
-XX:InitiatingHeapOccupancyPercent=35

# 类加载缓存
-XX:+UnlockDiagnosticVMOptions \
-XX:+TraceClassLoading

# 内存配置
-Xms512m -Xmx2048m
```

### 数据库连接池优化

```yaml
spring:
  datasource:
    hikari:
      maximum-pool-size: 20
      minimum-idle: 5
      connection-timeout: 30000
      idle-timeout: 600000
      max-lifetime: 1800000
      auto-commit: true
```

### 缓存配置

```yaml
spring:
  cache:
    type: redis
    redis:
      time-to-live: 600000

  redis:
    jedis:
      pool:
        max-active: 16
        max-idle: 8
        min-idle: 0
```

## 灾难恢复

### 备份策略

```bash
# 定期备份数据库
mysqldump -u root -p mydb > backup-$(date +%Y%m%d).sql

# 备份应用配置
tar -czf config-backup-$(date +%Y%m%d).tar.gz /etc/config/

# 备份 Redis 数据
redis-cli BGSAVE
```

### 应用重启策略

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: myapp
spec:
  restartPolicy: Always  # Always, OnFailure, Never
  containers:
  - name: app
    image: myapp:1.0
```

## 发布策略

### 蓝绿部署

```bash
# 启动新版本（绿）
docker run -d --name myapp-green myapp:2.0

# 验证新版本
curl http://localhost:8080/actuator/health

# 切换流量到新版本
nginx reload  # 更新 nginx 配置指向新版本

# 停止旧版本（蓝）
docker stop myapp-blue
```

### 金丝雀部署（Canary）

```yaml
apiVersion: fluxcd.io/v1alpha1
kind: Canary
metadata:
  name: myapp
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp
  progressDeadlineSeconds: 60
  service:
    port: 8080
  analysis:
    interval: 1m
    threshold: 5
    metrics:
    - name: error-rate
      thresholdRange:
        max: 1
  skipAnalysis: false
```

## 总结

成功的部署需要：

1. ✅ 合理的打包和构建流程
2. ✅ 充分的监控和日志
3. ✅ 定期的备份和恢复测试
4. ✅ 适当的灾难恢复计划
5. ✅ 稳妥的发布策略
6. ✅ 性能和资源优化

---

**提示**：始终在生产环境前进行充分的测试和验证！
