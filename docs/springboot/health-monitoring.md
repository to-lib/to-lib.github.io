---
sidebar_position: 16
---

# 健康检查与监控

## Spring Boot Actuator

### 依赖和配置

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>
```

```yaml
management:
  endpoints:
    web:
      exposure:
        include: health,info,metrics,env,configprops,loggers,threaddump
      base-path: /actuator
      base-path-mapping:
        health: /health
  
  endpoint:
    health:
      show-details: always
      show-components: always
    shutdown:
      enabled: true
  
  server:
    port: 9000  # 独立的管理端口（可选）
```

### 常用端点

| 端点 | 说明 |
|------|------|
| `/actuator/health` | 应用健康状态 |
| `/actuator/info` | 应用信息 |
| `/actuator/metrics` | 应用指标 |
| `/actuator/env` | 环境属性 |
| `/actuator/configprops` | 配置属性 |
| `/actuator/loggers` | 日志级别 |
| `/actuator/threaddump` | 线程转储 |
| `/actuator/prometheus` | Prometheus 格式指标 |

## 健康检查

### 内置健康指示器

```bash
# 检查健康状态
curl http://localhost:8080/actuator/health

# 响应示例
{
  "status": "UP",
  "components": {
    "db": {
      "status": "UP",
      "details": {
        "database": "MySQL",
        "validationQuery": "isValid()"
      }
    },
    "redis": {
      "status": "UP"
    },
    "diskSpace": {
      "status": "UP",
      "details": {
        "total": 1000000000,
        "free": 500000000,
        "threshold": 10485760,
        "exists": true
      }
    }
  }
}
```

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
            // 检查第三方服务
            boolean serviceHealthy = checkExternalService();
            
            if (serviceHealthy) {
                return Health.up()
                    .withDetail("service", "Running")
                    .withDetail("responseTime", "100ms")
                    .build();
            } else {
                return Health.down()
                    .withDetail("service", "Not responding")
                    .withDetail("lastCheck", System.currentTimeMillis())
                    .build();
            }
        } catch (Exception e) {
            return Health.outOfService()
                .withDetail("error", e.getMessage())
                .build();
        }
    }
    
    private boolean checkExternalService() {
        // 实现检查逻辑
        return true;
    }
}
```

### 就绪性和存活性探针

```java
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;

// 存活性探针（Liveness）- 应用是否还在运行
@Component
public class LivenessHealthIndicator implements HealthIndicator {
    
    @Override
    public Health health() {
        // 检查应用进程是否活跃
        return Health.up()
            .withDetail("status", "Application is running")
            .build();
    }
}

// 就绪性探针（Readiness）- 应用是否准备好处理请求
@Component
public class ReadinessHealthIndicator implements HealthIndicator {
    
    @Override
    public Health health() {
        try {
            // 检查数据库连接
            // 检查缓存可用性
            // 检查依赖服务
            return Health.up()
                .withDetail("status", "Application is ready to handle requests")
                .build();
        } catch (Exception e) {
            return Health.down()
                .withDetail("error", e.getMessage())
                .build();
        }
    }
}
```

### 配置存活性和就绪性

```yaml
management:
  health:
    livenessState:
      enabled: true
    readinessState:
      enabled: true
```

```bash
# 存活性检查
curl http://localhost:8080/actuator/health/liveness

# 就绪性检查
curl http://localhost:8080/actuator/health/readiness
```

## 性能指标

### Micrometer 集成

```xml
<dependency>
    <groupId>io.micrometer</groupId>
    <artifactId>micrometer-core</artifactId>
</dependency>

<!-- Prometheus 支持 -->
<dependency>
    <groupId>io.micrometer</groupId>
    <artifactId>micrometer-registry-prometheus</artifactId>
</dependency>
```

### 自定义指标

```java
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.Counter;
import org.springframework.stereotype.Service;

@Service
public class CustomMetricsService {
    
    private final MeterRegistry meterRegistry;
    private final Counter requestCounter;
    private final Timer requestTimer;
    
    public CustomMetricsService(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
        
        // 创建计数器
        this.requestCounter = Counter.builder("custom.requests.total")
            .description("Total number of requests")
            .register(meterRegistry);
        
        // 创建计时器
        this.requestTimer = Timer.builder("custom.request.duration")
            .description("Request duration")
            .publishPercentiles(0.5, 0.95, 0.99)
            .register(meterRegistry);
    }
    
    public void handleRequest() {
        requestCounter.increment();
        
        requestTimer.record(() -> {
            // 处理请求
        });
    }
    
    // Gauge - 测量当前值
    public void registerGauge() {
        meterRegistry.gauge("custom.queue.size", 
            () -> getQueueSize());
    }
    
    private int getQueueSize() {
        return 10;
    }
}
```

### 访问指标

```bash
# 获取所有指标
curl http://localhost:8080/actuator/metrics

# 获取特定指标
curl http://localhost:8080/actuator/metrics/http.server.requests

# Prometheus 格式
curl http://localhost:8080/actuator/prometheus
```

## Prometheus 和 Grafana

### Prometheus 配置

创建 `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'spring-boot'
    metrics_path: '/actuator/prometheus'
    static_configs:
      - targets: ['localhost:8080']
```

### Docker 运行 Prometheus

```bash
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

### Grafana 仪表板

```bash
docker run -d \
  --name grafana \
  -p 3000:3000 \
  grafana/grafana
```

在 Grafana 中：
1. 添加 Prometheus 数据源：`http://prometheus:9090`
2. 导入 Spring Boot 仪表板（ID: 6417）

## 日志管理

### 动态修改日志级别

```bash
# 查看日志配置
curl http://localhost:8080/actuator/loggers

# 获取特定日志级别
curl http://localhost:8080/actuator/loggers/com.example

# 修改日志级别
curl -X POST http://localhost:8080/actuator/loggers/com.example \
  -H "Content-Type: application/json" \
  -d '{"configuredLevel":"DEBUG"}'
```

### 日志配置类

```java
@Component
@Slf4j
public class LoggingConfigService {
    
    @Autowired
    private LoggingSystem loggingSystem;
    
    public void setLogLevel(String loggerName, LogLevel level) {
        loggingSystem.setLogLevel(loggerName, level);
        log.info("Set {} log level to {}", loggerName, level);
    }
}
```

## 应用信息

### 配置应用信息

```yaml
app:
  name: My Application
  version: 1.0.0
  description: This is my Spring Boot application
  author: Your Name
  contact:
    email: contact@example.com
    url: https://example.com

spring:
  application:
    name: myapp
```

### 创建 Info 端点

```java
import org.springframework.boot.actuate.info.Info;
import org.springframework.boot.actuate.info.InfoContributor;
import org.springframework.stereotype.Component;

@Component
public class CustomInfoContributor implements InfoContributor {
    
    @Override
    public void contribute(Info.Builder builder) {
        builder.withDetail("app", Map.of(
            "name", "My Application",
            "version", "1.0.0",
            "buildTime", new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
                .format(new Date()),
            "javaVersion", System.getProperty("java.version"),
            "osName", System.getProperty("os.name")
        ));
    }
}
```

访问信息：

```bash
curl http://localhost:8080/actuator/info
```

## 性能监控

### JVM 监控

```java
import io.micrometer.core.instrument.MeterRegistry;
import com.sun.management.OperatingSystemMXBean;
import java.lang.management.ManagementFactory;

@Component
public class JvmMonitor {
    
    private final MeterRegistry meterRegistry;
    
    @Autowired
    public JvmMonitor(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
        initializeMetrics();
    }
    
    private void initializeMetrics() {
        OperatingSystemMXBean osBean = 
            (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
        
        // CPU 使用率
        meterRegistry.gauge("jvm.cpu.usage", 
            osBean::getProcessCpuLoad);
        
        // 系统 CPU 使用率
        meterRegistry.gauge("system.cpu.usage", 
            osBean::getSystemCpuLoad);
        
        // 进程 CPU 时间
        meterRegistry.gauge("jvm.cpu.time", 
            () -> osBean.getProcessCpuTime() / 1_000_000_000);
    }
}
```

### HTTP 请求监控

```java
@Component
public class HttpMetricsFilter extends OncePerRequestFilter {
    
    private final MeterRegistry meterRegistry;
    
    @Override
    protected void doFilterInternal(HttpServletRequest request, 
                                  HttpServletResponse response, 
                                  FilterChain filterChain) throws ServletException, IOException {
        long startTime = System.currentTimeMillis();
        
        try {
            filterChain.doFilter(request, response);
        } finally {
            long duration = System.currentTimeMillis() - startTime;
            
            meterRegistry.timer("http.request.duration",
                "method", request.getMethod(),
                "path", request.getRequestURI(),
                "status", String.valueOf(response.getStatus())
            ).record(duration, TimeUnit.MILLISECONDS);
        }
    }
}
```

## 总结

监控和健康检查的关键点：

1. ✅ **Actuator 端点** - 提供应用的各种信息和指标
2. ✅ **自定义健康检查** - 监控关键的业务系统
3. ✅ **性能指标** - 使用 Micrometer 收集性能数据
4. ✅ **Prometheus 集成** - 导出 Prometheus 格式的指标
5. ✅ **Grafana 可视化** - 在仪表板中展示指标
6. ✅ **存活性和就绪性** - Kubernetes 健康检查支持
7. ✅ **日志管理** - 动态调整日志级别

---

**提示**：定期监控应用的健康状态和性能指标是保证应用稳定运行的关键！
