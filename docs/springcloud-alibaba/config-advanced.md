---
id: config-advanced
title: Nacos 配置中心高级特性
sidebar_label: 配置高级特性
sidebar_position: 10
---

# Nacos 配置中心高级特性

> [!TIP]
> **深入配置管理**: 探索 Nacos 配置中心的高级功能,包括配置加密、灰度发布、版本管理等企业级特性。

## 1. 配置加密

### 使用 Jasypt 加密

**添加依赖**:

```xml
<dependency>
    <groupId>com.github.ulisesbocchio</groupId>
    <artifactId>jasypt-spring-boot-starter</artifactId>
    <version>3.0.5</version>
</dependency>
```

**配置密钥**:

```yaml
jasypt:
  encryptor:
    password: ${JASYPT_ENCRYPTOR_PASSWORD}  # 从环境变量获取
    algorithm: PBEWithMD5AndDES
```

**加密敏感信息**:

```java
import org.jasypt.encryption.StringEncryptor;

@SpringBootTest
public class EncryptTest {

    @Autowired
    private StringEncryptor stringEncryptor;

    @Test
    public void testEncrypt() {
        String password = "myPassword123";
        String encrypted = stringEncryptor.encrypt(password);
        System.out.println("加密后: " + encrypted);
    }
}
```

**在 Nacos 中使用加密配置**:

```yaml
spring:
  datasource:
    password: ENC(G9FDaZ/oiJKdT7TqZvGmKg==)  # 加密后的密码
```

### 自定义加密解密

```java
@Component
public class CustomEncryptor {

    public String encrypt(String plainText) {
        // 自定义加密逻辑
        return Base64.getEncoder().encodeToString(plainText.getBytes());
    }

    public String decrypt(String encryptedText) {
        // 自定义解密逻辑
        return new String(Base64.getDecoder().decode(encryptedText));
    }
}

@Component
public class EncryptedConfigProcessor implements EnvironmentPostProcessor {

    @Override
    public void postProcessEnvironment(ConfigurableEnvironment environment, 
                                       SpringApplication application) {
        // 处理加密配置
        for (PropertySource<?> propertySource : environment.getPropertySources()) {
            if (propertySource instanceof MapPropertySource) {
                processPropertySource((MapPropertySource) propertySource);
            }
        }
    }
}
```

## 2. 配置灰度发布

### Beta 发布

**在 Nacos 控制台操作**:

1. 编辑配置
2. 点击"Beta发布"
3. 输入 Beta IP 列表 (如: `192.168.1.100,192.168.1.101`)
4. 发布 Beta 配置

**验证 Beta 配置**:

```java
@RestController
@RefreshScope
public class ConfigController {

    @Value("${app.version:默认版本}")
    private String version;

    @GetMapping("/config/version")
    public String getVersion() {
        return "当前版本: " + version;
    }
}
```

**发布流程**:

```
1. Beta发布 → 验证 Beta 环境
2. 正式发布 → 所有实例生效
3. 回滚 → 恢复上一版本
```

### 金丝雀发布

**自定义金丝雀策略**:

```java
@Component
public class CanaryConfigListener {

    @Autowired
    private NacosConfigManager nacosConfigManager;

    @PostConstruct
    public void init() throws NacosException {
        String dataId = "app-config.yaml";
        String group = "DEFAULT_GROUP";

        // 监听配置变更
        nacosConfigManager.getConfigService().addListener(
            dataId,
            group,
            new Listener() {
                @Override
                public void receiveConfigInfo(String configInfo) {
                    // 金丝雀策略: 随机 10% 流量使用新配置
                    if (Math.random() < 0.1) {
                        applyNewConfig(configInfo);
                    } else {
                        applyOldConfig();
                    }
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

## 3. 配置版本管理

### 查看历史版本

**Nacos 控制台**:

- 配置管理 → 配置列表 → 点击配置 → 历史版本

**API 方式**:

```java
@Service
public class ConfigHistoryService {

    @Autowired
    private NacosConfigProperties nacosConfigProperties;

    public List<ConfigHistory> getConfigHistory(String dataId, String group) {
        String serverAddr = nacosConfigProperties.getServerAddr();
        String url = String.format(
            "http://%s/nacos/v1/cs/history?dataId=%s&group=%s&search=accurate",
            serverAddr, dataId, group
        );

        // 调用 API 获取历史版本
        RestTemplate restTemplate = new RestTemplate();
        return restTemplate.getForObject(url, List.class);
    }
}
```

### 配置回滚

```java
public void rollbackConfig(String dataId, String group, String nid) {
    String serverAddr = nacosConfigProperties.getServerAddr();
    String url = String.format(
        "http://%s/nacos/v1/cs/history?dataId=%s&group=%s&nid=%s&rollback=true",
        serverAddr, dataId, group, nid
    );

    RestTemplate restTemplate = new RestTemplate();
    restTemplate.postForObject(url, null, String.class);
}
```

## 4. 配置监听与回调

### 实时监听配置变更

```java
@Component
public class ConfigChangeListener {

    @Autowired
    private NacosConfigManager nacosConfigManager;

    @PostConstruct
    public void registerListener() throws NacosException {
        String dataId = "user-service.yaml";
        String group = "DEFAULT_GROUP";

        nacosConfigManager.getConfigService().addListener(
            dataId,
            group,
            new AbstractConfigChangeListener() {
                @Override
                public void receiveConfigChange(ConfigChangeEvent event) {
                    for (ConfigChangeItem item : event.getChangeItems()) {
                        log.info("配置变更: key={}, oldValue={}, newValue={}, type={}",
                            item.getKey(),
                            item.getOldValue(),
                            item.getNewValue(),
                            item.getType()
                        );
                        
                        // 根据变更类型处理
                        switch (item.getType()) {
                            case ADDED:
                                handleAdded(item);
                                break;
                            case MODIFIED:
                                handleModified(item);
                                break;
                            case DELETED:
                                handleDeleted(item);
                                break;
                        }
                    }
                }
            }
        );
    }

    private void handleAdded(ConfigChangeItem item) {
        log.info("新增配置: {}", item.getKey());
    }

    private void handleModified(ConfigChangeItem item) {
        log.info("修改配置: {}", item.getKey());
        // 可以触发特定逻辑,如清除缓存
        if ("cache.enabled".equals(item.getKey())) {
            clearCache();
        }
    }

    private void handleDeleted(ConfigChangeItem item) {
        log.info("删除配置: {}", item.getKey());
    }
}
```

### 配置变更通知

```java
@Component
public class ConfigChangeNotifier {

    @Autowired
    private JavaMailSender mailSender;

    @EventListener
    public void onConfigChange(ConfigChangeEvent event) {
        // 发送邮件通知
        sendEmail("配置变更通知", event.toString());

        // 发送钉钉通知
        sendDingTalk("配置变更", event.toString());
    }

    private void sendEmail(String subject, String content) {
        SimpleMailMessage message = new SimpleMailMessage();
        message.setTo("ops@example.com");
        message.setSubject(subject);
        message.setText(content);
        mailSender.send(message);
    }
}
```

## 5. 多环境配置管理

### 配置层级结构

```
Namespace (环境)
├── dev (开发)
│   ├── DEFAULT_GROUP
│   │   ├── application.yaml (公共配置)
│   │   ├── user-service.yaml
│   │   └── order-service.yaml
│   └── DATABASE_GROUP
│       ├── mysql.yaml
│       └── redis.yaml
├── test (测试)
└── prod (生产)
```

### 配置继承

**公共配置 (shared-config)**:

```yaml
# common.yaml
logging:
  level:
    root: INFO

management:
  endpoints:
    web:
      exposure:
        include: "*"
```

**应用配置**:

```yaml
spring:
  cloud:
    nacos:
      config:
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
          - data-id: mysql.yaml
            group: DATABASE_GROUP
            refresh: true
```

### 配置优先级

**加载顺序(从高到低)**:

```
1. application-${profile}.${file-extension}
2. application.${file-extension}
3. extension-configs (从下到上)
4. shared-configs (从下到上)
5. bootstrap.yml
```

## 6. 配置推送

### 服务端推送

Nacos 使用长轮询机制实现配置实时推送:

```java
// Nacos Client 内部实现
public class ClientWorker {
    
    // 长轮询
    private void checkUpdateDataIds(List<CacheData> cacheDatas, long timeout) {
        // 1. 构建请求参数
        StringBuilder sb = new StringBuilder();
        for (CacheData cacheData : cacheDatas) {
            sb.append(cacheData.dataId).append(WORD_SEPARATOR);
            sb.append(cacheData.group).append(WORD_SEPARATOR);
            sb.append(cacheData.getMd5()).append(LINE_SEPARATOR);
        }

        // 2. 发送长轮询请求(默认30秒)
        String result = agent.httpPost(
            Constants.CONFIG_CONTROLLER_PATH + "/listener",
            headers,
            params,
            timeout
        );

        // 3. 解析变更的配置
        parseUpdateDataIdResponse(result);
    }
}
```

### 客户端拉取

```java
@Service
public class ConfigPullService {

    @Autowired
    private ConfigService configService;

    public String getConfig(String dataId, String group, long timeout) 
        throws NacosException {
        return configService.getConfig(dataId, group, timeout);
    }

    @Scheduled(fixedRate = 60000)  // 每分钟拉取一次
    public void pullConfig() {
        try {
            String config = getConfig("app-config.yaml", "DEFAULT_GROUP", 3000);
            processConfig(config);
        } catch (NacosException e) {
            log.error("拉取配置失败", e);
        }
    }
}
```

## 7. 配置导入导出

### 导出配置

```bash
# 导出单个配置
curl -X GET "http://localhost:8848/nacos/v1/cs/configs?dataId=user-service.yaml&group=DEFAULT_GROUP" \
  > user-service.yaml

# 导出所有配置 (需要登录)
# 在 Nacos 控制台: 配置管理 → 导出配置
```

### 导入配置

**批量导入**:

```java
@Service
public class ConfigImportService {

    @Autowired
    private ConfigService configService;

    public void importConfigs(List<ConfigDTO> configs) {
        for (ConfigDTO config : configs) {
            try {
                boolean result = configService.publishConfig(
                    config.getDataId(),
                    config.getGroup(),
                    config.getContent(),
                    config.getType()
                );
                
                if (result) {
                    log.info("导入配置成功: {}", config.getDataId());
                } else {
                    log.error("导入配置失败: {}", config.getDataId());
                }
            } catch (NacosException e) {
                log.error("导入配置异常: {}", config.getDataId(), e);
            }
        }
    }
}
```

## 8. 配置中心高可用

### 集群部署

**集群架构**:

```
                  Load Balancer
                       |
      +----------------+----------------+
      |                |                |
   Nacos 1          Nacos 2         Nacos 3
      |                |                |
      +----------------+----------------+
                       |
                    MySQL (主从)
```

**客户端配置**:

```yaml
spring:
  cloud:
    nacos:
      config:
        server-addr: 192.168.1.101:8848,192.168.1.102:8848,192.168.1.103:8848
```

### 数据持久化

**MySQL 配置**:

```properties
# application.properties
spring.datasource.platform=mysql

db.num=1
db.url.0=jdbc:mysql://localhost:3306/nacos?characterEncoding=utf8
db.user.0=nacos
db.password.0=nacos
```

### 故障转移

Nacos Client 支持自动故障转移:

```java
// 当某个 Nacos 节点故障时,自动切换到其他节点
public class ServerListManager {
    
    private void refreshServerListIfNeed() {
        // 检测服务器健康状态
        for (String server : serverList) {
            if (!isServerAvailable(server)) {
                // 从列表中移除故障节点
                serverList.remove(server);
            }
        }
        
        // 选择健康的服务器
        currentServer = selectHealthyServer();
    }
}
```

## 9. 配置审计

### 操作记录

Nacos 自动记录所有配置操作:

```sql
-- 查询配置操作记录
SELECT * FROM config_info_aggr 
WHERE data_id = 'user-service.yaml'
ORDER BY gmt_modified DESC;
```

### 自定义审计

```java
@Aspect
@Component
public class ConfigAuditAspect {

    @Around("@annotation(com.example.ConfigOperation)")
    public Object auditConfigOperation(ProceedingJoinPoint joinPoint) throws Throwable {
        // 记录操作前状态
        String before = getCurrentConfig();
        
        // 执行操作
        Object result = joinPoint.proceed();
        
        // 记录操作后状态
        String after = getCurrentConfig();
        
        // 保存审计日志
        saveAuditLog(before, after);
        
        return result;
    }
}
```

## 10. 最佳实践

### 配置命名规范

```
格式: ${应用名}-${环境}.${文件扩展名}

示例:
user-service-dev.yaml
user-service-prod.yaml
order-service-dev.yaml
```

### 配置分类

| 配置类型   | 存放位置        | 刷新策略   |
| ---------- | --------------- | ---------- |
| 基础配置   | bootstrap.yml   | 不刷新     |
| 业务配置   | Nacos共享配置   | 动态刷新   |
| 环境配置   | Nacos扩展配置   | 动态刷新   |
| 敏感配置   | 环境变量        | 不刷新     |

### 配置变更流程

```
1. 开发环境测试
2. 测试环境验证
3. 灰度发布(Beta)
4. 全量发布
5. 监控观察
6. 出现问题立即回滚
```

### 安全建议

- ✅ 敏感信息必须加密
- ✅ 生产环境配置权限控制
- ✅ 开启操作审计
- ✅ 定期备份配置
- ✅ 使用 HTTPS

---

**关键要点**:

- 配置加密保护敏感信息
- 灰度发布降低变更风险
- 版本管理便于回滚
- 监听机制实时响应变更
- 集群部署保证高可用

**下一步**: 学习 [服务治理高级](/docs/springcloud-alibaba/service-governance)
