---
id: faq
title: 常见问题解答
sidebar_label: 常见问题
sidebar_position: 97
---

# Spring Framework 常见问题解答

## 1. 依赖注入相关

### Q1: 什么时候应该使用@Autowired？

**A:** 在Spring管理的Bean中注入其他Bean时使用。但最好的实践是使用构造函数注入。

```java
// ✅ 最佳实践
@Component
public class MyService {
    private final MyDependency dependency;
    
    public MyService(MyDependency dependency) {
        this.dependency = dependency;
    }
}

// 次选
@Component
public class MyService {
    @Autowired
    public void setDependency(MyDependency dependency) {
        this.dependency = dependency;
    }
}
```

### Q2: 如何解决循环依赖问题？

**A:** 重构设计，提取公共逻辑到第三个类。

```java
// ❌ 问题：循环依赖
@Component
public class ServiceA {
    @Autowired
    private ServiceB serviceB;
}

@Component
public class ServiceB {
    @Autowired
    private ServiceA serviceA;
}

// ✅ 解决
@Component
public class CommonService {
    // 公共逻辑
}

@Component
public class ServiceA {
    @Autowired
    private CommonService common;
}

@Component
public class ServiceB {
    @Autowired
    private CommonService common;
}
```

### Q3: 如何处理Bean不存在的情况？

**A:** 使用Optional或ObjectProvider。

```java
// 方式1：使用Optional
@Component
public class MyService {
    @Autowired
    private Optional<OptionalDependency> dependency;
    
    public void process() {
        dependency.ifPresent(dep -> dep.execute());
    }
}

// 方式2：使用ObjectProvider
@Component
public class MyService {
    @Autowired
    private ObjectProvider<OptionalDependency> dependency;
    
    public void process() {
        ObjectProvider<OptionalDependency> dep = dependency.getIfAvailable();
        if (dep != null) {
            dep.execute();
        }
    }
}
```

### Q4: 多个Bean实现同一接口怎么办？

**A:** 使用@Qualifier或@Primary。

```java
@Component
@Qualifier("mysql")
public class MysqlRepository implements UserRepository {
}

@Component
@Primary  // 设置为首选
public class MongoRepository implements UserRepository {
}

@Component
public class UserService {
    // 方式1：使用@Qualifier
    @Autowired
    @Qualifier("mysql")
    private UserRepository repo1;
    
    // 方式2：使用@Primary
    @Autowired
    private UserRepository repo2;  // 默认注入MongoRepository
}
```

## 2. Bean生命周期相关

### Q5: Bean的初始化顺序是什么？

**A:** 
1. 实例化
2. 设置属性
3. @PostConstruct
4. InitializingBean.afterPropertiesSet()
5. 自定义init-method
6. 使用中
7. @PreDestroy
8. DisposableBean.destroy()

```java
@Component
public class MyBean implements InitializingBean, DisposableBean {
    
    @PostConstruct
    public void init1() {
        System.out.println("1. @PostConstruct");
    }
    
    @Override
    public void afterPropertiesSet() {
        System.out.println("2. afterPropertiesSet");
    }
    
    public void init2() {
        System.out.println("3. init-method");
    }
    
    @PreDestroy
    public void destroy1() {
        System.out.println("4. @PreDestroy");
    }
    
    @Override
    public void destroy() {
        System.out.println("5. destroy");
    }
}
```

### Q6: 如何在Bean初始化后执行某些操作？

**A:** 使用@PostConstruct注解。

```java
@Component
public class DatabaseInitializer {
    
    @Autowired
    private JdbcTemplate jdbcTemplate;
    
    @PostConstruct
    public void init() {
        // 在Bean创建后执行
        jdbcTemplate.update("CREATE TABLE IF NOT EXISTS ...");
        System.out.println("Database initialized");
    }
}
```

## 3. 配置相关

### Q7: 如何切换不同的配置？

**A:** 使用Spring Profiles。

```yaml
# application.yml
spring:
  profiles:
    active: dev

---
# application-dev.yml
spring:
  profiles: dev
  datasource:
    url: jdbc:mysql://localhost:3306/mydb_dev

---
# application-prod.yml
spring:
  profiles: prod
  datasource:
    url: jdbc:mysql://prod-server:3306/mydb_prod
```

运行时指定：
```bash
java -jar app.jar --spring.profiles.active=prod
```

### Q8: 如何注入配置值？

**A:** 使用@Value或@ConfigurationProperties。

```java
// 方式1：@Value
@Component
public class AppConfig {
    @Value("${app.name:Default}")
    private String appName;
    
    @Value("${app.version}")
    private String appVersion;
}

// 方式2：@ConfigurationProperties（推荐）
@Configuration
@ConfigurationProperties(prefix = "app")
@Data
public class AppProperties {
    private String name;
    private String version;
}

@Component
public class MyService {
    @Autowired
    private AppProperties appProperties;
}
```

## 4. 事务相关

### Q9: @Transactional不生效怎么办？

**A:** 检查以下几点：

1. 方法必须是public
2. 必须通过Spring容器调用
3. 确保启用了事务管理

```java
// ❌ 不生效：private方法
@Service
public class UserService {
    @Transactional
    private void saveUser(User user) {  // private不生效
    }
}

// ✅ 改为public
@Service
public class UserService {
    @Transactional
    public void saveUser(User user) {
    }
}

// ❌ 不生效：self-invocation
@Service
public class UserService {
    @Transactional
    public void save(User user) {
    }
    
    public void register(User user) {
        save(user);  // 事务不生效，应该用代理调用
    }
}

// ✅ 解决：注入自己
@Service
public class UserService {
    @Autowired
    private UserService self;
    
    @Transactional
    public void save(User user) {
    }
    
    public void register(User user) {
        self.save(user);  // 通过代理调用
    }
}
```

### Q10: 事务回滚不起作用？

**A:** 检查异常类型和配置。

```java
// 默认只对RuntimeException回滚
@Transactional
public void saveUser(User user) throws Exception {
    userRepository.save(user);
    throw new Exception("Error");  // 不会回滚
}

// 解决：指定回滚异常
@Transactional(rollbackFor = Exception.class)
public void saveUser(User user) throws Exception {
    userRepository.save(user);
    throw new Exception("Error");  // 会回滚
}
```

## 5. Web相关

### Q11: 如何处理CORS问题？

**A:** 配置CORS。

```java
// 全局CORS配置
@Configuration
public class CorsConfig implements WebMvcConfigurer {
    
    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/api/**")
            .allowedOrigins("http://localhost:3000", "http://localhost:4200")
            .allowedMethods("GET", "POST", "PUT", "DELETE")
            .allowedHeaders("*")
            .allowCredentials(true)
            .maxAge(3600);
    }
}

// 或在Controller上
@RestController
@RequestMapping("/api/users")
@CrossOrigin(origins = "http://localhost:3000")
public class UserController {
}

// 或在方法上
@GetMapping("/{id}")
@CrossOrigin
public User getUser(@PathVariable Long id) {
}
```

### Q12: 如何验证请求参数？

**A:** 使用Bean Validation。

```java
@Data
public class UserDTO {
    @NotNull(message = "ID不能为空")
    private Long id;
    
    @NotBlank(message = "名称不能为空")
    private String name;
    
    @Email(message = "邮箱格式不正确")
    private String email;
}

@RestController
@RequestMapping("/api/users")
public class UserController {
    
    @PostMapping
    public ResponseEntity<User> createUser(@Valid @RequestBody UserDTO userDTO) {
        // 如果验证失败，自动返回400
        return ResponseEntity.ok(new User(userDTO));
    }
}
```

### Q13: 如何自定义参数解析？

**A:** 实现Converter或使用@RequestParam。

```java
// 自定义Converter
@Component
public class StringToLocalDateConverter implements Converter<String, LocalDate> {
    @Override
    public LocalDate convert(String source) {
        return LocalDate.parse(source, DateTimeFormatter.ISO_LOCAL_DATE);
    }
}

// 注册Converter
@Configuration
public class WebConfig implements WebMvcConfigurer {
    
    @Override
    public void addFormatters(FormatterRegistry registry) {
        registry.addConverter(new StringToLocalDateConverter());
    }
}

// 使用
@GetMapping("/search")
public String search(
    @RequestParam String keyword,
    @RequestParam LocalDate startDate) {
    
    return String.format("Search: %s from %s", keyword, startDate);
}
```

## 6. AOP相关

### Q14: 为什么AOP不生效？

**A:** 检查以下几点：

1. 被代理对象必须是Spring Bean
2. 方法必须是public
3. 不能是static方法
4. 确保aspect已启用

```java
// ❌ 问题：static方法不被AOP处理
@Component
public class MyService {
    @Transactional
    public static void staticMethod() {
    }
}

// ✅ 改为非static
@Component
public class MyService {
    @Transactional
    public void publicMethod() {
    }
}
```

### Q15: 如何在AOP中获取方法参数？

**A:** 使用JoinPoint。

```java
@Aspect
@Component
public class LoggingAspect {
    
    @Before("execution(* com.example.service.*.*(..))")
    public void before(JoinPoint joinPoint) {
        String methodName = joinPoint.getSignature().getName();
        Object[] args = joinPoint.getArgs();
        
        System.out.println(String.format(
            "Calling %s with args: %s",
            methodName,
            Arrays.toString(args)
        ));
    }
}
```

## 7. 其他问题

### Q16: 如何访问ApplicationContext？

**A:** 注入或实现ApplicationContextAware。

```java
// 方式1：直接注入
@Component
public class MyComponent {
    @Autowired
    private ApplicationContext applicationContext;
    
    public void getBean() {
        MyBean bean = applicationContext.getBean(MyBean.class);
    }
}

// 方式2：实现Aware接口
@Component
public class MyComponent implements ApplicationContextAware {
    
    private ApplicationContext applicationContext;
    
    @Override
    public void setApplicationContext(ApplicationContext applicationContext) {
        this.applicationContext = applicationContext;
    }
}
```

### Q17: 如何延迟初始化Bean？

**A:** 使用@Lazy注解。

```java
@Component
@Lazy
public class ExpensiveService {
    
    public ExpensiveService() {
        System.out.println("ExpensiveService created");
    }
}

@Component
public class MyService {
    
    @Autowired
    @Lazy
    private ExpensiveService expensiveService;
    
    public void process() {
        // 第一次使用时才创建Bean
        expensiveService.execute();
    }
}
```

### Q18: 如何条件化创建Bean？

**A:** 使用@Conditional或Spring Boot条件注解。

```java
// 自定义条件
public class RedisEnabledCondition implements Condition {
    @Override
    public boolean matches(ConditionContext context, AnnotatedTypeMetadata metadata) {
        return context.getEnvironment().containsProperty("redis.enabled");
    }
}

// 使用条件
@Configuration
public class CacheConfig {
    
    @Bean
    @Conditional(RedisEnabledCondition.class)
    public CacheManager redisCacheManager() {
        return new RedisCacheManager();
    }
    
    @Bean
    @ConditionalOnMissingBean
    public CacheManager defaultCacheManager() {
        return new SimpleCacheManager();
    }
}
```

### Q19: 如何避免NullPointerException？

**A:** 使用Optional或进行null检查。

```java
// 坏的做法
@Service
public class UserService {
    
    @Autowired
    private UserRepository repository;  // 可能为null
    
    public User getUser(Long id) {
        return repository.findById(id).orElse(null);
    }
}

// 好的做法
@Service
public class UserService {
    
    private final UserRepository repository;
    
    public UserService(UserRepository repository) {
        this.repository = Objects.requireNonNull(repository, "repository cannot be null");
    }
    
    public User getUser(Long id) {
        return repository.findById(id).orElse(null);
    }
}
```

### Q20: Spring和Spring Boot的区别？

**A:** Spring Boot是Spring的快速开发框架。

| 特性 | Spring | Spring Boot |
|------|--------|-----------|
| 配置 | XML或JavaConfig | 自动配置 |
| 依赖 | 需要手动添加 | starter自动管理 |
| 服务器 | 需要外部服务器 | 内嵌服务器 |
| 部署 | WAR文件 | JAR文件 |
| 开发效率 | 较低 | 高 |

## 总结

常见问题通常与以下方面有关：
- **依赖注入** - 选择正确的注入方式
- **Bean生命周期** - 理解初始化和销毁顺序
- **事务管理** - 注意@Transactional的限制
- **AOP** - 理解代理的工作原理
- **Web开发** - 正确处理请求和响应
- **配置** - 合理管理配置文件

---

**建议**：
1. 熟悉Spring的核心概念
2. 阅读相关文档和源码
3. 实践常见问题的解决方案
4. 遵循最佳实践

**相关文档**：
- [核心概念](./core-concepts.md)
- [依赖注入](./dependency-injection.md)
- [最佳实践](./best-practices.md)
