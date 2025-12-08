---
id: quick-reference
title: 快速参考
sidebar_label: 快速参考
sidebar_position: 99
---

# Spring Framework 快速参考

## 常用注解

### 组件注解

| 注解 | 说明 | 用途 |
|------|------|------|
| `@Component` | 通用组件 | 标记任何组件 |
| `@Service` | 业务逻辑层 | 标记服务类 |
| `@Repository` | 数据访问层 | 标记DAO类 |
| `@Controller` | 控制层 | 标记控制器 |
| `@Configuration` | 配置类 | 定义配置和Bean |

### 依赖注入注解

| 注解 | 说明 |
|------|------|
| `@Autowired` | 自动装配依赖 |
| `@Qualifier` | 指定要装配的Bean名称 |
| `@Resource` | JSR-250注解，按名称注入 |
| `@Inject` | JSR-330注解 |
| `@Value` | 注入配置值 |

### Bean相关注解

| 注解 | 说明 |
|------|------|
| `@Bean` | 声明Bean |
| `@Scope` | 指定Bean作用域 |
| `@Primary` | 标记首选Bean |
| `@Lazy` | 延迟加载 |
| `@Conditional` | 条件化Bean |

### 生命周期注解

| 注解 | 说明 |
|------|------|
| `@PostConstruct` | Bean初始化后调用 |
| `@PreDestroy` | Bean销毁前调用 |

### 配置相关注解

| 注解 | 说明 |
|------|------|
| `@ComponentScan` | 扫描组件 |
| `@Import` | 导入其他配置类 |
| `@PropertySource` | 加载属性文件 |
| `@ConfigurationProperties` | 绑定配置属性 |

## Bean定义速查表

### 基本Bean定义

```java
// 1. 使用@Component
@Component
public class MyService {
}

// 2. 使用@Bean
@Configuration
public class AppConfig {
    @Bean
    public MyService myService() {
        return new MyService();
    }
}

// 3. XML配置
<bean id="myService" class="com.example.MyService" />
```

### 依赖注入

```java
// 构造函数注入
public MyService(Dependency dependency) {
    this.dependency = dependency;
}

// Setter注入
@Autowired
public void setDependency(Dependency dependency) {
    this.dependency = dependency;
}

// 字段注入
@Autowired
private Dependency dependency;
```

## 作用域快速参考

| 作用域 | 说明 | 用法 |
|-------|------|------|
| singleton | 单例（默认） | `@Scope("singleton")` |
| prototype | 原型 | `@Scope("prototype")` |
| request | 请求级别 | `@Scope("request")` |
| session | 会话级别 | `@Scope("session")` |

## 常用配置

### 扫描包

```java
@Configuration
@ComponentScan("com.example")
public class AppConfig {
}
```

### 导入其他配置

```java
@Configuration
@Import({DataSourceConfig.class, CacheConfig.class})
public class AppConfig {
}
```

### 加载属性文件

```java
@Configuration
@PropertySource("classpath:application.properties")
public class AppConfig {
    @Value("${app.name}")
    private String appName;
}
```

## AOP注解

| 注解 | 说明 |
|------|------|
| `@Aspect` | 标记切面类 |
| `@Pointcut` | 定义切点 |
| `@Before` | 前置通知 |
| `@After` | 后置通知 |
| `@AfterReturning` | 返回通知 |
| `@AfterThrowing` | 异常通知 |
| `@Around` | 环绕通知 |

### AOP示例

```java
@Aspect
@Component
public class LoggingAspect {
    
    @Pointcut("execution(* com.example.service.*.*(..))")
    public void serviceLayer() {}
    
    @Before("serviceLayer()")
    public void before(JoinPoint joinPoint) {
        System.out.println("Before: " + joinPoint.getSignature());
    }
    
    @After("serviceLayer()")
    public void after() {
        System.out.println("After");
    }
}
```

## 事务相关

### 事务注解

```java
@Transactional  // 声明式事务
public void save(User user) {
    userRepository.save(user);
}

@Transactional(
    readOnly = false,  // 非只读
    propagation = Propagation.REQUIRED,  // 传播行为
    isolation = Isolation.READ_COMMITTED,  // 隔离级别
    timeout = 30,  // 超时时间
    rollbackFor = Exception.class  // 回滚异常
)
public void update(User user) {
    userRepository.update(user);
}
```

## 常见代码片段

### ApplicationContext获取Bean

```java
// Spring Boot自动注入
@Autowired
private ApplicationContext applicationContext;

// 或在代码中获取
ApplicationContext context = new ClassPathXmlApplicationContext("applicationContext.xml");
UserService service = context.getBean(UserService.class);
```

### 获取配置值

```java
// 方式1：@Value
@Value("${server.port:8080}")
private int port;

// 方式2：Environment
@Autowired
private Environment environment;

public void getConfig() {
    String port = environment.getProperty("server.port", "8080");
}

// 方式3：@ConfigurationProperties
@ConfigurationProperties(prefix = "app")
public class AppProperties {
    private String name;
    private String version;
}
```

### 事件发布和监听

```java
// 发布事件
@Component
public class EventPublisher {
    @Autowired
    private ApplicationEventPublisher publisher;
    
    public void publishEvent() {
        publisher.publishEvent(new MyEvent(this, "data"));
    }
}

// 监听事件
@Component
public class EventListener {
    @EventListener
    public void onMyEvent(MyEvent event) {
        System.out.println("Event received: " + event.getMessage());
    }
}
```

### 条件判断

```java
// 根据条件创建Bean
@Configuration
public class DatabaseConfig {
    
    @Bean
    @ConditionalOnProperty(name = "db.type", havingValue = "mysql")
    public DataSource mysqlDataSource() {
        return new MysqlDataSource();
    }
    
    @Bean
    @ConditionalOnProperty(name = "db.type", havingValue = "postgresql")
    public DataSource postgresqlDataSource() {
        return new PostgresqlDataSource();
    }
}
```

## 常见错误处理

### 循环依赖

```java
// ❌ 错误：循环依赖
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

// ✅ 解决：重构设计
@Component
public class CommonService {
    // 公共逻辑
}

@Component
public class ServiceA {
    @Autowired
    private CommonService commonService;
}

@Component
public class ServiceB {
    @Autowired
    private CommonService commonService;
}
```

### Bean不存在

```java
// ✅ 使用Optional
@Autowired
private Optional<UserService> userService;

public void process() {
    userService.ifPresent(service -> service.execute());
}

// ✅ 使用ObjectProvider
@Autowired
private ObjectProvider<UserService> userServiceProvider;

public void process() {
    UserService service = userServiceProvider.getIfAvailable();
}
```

### 多个实现选择

```java
// ❌ 错误：不知道注入哪个
@Autowired
private UserRepository userRepository;  // 有多个实现

// ✅ 方案1：使用@Qualifier
@Autowired
@Qualifier("mysqlRepository")
private UserRepository userRepository;

// ✅ 方案2：使用@Primary
@Component
@Primary
public class MysqlUserRepository implements UserRepository {
}

// ✅ 方案3：注入所有
@Autowired
private List<UserRepository> repositories;
```

## 命令行参数

### 运行Spring Boot应用

```bash
# 指定端口
java -jar app.jar --server.port=9090

# 指定活跃配置
java -jar app.jar --spring.profiles.active=dev

# 指定属性
java -jar app.jar --app.name=MyApp --db.host=localhost
```

## 调试技巧

### 打印所有Bean

```java
@Component
public class BeanInspector implements ApplicationContextAware {
    
    @Override
    public void setApplicationContext(ApplicationContext applicationContext) {
        String[] names = applicationContext.getBeanDefinitionNames();
        Arrays.stream(names)
            .sorted()
            .forEach(System.out::println);
    }
}
```

### 启用Debug日志

在`application.properties`：
```properties
logging.level.org.springframework=DEBUG
logging.level.org.springframework.web=DEBUG
```

### 查看Bean定义信息

```java
@Autowired
private ApplicationContext context;

public void inspectBean() {
    BeanDefinition definition = context.getBeanDefinition("userService");
    System.out.println("Scope: " + definition.getScope());
    System.out.println("Lazy: " + definition.isLazyInit());
    System.out.println("Primary: " + definition.isPrimary());
}
```

---

**更新时间**: 2025年12月  
**Spring版本**: 6.x
