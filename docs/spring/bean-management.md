---
id: bean-management
title: Bean管理
sidebar_label: Bean管理
sidebar_position: 4
---

# Bean管理

> [!TIP]
> **Bean 生命周期**: 掌握 Bean 的生命周期和作用域对于正确使用 Spring 至关重要。默认为单例(singleton),注意线程安全问题。

### 1.1 使用@Bean注解

```java
@Configuration
public class AppConfig {
    // 基本Bean定义
    @Bean
    public UserRepository userRepository() {
        return new UserRepository();
    }
    
    // 依赖注入
    @Bean
    public UserService userService(UserRepository userRepository) {
        return new UserService(userRepository);
    }
    
    // 使用方法参数获取依赖
    @Bean
    public OrderService orderService(UserService userService, ProductService productService) {
        return new OrderService(userService, productService);
    }
}
```

### 1.2 使用@Component注解

```java
@Component
public class UserRepository {
    // 自动注册为Bean
}

@Service  // @Component的特化
public class UserService {
    @Autowired
    private UserRepository userRepository;
}

@Repository  // @Component的特化
public class UserRepository {
    // 数据访问层组件
}

@Controller  // @Component的特化
public class UserController {
    // 控制层组件
}
```

### 1.3 XML配置

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans
       http://www.springframework.org/schema/beans/spring-beans.xsd">
    
    <!-- 基本Bean定义 -->
    <bean id="userRepository" class="com.example.repository.UserRepository" />
    
    <!-- 使用constructor-arg注入 -->
    <bean id="userService" class="com.example.service.UserService">
        <constructor-arg ref="userRepository" />
    </bean>
    
    <!-- 使用property注入 -->
    <bean id="orderService" class="com.example.service.OrderService">
        <property name="userService" ref="userService" />
        <property name="productService" ref="productService" />
    </bean>
    
    <!-- 工厂方法创建Bean -->
    <bean id="userFactory" class="com.example.factory.UserFactory" />
    <bean id="user" factory-bean="userFactory" factory-method="createUser" />
</beans>
```

## 2. Bean的命名

### 2.1 指定Bean名称

```java
// 使用@Bean指定名称
@Configuration
public class AppConfig {
    @Bean(name = "userRepo")
    public UserRepository userRepository() {
        return new UserRepository();
    }
    
    // 多个别名
    @Bean(name = {"userRepository", "userRepo", "repo"})
    public UserRepository userRepository() {
        return new UserRepository();
    }
}

// 使用@Component指定名称
@Component("userService")
public class UserService {
}

// 使用@Component默认名称（类名首字母小写）
@Component
public class UserService {  // Bean名称为 userService
}
```

### 2.2 访问Bean

```java
@Autowired
ApplicationContext applicationContext;

public void accessBeans() {
    // 按类型获取
    UserService userService = applicationContext.getBean(UserService.class);
    
    // 按名称获取
    UserService userService = (UserService) applicationContext.getBean("userService");
    
    // 按名称和类型获取
    UserService userService = applicationContext.getBean("userService", UserService.class);
}
```

## 3. Bean的作用域

### 3.1 单例作用域（Singleton）

```java
@Component
@Scope("singleton")  // 或使用 @Scope(ConfigurableBeanFactory.SCOPE_SINGLETON)
public class UserService {
    // 容器中只有一个实例
}

// 验证单例
@Test
public void testSingleton() {
    UserService service1 = context.getBean(UserService.class);
    UserService service2 = context.getBean(UserService.class);
    
    assertSame(service1, service2);  // 同一个实例
}
```

### 3.2 原型作用域（Prototype）

```java
@Component
@Scope("prototype")  // 或使用 @Scope(ConfigurableBeanFactory.SCOPE_PROTOTYPE)
public class RequestData {
    // 每次获取都创建新实例
}

// 验证原型
@Test
public void testPrototype() {
    RequestData data1 = context.getBean(RequestData.class);
    RequestData data2 = context.getBean(RequestData.class);
    
    assertNotSame(data1, data2);  // 不同的实例
}
```

### 3.3 Web应用作用域

在Spring MVC/Web环境中可用。

```java
// 请求作用域 - 每个HTTP请求一个实例
@Component
@Scope(value = "request", proxyMode = ScopedProxyMode.TARGET_CLASS)
public class RequestContext {
    private String requestId;
    
    @PostConstruct
    public void init() {
        requestId = UUID.randomUUID().toString();
    }
}

// 会话作用域 - 每个HTTP会话一个实例
@Component
@Scope(value = "session", proxyMode = ScopedProxyMode.TARGET_CLASS)
public class UserSession {
    private User currentUser;
}

// 应用作用域 - 应用启动到关闭期间一个实例
@Component
@Scope("application")
public class ApplicationConfig {
    private String appName;
}
```

### 3.4 在单例Bean中注入原型Bean

问题：单例Bean中的原型Bean只被注入一次。

```java
@Component
public class SingletonService {
    @Autowired
    private PrototypeService prototypeService;  // 只注入一次
    
    public void process() {
        prototypeService.execute();  // 每次都使用同一个实例
    }
}
```

解决方案1：使用ObjectFactory

```java
@Component
public class SingletonService {
    @Autowired
    private ObjectFactory<PrototypeService> prototypeBeanFactory;
    
    public void process() {
        PrototypeService prototypeService = prototypeBeanFactory.getObject();
        prototypeService.execute();  // 每次都获取新实例
    }
}
```

解决方案2：使用ObjectProvider

```java
@Component
public class SingletonService {
    @Autowired
    private ObjectProvider<PrototypeService> prototypeBeanProvider;
    
    public void process() {
        PrototypeService prototypeService = prototypeBeanProvider.getObject();
        prototypeService.execute();  // 每次都获取新实例
    }
}
```

解决方案3：使用Lookup Method

```java
@Component
public abstract class SingletonService {
    @Lookup
    protected abstract PrototypeService getPrototypeService();
    
    public void process() {
        PrototypeService service = getPrototypeService();  // 每次都获取新实例
        service.execute();
    }
}
```

## 4. 懒加载

```java
// 容器启动时不创建Bean，而是在首次使用时创建
@Component
@Lazy
public class ExpensiveService {
    public ExpensiveService() {
        System.out.println("ExpensiveService created!");
    }
}

// @Bean也可以使用@Lazy
@Configuration
public class AppConfig {
    @Bean
    @Lazy
    public ExpensiveService expensiveService() {
        System.out.println("Creating ExpensiveService");
        return new ExpensiveService();
    }
}
```

## 5. 条件化Bean

### 5.1 @Conditional注解

```java
// 自定义条件
public class RedisCondition implements Condition {
    @Override
    public boolean matches(ConditionContext context, AnnotatedTypeMetadata metadata) {
        return context.getEnvironment().containsProperty("redis.host");
    }
}

// 使用条件
@Configuration
public class CacheConfig {
    @Bean
    @Conditional(RedisCondition.class)
    public CacheManager redisCacheManager() {
        return new RedisCacheManager();
    }
    
    @Bean
    @ConditionalOnMissingBean
    public CacheManager defaultCacheManager() {
        return new DefaultCacheManager();
    }
}
```

### 5.2 Spring Boot条件注解

```java
// 当类在classpath上时
@ConditionalOnClass(RedisTemplate.class)
public class RedisAutoConfiguration {
}

// 当属性存在时
@ConditionalOnProperty(name = "cache.enabled", havingValue = "true")
@Configuration
public class CacheConfiguration {
}

// 当Bean不存在时
@ConditionalOnMissingBean(CacheManager.class)
@Bean
public CacheManager defaultCacheManager() {
    return new DefaultCacheManager();
}

// 当容器中有指定Bean时
@ConditionalOnBean(DataSource.class)
@Configuration
public class JdbcConfiguration {
}
```

## 6. Bean的初始化和销毁

### 6.1 使用生命周期注解

```java
@Component
public class DataSource {
    private Connection connection;
    
    @PostConstruct
    public void init() {
        System.out.println("初始化连接");
        this.connection = createConnection();
    }
    
    @PreDestroy
    public void cleanup() {
        System.out.println("关闭连接");
        if (connection != null) {
            connection.close();
        }
    }
}
```

### 6.2 使用InitializingBean和DisposableBean

```java
@Component
public class DataSource implements InitializingBean, DisposableBean {
    private Connection connection;
    
    @Override
    public void afterPropertiesSet() throws Exception {
        System.out.println("属性设置完成，进行初始化");
        this.connection = createConnection();
    }
    
    @Override
    public void destroy() throws Exception {
        System.out.println("销毁时调用");
        if (connection != null) {
            connection.close();
        }
    }
}
```

### 6.3 使用initMethod和destroyMethod

```java
@Configuration
public class AppConfig {
    @Bean(initMethod = "init", destroyMethod = "cleanup")
    public DataSource dataSource() {
        return new DataSource();
    }
}

// 或在XML中
// <bean id="dataSource" class="com.example.DataSource"
//       init-method="init"
//       destroy-method="cleanup" />
```

### 6.4 完整示例

```java
@Component
public class DatabaseConnection implements InitializingBean, DisposableBean {
    private Connection connection;
    private String host;
    private int port;
    
    @Autowired
    private ApplicationContext applicationContext;
    
    @PostConstruct
    public void init() {
        System.out.println("1. @PostConstruct - 初始化开始");
    }
    
    @Override
    public void afterPropertiesSet() throws Exception {
        System.out.println("2. afterPropertiesSet - 属性设置完成");
        this.connection = DriverManager.getConnection(
            String.format("jdbc:mysql://%s:%d/test", host, port));
    }
    
    public void customInit() {
        System.out.println("3. customInit - 自定义初始化");
    }
    
    @PreDestroy
    public void preDestroy() {
        System.out.println("4. @PreDestroy - 销毁前");
    }
    
    @Override
    public void destroy() throws Exception {
        System.out.println("5. destroy - 销毁资源");
        if (connection != null) {
            connection.close();
        }
    }
}
```

生命周期顺序：

1. @PostConstruct
2. afterPropertiesSet()
3. 自定义init-method/initMethod
4. 使用中...
5. @PreDestroy
6. destroy()

## 7. Primary Bean和Qualifier

```java
// 定义多个实现
@Component
public class MysqlRepository implements UserRepository {
}

@Component
@Primary  // 标记为主要实现
public class MongoRepository implements UserRepository {
}

// 注入时
@Component
public class UserService {
    // 默认注入MongoRepository（因为是@Primary）
    @Autowired
    private UserRepository repository;
    
    // 指定注入MysqlRepository
    @Autowired
    @Qualifier("mysqlRepository")
    private UserRepository mysqlRepository;
}
```

## 8. 获取Bean信息

```java
@Autowired
private ApplicationContext applicationContext;

public void inspectBeans() {
    // 获取所有Bean名称
    String[] beanNames = applicationContext.getBeanDefinitionNames();
    Arrays.stream(beanNames).forEach(System.out::println);
    
    // 检查Bean是否存在
    boolean exists = applicationContext.containsBean("userService");
    
    // 获取Bean定义
    BeanDefinition definition = applicationContext.getBeanDefinition("userService");
    System.out.println("Scope: " + definition.getScope());
    
    // 获取所有特定类型的Bean
    Map<String, UserRepository> repos = applicationContext.getBeansOfType(UserRepository.class);
    repos.forEach((name, repo) -> System.out.println(name + ": " + repo));
    
    // 获取带指定注解的Bean
    Map<String, Object> beans = applicationContext.getBeansWithAnnotation(Component.class);
}
```

## 9. 总结

| 功能 | 说明 | 推荐用法 |
|------|------|--------|
| @Bean | 在配置类中定义Bean | 第三方类 |
| @Component | 标记组件 | 自己的类 |
| 作用域 | Bean的生命周期范围 | 根据需要选择 |
| 初始化 | Bean创建后的初始化 | @PostConstruct |
| 销毁 | Bean销毁前的清理 | @PreDestroy |
| Lazy | 延迟加载 | 资源密集型Bean |
| Conditional | 条件化Bean创建 | 环境相关 |

---

**关键要点**：

- 优先使用@Component系列注解
- 合理使用@Bean处理第三方对象
- 选择合适的作用域
- 正确处理资源的初始化和销毁
- 使用ObjectFactory/ObjectProvider在单例中获取原型Bean

**下一步**：学习[面向切面编程(AOP)](./aop)
