---
id: dependency-injection
title: 依赖注入详解
sidebar_label: 依赖注入
sidebar_position: 3
---

# 依赖注入

> [!TIP]
> **DI 最佳实践**: 优先使用构造器注入,避免字段注入。使用 @Autowired 时注意循环依赖问题。详解

## 1. 依赖注入的三种方式

### 1.1 构造函数注入（推荐）

通过构造函数的参数注入依赖。

```java
@Component
public class UserService {
    private final UserRepository userRepository;
    private final EmailService emailService;
    
    // 推荐：使用构造函数注入
    public UserService(UserRepository userRepository, EmailService emailService) {
        this.userRepository = userRepository;
        this.emailService = emailService;
    }
    
    public void registerUser(User user) {
        userRepository.save(user);
        emailService.sendWelcomeEmail(user);
    }
}
```

**优点**：

- ✅ 依赖关系明确，在构造时就知道需要什么
- ✅ 对象创建后立即可用，不存在未初始化的状态
- ✅ 易于测试，可以轻松传入Mock对象
- ✅ 不可变性：使用final关键字，线程安全

**缺点**：

- ❌ 如果依赖很多，构造函数参数会很长
- ❌ 不能处理循环依赖

**使用场景**：

- 必需的依赖
- 不可变的对象
- 易于测试的代码

### 1.2 Setter注入

通过Setter方法注入依赖。

```java
@Component
public class UserService {
    private UserRepository userRepository;
    private EmailService emailService;
    
    @Autowired
    public void setUserRepository(UserRepository userRepository) {
        this.userRepository = userRepository;
    }
    
    @Autowired
    public void setEmailService(EmailService emailService) {
        this.emailService = emailService;
    }
}
```

**优点**：

- ✅ 依赖是可选的
- ✅ 可以在对象创建后更改依赖
- ✅ 减少构造函数参数
- ✅ 可以处理循环依赖

**缺点**：

- ❌ 对象创建后不一定完全初始化
- ❌ 依赖关系不明确
- ❌ 不易于测试

**使用场景**：

- 可选的依赖
- 需要处理循环依赖
- 旧代码改造

### 1.3 字段注入

直接在字段上使用@Autowired注解。

```java
@Component
public class UserService {
    @Autowired
    private UserRepository userRepository;
    
    @Autowired
    private EmailService emailService;
    
    public void registerUser(User user) {
        userRepository.save(user);
        emailService.sendWelcomeEmail(user);
    }
}
```

**优点**：

- ✅ 代码简洁
- ✅ 减少样板代码

**缺点**：

- ❌ 无法用于构造函数参数（真正的不可变）
- ❌ 不利于单元测试，难以注入Mock
- ❌ 隐藏了真实的依赖
- ❌ 字段为null时容易产生NullPointerException

**使用场景**：

- 快速原型开发
- 不需要测试的代码

## 2. 推荐最佳实践

### 2.1 依赖注入优先级

```
1. 构造函数注入 (首选)
   ↓
2. Setter注入 (次选)
   ↓
3. 字段注入 (避免)
```

### 2.2 实战示例 - 完整的最佳实践

```java
@Component
public class UserService {
    // 使用final，保证不可变性
    private final UserRepository userRepository;
    private final EmailService emailService;
    private final UserValidator userValidator;
    
    // 使用构造函数注入，使用@Autowired是可选的（Spring 4.3+）
    public UserService(
        UserRepository userRepository,
        EmailService emailService,
        UserValidator userValidator) {
        this.userRepository = userRepository;
        this.emailService = emailService;
        this.userValidator = userValidator;
    }
    
    public void registerUser(User user) {
        // 验证
        if (!userValidator.isValid(user)) {
            throw new IllegalArgumentException("Invalid user");
        }
        
        // 保存
        userRepository.save(user);
        
        // 发送邮件
        emailService.sendWelcomeEmail(user);
    }
}
```

### 2.3 处理可选依赖

当某个依赖是可选时：

```java
@Component
public class NotificationService {
    private final EmailService emailService;
    private final Optional<SmsService> smsService;
    
    public NotificationService(
        EmailService emailService,
        Optional<SmsService> smsService) {
        this.emailService = emailService;
        this.smsService = smsService;
    }
    
    public void notify(String message) {
        emailService.send(message);
        
        // 如果SMS服务可用，则发送短信
        smsService.ifPresent(service -> service.send(message));
    }
}
```

## 3. 多个实现的处理

当接口有多个实现类时，处理方式：

### 3.1 使用@Qualifier

```java
// 实现1
@Component
@Qualifier("mysql")
public class MysqlUserRepository implements UserRepository {
}

// 实现2
@Component
@Qualifier("mongodb")
public class MongodbUserRepository implements UserRepository {
}

// 注入时指定
@Component
public class UserService {
    private final UserRepository userRepository;
    
    public UserService(@Qualifier("mysql") UserRepository userRepository) {
        this.userRepository = userRepository;
    }
}
```

### 3.2 使用@Primary

```java
// 标记为主要的实现
@Component
@Primary
public class MysqlUserRepository implements UserRepository {
}

// 其他实现
@Component
public class MongodbUserRepository implements UserRepository {
}

// 注入时，如果不指定则使用@Primary标记的
@Component
public class UserService {
    private final UserRepository userRepository;
    
    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;  // 默认使用MysqlUserRepository
    }
}
```

### 3.3 注入所有实现

```java
@Component
public class UserService {
    private final List<UserRepository> repositories;
    
    public UserService(List<UserRepository> repositories) {
        this.repositories = repositories;  // 获取所有实现
    }
    
    public void syncToAllRepositories(User user) {
        repositories.forEach(repo -> repo.save(user));
    }
}
```

### 3.4 使用Map注入

```java
@Component
public class UserService {
    private final Map<String, UserRepository> repositories;
    
    public UserService(Map<String, UserRepository> repositories) {
        this.repositories = repositories;
    }
    
    public void saveUser(String type, User user) {
        UserRepository repo = repositories.get(type);
        repo.save(user);
    }
}
```

## 4. 循环依赖

### 4.1 问题示例

```java
@Component
public class ServiceA {
    private final ServiceB serviceB;
    
    public ServiceA(ServiceB serviceB) {
        this.serviceB = serviceB;
    }
}

@Component
public class ServiceB {
    private final ServiceA serviceA;  // 形成循环依赖
    
    public ServiceB(ServiceA serviceA) {
        this.serviceA = serviceA;
    }
}
```

### 4.2 解决方案

**方案1：改变设计**

```java
// 将公共逻辑提取到第三个类
@Component
public class CommonService {
    public void commonMethod() {
        // 公共方法
    }
}

@Component
public class ServiceA {
    private final CommonService commonService;
    
    public ServiceA(CommonService commonService) {
        this.commonService = commonService;
    }
}

@Component
public class ServiceB {
    private final CommonService commonService;
    
    public ServiceB(CommonService commonService) {
        this.commonService = commonService;
    }
}
```

**方案2：使用Setter注入（临时方案）**

```java
@Component
public class ServiceA {
    private ServiceB serviceB;
    
    @Autowired
    public void setServiceB(ServiceB serviceB) {
        this.serviceB = serviceB;
    }
}

@Component
public class ServiceB {
    private ServiceA serviceA;
    
    @Autowired
    public void setServiceA(ServiceA serviceA) {
        this.serviceA = serviceA;
    }
}
```

**方案3：延迟初始化**

```java
@Component
public class ServiceA {
    private final ServiceB serviceB;
    
    public ServiceA(ObjectProvider<ServiceB> serviceBProvider) {
        this.serviceB = serviceBProvider.getIfAvailable();
    }
}
```

## 5. 泛型依赖注入

```java
// 定义泛型类
public abstract class GenericRepository<T> {
    abstract void save(T entity);
}

@Component
public class UserRepository extends GenericRepository<User> {
    @Override
    void save(User entity) {
        System.out.println("Saving user: " + entity);
    }
}

// 注入泛型依赖
@Component
public class UserService {
    private final GenericRepository<User> userRepository;
    
    public UserService(GenericRepository<User> userRepository) {
        this.userRepository = userRepository;
    }
}
```

## 6. ObjectProvider的高级用法

```java
@Component
public class AdvancedService {
    private final ObjectProvider<DatabaseService> databaseService;
    
    public AdvancedService(ObjectProvider<DatabaseService> databaseService) {
        this.databaseService = databaseService;
    }
    
    public void process() {
        // 如果Bean存在，则使用；否则使用默认值
        databaseService.ifAvailable(service -> service.execute());
        
        // 如果有多个实现，获取第一个
        DatabaseService service = databaseService.getIfAvailable();
        
        // 获取所有实现
        databaseService.stream().forEach(s -> s.execute());
    }
}
```

## 7. 注入配置值

### 7.1 @Value注入

```java
@Component
public class DatabaseConfig {
    @Value("${db.host:localhost}")
    private String host;
    
    @Value("${db.port:5432}")
    private int port;
    
    @Value("${db.username}")
    private String username;
    
    // 注入系统属性
    @Value("${java.version}")
    private String javaVersion;
    
    // 注入表达式
    @Value("#{systemProperties['user.timezone']}")
    private String timezone;
}
```

### 7.2 @ConfigurationProperties

```java
@Component
@ConfigurationProperties(prefix = "database")
public class DatabaseProperties {
    private String host;
    private int port;
    private String username;
    private String password;
    
    // getters and setters
}
```

## 8. 总结

| 方式 | 推荐 | 说明 |
|------|------|------|
| 构造函数注入 | ⭐⭐⭐⭐⭐ | 最佳实践，优先使用 |
| Setter注入 | ⭐⭐⭐ | 用于可选依赖或循环依赖 |
| 字段注入 | ⭐ | 尽量避免使用 |

---

**关键原则**：

1. 优先使用构造函数注入
2. 依赖应该使用final修饰符
3. 避免循环依赖，通过重新设计解决
4. 充分利用Optional和ObjectProvider处理可选依赖

**下一步**：学习[Bean管理](/docs/spring/bean-management)
