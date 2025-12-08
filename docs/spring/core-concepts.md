---
id: core-concepts
title: Spring核心概念
sidebar_label: 核心概念
sidebar_position: 2
---

# Spring核心概念

## 1. 概述

Spring框架的核心概念围绕两个主要的设计原则：
- **IoC（Inversion of Control）** - 控制反转
- **DI（Dependency Injection）** - 依赖注入

这些概念构成了Spring应用的基础。

## 2. IoC - 控制反转

### 什么是IoC？

**控制反转**是一种设计原则，它反转了程序中对象创建和生命周期管理的控制流。

传统方式（没有IoC）：
```java
public class UserService {
    // 手动创建依赖对象
    private UserRepository userRepository = new UserRepository();
    
    public void saveUser(User user) {
        userRepository.save(user);
    }
}
```

使用IoC方式：
```java
public class UserService {
    private UserRepository userRepository;
    
    // 由容器注入依赖
    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }
    
    public void saveUser(User user) {
        userRepository.save(user);
    }
}
```

### IoC的优势

| 优势 | 说明 |
|------|------|
| 降低耦合度 | 对象之间不直接依赖，通过容器管理 |
| 提高可测试性 | 易于注入Mock对象进行单元测试 |
| 提高可维护性 | 对象依赖关系清晰，便于修改 |
| 代码重用性好 | 对象可独立使用，不依赖具体实现 |

## 3. DI - 依赖注入

### 什么是DI？

**依赖注入**是实现IoC的一种方式，它通过将对象的依赖注入到对象中，而不是在对象内部创建依赖。

### DI的三种方式

#### 3.1 构造函数注入

```java
@Component
public class UserService {
    private final UserRepository userRepository;
    private final UserValidator userValidator;
    
    // 通过构造函数注入
    public UserService(UserRepository userRepository, UserValidator userValidator) {
        this.userRepository = userRepository;
        this.userValidator = userValidator;
    }
}
```

**优势**：
- 依赖清晰，必须依赖在构造时提供
- 对象创建时就是完全初始化状态
- 易于测试

#### 3.2 Setter注入

```java
@Component
public class UserService {
    private UserRepository userRepository;
    
    @Autowired
    public void setUserRepository(UserRepository userRepository) {
        this.userRepository = userRepository;
    }
}
```

**优势**：
- 依赖可选
- 可在对象创建后更改依赖

**劣势**：
- 对象创建时不一定完全初始化
- 依赖关系不明确

#### 3.3 字段注入

```java
@Component
public class UserService {
    @Autowired
    private UserRepository userRepository;
}
```

**优势**：
- 代码简洁

**劣势**：
- 无法用于构造函数参数
- 不利于单元测试
- 隐藏了依赖关系

### 推荐做法

**优先使用构造函数注入**，它最明确、最安全。

## 4. Spring容器

### 容器的作用

Spring容器是IoC的具体实现，主要职责：

1. **对象创建** - 根据配置创建Bean
2. **依赖管理** - 管理对象之间的依赖关系
3. **生命周期管理** - 管理Bean的初始化和销毁
4. **配置管理** - 加载和管理应用配置

### 两种容器类型

#### 4.1 BeanFactory

基础的容器接口，功能简洁。

```java
Resource resource = new ClassPathResource("applicationContext.xml");
BeanFactory beanFactory = new XmlBeanFactory(resource);
UserService userService = beanFactory.getBean(UserService.class);
```

#### 4.2 ApplicationContext

扩展的容器接口，功能更强大。

```java
// 基于XML配置
ApplicationContext context = new ClassPathXmlApplicationContext("applicationContext.xml");

// 基于注解配置
ApplicationContext context = new AnnotationConfigApplicationContext(AppConfig.class);

UserService userService = context.getBean(UserService.class);
```

### ApplicationContext的优势

- 国际化支持
- 事件发布机制
- 资源加载
- 环境抽象

## 5. Bean的概念

### 什么是Bean？

**Bean**是由Spring容器创建、组装和管理的对象。

### Bean的配置方式

#### 5.1 XML配置

```xml
<bean id="userRepository" class="com.example.repository.UserRepository" />

<bean id="userService" class="com.example.service.UserService">
    <constructor-arg ref="userRepository" />
</bean>
```

#### 5.2 注解配置

```java
@Configuration
public class AppConfig {
    @Bean
    public UserRepository userRepository() {
        return new UserRepository();
    }
    
    @Bean
    public UserService userService(UserRepository userRepository) {
        return new UserService(userRepository);
    }
}
```

#### 5.3 组件扫描

```java
@Configuration
@ComponentScan("com.example")
public class AppConfig {
}

@Component
public class UserService {
    @Autowired
    private UserRepository userRepository;
}
```

## 6. Bean的生命周期

Spring Bean的完整生命周期包括以下阶段：

```
1. 实例化 (Instantiation)
   ↓
2. 属性赋值 (Property Assignment)
   ↓
3. 初始化前处理 (BeanPostProcessor.postProcessBeforeInitialization)
   ↓
4. 初始化 (Initialization)
   - 调用 InitializingBean.afterPropertiesSet()
   - 或调用自定义的 init-method
   ↓
5. 初始化后处理 (BeanPostProcessor.postProcessAfterInitialization)
   ↓
6. 使用 (In Use)
   ↓
7. 销毁前处理 (DisposableBean.destroy)
   - 或调用自定义的 destroy-method
```

### 生命周期回调示例

```java
@Component
public class UserService implements InitializingBean, DisposableBean {
    
    // 初始化后调用
    @PostConstruct
    public void init() {
        System.out.println("初始化");
    }
    
    // 销毁前调用
    @PreDestroy
    public void cleanup() {
        System.out.println("清理资源");
    }
    
    @Override
    public void afterPropertiesSet() throws Exception {
        System.out.println("属性设置完成");
    }
    
    @Override
    public void destroy() throws Exception {
        System.out.println("对象销毁");
    }
}
```

## 7. Bean的作用域

Spring提供了多种Bean作用域：

| 作用域 | 说明 | 使用场景 |
|-------|------|--------|
| singleton | 单例，容器中只有一个Bean实例 | 无状态的服务类 |
| prototype | 原型，每次获取都创建新实例 | 有状态的对象 |
| request | 请求作用域，每个HTTP请求一个实例 | Web应用中的请求对象 |
| session | 会话作用域，每个HTTP会话一个实例 | 用户会话数据 |
| application | 应用作用域，整个应用生命周期一个实例 | 全局配置 |
| websocket | WebSocket作用域，每个WebSocket连接一个实例 | WebSocket应用 |

### 作用域声明

```java
@Component
@Scope("prototype")
public class UserRequest {
    // 每次都创建新实例
}

@Component
@Scope(value = "request", proxyMode = ScopedProxyMode.TARGET_CLASS)
public class RequestData {
    // 请求作用域
}
```

## 8. 自动装配

### @Autowired

自动注入依赖对象：

```java
@Component
public class UserService {
    @Autowired
    private UserRepository userRepository;
    
    @Autowired
    public void setValidator(UserValidator validator) {
        this.validator = validator;
    }
}
```

### @Qualifier

当有多个Bean实现同一接口时，使用@Qualifier指定：

```java
@Component
@Qualifier("mysqlRepository")
public class MysqlUserRepository implements UserRepository {
}

@Component
public class UserService {
    @Autowired
    @Qualifier("mysqlRepository")
    private UserRepository userRepository;
}
```

### @Resource

JSR-250标准注解，功能类似@Autowired：

```java
@Component
public class UserService {
    @Resource(name = "userRepository")
    private UserRepository userRepository;
}
```

## 9. 总结

| 概念 | 说明 |
|------|------|
| IoC | 将对象创建和生命周期管理权交给容器 |
| DI | IoC的具体实现，通过注入依赖 |
| Spring容器 | 实现IoC的核心组件 |
| Bean | 容器管理的对象 |
| 生命周期 | Bean从创建到销毁的完整过程 |
| 作用域 | Bean的存活范围 |
| 自动装配 | 自动注入依赖 |

---

**关键要点**：
- 使用构造函数注入是最佳实践
- 理解Bean的生命周期很重要
- 合理选择Bean的作用域
- 避免循环依赖

**下一步**：学习[依赖注入详解](./dependency-injection.md)
