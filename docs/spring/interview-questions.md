---
sidebar_position: 100
title: Spring 面试题精选
---

# Spring 面试题精选

> [!TIP]
> 本文精选了 Spring Framework 常见面试题，涵盖 IoC、DI、AOP、事务管理等核心知识点。

## 🎯 核心概念

### 1. 什么是 Spring IoC？有什么优势？

**答案要点：**

**IoC（Inversion of Control，控制反转）：**

- 将对象的创建和依赖关系管理交给 Spring 容器
- 开发者不再手动 new 对象，而是从容器中获取

**优势：**

- 降低耦合度
- 提高代码可测试性
- 便于维护和扩展
- 支持 AOP 等高级特性

**示例对比：**

```java
// 传统方式 - 高耦合
public class UserService {
    private UserDao userDao = new UserDaoImpl();  // 紧耦合
}

// Spring IoC - 低耦合
@Service
public class UserService {
    @Autowired
    private UserDao userDao;  // 由容器注入
}
```

**延伸：** 参考 [核心概念](./core-concepts)

---

### 2. Spring 中的依赖注入有几种方式？推荐哪种？

**答案要点：**

**三种方式：**

**1. 构造器注入（推荐）**

```java
@Service
public class UserService {
    private final UserDao userDao;

    @Autowired  // Spring 4.3+ 单构造器可省略
    public UserService(UserDao userDao) {
        this.userDao = userDao;
    }
}
```

**优点：** 依赖不可变、强制依赖、便于测试

**2. Setter 注入**

```java
@Service
public class UserService {
    private UserDao userDao;

    @Autowired
    public void setUserDao(UserDao userDao) {
        this.userDao = userDao;
    }
}
```

**优点：** 可选依赖、灵活

**3. 字段注入（不推荐）**

```java
@Service
public class UserService {
    @Autowired
    private UserDao userDao;
}
```

**缺点：** 难以测试、隐藏依赖、不能用于 final 字段

**推荐：** 必需依赖用构造器注入，可选依赖用 Setter 注入

**延伸：** 参考 [依赖注入详解](./dependency-injection)

---

### 3. @Autowired 和 @Resource 的区别？

**答案要点：**

| 特性         | @Autowired           | @Resource          |
| ------------ | -------------------- | ------------------ |
| 来源         | Spring               | Java EE（JSR-250） |
| 默认装配方式 | byType（按类型）     | byName（按名称）   |
| 指定名称     | 配合@Qualifier       | name 属性          |
| 支持位置     | 字段、setter、构造器 | 字段、setter       |

**示例：**

```java
// @Autowired - 按类型匹配
@Autowired
private UserDao userDao;

// 多个同类型Bean时，配合@Qualifier
@Autowired
@Qualifier("userDaoImpl")
private UserDao userDao;

// @Resource - 按名称匹配
@Resource(name = "userDaoImpl")
private UserDao userDao;
```

**延伸：** 参考 [依赖注入](./dependency-injection)

---

### 4. Bean 的作用域有哪些？

**答案要点：**

**六种作用域：**

| 作用域      | 说明                         | 使用场景       |
| ----------- | ---------------------------- | -------------- |
| singleton   | 单例（默认）                 | 无状态 Bean    |
| prototype   | 每次请求创建新实例           | 有状态 Bean    |
| request     | 每个 HTTP 请求一个实例       | Web 应用       |
| session     | 每个 HTTP 会话一个实例       | 用户会话数据   |
| application | 整个 ServletContext 一个实例 | 全局应用数据   |
| websocket   | 每个 WebSocket 会话一个实例  | WebSocket 应用 |

**配置示例：**

```java
// 单例（默认）
@Service
@Scope("singleton")  // 可省略
public class UserService { }

// 原型
@Service
@Scope("prototype")
public class OrderService { }

// Web相关作用域
@Controller
@Scope("request")
public class LoginController { }
```

**延伸：** 参考 [Bean 管理](./bean-management)

---

### 5. Bean 的生命周期？

**答案要点：**

**完整生命周期流程：**

1. **实例化：** 调用构造方法创建 Bean 实例
2. **属性赋值：** 注入依赖（@Autowired 等）
3. **初始化前：** 调用 BeanPostProcessor 的 postProcessBeforeInitialization
4. **初始化：**
   - 调用@PostConstruct 方法
   - 调用 InitializingBean 的 afterPropertiesSet 方法
   - 调用自定义 init-method
5. **初始化后：** 调用 BeanPostProcessor 的 postProcessAfterInitialization
6. **使用：** Bean 可以被使用
7. **销毁：**
   - 调用@PreDestroy 方法
   - 调用 DisposableBean 的 destroy 方法
   - 调用自定义 destroy-method

**代码示例：**

```java
@Component
public class MyBean implements InitializingBean, DisposableBean {

    public MyBean() {
        System.out.println("1. 构造方法");
    }

    @Autowired
    public void setDependency(Dependency dep) {
        System.out.println("2. 属性注入");
    }

    @PostConstruct
    public void postConstruct() {
        System.out.println("3. @PostConstruct");
    }

    @Override
    public void afterPropertiesSet() {
        System.out.println("4. afterPropertiesSet");
    }

    @PreDestroy
    public void preDestroy() {
        System.out.println("5. @PreDestroy");
    }

    @Override
    public void destroy() {
        System.out.println("6. destroy");
    }
}
```

**延伸：** 参考 [Bean 管理](./bean-management)

---

## 🎯 AOP 面向切面

### 6. 什么是 AOP？有哪些应用场景？

**答案要点：**

**AOP（Aspect-Oriented Programming）：** 面向切面编程，将横切关注点从业务逻辑中分离

**核心概念：**

- **切面（Aspect）：** 横切关注点的模块化
- **连接点（Join Point）：** 程序执行的某个点（方法调用）
- **切点（Pointcut）：** 匹配连接点的表达式
- **通知（Advice）：** 在切点执行的动作

**典型应用场景：**

- 日志记录
- 事务管理
- 权限控制
- 性能监控
- 异常处理

**示例：**

```java
@Aspect
@Component
public class LoggingAspect {

    @Before("execution(* com.example.service.*.*(..))")
    public void logBefore(JoinPoint joinPoint) {
        System.out.println("执行方法: " + joinPoint.getSignature().getName());
    }

    @AfterReturning(pointcut = "execution(* com.example.service.*.*(..))",
                    returning = "result")
    public void logAfterReturning(JoinPoint joinPoint, Object result) {
        System.out.println("方法返回值: " + result);
    }
}
```

**延伸：** 参考 [AOP 详解](./aop)

---

### 7. Spring AOP 和 AspectJ AOP 的区别？

**答案要点：**

| 特性     | Spring AOP            | AspectJ AOP          |
| -------- | --------------------- | -------------------- |
| 实现方式 | 动态代理（JDK/CGLIB） | 编译时/加载时织入    |
| 功能     | 仅支持方法级别        | 支持字段、构造器等   |
| 性能     | 运行时代理，略慢      | 编译时织入，更快     |
| 易用性   | 简单，Spring 原生支持 | 复杂，需要特殊编译器 |
| 适用场景 | 一般企业应用          | 复杂 AOP 需求        |

**Spring AOP 代理方式：**

- 接口 → JDK 动态代理
- 类 → CGLIB 代理

```java
// 强制使用CGLIB
@EnableAspectJAutoProxy(proxyTargetClass = true)
```

**延伸：** 参考 [AOP 详解](./aop)

---

### 8. @Before、@After、@Around 的执行顺序？

**答案要点：**

**正常执行顺序：**

1. @Around（前半部分）
2. @Before
3. 目标方法执行
4. @AfterReturning
5. @After
6. @Around（后半部分）

**异常执行顺序：**

1. @Around（前半部分）
2. @Before
3. 目标方法执行（抛异常）
4. @AfterThrowing
5. @After
6. @Around 异常处理

**示例：**

```java
@Aspect
@Component
public class OrderAspect {

    @Around("execution(* com.example.service.OrderService.*(..))")
    public Object around(ProceedingJoinPoint pjp) throws Throwable {
        System.out.println("Around - 前");
        Object result = pjp.proceed();
        System.out.println("Around - 后");
        return result;
    }

    @Before("execution(* com.example.service.OrderService.*(..))")
    public void before() {
        System.out.println("Before");
    }

    @After("execution(* com.example.service.OrderService.*(..))")
    public void after() {
        System.out.println("After - 总是执行");
    }

    @AfterReturning("execution(* com.example.service.OrderService.*(..))")
    public void afterReturning() {
        System.out.println("AfterReturning - 正常返回");
    }

    @AfterThrowing("execution(* com.example.service.OrderService.*(..))")
    public void afterThrowing() {
        System.out.println("AfterThrowing - 异常抛出");
    }
}
```

**延伸：** 参考 [AOP 详解](./aop)

---

## 🎯 事务管理

### 9. Spring 事务的传播行为有哪些？

**答案要点：**

**七种传播行为：**

| 传播行为             | 说明                         |
| -------------------- | ---------------------------- |
| **REQUIRED（默认）** | 有事务则加入，无则新建       |
| **REQUIRES_NEW**     | 总是新建事务，挂起当前事务   |
| **SUPPORTS**         | 有事务则加入，无则非事务执行 |
| **NOT_SUPPORTED**    | 总是非事务执行，挂起当前事务 |
| **MANDATORY**        | 必须在事务中执行，否则抛异常 |
| **NEVER**            | 不能在事务中执行，否则抛异常 |
| **NESTED**           | 嵌套事务，有保存点           |

**常用场景示例：**

```java
@Service
public class OrderService {

    // 默认：加入外层事务或新建
    @Transactional(propagation = Propagation.REQUIRED)
    public void createOrder() {
        // ...
        logService.log();  // 如果log抛异常，整个order事务回滚
    }
}

@Service
public class LogService {

    // 独立事务：即使失败也不影响外层
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void log() {
        // 日志记录失败不应该影响订单创建
    }
}
```

**延伸：** 参考 [事务管理](./transactions)

---

### 10. @Transactional 失效的场景有哪些？

**答案要点：**

**常见失效场景：**

**1. 方法不是 public**

```java
@Transactional
private void save() { }  // ✗ 私有方法，事务失效
```

**2. 同类内部调用**

```java
@Service
public class UserService {
    public void methodA() {
        this.methodB();  // ✗ 内部调用，事务失效
    }

    @Transactional
    public void methodB() { }
}
```

**解决：** 注入自己或使用 AopContext.currentProxy()

**3. 异常被捕获**

```java
@Transactional
public void save() {
    try {
        // ...
    } catch (Exception e) {
        // ✗ 异常被吞掉，不会回滚
    }
}
```

**4. 异常类型不匹配**

```java
@Transactional  // 默认只回滚RuntimeException和Error
public void save() throws Exception {
    throw new Exception();  // ✗ 检查异常不回滚
}

// 解决：指定回滚异常
@Transactional(rollbackFor = Exception.class)
```

**5. 数据库引擎不支持事务**

- MyISAM 不支持事务，必须使用 InnoDB

**延伸：** 参考 [常见问题](./faq)

---

### 11. 事务的隔离级别有哪些？

**答案要点：**

**四种隔离级别：**

| 隔离级别         | 脏读 | 不可重复读 | 幻读 |
| ---------------- | ---- | ---------- | ---- |
| READ_UNCOMMITTED | ✓    | ✓          | ✓    |
| READ_COMMITTED   | ✗    | ✓          | ✓    |
| REPEATABLE_READ  | ✗    | ✗          | ✓    |
| SERIALIZABLE     | ✗    | ✗          | ✗    |

**MySQL 默认：** REPEATABLE_READ  
**Oracle 默认：** READ_COMMITTED

**问题说明：**

- **脏读：** 读到未提交的数据
- **不可重复读：** 同一查询两次结果不同（UPDATE）
- **幻读：** 同一查询两次行数不同（INSERT/DELETE）

**配置示例：**

```java
@Transactional(isolation = Isolation.READ_COMMITTED)
public void transfer() {
    // 读已提交，避免脏读
}
```

**延伸：** 参考 [事务管理](./transactions)

---

## 🎯 Spring MVC

### 12. Spring MVC 的请求处理流程？

**答案要点：**

**核心流程：**

1. **DispatcherServlet** 接收请求
2. **HandlerMapping** 查找处理器（Controller）
3. **HandlerAdapter** 调用处理器方法
4. **Controller** 执行业务逻辑，返回 ModelAndView
5. **ViewResolver** 解析视图名称
6. **View** 渲染视图
7. **DispatcherServlet** 返回响应

**流程图：**

```
Request → DispatcherServlet → HandlerMapping → HandlerAdapter
       → Controller → ModelAndView → ViewResolver → View
       → Response
```

**代码示例：**

```java
@Controller
public class UserController {

    @GetMapping("/users/{id}")
    public String getUser(@PathVariable Long id, Model model) {
        User user = userService.getById(id);
        model.addAttribute("user", user);
        return "user/detail";  // 视图名称
    }
}
```

**延伸：** 参考 [Spring MVC](./spring-mvc)

---

### 13. @RequestParam 和 @PathVariable 的区别？

**答案要点：**

| 注解          | 用途         | 示例          |
| ------------- | ------------ | ------------- |
| @RequestParam | 获取查询参数 | `/users?id=1` |
| @PathVariable | 获取路径变量 | `/users/1`    |

**代码示例：**

```java
// @RequestParam - 查询参数
@GetMapping("/users")
public List<User> search(@RequestParam String name,
                        @RequestParam(required = false) Integer age) {
    // GET /users?name=Tom&age=20
}

// @PathVariable - 路径变量
@GetMapping("/users/{id}")
public User getUser(@PathVariable Long id) {
    // GET /users/123
}

// 组合使用
@GetMapping("/users/{id}/orders")
public List<Order> getUserOrders(@PathVariable Long id,
                                @RequestParam String status) {
    // GET /users/123/orders?status=PENDING
}
```

**延伸：** 参考 [Spring MVC](./spring-mvc)

---

## 🎯 配置与高级

### 14. @Component、@Service、@Repository、@Controller 的区别？

**答案要点：**

**本质上都是 @Component：**

- `@Component`：通用组件
- `@Service`：业务逻辑层
- `@Repository`：数据访问层（额外支持异常转换）
- `@Controller`：控制层

**语义区分，便于分层：**

```java
@Repository  // DAO层
public class UserDao { }

@Service  // Service层
public class UserService { }

@Controller  // Controller层
public class UserController { }

@Component  // 通用组件
public class EmailSender { }
```

**@Repository 的特殊之处：**

- 会将数据库异常转换为 Spring 的 DataAccessException

**延伸：** 参考 [核心概念](./core-concepts)

---

### 15. Spring Boot 自动配置原理？

**答案要点：**

**核心机制：**

1. **@SpringBootApplication** 包含三个注解：

   - `@SpringBootConfiguration`：配置类
   - `@EnableAutoConfiguration`：启用自动配置
   - `@ComponentScan`：组件扫描

2. **@EnableAutoConfiguration** 通过 `@Import` 导入配置

3. **spring.factories** 文件中定义自动配置类

4. **@Conditional** 条件注解控制是否生效

**自动配置示例：**

```java
@Configuration
@ConditionalOnClass(DataSource.class)  // 类路径有DataSource
@ConditionalOnMissingBean(DataSource.class)  // 未自定义Bean
public class DataSourceAutoConfiguration {

    @Bean
    public DataSource dataSource() {
        // 自动配置数据源
    }
}
```

**常用条件注解：**

- `@ConditionalOnClass`：类存在
- `@ConditionalOnMissingBean`：Bean 不存在
- `@ConditionalOnProperty`：配置属性存在

**延伸：** 参考 [Spring Boot 自动配置](../springboot)

---

## 📌 总结与建议

### 高频考点

1. **IoC/DI** - Bean 的生命周期、作用域、注入方式
2. **AOP** - 代理机制、通知类型、应用场景
3. **事务** - 传播行为、隔离级别、失效场景
4. **MVC** - 请求处理流程、参数绑定
5. **自动配置** - Spring Boot 的自动配置原理

### 学习建议

- **理解原理** > 记忆 API
- **动手实践** > 纸上谈兵
- **源码阅读** > 文档浏览
- **项目应用** > 孤立学习

### 相关资源

- [Spring Framework 学习指南](./index.md)
- [核心概念](./core-concepts)
- [AOP 详解](./aop)
- [事务管理](./transactions)
- [最佳实践](./best-practices)

---

**持续更新中...** 欢迎反馈和补充！
