---
id: aop
title: 面向切面编程(AOP)
sidebar_label: AOP
sidebar_position: 5
---

# 面向切面编程(AOP)

> [!IMPORTANT]
> **AOP 核心作用**: AOP 用于分离横切关注点(日志、事务、安全等),避免代码重复。理解切点(Pointcut)和通知(Advice)是关键。

**面向切面编程(Aspect-Oriented Programming)**是一种编程范式，它将横切关注点（如日志、事务、安全等）从业务逻辑中分离出来，通过**切面**来管理这些关注点。

### 1.1 AOP的好处

- **降低代码耦合度** - 业务逻辑和横切关注点分离
- **提高代码重用** - 横切关注点可以被多个模块共用
- **便于维护** - 修改横切逻辑只需修改一处
- **易于测试** - 可以独立测试业务逻辑

### 1.2 AOP的核心概念

| 概念 | 说明 |
|------|------|
| **Aspect** | 切面，包含切点和通知的组合 |
| **Joinpoint** | 连接点，程序执行中可以进行拦截的点（如方法调用） |
| **Pointcut** | 切点，匹配特定连接点的表达式 |
| **Advice** | 通知，在切点处执行的代码 |
| **Weaving** | 织入，将切面与目标对象结合的过程 |
| **Target** | 目标对象，被代理的对象 |
| **Proxy** | 代理对象，包含了切面逻辑的对象 |

## 2. 切点表达式

### 2.1 execution表达式（最常用）

```java
// 语法：execution(修饰符 返回类型 包名.类名.方法名(参数))

// 匹配UserService中所有public方法
@Pointcut("execution(public * com.example.service.UserService.*(..))")

// 匹配com.example.service包下所有类的所有方法
@Pointcut("execution(* com.example.service.*.*(..))")

// 匹配所有save方法
@Pointcut("execution(* *.save(..))")

// 匹配返回User对象的方法
@Pointcut("execution(com.example.entity.User com.example.service.*.*(..))")

// 匹配参数为User的方法
@Pointcut("execution(* com.example.service.*.*(com.example.entity.User))")

// 匹配任意参数个数的方法
@Pointcut("execution(* com.example.service.*.*(..))")

// 匹配单个参数的方法
@Pointcut("execution(* com.example.service.*.*(*))")
```

### 2.2 其他切点类型

```java
// within - 匹配特定类中的方法
@Pointcut("within(com.example.service.UserService)")

// target - 匹配代理对象的类型
@Pointcut("target(com.example.service.UserService)")

// args - 匹配参数类型
@Pointcut("args(com.example.entity.User)")

// @annotation - 匹配有特定注解的方法
@Pointcut("@annotation(com.example.annotation.LogExecution)")

// @target - 匹配有特定注解的类
@Pointcut("@target(com.example.annotation.Service)")
```

### 2.3 切点表达式的组合

```java
// && - AND
@Pointcut("execution(public * com.example.service.*.*(..)) && @annotation(com.example.annotation.Cacheable)")

// || - OR
@Pointcut("execution(public * *.save(..)) || execution(public * *.update(..))")

// ! - NOT
@Pointcut("execution(public * *.*(..)) && !execution(public * *.get(..))")
```

## 3. 通知(Advice)

### 3.1 前置通知(Before)

在方法执行**之前**执行。

```java
@Aspect
@Component
public class LoggingAspect {
    
    @Before("execution(* com.example.service.*.*(..))")
    public void beforeMethod(JoinPoint joinPoint) {
        String className = joinPoint.getTarget().getClass().getSimpleName();
        String methodName = joinPoint.getSignature().getName();
        Object[] args = joinPoint.getArgs();
        
        System.out.println(String.format(
            "Calling: %s.%s with args: %s",
            className, methodName, Arrays.toString(args)
        ));
    }
}
```

### 3.2 后置通知(After)

在方法执行**之后**执行（无论成功还是失败）。

```java
@After("execution(* com.example.service.*.*(..))")
public void afterMethod(JoinPoint joinPoint) {
    String methodName = joinPoint.getSignature().getName();
    System.out.println(methodName + " executed");
}
```

### 3.3 返回通知(AfterReturning)

在方法**成功返回**后执行。

```java
@AfterReturning(
    pointcut = "execution(* com.example.service.UserService.*(..))",
    returning = "result"
)
public void afterReturning(JoinPoint joinPoint, Object result) {
    String methodName = joinPoint.getSignature().getName();
    System.out.println(String.format(
        "%s returned: %s",
        methodName, result
    ));
}
```

### 3.4 异常通知(AfterThrowing)

在方法**抛出异常**后执行。

```java
@AfterThrowing(
    pointcut = "execution(* com.example.service.*.*(..))",
    throwing = "ex"
)
public void afterThrowing(JoinPoint joinPoint, Exception ex) {
    String methodName = joinPoint.getSignature().getName();
    System.out.println(String.format(
        "Exception in %s: %s",
        methodName, ex.getMessage()
    ));
}
```

### 3.5 环绕通知(Around)（最灵活）

在方法**执行前后**都可以执行，可以修改参数和返回值。

```java
@Around("execution(* com.example.service.*.*(..))")
public Object aroundMethod(ProceedingJoinPoint joinPoint) throws Throwable {
    String methodName = joinPoint.getSignature().getName();
    
    long startTime = System.currentTimeMillis();
    
    try {
        // 执行目标方法
        Object result = joinPoint.proceed();
        
        long duration = System.currentTimeMillis() - startTime;
        System.out.println(String.format(
            "%s executed in %d ms, result: %s",
            methodName, duration, result
        ));
        
        return result;
    } catch (Throwable ex) {
        long duration = System.currentTimeMillis() - startTime;
        System.out.println(String.format(
            "%s failed after %d ms with exception: %s",
            methodName, duration, ex.getMessage()
        ));
        throw ex;
    }
}
```

## 4. 实战示例

### 4.1 日志切面

```java
@Aspect
@Component
@Slf4j
public class LoggingAspect {
    
    @Pointcut("execution(* com.example.service.*.*(..))")
    public void serviceLayer() {}
    
    @Before("serviceLayer()")
    public void logBefore(JoinPoint joinPoint) {
        String methodName = joinPoint.getSignature().getName();
        Object[] args = joinPoint.getArgs();
        
        log.info(">>> Method: {}, Arguments: {}", methodName, args);
    }
    
    @AfterReturning(pointcut = "serviceLayer()", returning = "result")
    public void logAfterReturning(JoinPoint joinPoint, Object result) {
        String methodName = joinPoint.getSignature().getName();
        
        log.info("<<< Method: {}, Result: {}", methodName, result);
    }
    
    @AfterThrowing(pointcut = "serviceLayer()", throwing = "ex")
    public void logAfterThrowing(JoinPoint joinPoint, Exception ex) {
        String methodName = joinPoint.getSignature().getName();
        
        log.error("!!! Method: {}, Exception: {}", methodName, ex.getMessage());
    }
}
```

### 4.2 性能监控切面

```java
@Aspect
@Component
public class PerformanceAspect {
    
    @Around("execution(* com.example.service.*.*(..))")
    public Object monitorPerformance(ProceedingJoinPoint joinPoint) throws Throwable {
        String methodName = joinPoint.getSignature().getName();
        long startTime = System.currentTimeMillis();
        
        try {
            Object result = joinPoint.proceed();
            return result;
        } finally {
            long duration = System.currentTimeMillis() - startTime;
            if (duration > 1000) {
                log.warn("Method {} took {} ms (slow)", methodName, duration);
            } else {
                log.debug("Method {} took {} ms", methodName, duration);
            }
        }
    }
}
```

### 4.3 缓存切面

```java
@Aspect
@Component
public class CachingAspect {
    
    private Map<String, Object> cache = new HashMap<>();
    
    @Around("@annotation(com.example.annotation.Cacheable)")
    public Object cache(ProceedingJoinPoint joinPoint) throws Throwable {
        String key = generateKey(joinPoint);
        
        // 检查缓存
        if (cache.containsKey(key)) {
            log.info("Cache hit: {}", key);
            return cache.get(key);
        }
        
        // 执行方法
        Object result = joinPoint.proceed();
        
        // 存入缓存
        cache.put(key, result);
        log.info("Cached: {}", key);
        
        return result;
    }
    
    private String generateKey(ProceedingJoinPoint joinPoint) {
        String methodName = joinPoint.getSignature().getName();
        Object[] args = joinPoint.getArgs();
        return methodName + "_" + Arrays.toString(args);
    }
}
```

### 4.4 事务切面

```java
@Aspect
@Component
public class TransactionAspect {
    
    @Autowired
    private PlatformTransactionManager transactionManager;
    
    @Around("execution(* com.example.service.*.*(..))")
    public Object manageTransaction(ProceedingJoinPoint joinPoint) throws Throwable {
        TransactionStatus status = transactionManager.getTransaction(
            new DefaultTransactionDefinition()
        );
        
        try {
            Object result = joinPoint.proceed();
            transactionManager.commit(status);
            return result;
        } catch (Throwable ex) {
            transactionManager.rollback(status);
            throw ex;
        }
    }
}
```

## 5. JoinPoint和ProceedingJoinPoint

### 5.1 JoinPoint（所有通知都可用）

```java
@Before("execution(* com.example.service.*.*(..))")
public void before(JoinPoint joinPoint) {
    // 获取目标对象
    Object target = joinPoint.getTarget();
    
    // 获取方法签名
    MethodSignature signature = (MethodSignature) joinPoint.getSignature();
    String methodName = signature.getName();
    Class<?>[] parameterTypes = signature.getParameterTypes();
    
    // 获取参数
    Object[] args = joinPoint.getArgs();
    
    // 获取返回类型
    Class<?> returnType = signature.getReturnType();
    
    // 获取当前执行点
    StaticPart staticPart = joinPoint.getStaticPart();
}
```

### 5.2 ProceedingJoinPoint（仅在@Around中可用）

```java
@Around("execution(* com.example.service.*.*(..))")
public Object around(ProceedingJoinPoint joinPoint) throws Throwable {
    // 包含JoinPoint的所有方法
    
    // 执行目标方法
    Object result = joinPoint.proceed();
    
    // 执行带修改参数的目标方法
    Object resultWithArgs = joinPoint.proceed(new Object[]{/* new args */});
    
    return result;
}
```

## 6. 自定义注解切面

### 6.1 创建自定义注解

```java
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface LogExecution {
    String value() default "";
}

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface Cacheable {
    long ttl() default 3600;  // 过期时间（秒）
}
```

### 6.2 基于注解的切面

```java
@Aspect
@Component
public class AnnotationAspect {
    
    @Around("@annotation(logExecution)")
    public Object logExecution(ProceedingJoinPoint joinPoint, LogExecution logExecution) throws Throwable {
        String description = logExecution.value();
        String methodName = joinPoint.getSignature().getName();
        
        long startTime = System.currentTimeMillis();
        log.info("Executing {}: {}", methodName, description);
        
        try {
            Object result = joinPoint.proceed();
            long duration = System.currentTimeMillis() - startTime;
            log.info("Completed {} in {} ms", methodName, duration);
            return result;
        } catch (Exception ex) {
            log.error("Failed {}: {}", methodName, ex.getMessage());
            throw ex;
        }
    }
}
```

### 6.3 使用自定义注解

```java
@Service
public class UserService {
    
    @LogExecution("保存用户")
    public void saveUser(User user) {
        // 业务逻辑
    }
    
    @Cacheable(ttl = 1800)
    public User getUserById(Long id) {
        // 业务逻辑
        return user;
    }
}
```

## 7. AOP的注意事项

### 7.1 self-invocation问题

```java
@Component
public class UserService {
    
    @Transactional
    public void save(User user) {
        // 直接调用其他方法
        update(user);  // ❌ update方法的@Transactional不会生效
    }
    
    @Transactional
    public void update(User user) {
    }
}

// ✅ 解决方案1：注入自己
@Component
public class UserService {
    @Autowired
    private UserService self;
    
    @Transactional
    public void save(User user) {
        self.update(user);  // ✅ 通过代理对象调用
    }
    
    @Transactional
    public void update(User user) {
    }
}

// ✅ 解决方案2：使用AopContext
@Component
public class UserService {
    @Transactional
    public void save(User user) {
        UserService proxy = (UserService) AopContext.currentProxy();
        proxy.update(user);
    }
    
    @Transactional
    public void update(User user) {
    }
}
```

### 7.2 代理类型

```java
// Spring默认使用JDK动态代理（需要接口）
public interface UserService {
    void save(User user);
}

@Service
public class UserServiceImpl implements UserService {
    @Override
    public void save(User user) {
    }
}

// 如果没有接口，使用CGLIB代理
@Configuration
@EnableAspectJAutoProxy(proxyTargetClass = true)  // 强制使用CGLIB
public class AppConfig {
}
```

## 8. 总结

| 概念 | 说明 |
|------|------|
| Aspect | 切面，包含切点和通知 |
| Pointcut | 切点表达式，定义何处拦截 |
| Advice | 通知，定义如何拦截 |
| Weaving | 织入，将切面应用到目标 |

| 通知类型 | 时机 | 使用场景 |
|---------|------|--------|
| Before | 方法前 | 参数验证、权限检查 |
| After | 方法后 | 资源清理、日志 |
| AfterReturning | 返回后 | 返回值处理 |
| AfterThrowing | 异常后 | 异常处理、日志 |
| Around | 前后都执行 | 性能监控、缓存、事务 |

---

**关键要点**：

- 优先使用@Around处理复杂逻辑
- 使用自定义注解让代码更清晰
- 注意self-invocation问题
- 理解代理的工作原理

**下一步**：学习[事务管理](/docs/spring/transactions)
