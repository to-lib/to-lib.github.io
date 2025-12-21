---
sidebar_position: 6
title: 框架源码分析
---

# 🎯 框架源码分析（专家级）

## 21. Spring IoC 容器启动流程是怎样的？

**答案要点：**

**核心启动流程：**

```java
// AbstractApplicationContext.refresh() 方法
public void refresh() {
    // 1. 准备刷新
    prepareRefresh();
    
    // 2. 获取 BeanFactory
    ConfigurableListableBeanFactory beanFactory = obtainFreshBeanFactory();
    
    // 3. 准备 BeanFactory
    prepareBeanFactory(beanFactory);
    
    // 4. 后置处理 BeanFactory
    postProcessBeanFactory(beanFactory);
    
    // 5. 调用 BeanFactoryPostProcessor
    invokeBeanFactoryPostProcessors(beanFactory);
    
    // 6. 注册 BeanPostProcessor
    registerBeanPostProcessors(beanFactory);
    
    // 7. 初始化消息源
    initMessageSource();
    
    // 8. 初始化事件广播器
    initApplicationEventMulticaster();
    
    // 9. 子类扩展点
    onRefresh();
    
    // 10. 注册监听器
    registerListeners();
    
    // 11. 实例化所有非懒加载的单例 Bean
    finishBeanFactoryInitialization(beanFactory);
    
    // 12. 完成刷新
    finishRefresh();
}
```

**Bean 创建流程：**

```
getBean() 
    → doGetBean()
        → getSingleton() // 从缓存获取
        → createBean()
            → resolveBeforeInstantiation() // 实例化前处理
            → doCreateBean()
                → createBeanInstance() // 实例化
                → populateBean() // 属性填充
                → initializeBean() // 初始化
                    → invokeAwareMethods()
                    → applyBeanPostProcessorsBeforeInitialization()
                    → invokeInitMethods()
                    → applyBeanPostProcessorsAfterInitialization()
```

**三级缓存解决循环依赖：**

```java
// DefaultSingletonBeanRegistry
// 一级缓存：完整的 Bean
private final Map<String, Object> singletonObjects = new ConcurrentHashMap<>();

// 二级缓存：早期暴露的 Bean（未完成属性填充）
private final Map<String, Object> earlySingletonObjects = new ConcurrentHashMap<>();

// 三级缓存：Bean 工厂
private final Map<String, ObjectFactory<?>> singletonFactories = new HashMap<>();
```

**延伸：** 参考 [Spring 核心概念](/docs/spring/core-concepts)

---

## 22. Spring AOP 是如何实现的？

**答案要点：**

**AOP 实现方式：**

| 方式 | 条件 | 特点 |
|------|------|------|
| JDK 动态代理 | 目标类实现接口 | 基于接口代理 |
| CGLIB 代理 | 目标类无接口 | 基于继承代理 |

**JDK 动态代理原理：**

```java
public class JdkProxyDemo {
    public static void main(String[] args) {
        UserService target = new UserServiceImpl();
        
        UserService proxy = (UserService) Proxy.newProxyInstance(
            target.getClass().getClassLoader(),
            target.getClass().getInterfaces(),
            new InvocationHandler() {
                @Override
                public Object invoke(Object proxy, Method method, Object[] args) 
                        throws Throwable {
                    System.out.println("Before: " + method.getName());
                    Object result = method.invoke(target, args);
                    System.out.println("After: " + method.getName());
                    return result;
                }
            }
        );
        
        proxy.getUser("1");
    }
}
```

**CGLIB 代理原理：**

```java
public class CglibProxyDemo {
    public static void main(String[] args) {
        Enhancer enhancer = new Enhancer();
        enhancer.setSuperclass(UserServiceImpl.class);
        enhancer.setCallback(new MethodInterceptor() {
            @Override
            public Object intercept(Object obj, Method method, Object[] args, 
                    MethodProxy proxy) throws Throwable {
                System.out.println("Before: " + method.getName());
                Object result = proxy.invokeSuper(obj, args);
                System.out.println("After: " + method.getName());
                return result;
            }
        });
        
        UserServiceImpl proxy = (UserServiceImpl) enhancer.create();
        proxy.getUser("1");
    }
}
```

**Spring AOP 代理创建流程：**

```
@EnableAspectJAutoProxy
    → 注册 AnnotationAwareAspectJAutoProxyCreator
        → postProcessAfterInitialization()
            → wrapIfNecessary()
                → getAdvicesAndAdvisorsForBean() // 获取切面
                → createProxy() // 创建代理
                    → ProxyFactory.getProxy()
                        → JdkDynamicAopProxy 或 CglibAopProxy
```

**延伸：** 参考 [Spring AOP 详解](/docs/spring/aop)

---

## 23. Spring Boot 自动配置原理是什么？

**答案要点：**

**自动配置核心注解：**

```java
@SpringBootApplication
    ├── @SpringBootConfiguration  // 配置类
    ├── @EnableAutoConfiguration  // 启用自动配置
    │       └── @Import(AutoConfigurationImportSelector.class)
    └── @ComponentScan            // 组件扫描
```

**自动配置加载流程：**

```
1. @EnableAutoConfiguration
    ↓
2. AutoConfigurationImportSelector.selectImports()
    ↓
3. SpringFactoriesLoader.loadFactoryNames()
    ↓
4. 读取 META-INF/spring.factories
    ↓
5. 过滤条件注解（@ConditionalOnXxx）
    ↓
6. 加载符合条件的自动配置类
```

**spring.factories 示例：**

```properties
# META-INF/spring.factories
org.springframework.boot.autoconfigure.EnableAutoConfiguration=\
org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration,\
org.springframework.boot.autoconfigure.web.servlet.WebMvcAutoConfiguration,\
org.springframework.boot.autoconfigure.data.redis.RedisAutoConfiguration
```

**条件注解原理：**

```java
@Configuration
@ConditionalOnClass(DataSource.class)  // 类路径存在 DataSource
@ConditionalOnMissingBean(DataSource.class)  // 未自定义 DataSource Bean
@EnableConfigurationProperties(DataSourceProperties.class)
public class DataSourceAutoConfiguration {
    
    @Bean
    @ConfigurationProperties(prefix = "spring.datasource")
    public DataSource dataSource() {
        return DataSourceBuilder.create().build();
    }
}
```

**自定义 Starter：**

```java
// 1. 创建自动配置类
@Configuration
@ConditionalOnClass(MyService.class)
@EnableConfigurationProperties(MyProperties.class)
public class MyAutoConfiguration {
    
    @Bean
    @ConditionalOnMissingBean
    public MyService myService(MyProperties properties) {
        return new MyService(properties);
    }
}

// 2. 创建 spring.factories
// META-INF/spring.factories
org.springframework.boot.autoconfigure.EnableAutoConfiguration=\
com.example.MyAutoConfiguration
```

**延伸：** 参考 [Spring Boot 自动配置](/docs/springboot)

---

## 24. MyBatis 的执行流程和缓存机制是怎样的？

**答案要点：**

**MyBatis 执行流程：**

```
SqlSessionFactory
    ↓ openSession()
SqlSession
    ↓ getMapper()
MapperProxy（动态代理）
    ↓ invoke()
MapperMethod
    ↓ execute()
Executor（执行器）
    ↓ query/update
StatementHandler
    ↓ prepare/parameterize/query
ResultSetHandler
    ↓ handleResultSets
返回结果
```

**核心组件：**

```java
// 1. SqlSessionFactory 创建
SqlSessionFactory factory = new SqlSessionFactoryBuilder()
    .build(Resources.getResourceAsStream("mybatis-config.xml"));

// 2. 获取 SqlSession
try (SqlSession session = factory.openSession()) {
    // 3. 获取 Mapper 代理
    UserMapper mapper = session.getMapper(UserMapper.class);
    
    // 4. 执行查询
    User user = mapper.selectById(1L);
}
```

**缓存机制：**

```
┌─────────────────────────────────────────────────────────┐
│                    二级缓存（Mapper 级别）                │
│                    namespace 范围共享                    │
└─────────────────────────────────────────────────────────┘
                           ↓ 未命中
┌─────────────────────────────────────────────────────────┐
│                    一级缓存（SqlSession 级别）           │
│                    默认开启，同一 SqlSession 共享        │
└─────────────────────────────────────────────────────────┘
                           ↓ 未命中
┌─────────────────────────────────────────────────────────┐
│                         数据库                           │
└─────────────────────────────────────────────────────────┘
```

**二级缓存配置：**

```xml
<!-- mybatis-config.xml -->
<settings>
    <setting name="cacheEnabled" value="true"/>
</settings>

<!-- UserMapper.xml -->
<mapper namespace="com.example.mapper.UserMapper">
    <cache eviction="LRU" flushInterval="60000" size="512" readOnly="true"/>
    
    <select id="selectById" resultType="User" useCache="true">
        SELECT * FROM user WHERE id = #{id}
    </select>
</mapper>
```

**延伸：** 参考 [MyBatis 核心原理](/docs/spring)

---

## 25. Netty 的线程模型和核心组件是什么？

**答案要点：**

**Netty 线程模型（主从 Reactor）：**

```
                    ┌─────────────────┐
                    │   BossGroup     │  接收连接
                    │  (1个EventLoop) │
                    └────────┬────────┘
                             │ 分发
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐  ┌─────────▼─────────┐  ┌───────▼───────┐
│  WorkerGroup  │  │   WorkerGroup     │  │  WorkerGroup  │
│  EventLoop-1  │  │   EventLoop-2     │  │  EventLoop-N  │
│  (处理IO)     │  │   (处理IO)        │  │  (处理IO)     │
└───────────────┘  └───────────────────┘  └───────────────┘
```

**核心组件：**

| 组件 | 作用 |
|------|------|
| **Channel** | 网络连接通道 |
| **EventLoop** | 事件循环，处理 IO 事件 |
| **ChannelPipeline** | 处理器链 |
| **ChannelHandler** | 事件处理器 |
| **ByteBuf** | 字节缓冲区 |

**Netty 服务端示例：**

```java
public class NettyServer {
    public static void main(String[] args) throws Exception {
        EventLoopGroup bossGroup = new NioEventLoopGroup(1);
        EventLoopGroup workerGroup = new NioEventLoopGroup();
        
        try {
            ServerBootstrap bootstrap = new ServerBootstrap();
            bootstrap.group(bossGroup, workerGroup)
                .channel(NioServerSocketChannel.class)
                .option(ChannelOption.SO_BACKLOG, 128)
                .childOption(ChannelOption.SO_KEEPALIVE, true)
                .childHandler(new ChannelInitializer<SocketChannel>() {
                    @Override
                    protected void initChannel(SocketChannel ch) {
                        ChannelPipeline pipeline = ch.pipeline();
                        pipeline.addLast(new StringDecoder());
                        pipeline.addLast(new StringEncoder());
                        pipeline.addLast(new MyServerHandler());
                    }
                });
            
            ChannelFuture future = bootstrap.bind(8080).sync();
            future.channel().closeFuture().sync();
        } finally {
            bossGroup.shutdownGracefully();
            workerGroup.shutdownGracefully();
        }
    }
}

class MyServerHandler extends SimpleChannelInboundHandler<String> {
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, String msg) {
        System.out.println("Received: " + msg);
        ctx.writeAndFlush("Server: " + msg);
    }
}
```

**延伸：** 参考 [Netty 核心组件](/docs/netty/core-components)
