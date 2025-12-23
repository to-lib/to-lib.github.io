---
sidebar_position: 8
title: MyBatis Spring 集成
---

# MyBatis Spring 集成

本章介绍 MyBatis 与 Spring/Spring Boot 的集成方式，包括配置、事务管理和多数据源配置。

## Spring Boot 集成

### 添加依赖

```xml
<dependencies>
    <!-- MyBatis Spring Boot Starter -->
    <dependency>
        <groupId>org.mybatis.spring.boot</groupId>
        <artifactId>mybatis-spring-boot-starter</artifactId>
        <version>3.0.3</version>
    </dependency>
    
    <!-- MySQL 驱动 -->
    <dependency>
        <groupId>com.mysql</groupId>
        <artifactId>mysql-connector-j</artifactId>
        <scope>runtime</scope>
    </dependency>
</dependencies>
```

### 配置文件

**application.yml:**

```yaml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/mydb?useSSL=false&serverTimezone=UTC
    username: root
    password: 123456
    driver-class-name: com.mysql.cj.jdbc.Driver
    # HikariCP 连接池配置
    hikari:
      minimum-idle: 5
      maximum-pool-size: 20
      idle-timeout: 30000
      connection-timeout: 30000
      max-lifetime: 1800000

mybatis:
  # Mapper XML 文件位置
  mapper-locations: classpath:mapper/*.xml
  # 实体类包路径（类型别名）
  type-aliases-package: com.example.entity
  # MyBatis 配置文件（可选）
  config-location: classpath:mybatis-config.xml
  configuration:
    # 驼峰命名映射
    map-underscore-to-camel-case: true
    # 日志实现
    log-impl: org.apache.ibatis.logging.slf4j.Slf4jImpl
    # 缓存
    cache-enabled: true
    # 延迟加载
    lazy-loading-enabled: false
    # 空值调用 setter
    call-setters-on-nulls: true
```

**application.properties:**

```properties
# 数据源配置
spring.datasource.url=jdbc:mysql://localhost:3306/mydb
spring.datasource.username=root
spring.datasource.password=123456
spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver

# MyBatis 配置
mybatis.mapper-locations=classpath:mapper/*.xml
mybatis.type-aliases-package=com.example.entity
mybatis.configuration.map-underscore-to-camel-case=true
mybatis.configuration.log-impl=org.apache.ibatis.logging.slf4j.Slf4jImpl
```

### 项目结构

```
src/main/java
├── com.example
│   ├── Application.java
│   ├── entity
│   │   └── User.java
│   ├── mapper
│   │   └── UserMapper.java
│   └── service
│       └── UserService.java
src/main/resources
├── application.yml
├── mapper
│   └── UserMapper.xml
└── mybatis-config.xml (可选)
```

### Mapper 接口

```java
package com.example.mapper;

import com.example.entity.User;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper  // 标记为 Mapper 接口
public interface UserMapper {
    
    @Select("SELECT * FROM user WHERE id = #{id}")
    User selectById(Long id);
    
    List<User> selectByCondition(UserQuery query);
    
    @Insert("INSERT INTO user(name, email) VALUES(#{name}, #{email})")
    @Options(useGeneratedKeys = true, keyProperty = "id")
    int insert(User user);
    
    int update(User user);
    
    @Delete("DELETE FROM user WHERE id = #{id}")
    int deleteById(Long id);
}
```

### Mapper 扫描

**方式一：@Mapper 注解**

```java
@Mapper
public interface UserMapper {
    // ...
}
```

**方式二：@MapperScan 注解（推荐）**

```java
@SpringBootApplication
@MapperScan("com.example.mapper")  // 扫描 Mapper 包
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}
```

**方式三：多包扫描**

```java
@MapperScan(basePackages = {
    "com.example.mapper",
    "com.example.repository"
})
public class Application {
    // ...
}
```

### Service 层使用

```java
@Service
public class UserService {
    
    @Autowired
    private UserMapper userMapper;
    
    public User getById(Long id) {
        return userMapper.selectById(id);
    }
    
    @Transactional
    public void createUser(User user) {
        userMapper.insert(user);
    }
    
    @Transactional
    public void updateUser(User user) {
        userMapper.update(user);
    }
}
```

## 传统 Spring 集成

### 添加依赖

```xml
<dependencies>
    <dependency>
        <groupId>org.mybatis</groupId>
        <artifactId>mybatis</artifactId>
        <version>3.5.16</version>
    </dependency>
    <dependency>
        <groupId>org.mybatis</groupId>
        <artifactId>mybatis-spring</artifactId>
        <version>3.0.3</version>
    </dependency>
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-jdbc</artifactId>
        <version>6.1.5</version>
    </dependency>
</dependencies>
```

### XML 配置方式

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:tx="http://www.springframework.org/schema/tx"
       xsi:schemaLocation="
           http://www.springframework.org/schema/beans
           http://www.springframework.org/schema/beans/spring-beans.xsd
           http://www.springframework.org/schema/tx
           http://www.springframework.org/schema/tx/spring-tx.xsd">
    
    <!-- 数据源 -->
    <bean id="dataSource" class="com.zaxxer.hikari.HikariDataSource">
        <property name="driverClassName" value="com.mysql.cj.jdbc.Driver"/>
        <property name="jdbcUrl" value="jdbc:mysql://localhost:3306/mydb"/>
        <property name="username" value="root"/>
        <property name="password" value="123456"/>
    </bean>
    
    <!-- SqlSessionFactory -->
    <bean id="sqlSessionFactory" class="org.mybatis.spring.SqlSessionFactoryBean">
        <property name="dataSource" ref="dataSource"/>
        <property name="configLocation" value="classpath:mybatis-config.xml"/>
        <property name="mapperLocations" value="classpath:mapper/*.xml"/>
        <property name="typeAliasesPackage" value="com.example.entity"/>
    </bean>
    
    <!-- Mapper 扫描 -->
    <bean class="org.mybatis.spring.mapper.MapperScannerConfigurer">
        <property name="basePackage" value="com.example.mapper"/>
        <property name="sqlSessionFactoryBeanName" value="sqlSessionFactory"/>
    </bean>
    
    <!-- 事务管理器 -->
    <bean id="transactionManager" 
          class="org.springframework.jdbc.datasource.DataSourceTransactionManager">
        <property name="dataSource" ref="dataSource"/>
    </bean>
    
    <!-- 启用注解事务 -->
    <tx:annotation-driven transaction-manager="transactionManager"/>
    
</beans>
```

### Java 配置方式

```java
@Configuration
@MapperScan("com.example.mapper")
@EnableTransactionManagement
public class MyBatisConfig {
    
    @Bean
    public DataSource dataSource() {
        HikariDataSource dataSource = new HikariDataSource();
        dataSource.setDriverClassName("com.mysql.cj.jdbc.Driver");
        dataSource.setJdbcUrl("jdbc:mysql://localhost:3306/mydb");
        dataSource.setUsername("root");
        dataSource.setPassword("123456");
        return dataSource;
    }
    
    @Bean
    public SqlSessionFactory sqlSessionFactory(DataSource dataSource) throws Exception {
        SqlSessionFactoryBean factoryBean = new SqlSessionFactoryBean();
        factoryBean.setDataSource(dataSource);
        factoryBean.setTypeAliasesPackage("com.example.entity");
        factoryBean.setMapperLocations(
            new PathMatchingResourcePatternResolver()
                .getResources("classpath:mapper/*.xml"));
        
        // MyBatis 配置
        org.apache.ibatis.session.Configuration configuration = 
            new org.apache.ibatis.session.Configuration();
        configuration.setMapUnderscoreToCamelCase(true);
        configuration.setCacheEnabled(true);
        factoryBean.setConfiguration(configuration);
        
        return factoryBean.getObject();
    }
    
    @Bean
    public DataSourceTransactionManager transactionManager(DataSource dataSource) {
        return new DataSourceTransactionManager(dataSource);
    }
}
```

## 事务管理

### @Transactional 注解

```java
@Service
public class UserService {
    
    @Autowired
    private UserMapper userMapper;
    
    @Autowired
    private OrderMapper orderMapper;
    
    // 基本事务
    @Transactional
    public void createUser(User user) {
        userMapper.insert(user);
    }
    
    // 只读事务（优化查询性能）
    @Transactional(readOnly = true)
    public User getUser(Long id) {
        return userMapper.selectById(id);
    }
    
    // 指定回滚异常
    @Transactional(rollbackFor = Exception.class)
    public void updateUser(User user) throws Exception {
        userMapper.update(user);
        // 如果抛出异常，事务回滚
    }
    
    // 不回滚特定异常
    @Transactional(noRollbackFor = BusinessException.class)
    public void processUser(User user) {
        // BusinessException 不会导致回滚
    }
    
    // 超时设置
    @Transactional(timeout = 30)
    public void longOperation() {
        // 超过 30 秒自动回滚
    }
}
```

### 事务传播行为

```java
@Service
public class OrderService {
    
    @Autowired
    private OrderMapper orderMapper;
    
    @Autowired
    private UserService userService;
    
    // REQUIRED（默认）：有事务就加入，没有就新建
    @Transactional(propagation = Propagation.REQUIRED)
    public void createOrder(Order order) {
        orderMapper.insert(order);
        userService.updateUserPoints(order.getUserId());
    }
    
    // REQUIRES_NEW：总是新建事务，挂起当前事务
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void createOrderLog(OrderLog log) {
        // 独立事务，不受外层事务影响
    }
    
    // NESTED：嵌套事务，可以独立回滚
    @Transactional(propagation = Propagation.NESTED)
    public void updateInventory(Long productId, int quantity) {
        // 嵌套事务
    }
    
    // SUPPORTS：有事务就加入，没有就非事务执行
    @Transactional(propagation = Propagation.SUPPORTS)
    public Order getOrder(Long id) {
        return orderMapper.selectById(id);
    }
    
    // NOT_SUPPORTED：非事务执行，挂起当前事务
    @Transactional(propagation = Propagation.NOT_SUPPORTED)
    public void sendNotification() {
        // 不需要事务
    }
}
```

### 事务隔离级别

```java
@Transactional(isolation = Isolation.READ_COMMITTED)
public void updateBalance(Long userId, BigDecimal amount) {
    // 读已提交隔离级别
}
```

| 隔离级别 | 描述 |
|----------|------|
| `DEFAULT` | 使用数据库默认 |
| `READ_UNCOMMITTED` | 读未提交 |
| `READ_COMMITTED` | 读已提交 |
| `REPEATABLE_READ` | 可重复读 |
| `SERIALIZABLE` | 串行化 |

### 编程式事务

```java
@Service
public class UserService {
    
    @Autowired
    private TransactionTemplate transactionTemplate;
    
    @Autowired
    private UserMapper userMapper;
    
    public void createUserWithProgrammaticTx(User user) {
        transactionTemplate.execute(status -> {
            try {
                userMapper.insert(user);
                // 其他操作
                return true;
            } catch (Exception e) {
                status.setRollbackOnly();
                return false;
            }
        });
    }
}
```

## 多数据源配置

### 配置文件

```yaml
spring:
  datasource:
    primary:
      url: jdbc:mysql://localhost:3306/db_primary
      username: root
      password: 123456
      driver-class-name: com.mysql.cj.jdbc.Driver
    secondary:
      url: jdbc:mysql://localhost:3306/db_secondary
      username: root
      password: 123456
      driver-class-name: com.mysql.cj.jdbc.Driver
```

### 数据源配置类

```java
@Configuration
public class DataSourceConfig {
    
    @Bean
    @Primary
    @ConfigurationProperties(prefix = "spring.datasource.primary")
    public DataSource primaryDataSource() {
        return DataSourceBuilder.create().build();
    }
    
    @Bean
    @ConfigurationProperties(prefix = "spring.datasource.secondary")
    public DataSource secondaryDataSource() {
        return DataSourceBuilder.create().build();
    }
}
```

### 主数据源 MyBatis 配置

```java
@Configuration
@MapperScan(
    basePackages = "com.example.mapper.primary",
    sqlSessionFactoryRef = "primarySqlSessionFactory"
)
public class PrimaryMyBatisConfig {
    
    @Bean
    @Primary
    public SqlSessionFactory primarySqlSessionFactory(
            @Qualifier("primaryDataSource") DataSource dataSource) throws Exception {
        SqlSessionFactoryBean factoryBean = new SqlSessionFactoryBean();
        factoryBean.setDataSource(dataSource);
        factoryBean.setMapperLocations(
            new PathMatchingResourcePatternResolver()
                .getResources("classpath:mapper/primary/*.xml"));
        return factoryBean.getObject();
    }
    
    @Bean
    @Primary
    public DataSourceTransactionManager primaryTransactionManager(
            @Qualifier("primaryDataSource") DataSource dataSource) {
        return new DataSourceTransactionManager(dataSource);
    }
}
```

### 从数据源 MyBatis 配置

```java
@Configuration
@MapperScan(
    basePackages = "com.example.mapper.secondary",
    sqlSessionFactoryRef = "secondarySqlSessionFactory"
)
public class SecondaryMyBatisConfig {
    
    @Bean
    public SqlSessionFactory secondarySqlSessionFactory(
            @Qualifier("secondaryDataSource") DataSource dataSource) throws Exception {
        SqlSessionFactoryBean factoryBean = new SqlSessionFactoryBean();
        factoryBean.setDataSource(dataSource);
        factoryBean.setMapperLocations(
            new PathMatchingResourcePatternResolver()
                .getResources("classpath:mapper/secondary/*.xml"));
        return factoryBean.getObject();
    }
    
    @Bean
    public DataSourceTransactionManager secondaryTransactionManager(
            @Qualifier("secondaryDataSource") DataSource dataSource) {
        return new DataSourceTransactionManager(dataSource);
    }
}
```

### 使用多数据源

```java
@Service
public class UserService {
    
    @Autowired
    private PrimaryUserMapper primaryUserMapper;  // 主库
    
    @Autowired
    private SecondaryUserMapper secondaryUserMapper;  // 从库
    
    // 使用主库事务
    @Transactional("primaryTransactionManager")
    public void createUser(User user) {
        primaryUserMapper.insert(user);
    }
    
    // 使用从库事务
    @Transactional("secondaryTransactionManager")
    public User getUser(Long id) {
        return secondaryUserMapper.selectById(id);
    }
}
```

## 动态数据源

### 动态数据源实现

```java
public class DynamicDataSource extends AbstractRoutingDataSource {
    
    @Override
    protected Object determineCurrentLookupKey() {
        return DataSourceContextHolder.getDataSource();
    }
}

public class DataSourceContextHolder {
    
    private static final ThreadLocal<String> CONTEXT = new ThreadLocal<>();
    
    public static void setDataSource(String dataSource) {
        CONTEXT.set(dataSource);
    }
    
    public static String getDataSource() {
        return CONTEXT.get();
    }
    
    public static void clear() {
        CONTEXT.remove();
    }
}
```

### 配置动态数据源

```java
@Configuration
public class DynamicDataSourceConfig {
    
    @Bean
    public DataSource dynamicDataSource(
            @Qualifier("primaryDataSource") DataSource primary,
            @Qualifier("secondaryDataSource") DataSource secondary) {
        
        Map<Object, Object> targetDataSources = new HashMap<>();
        targetDataSources.put("primary", primary);
        targetDataSources.put("secondary", secondary);
        
        DynamicDataSource dynamicDataSource = new DynamicDataSource();
        dynamicDataSource.setTargetDataSources(targetDataSources);
        dynamicDataSource.setDefaultTargetDataSource(primary);
        
        return dynamicDataSource;
    }
}
```

### 使用注解切换数据源

```java
@Target({ElementType.METHOD, ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
public @interface DataSource {
    String value() default "primary";
}

@Aspect
@Component
public class DataSourceAspect {
    
    @Before("@annotation(dataSource)")
    public void switchDataSource(JoinPoint point, DataSource dataSource) {
        DataSourceContextHolder.setDataSource(dataSource.value());
    }
    
    @After("@annotation(DataSource)")
    public void clearDataSource(JoinPoint point) {
        DataSourceContextHolder.clear();
    }
}
```

```java
@Service
public class UserService {
    
    @DataSource("primary")
    public void writeUser(User user) {
        // 使用主库
    }
    
    @DataSource("secondary")
    public User readUser(Long id) {
        // 使用从库
    }
}
```

## 完整项目示例

### pom.xml

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.2.0</version>
    </parent>
    
    <groupId>com.example</groupId>
    <artifactId>mybatis-demo</artifactId>
    <version>1.0.0</version>
    
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <dependency>
            <groupId>org.mybatis.spring.boot</groupId>
            <artifactId>mybatis-spring-boot-starter</artifactId>
            <version>3.0.3</version>
        </dependency>
        <dependency>
            <groupId>com.mysql</groupId>
            <artifactId>mysql-connector-j</artifactId>
            <scope>runtime</scope>
        </dependency>
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <optional>true</optional>
        </dependency>
    </dependencies>
</project>
```

### 启动类

```java
@SpringBootApplication
@MapperScan("com.example.mapper")
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}
```

### 实体类

```java
@Data
public class User implements Serializable {
    private Long id;
    private String name;
    private String email;
    private Integer age;
    private Integer status;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
}
```

### Mapper 接口

```java
@Mapper
public interface UserMapper {
    
    User selectById(Long id);
    
    List<User> selectByCondition(UserQuery query);
    
    int insert(User user);
    
    int update(User user);
    
    int deleteById(Long id);
}
```

### Mapper XML

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN"
  "https://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.example.mapper.UserMapper">
    
    <sql id="baseColumns">
        id, name, email, age, status, create_time, update_time
    </sql>
    
    <select id="selectById" resultType="User">
        SELECT <include refid="baseColumns"/>
        FROM user WHERE id = #{id}
    </select>
    
    <select id="selectByCondition" resultType="User">
        SELECT <include refid="baseColumns"/>
        FROM user
        <where>
            <if test="name != null">AND name LIKE CONCAT('%', #{name}, '%')</if>
            <if test="status != null">AND status = #{status}</if>
        </where>
        ORDER BY create_time DESC
    </select>
    
    <insert id="insert" useGeneratedKeys="true" keyProperty="id">
        INSERT INTO user(name, email, age, status, create_time)
        VALUES(#{name}, #{email}, #{age}, #{status}, NOW())
    </insert>
    
    <update id="update">
        UPDATE user
        <set>
            <if test="name != null">name = #{name},</if>
            <if test="email != null">email = #{email},</if>
            <if test="age != null">age = #{age},</if>
            <if test="status != null">status = #{status},</if>
            update_time = NOW()
        </set>
        WHERE id = #{id}
    </update>
    
    <delete id="deleteById">
        DELETE FROM user WHERE id = #{id}
    </delete>
    
</mapper>
```

### Service 层

```java
@Service
@Transactional(readOnly = true)
public class UserService {
    
    @Autowired
    private UserMapper userMapper;
    
    public User getById(Long id) {
        return userMapper.selectById(id);
    }
    
    public List<User> search(UserQuery query) {
        return userMapper.selectByCondition(query);
    }
    
    @Transactional
    public Long create(User user) {
        userMapper.insert(user);
        return user.getId();
    }
    
    @Transactional
    public void update(User user) {
        userMapper.update(user);
    }
    
    @Transactional
    public void delete(Long id) {
        userMapper.deleteById(id);
    }
}
```

## 相关链接

- [配置详解](/docs/mybatis/configuration) - MyBatis 配置
- [缓存机制](/docs/mybatis/caching) - 缓存配置
- [最佳实践](/docs/mybatis/best-practices) - 项目最佳实践
- [Spring Framework](/docs/spring) - Spring 框架
- [Spring Boot](/docs/springboot) - Spring Boot 框架

---

**最后更新**: 2025 年 12 月
