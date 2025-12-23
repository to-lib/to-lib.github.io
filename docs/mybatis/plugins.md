---
sidebar_position: 9
title: MyBatis 插件机制
---

# MyBatis 插件机制

MyBatis 提供了强大的插件机制，允许在 SQL 执行的各个阶段进行拦截和扩展。本章介绍插件原理和自定义插件开发。

## 插件原理

### 可拦截的四大对象

MyBatis 允许拦截以下四个核心对象的方法：

| 对象 | 描述 | 可拦截方法 |
|------|------|------------|
| `Executor` | SQL 执行器 | update, query, commit, rollback, close 等 |
| `StatementHandler` | 语句处理器 | prepare, parameterize, batch, update, query |
| `ParameterHandler` | 参数处理器 | getParameterObject, setParameters |
| `ResultSetHandler` | 结果集处理器 | handleResultSets, handleOutputParameters |

### 拦截器链

```
┌─────────────────────────────────────────────────────────┐
│                      SqlSession                          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐   │
│  │              Executor (代理)                      │   │
│  │  ┌─────────────────────────────────────────┐    │   │
│  │  │         Plugin 1 (拦截器)                │    │   │
│  │  │  ┌─────────────────────────────────┐    │    │   │
│  │  │  │      Plugin 2 (拦截器)           │    │    │   │
│  │  │  │  ┌─────────────────────────┐    │    │    │   │
│  │  │  │  │   真实 Executor         │    │    │    │   │
│  │  │  │  └─────────────────────────┘    │    │    │   │
│  │  │  └─────────────────────────────────┘    │    │   │
│  │  └─────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 插件执行顺序

配置顺序：Plugin1 → Plugin2 → Plugin3

执行顺序：Plugin3 → Plugin2 → Plugin1 → 目标方法 → Plugin1 → Plugin2 → Plugin3

## 自定义插件开发

### 基本结构

```java
@Intercepts({
    @Signature(
        type = Executor.class,                    // 拦截的对象类型
        method = "query",                         // 拦截的方法名
        args = {MappedStatement.class, Object.class, 
                RowBounds.class, ResultHandler.class}  // 方法参数类型
    )
})
public class MyPlugin implements Interceptor {
    
    @Override
    public Object intercept(Invocation invocation) throws Throwable {
        // 前置处理
        System.out.println("Before query...");
        
        // 执行原方法
        Object result = invocation.proceed();
        
        // 后置处理
        System.out.println("After query...");
        
        return result;
    }
    
    @Override
    public Object plugin(Object target) {
        // 生成代理对象
        return Plugin.wrap(target, this);
    }
    
    @Override
    public void setProperties(Properties properties) {
        // 读取配置属性
        String prop = properties.getProperty("myProp");
    }
}
```

### 注册插件

**XML 配置：**

```xml
<plugins>
    <plugin interceptor="com.example.plugin.MyPlugin">
        <property name="myProp" value="myValue"/>
    </plugin>
</plugins>
```

**Spring Boot 配置：**

```java
@Configuration
public class MyBatisConfig {
    
    @Bean
    public MyPlugin myPlugin() {
        return new MyPlugin();
    }
}
```

## 实用插件示例

### SQL 执行时间监控插件

```java
@Intercepts({
    @Signature(type = StatementHandler.class, method = "query", 
               args = {Statement.class, ResultHandler.class}),
    @Signature(type = StatementHandler.class, method = "update", 
               args = {Statement.class})
})
@Slf4j
public class SqlExecutionTimePlugin implements Interceptor {
    
    private long slowSqlThreshold = 1000; // 慢 SQL 阈值（毫秒）
    
    @Override
    public Object intercept(Invocation invocation) throws Throwable {
        long startTime = System.currentTimeMillis();
        
        try {
            return invocation.proceed();
        } finally {
            long executionTime = System.currentTimeMillis() - startTime;
            
            StatementHandler handler = (StatementHandler) invocation.getTarget();
            BoundSql boundSql = handler.getBoundSql();
            String sql = boundSql.getSql().replaceAll("\\s+", " ");
            
            if (executionTime >= slowSqlThreshold) {
                log.warn("慢 SQL [{}ms]: {}", executionTime, sql);
            } else {
                log.debug("SQL [{}ms]: {}", executionTime, sql);
            }
        }
    }
    
    @Override
    public Object plugin(Object target) {
        return Plugin.wrap(target, this);
    }
    
    @Override
    public void setProperties(Properties properties) {
        String threshold = properties.getProperty("slowSqlThreshold");
        if (threshold != null) {
            this.slowSqlThreshold = Long.parseLong(threshold);
        }
    }
}
```

### SQL 打印插件

```java
@Intercepts({
    @Signature(type = Executor.class, method = "update",
               args = {MappedStatement.class, Object.class}),
    @Signature(type = Executor.class, method = "query",
               args = {MappedStatement.class, Object.class, 
                       RowBounds.class, ResultHandler.class})
})
@Slf4j
public class SqlPrintPlugin implements Interceptor {
    
    @Override
    public Object intercept(Invocation invocation) throws Throwable {
        MappedStatement ms = (MappedStatement) invocation.getArgs()[0];
        Object parameter = invocation.getArgs()[1];
        
        BoundSql boundSql = ms.getBoundSql(parameter);
        String sql = boundSql.getSql();
        
        // 获取完整 SQL（替换参数）
        String fullSql = getFullSql(boundSql, ms.getConfiguration());
        
        log.info("执行 SQL [{}]: {}", ms.getId(), fullSql);
        
        return invocation.proceed();
    }
    
    private String getFullSql(BoundSql boundSql, Configuration configuration) {
        Object parameterObject = boundSql.getParameterObject();
        List<ParameterMapping> parameterMappings = boundSql.getParameterMappings();
        String sql = boundSql.getSql().replaceAll("\\s+", " ");
        
        if (parameterMappings.isEmpty() || parameterObject == null) {
            return sql;
        }
        
        TypeHandlerRegistry typeHandlerRegistry = configuration.getTypeHandlerRegistry();
        
        if (typeHandlerRegistry.hasTypeHandler(parameterObject.getClass())) {
            sql = sql.replaceFirst("\\?", getParameterValue(parameterObject));
        } else {
            MetaObject metaObject = configuration.newMetaObject(parameterObject);
            for (ParameterMapping parameterMapping : parameterMappings) {
                String propertyName = parameterMapping.getProperty();
                if (metaObject.hasGetter(propertyName)) {
                    Object obj = metaObject.getValue(propertyName);
                    sql = sql.replaceFirst("\\?", getParameterValue(obj));
                } else if (boundSql.hasAdditionalParameter(propertyName)) {
                    Object obj = boundSql.getAdditionalParameter(propertyName);
                    sql = sql.replaceFirst("\\?", getParameterValue(obj));
                }
            }
        }
        return sql;
    }
    
    private String getParameterValue(Object obj) {
        if (obj == null) {
            return "null";
        }
        if (obj instanceof String) {
            return "'" + obj + "'";
        }
        if (obj instanceof Date) {
            return "'" + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(obj) + "'";
        }
        return obj.toString();
    }
    
    @Override
    public Object plugin(Object target) {
        return Plugin.wrap(target, this);
    }
    
    @Override
    public void setProperties(Properties properties) {}
}
```

### 数据权限插件

```java
@Intercepts({
    @Signature(type = StatementHandler.class, method = "prepare",
               args = {Connection.class, Integer.class})
})
public class DataPermissionPlugin implements Interceptor {
    
    @Override
    public Object intercept(Invocation invocation) throws Throwable {
        StatementHandler handler = (StatementHandler) invocation.getTarget();
        
        // 获取原始 SQL
        MetaObject metaObject = SystemMetaObject.forObject(handler);
        MappedStatement ms = (MappedStatement) metaObject.getValue("delegate.mappedStatement");
        
        // 只处理 SELECT 语句
        if (ms.getSqlCommandType() != SqlCommandType.SELECT) {
            return invocation.proceed();
        }
        
        BoundSql boundSql = handler.getBoundSql();
        String originalSql = boundSql.getSql();
        
        // 获取当前用户的数据权限
        Long userId = SecurityContextHolder.getCurrentUserId();
        String deptIds = SecurityContextHolder.getCurrentUserDeptIds();
        
        // 添加数据权限条件
        String newSql = addDataPermission(originalSql, userId, deptIds);
        
        // 替换 SQL
        metaObject.setValue("delegate.boundSql.sql", newSql);
        
        return invocation.proceed();
    }
    
    private String addDataPermission(String sql, Long userId, String deptIds) {
        // 简单示例：添加部门过滤条件
        if (deptIds != null && !deptIds.isEmpty()) {
            return sql + " AND dept_id IN (" + deptIds + ")";
        }
        return sql;
    }
    
    @Override
    public Object plugin(Object target) {
        return Plugin.wrap(target, this);
    }
    
    @Override
    public void setProperties(Properties properties) {}
}
```

### 乐观锁插件

```java
@Intercepts({
    @Signature(type = Executor.class, method = "update",
               args = {MappedStatement.class, Object.class})
})
public class OptimisticLockPlugin implements Interceptor {
    
    @Override
    public Object intercept(Invocation invocation) throws Throwable {
        MappedStatement ms = (MappedStatement) invocation.getArgs()[0];
        Object parameter = invocation.getArgs()[1];
        
        // 只处理 UPDATE 语句
        if (ms.getSqlCommandType() != SqlCommandType.UPDATE) {
            return invocation.proceed();
        }
        
        // 检查是否有 version 字段
        MetaObject metaObject = SystemMetaObject.forObject(parameter);
        if (!metaObject.hasGetter("version")) {
            return invocation.proceed();
        }
        
        // 获取当前版本号
        Object currentVersion = metaObject.getValue("version");
        if (currentVersion == null) {
            return invocation.proceed();
        }
        
        // 执行更新
        int rows = (int) invocation.proceed();
        
        // 检查更新结果
        if (rows == 0) {
            throw new OptimisticLockException("数据已被其他用户修改，请刷新后重试");
        }
        
        // 更新版本号
        if (currentVersion instanceof Integer) {
            metaObject.setValue("version", (Integer) currentVersion + 1);
        } else if (currentVersion instanceof Long) {
            metaObject.setValue("version", (Long) currentVersion + 1);
        }
        
        return rows;
    }
    
    @Override
    public Object plugin(Object target) {
        return Plugin.wrap(target, this);
    }
    
    @Override
    public void setProperties(Properties properties) {}
}
```

## PageHelper 分页插件

PageHelper 是最流行的 MyBatis 分页插件。

### 添加依赖

```xml
<dependency>
    <groupId>com.github.pagehelper</groupId>
    <artifactId>pagehelper-spring-boot-starter</artifactId>
    <version>2.1.0</version>
</dependency>
```

### 配置

```yaml
pagehelper:
  helper-dialect: mysql
  reasonable: true
  support-methods-arguments: true
  params: count=countSql
```

### 使用方式

```java
@Service
public class UserService {
    
    @Autowired
    private UserMapper userMapper;
    
    // 方式一：PageHelper.startPage
    public PageInfo<User> getUsers(int pageNum, int pageSize) {
        PageHelper.startPage(pageNum, pageSize);
        List<User> users = userMapper.selectAll();
        return new PageInfo<>(users);
    }
    
    // 方式二：使用 PageInfo
    public PageInfo<User> getUsersWithSort(int pageNum, int pageSize) {
        PageHelper.startPage(pageNum, pageSize, "create_time desc");
        List<User> users = userMapper.selectAll();
        return new PageInfo<>(users);
    }
    
    // 方式三：Lambda 方式
    public PageInfo<User> getUsersLambda(int pageNum, int pageSize) {
        return PageHelper.startPage(pageNum, pageSize)
            .doSelectPageInfo(() -> userMapper.selectAll());
    }
    
    // 方式四：只获取总数
    public long countUsers() {
        return PageHelper.count(() -> userMapper.selectAll());
    }
}
```

### PageInfo 属性

```java
PageInfo<User> pageInfo = new PageInfo<>(users);

pageInfo.getPageNum();      // 当前页码
pageInfo.getPageSize();     // 每页数量
pageInfo.getTotal();        // 总记录数
pageInfo.getPages();        // 总页数
pageInfo.getList();         // 当前页数据
pageInfo.isHasNextPage();   // 是否有下一页
pageInfo.isHasPreviousPage(); // 是否有上一页
pageInfo.isIsFirstPage();   // 是否第一页
pageInfo.isIsLastPage();    // 是否最后一页
```

## MyBatis-Plus

MyBatis-Plus 是 MyBatis 的增强工具，提供了更多便捷功能。

### 添加依赖

```xml
<dependency>
    <groupId>com.baomidou</groupId>
    <artifactId>mybatis-plus-spring-boot3-starter</artifactId>
    <version>3.5.5</version>
</dependency>
```

### 实体类

```java
@Data
@TableName("user")
public class User {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    @TableField("user_name")
    private String name;
    
    private String email;
    
    private Integer age;
    
    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;
    
    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updateTime;
    
    @TableLogic
    private Integer deleted;
    
    @Version
    private Integer version;
}
```

### Mapper 接口

```java
@Mapper
public interface UserMapper extends BaseMapper<User> {
    // 继承 BaseMapper 即可获得基本 CRUD 方法
    // 可以添加自定义方法
}
```

### Service 层

```java
public interface UserService extends IService<User> {
}

@Service
public class UserServiceImpl extends ServiceImpl<UserMapper, User> 
        implements UserService {
}
```

### 使用示例

```java
@Service
public class UserService {
    
    @Autowired
    private UserMapper userMapper;
    
    // 基本 CRUD
    public User getById(Long id) {
        return userMapper.selectById(id);
    }
    
    public List<User> getByIds(List<Long> ids) {
        return userMapper.selectBatchIds(ids);
    }
    
    public int insert(User user) {
        return userMapper.insert(user);
    }
    
    public int update(User user) {
        return userMapper.updateById(user);
    }
    
    public int delete(Long id) {
        return userMapper.deleteById(id);
    }
    
    // 条件查询
    public List<User> search(String name, Integer minAge) {
        LambdaQueryWrapper<User> wrapper = new LambdaQueryWrapper<>();
        wrapper.like(name != null, User::getName, name)
               .ge(minAge != null, User::getAge, minAge)
               .orderByDesc(User::getCreateTime);
        return userMapper.selectList(wrapper);
    }
    
    // 分页查询
    public IPage<User> page(int pageNum, int pageSize) {
        Page<User> page = new Page<>(pageNum, pageSize);
        return userMapper.selectPage(page, null);
    }
    
    // 条件更新
    public int updateByCondition(String name, Integer status) {
        LambdaUpdateWrapper<User> wrapper = new LambdaUpdateWrapper<>();
        wrapper.eq(User::getName, name)
               .set(User::getStatus, status);
        return userMapper.update(null, wrapper);
    }
}
```

### 常用注解

| 注解 | 描述 |
|------|------|
| `@TableName` | 表名映射 |
| `@TableId` | 主键映射 |
| `@TableField` | 字段映射 |
| `@TableLogic` | 逻辑删除 |
| `@Version` | 乐观锁版本号 |
| `@EnumValue` | 枚举值映射 |

### 内置插件

```java
@Configuration
public class MyBatisPlusConfig {
    
    @Bean
    public MybatisPlusInterceptor mybatisPlusInterceptor() {
        MybatisPlusInterceptor interceptor = new MybatisPlusInterceptor();
        
        // 分页插件
        interceptor.addInnerInterceptor(
            new PaginationInnerInterceptor(DbType.MYSQL));
        
        // 乐观锁插件
        interceptor.addInnerInterceptor(new OptimisticLockerInnerInterceptor());
        
        // 防全表更新删除插件
        interceptor.addInnerInterceptor(new BlockAttackInnerInterceptor());
        
        return interceptor;
    }
}
```

## 相关链接

- [核心概念](/docs/mybatis/core-concepts) - 了解 MyBatis 架构
- [配置详解](/docs/mybatis/configuration) - 插件配置
- [Spring 集成](/docs/mybatis/spring-integration) - Spring Boot 中使用插件

---

**最后更新**: 2025 年 12 月
