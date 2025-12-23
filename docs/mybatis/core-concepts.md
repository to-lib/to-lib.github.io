---
sidebar_position: 2
title: MyBatis 核心概念
---

# MyBatis 核心概念

本章介绍 MyBatis 的核心架构、组件和工作原理，帮助你深入理解 MyBatis 的设计思想。

## MyBatis 简介

MyBatis 是一款优秀的持久层框架，它支持自定义 SQL、存储过程以及高级映射。MyBatis 免除了几乎所有的 JDBC 代码以及设置参数和获取结果集的工作。

### 核心特点

- **SQL 与代码分离** - SQL 语句写在 XML 文件或注解中，便于维护
- **灵活的映射** - 支持简单的 POJO 映射到复杂的关联映射
- **动态 SQL** - 强大的动态 SQL 生成能力
- **插件机制** - 可扩展的插件架构
- **缓存支持** - 内置一级、二级缓存

## 架构概述

MyBatis 的整体架构分为三层：

```
┌─────────────────────────────────────────────────────────┐
│                    接口层 (API Layer)                    │
│              SqlSession / SqlSessionFactory              │
├─────────────────────────────────────────────────────────┤
│                  核心处理层 (Core Layer)                  │
│  配置解析 │ SQL解析 │ SQL执行 │ 结果映射 │ 插件机制      │
├─────────────────────────────────────────────────────────┤
│                  基础支撑层 (Support Layer)               │
│  连接管理 │ 事务管理 │ 缓存机制 │ 类型处理 │ 日志模块     │
└─────────────────────────────────────────────────────────┘
```

### 接口层

提供给开发者使用的 API，主要包括：
- `SqlSessionFactory` - 创建 SqlSession 的工厂
- `SqlSession` - 执行 SQL 的核心接口
- `Mapper` 接口 - 定义数据访问方法

### 核心处理层

负责 SQL 的解析、执行和结果映射：
- **配置解析** - 解析 mybatis-config.xml 和 Mapper XML
- **SQL 解析** - 解析动态 SQL，生成最终 SQL
- **SQL 执行** - 通过 Executor 执行 SQL
- **结果映射** - 将结果集映射为 Java 对象

### 基础支撑层

提供基础功能支持：
- **连接管理** - 数据库连接池管理
- **事务管理** - 事务的提交和回滚
- **缓存机制** - 一级缓存和二级缓存
- **类型处理** - Java 类型与 JDBC 类型转换

## 核心组件

### SqlSessionFactoryBuilder

用于创建 `SqlSessionFactory` 的构建器，通常只在应用启动时使用一次。

```java
// 从 XML 配置文件创建
String resource = "mybatis-config.xml";
InputStream inputStream = Resources.getResourceAsStream(resource);
SqlSessionFactory sqlSessionFactory = new SqlSessionFactoryBuilder().build(inputStream);

// 从 Java 配置创建
DataSource dataSource = getDataSource();
TransactionFactory transactionFactory = new JdbcTransactionFactory();
Environment environment = new Environment("development", transactionFactory, dataSource);
Configuration configuration = new Configuration(environment);
configuration.addMapper(UserMapper.class);
SqlSessionFactory sqlSessionFactory = new SqlSessionFactoryBuilder().build(configuration);
```

### SqlSessionFactory

创建 `SqlSession` 的工厂，是线程安全的，应该在应用中作为单例存在。

```java
// SqlSessionFactory 的核心方法
public interface SqlSessionFactory {
    SqlSession openSession();                    // 默认不自动提交
    SqlSession openSession(boolean autoCommit);  // 指定是否自动提交
    SqlSession openSession(Connection connection);
    SqlSession openSession(TransactionIsolationLevel level);
    SqlSession openSession(ExecutorType execType);
    Configuration getConfiguration();
}
```

**生命周期**：应用级别，整个应用运行期间只需要一个实例。

### SqlSession

执行 SQL 的核心接口，提供了所有执行 SQL 语句、获取映射器和管理事务的方法。

```java
public interface SqlSession extends Closeable {
    // 查询方法
    <T> T selectOne(String statement);
    <T> T selectOne(String statement, Object parameter);
    <E> List<E> selectList(String statement);
    <E> List<E> selectList(String statement, Object parameter);
    <K, V> Map<K, V> selectMap(String statement, String mapKey);
    <T> Cursor<T> selectCursor(String statement);
    void select(String statement, ResultHandler handler);
    
    // 增删改方法
    int insert(String statement);
    int insert(String statement, Object parameter);
    int update(String statement);
    int update(String statement, Object parameter);
    int delete(String statement);
    int delete(String statement, Object parameter);
    
    // 事务方法
    void commit();
    void commit(boolean force);
    void rollback();
    void rollback(boolean force);
    
    // 获取 Mapper
    <T> T getMapper(Class<T> type);
    
    // 获取连接
    Connection getConnection();
}
```

**生命周期**：请求/方法级别，每次数据库操作都应该创建新的 SqlSession。

```java
// 推荐使用 try-with-resources
try (SqlSession session = sqlSessionFactory.openSession()) {
    UserMapper mapper = session.getMapper(UserMapper.class);
    User user = mapper.selectById(1L);
    // 处理业务逻辑
    session.commit();
} // 自动关闭
```

### Mapper 接口

定义数据访问方法的接口，MyBatis 会为其生成代理实现。

```java
public interface UserMapper {
    
    @Select("SELECT * FROM user WHERE id = #{id}")
    User selectById(Long id);
    
    @Insert("INSERT INTO user(name, email) VALUES(#{name}, #{email})")
    @Options(useGeneratedKeys = true, keyProperty = "id")
    int insert(User user);
    
    @Update("UPDATE user SET name = #{name} WHERE id = #{id}")
    int update(User user);
    
    @Delete("DELETE FROM user WHERE id = #{id}")
    int delete(Long id);
    
    // 也可以使用 XML 映射
    List<User> selectByCondition(UserQuery query);
}
```

### Configuration

MyBatis 的核心配置类，包含所有配置信息。

```java
public class Configuration {
    // 环境配置
    protected Environment environment;
    
    // 全局设置
    protected boolean cacheEnabled = true;
    protected boolean lazyLoadingEnabled = false;
    protected boolean useGeneratedKeys = false;
    protected Integer defaultStatementTimeout;
    
    // 类型别名注册
    protected final TypeAliasRegistry typeAliasRegistry;
    
    // 类型处理器注册
    protected final TypeHandlerRegistry typeHandlerRegistry;
    
    // Mapper 注册
    protected final MapperRegistry mapperRegistry;
    
    // 已映射的语句
    protected final Map<String, MappedStatement> mappedStatements;
    
    // 缓存
    protected final Map<String, Cache> caches;
    
    // 结果映射
    protected final Map<String, ResultMap> resultMaps;
    
    // 插件
    protected final InterceptorChain interceptorChain;
}
```

## MyBatis vs JDBC

### 传统 JDBC 代码

```java
public User findById(Long id) {
    Connection conn = null;
    PreparedStatement ps = null;
    ResultSet rs = null;
    User user = null;
    
    try {
        // 1. 获取连接
        conn = dataSource.getConnection();
        
        // 2. 创建 PreparedStatement
        String sql = "SELECT id, name, email, age FROM user WHERE id = ?";
        ps = conn.prepareStatement(sql);
        
        // 3. 设置参数
        ps.setLong(1, id);
        
        // 4. 执行查询
        rs = ps.executeQuery();
        
        // 5. 处理结果集
        if (rs.next()) {
            user = new User();
            user.setId(rs.getLong("id"));
            user.setName(rs.getString("name"));
            user.setEmail(rs.getString("email"));
            user.setAge(rs.getInt("age"));
        }
    } catch (SQLException e) {
        throw new RuntimeException(e);
    } finally {
        // 6. 关闭资源
        try {
            if (rs != null) rs.close();
            if (ps != null) ps.close();
            if (conn != null) conn.close();
        } catch (SQLException e) {
            // ignore
        }
    }
    return user;
}
```

### MyBatis 代码

```java
// Mapper 接口
public interface UserMapper {
    User findById(Long id);
}
```

```xml
<!-- UserMapper.xml -->
<mapper namespace="com.example.mapper.UserMapper">
    <select id="findById" resultType="User">
        SELECT id, name, email, age FROM user WHERE id = #{id}
    </select>
</mapper>
```

```java
// 使用
try (SqlSession session = sqlSessionFactory.openSession()) {
    UserMapper mapper = session.getMapper(UserMapper.class);
    User user = mapper.findById(1L);
}
```

### 对比总结

| 方面 | JDBC | MyBatis |
|------|------|---------|
| 代码量 | 大量样板代码 | 简洁，只需定义接口和 SQL |
| SQL 管理 | 散落在代码中 | 集中在 XML 或注解中 |
| 参数设置 | 手动设置每个参数 | 自动参数映射 |
| 结果映射 | 手动从 ResultSet 取值 | 自动映射到对象 |
| 资源管理 | 手动管理连接、关闭资源 | 框架自动管理 |
| 异常处理 | 需要处理 SQLException | 统一转换为运行时异常 |
| 缓存 | 需要自己实现 | 内置一级、二级缓存 |
| 动态 SQL | 字符串拼接，易出错 | 强大的动态 SQL 标签 |

## 工作流程

### 初始化流程

```
1. 加载配置文件 (mybatis-config.xml)
         ↓
2. 解析配置，创建 Configuration 对象
         ↓
3. 解析 Mapper XML，注册 MappedStatement
         ↓
4. 创建 SqlSessionFactory
         ↓
5. 应用启动完成，等待请求
```

### SQL 执行流程

```
1. 获取 SqlSession
         ↓
2. 获取 Mapper 代理对象
         ↓
3. 调用 Mapper 方法
         ↓
4. MapperProxy 拦截方法调用
         ↓
5. 根据方法名找到 MappedStatement
         ↓
6. Executor 执行 SQL
         ↓
7. StatementHandler 处理 SQL 语句
         ↓
8. ParameterHandler 设置参数
         ↓
9. 执行 SQL，获取 ResultSet
         ↓
10. ResultSetHandler 映射结果
         ↓
11. 返回结果对象
```

### 核心执行组件

```java
// Executor - SQL 执行器
public interface Executor {
    int update(MappedStatement ms, Object parameter);
    <E> List<E> query(MappedStatement ms, Object parameter, 
                      RowBounds rowBounds, ResultHandler resultHandler);
    void commit(boolean required);
    void rollback(boolean required);
    void close(boolean forceRollback);
}

// StatementHandler - 语句处理器
public interface StatementHandler {
    Statement prepare(Connection connection, Integer transactionTimeout);
    void parameterize(Statement statement);
    int update(Statement statement);
    <E> List<E> query(Statement statement, ResultHandler resultHandler);
}

// ParameterHandler - 参数处理器
public interface ParameterHandler {
    Object getParameterObject();
    void setParameters(PreparedStatement ps);
}

// ResultSetHandler - 结果集处理器
public interface ResultSetHandler {
    <E> List<E> handleResultSets(Statement stmt);
    void handleOutputParameters(CallableStatement cs);
}
```

## Executor 类型

MyBatis 提供三种 Executor 实现：

### SimpleExecutor

默认的执行器，每次执行都会创建新的 Statement。

```java
// 每次查询都创建新的 PreparedStatement
SqlSession session = sqlSessionFactory.openSession(ExecutorType.SIMPLE);
```

### ReuseExecutor

重用 Statement 的执行器，相同的 SQL 会重用 PreparedStatement。

```java
// 相同 SQL 重用 PreparedStatement
SqlSession session = sqlSessionFactory.openSession(ExecutorType.REUSE);
```

### BatchExecutor

批量执行器，用于批量更新操作。

```java
// 批量执行，适合大量 insert/update
SqlSession session = sqlSessionFactory.openSession(ExecutorType.BATCH);
try {
    UserMapper mapper = session.getMapper(UserMapper.class);
    for (User user : users) {
        mapper.insert(user);
    }
    session.flushStatements(); // 刷新批量语句
    session.commit();
} finally {
    session.close();
}
```

### 对比

| Executor 类型 | 特点 | 适用场景 |
|--------------|------|----------|
| SIMPLE | 每次创建新 Statement | 默认，适合大多数场景 |
| REUSE | 重用 Statement | 相同 SQL 频繁执行 |
| BATCH | 批量执行 | 大量 insert/update 操作 |

## 类型处理器 (TypeHandler)

TypeHandler 负责 Java 类型与 JDBC 类型之间的转换。

### 内置类型处理器

MyBatis 内置了大量类型处理器：

| Java 类型 | JDBC 类型 | TypeHandler |
|-----------|-----------|-------------|
| String | VARCHAR | StringTypeHandler |
| Integer/int | INTEGER | IntegerTypeHandler |
| Long/long | BIGINT | LongTypeHandler |
| Double/double | DOUBLE | DoubleTypeHandler |
| Boolean/boolean | BOOLEAN | BooleanTypeHandler |
| Date | TIMESTAMP | DateTypeHandler |
| LocalDateTime | TIMESTAMP | LocalDateTimeTypeHandler |
| Enum | VARCHAR/INTEGER | EnumTypeHandler |

### 自定义类型处理器

```java
@MappedTypes(MyEnum.class)
@MappedJdbcTypes(JdbcType.VARCHAR)
public class MyEnumTypeHandler extends BaseTypeHandler<MyEnum> {
    
    @Override
    public void setNonNullParameter(PreparedStatement ps, int i, 
            MyEnum parameter, JdbcType jdbcType) throws SQLException {
        ps.setString(i, parameter.getCode());
    }
    
    @Override
    public MyEnum getNullableResult(ResultSet rs, String columnName) 
            throws SQLException {
        String code = rs.getString(columnName);
        return MyEnum.fromCode(code);
    }
    
    @Override
    public MyEnum getNullableResult(ResultSet rs, int columnIndex) 
            throws SQLException {
        String code = rs.getString(columnIndex);
        return MyEnum.fromCode(code);
    }
    
    @Override
    public MyEnum getNullableResult(CallableStatement cs, int columnIndex) 
            throws SQLException {
        String code = cs.getString(columnIndex);
        return MyEnum.fromCode(code);
    }
}
```

注册自定义类型处理器：

```xml
<typeHandlers>
    <typeHandler handler="com.example.handler.MyEnumTypeHandler"/>
</typeHandlers>
```

## 对象工厂 (ObjectFactory)

ObjectFactory 负责创建结果对象实例。

```java
public class CustomObjectFactory extends DefaultObjectFactory {
    
    @Override
    public <T> T create(Class<T> type) {
        // 自定义对象创建逻辑
        return super.create(type);
    }
    
    @Override
    public <T> T create(Class<T> type, List<Class<?>> constructorArgTypes, 
            List<Object> constructorArgs) {
        return super.create(type, constructorArgTypes, constructorArgs);
    }
    
    @Override
    public void setProperties(Properties properties) {
        super.setProperties(properties);
    }
}
```

## 总结

MyBatis 的核心概念包括：

1. **SqlSessionFactory** - 应用级单例，创建 SqlSession
2. **SqlSession** - 请求级，执行 SQL 的核心接口
3. **Mapper** - 数据访问接口，由 MyBatis 生成代理实现
4. **Configuration** - 核心配置，包含所有配置信息
5. **Executor** - SQL 执行器，有 SIMPLE、REUSE、BATCH 三种
6. **TypeHandler** - 类型处理器，Java 与 JDBC 类型转换

理解这些核心概念是掌握 MyBatis 的基础，后续章节将深入讲解配置、映射、动态 SQL 等内容。

## 相关链接

- [配置详解](/docs/mybatis/configuration) - 深入了解 MyBatis 配置
- [XML 映射](/docs/mybatis/xml-mapping) - 学习 Mapper XML 编写
- [Spring 集成](/docs/mybatis/spring-integration) - 在 Spring 中使用 MyBatis

---

**最后更新**: 2025 年 12 月
