---
sidebar_position: 3
title: MyBatis 配置详解
---

# MyBatis 配置详解

本章详细介绍 MyBatis 的配置文件结构和常用配置项，帮助你正确配置 MyBatis 应用。

## 配置文件结构

MyBatis 的配置文件 `mybatis-config.xml` 包含以下顶级元素（按顺序）：

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE configuration
  PUBLIC "-//mybatis.org//DTD Config 3.0//EN"
  "https://mybatis.org/dtd/mybatis-3-config.dtd">
<configuration>
    <!-- 属性配置 -->
    <properties/>
    
    <!-- 全局设置 -->
    <settings/>
    
    <!-- 类型别名 -->
    <typeAliases/>
    
    <!-- 类型处理器 -->
    <typeHandlers/>
    
    <!-- 对象工厂 -->
    <objectFactory/>
    
    <!-- 对象包装工厂 -->
    <objectWrapperFactory/>
    
    <!-- 反射工厂 -->
    <reflectorFactory/>
    
    <!-- 插件 -->
    <plugins/>
    
    <!-- 环境配置 -->
    <environments/>
    
    <!-- 数据库厂商标识 -->
    <databaseIdProvider/>
    
    <!-- 映射器 -->
    <mappers/>
</configuration>
```

> [!IMPORTANT]
> 配置元素必须按照上述顺序排列，否则会报错。

## properties（属性配置）

用于定义可外部化的属性，支持从外部文件加载。

### 内联定义

```xml
<properties>
    <property name="driver" value="com.mysql.cj.jdbc.Driver"/>
    <property name="url" value="jdbc:mysql://localhost:3306/mydb"/>
    <property name="username" value="root"/>
    <property name="password" value="123456"/>
</properties>
```

### 外部文件

```xml
<!-- 从 classpath 加载 -->
<properties resource="db.properties"/>

<!-- 从文件系统加载 -->
<properties url="file:///path/to/db.properties"/>
```

`db.properties` 文件：

```properties
driver=com.mysql.cj.jdbc.Driver
url=jdbc:mysql://localhost:3306/mydb?useSSL=false&serverTimezone=UTC
username=root
password=123456
```

### 使用属性

```xml
<dataSource type="POOLED">
    <property name="driver" value="${driver}"/>
    <property name="url" value="${url}"/>
    <property name="username" value="${username}"/>
    <property name="password" value="${password}"/>
</dataSource>
```

### 属性优先级

属性加载顺序（后加载的覆盖先加载的）：

1. `<properties>` 元素内定义的属性
2. `resource` 或 `url` 指定的外部文件属性
3. 方法参数传递的属性

```java
// 通过方法参数传递属性（优先级最高）
Properties props = new Properties();
props.setProperty("username", "admin");
SqlSessionFactory factory = new SqlSessionFactoryBuilder()
    .build(inputStream, props);
```

### 默认值

MyBatis 3.4.2+ 支持属性默认值：

```xml
<properties>
    <!-- 启用默认值功能 -->
    <property name="org.apache.ibatis.parsing.PropertyParser.enable-default-value" 
              value="true"/>
</properties>

<dataSource type="POOLED">
    <!-- 使用默认值语法 -->
    <property name="username" value="${db.username:root}"/>
    <property name="password" value="${db.password:123456}"/>
</dataSource>
```

## settings（全局设置）

settings 是 MyBatis 中最重要的配置，影响 MyBatis 的运行时行为。

### 完整配置示例

```xml
<settings>
    <!-- 缓存 -->
    <setting name="cacheEnabled" value="true"/>
    
    <!-- 延迟加载 -->
    <setting name="lazyLoadingEnabled" value="false"/>
    <setting name="aggressiveLazyLoading" value="false"/>
    
    <!-- 多结果集 -->
    <setting name="multipleResultSetsEnabled" value="true"/>
    
    <!-- 列标签 -->
    <setting name="useColumnLabel" value="true"/>
    
    <!-- 自增主键 -->
    <setting name="useGeneratedKeys" value="false"/>
    
    <!-- 自动映射 -->
    <setting name="autoMappingBehavior" value="PARTIAL"/>
    <setting name="autoMappingUnknownColumnBehavior" value="WARNING"/>
    
    <!-- 默认执行器 -->
    <setting name="defaultExecutorType" value="SIMPLE"/>
    
    <!-- 超时设置 -->
    <setting name="defaultStatementTimeout" value="25"/>
    <setting name="defaultFetchSize" value="100"/>
    
    <!-- 结果集类型 -->
    <setting name="defaultResultSetType" value="DEFAULT"/>
    
    <!-- 驼峰命名映射 -->
    <setting name="mapUnderscoreToCamelCase" value="true"/>
    
    <!-- 本地缓存作用域 -->
    <setting name="localCacheScope" value="SESSION"/>
    
    <!-- 空值 JDBC 类型 -->
    <setting name="jdbcTypeForNull" value="OTHER"/>
    
    <!-- 延迟加载触发方法 -->
    <setting name="lazyLoadTriggerMethods" value="equals,clone,hashCode,toString"/>
    
    <!-- 安全的结果处理器 -->
    <setting name="safeResultHandlerEnabled" value="true"/>
    <setting name="safeRowBoundsEnabled" value="false"/>
    
    <!-- 日志实现 -->
    <setting name="logImpl" value="SLF4J"/>
    
    <!-- 日志前缀 -->
    <setting name="logPrefix" value="mybatis."/>
    
    <!-- 空行为 -->
    <setting name="callSettersOnNulls" value="false"/>
    <setting name="returnInstanceForEmptyRow" value="false"/>
</settings>
```

### 常用设置详解

| 设置名 | 描述 | 默认值 | 可选值 |
|--------|------|--------|--------|
| `cacheEnabled` | 全局开启二级缓存 | true | true/false |
| `lazyLoadingEnabled` | 延迟加载开关 | false | true/false |
| `useGeneratedKeys` | 使用 JDBC 自增主键 | false | true/false |
| `mapUnderscoreToCamelCase` | 下划线转驼峰 | false | true/false |
| `defaultExecutorType` | 默认执行器类型 | SIMPLE | SIMPLE/REUSE/BATCH |
| `defaultStatementTimeout` | SQL 超时时间（秒） | null | 正整数 |
| `localCacheScope` | 本地缓存作用域 | SESSION | SESSION/STATEMENT |
| `logImpl` | 日志实现 | 未设置 | SLF4J/LOG4J/LOG4J2/... |

### 驼峰命名映射

开启后，数据库字段 `user_name` 自动映射到 Java 属性 `userName`：

```xml
<setting name="mapUnderscoreToCamelCase" value="true"/>
```

```java
// 数据库字段: user_name, create_time
// Java 属性: userName, createTime
public class User {
    private String userName;      // 自动映射 user_name
    private Date createTime;      // 自动映射 create_time
}
```

### 日志配置

```xml
<!-- 指定日志实现 -->
<setting name="logImpl" value="SLF4J"/>
```

可选的日志实现：
- `SLF4J` - 推荐，需要 slf4j-api 依赖
- `LOG4J` - Log4j 1.x
- `LOG4J2` - Log4j 2.x
- `JDK_LOGGING` - JDK 内置日志
- `COMMONS_LOGGING` - Apache Commons Logging
- `STDOUT_LOGGING` - 标准输出
- `NO_LOGGING` - 不记录日志

## typeAliases（类型别名）

类型别名为 Java 类型设置简短的名字，减少 XML 配置中的冗余。

### 单个别名

```xml
<typeAliases>
    <typeAlias alias="User" type="com.example.entity.User"/>
    <typeAlias alias="Order" type="com.example.entity.Order"/>
</typeAliases>
```

### 包扫描

```xml
<typeAliases>
    <!-- 扫描包下所有类，别名为类名（首字母小写） -->
    <package name="com.example.entity"/>
</typeAliases>
```

### @Alias 注解

```java
@Alias("user")
public class User {
    // ...
}
```

### 内置别名

MyBatis 内置了常用类型的别名：

| 别名 | Java 类型 |
|------|-----------|
| _byte | byte |
| _short | short |
| _int | int |
| _long | long |
| _float | float |
| _double | double |
| _boolean | boolean |
| string | String |
| byte | Byte |
| short | Short |
| int/integer | Integer |
| long | Long |
| float | Float |
| double | Double |
| boolean | Boolean |
| date | Date |
| object | Object |
| map | Map |
| hashmap | HashMap |
| list | List |
| arraylist | ArrayList |
| collection | Collection |

## typeHandlers（类型处理器）

类型处理器负责 Java 类型与 JDBC 类型之间的转换。

### 注册自定义类型处理器

```xml
<typeHandlers>
    <!-- 单个注册 -->
    <typeHandler handler="com.example.handler.JsonTypeHandler"/>
    
    <!-- 指定处理的 Java 类型和 JDBC 类型 -->
    <typeHandler handler="com.example.handler.MyEnumTypeHandler"
                 javaType="com.example.enums.Status"
                 jdbcType="VARCHAR"/>
    
    <!-- 包扫描 -->
    <package name="com.example.handler"/>
</typeHandlers>
```

### 自定义 JSON 类型处理器

```java
@MappedTypes(JsonObject.class)
@MappedJdbcTypes(JdbcType.VARCHAR)
public class JsonTypeHandler extends BaseTypeHandler<JsonObject> {
    
    private static final ObjectMapper mapper = new ObjectMapper();
    
    @Override
    public void setNonNullParameter(PreparedStatement ps, int i, 
            JsonObject parameter, JdbcType jdbcType) throws SQLException {
        try {
            ps.setString(i, mapper.writeValueAsString(parameter));
        } catch (JsonProcessingException e) {
            throw new SQLException("JSON 序列化失败", e);
        }
    }
    
    @Override
    public JsonObject getNullableResult(ResultSet rs, String columnName) 
            throws SQLException {
        return parseJson(rs.getString(columnName));
    }
    
    @Override
    public JsonObject getNullableResult(ResultSet rs, int columnIndex) 
            throws SQLException {
        return parseJson(rs.getString(columnIndex));
    }
    
    @Override
    public JsonObject getNullableResult(CallableStatement cs, int columnIndex) 
            throws SQLException {
        return parseJson(cs.getString(columnIndex));
    }
    
    private JsonObject parseJson(String json) throws SQLException {
        if (json == null) return null;
        try {
            return mapper.readValue(json, JsonObject.class);
        } catch (JsonProcessingException e) {
            throw new SQLException("JSON 解析失败", e);
        }
    }
}
```

## plugins（插件）

插件可以拦截 MyBatis 的核心方法调用。

```xml
<plugins>
    <!-- 分页插件 -->
    <plugin interceptor="com.github.pagehelper.PageInterceptor">
        <property name="helperDialect" value="mysql"/>
        <property name="reasonable" value="true"/>
    </plugin>
    
    <!-- 自定义插件 -->
    <plugin interceptor="com.example.plugin.SqlLogPlugin">
        <property name="showSql" value="true"/>
    </plugin>
</plugins>
```

详细内容请参考 [插件机制](/docs/mybatis/plugins)。

## environments（环境配置）

MyBatis 支持配置多个环境，但每个 SqlSessionFactory 只能选择一个。

```xml
<environments default="development">
    <!-- 开发环境 -->
    <environment id="development">
        <transactionManager type="JDBC"/>
        <dataSource type="POOLED">
            <property name="driver" value="${dev.driver}"/>
            <property name="url" value="${dev.url}"/>
            <property name="username" value="${dev.username}"/>
            <property name="password" value="${dev.password}"/>
        </dataSource>
    </environment>
    
    <!-- 生产环境 -->
    <environment id="production">
        <transactionManager type="JDBC"/>
        <dataSource type="POOLED">
            <property name="driver" value="${prod.driver}"/>
            <property name="url" value="${prod.url}"/>
            <property name="username" value="${prod.username}"/>
            <property name="password" value="${prod.password}"/>
        </dataSource>
    </environment>
</environments>
```

### 事务管理器类型

| 类型 | 描述 |
|------|------|
| `JDBC` | 直接使用 JDBC 的提交和回滚，依赖数据源连接管理事务 |
| `MANAGED` | 不提交或回滚连接，让容器管理事务（如 Spring） |

### 数据源类型

| 类型 | 描述 |
|------|------|
| `UNPOOLED` | 每次请求都创建新连接，不使用连接池 |
| `POOLED` | 使用连接池，推荐用于生产环境 |
| `JNDI` | 从 JNDI 获取数据源，用于 EJB 或应用服务器 |

### POOLED 数据源配置

```xml
<dataSource type="POOLED">
    <!-- 基本配置 -->
    <property name="driver" value="com.mysql.cj.jdbc.Driver"/>
    <property name="url" value="jdbc:mysql://localhost:3306/mydb"/>
    <property name="username" value="root"/>
    <property name="password" value="123456"/>
    
    <!-- 连接池配置 -->
    <property name="poolMaximumActiveConnections" value="10"/>
    <property name="poolMaximumIdleConnections" value="5"/>
    <property name="poolMaximumCheckoutTime" value="20000"/>
    <property name="poolTimeToWait" value="20000"/>
    <property name="poolMaximumLocalBadConnectionTolerance" value="3"/>
    <property name="poolPingQuery" value="SELECT 1"/>
    <property name="poolPingEnabled" value="false"/>
    <property name="poolPingConnectionsNotUsedFor" value="0"/>
</dataSource>
```

| 属性 | 描述 | 默认值 |
|------|------|--------|
| `poolMaximumActiveConnections` | 最大活动连接数 | 10 |
| `poolMaximumIdleConnections` | 最大空闲连接数 | 5 |
| `poolMaximumCheckoutTime` | 连接最大借出时间（毫秒） | 20000 |
| `poolTimeToWait` | 获取连接等待时间（毫秒） | 20000 |
| `poolPingQuery` | 连接检测 SQL | 无 |
| `poolPingEnabled` | 是否启用连接检测 | false |

## databaseIdProvider（数据库厂商标识）

用于支持多数据库厂商，根据不同数据库执行不同 SQL。

```xml
<databaseIdProvider type="DB_VENDOR">
    <property name="MySQL" value="mysql"/>
    <property name="Oracle" value="oracle"/>
    <property name="PostgreSQL" value="postgresql"/>
</databaseIdProvider>
```

在 Mapper XML 中使用：

```xml
<!-- MySQL 专用 -->
<select id="selectAll" resultType="User" databaseId="mysql">
    SELECT * FROM user LIMIT #{limit}
</select>

<!-- Oracle 专用 -->
<select id="selectAll" resultType="User" databaseId="oracle">
    SELECT * FROM user WHERE ROWNUM &lt;= #{limit}
</select>

<!-- 通用（无 databaseId） -->
<select id="selectById" resultType="User">
    SELECT * FROM user WHERE id = #{id}
</select>
```

## mappers（映射器）

配置 Mapper XML 文件或 Mapper 接口的位置。

### 资源路径

```xml
<mappers>
    <!-- 相对于 classpath -->
    <mapper resource="mapper/UserMapper.xml"/>
    <mapper resource="mapper/OrderMapper.xml"/>
</mappers>
```

### 文件路径

```xml
<mappers>
    <!-- 绝对路径 -->
    <mapper url="file:///path/to/mapper/UserMapper.xml"/>
</mappers>
```

### 接口类

```xml
<mappers>
    <!-- 指定 Mapper 接口 -->
    <mapper class="com.example.mapper.UserMapper"/>
</mappers>
```

### 包扫描

```xml
<mappers>
    <!-- 扫描包下所有 Mapper 接口 -->
    <package name="com.example.mapper"/>
</mappers>
```

> [!TIP]
> 使用包扫描时，Mapper XML 文件需要与接口在同一包下，或者在 resources 目录下的相同路径。

## 完整配置示例

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE configuration
  PUBLIC "-//mybatis.org//DTD Config 3.0//EN"
  "https://mybatis.org/dtd/mybatis-3-config.dtd">
<configuration>
    <!-- 加载外部属性文件 -->
    <properties resource="db.properties">
        <property name="org.apache.ibatis.parsing.PropertyParser.enable-default-value" 
                  value="true"/>
    </properties>
    
    <!-- 全局设置 -->
    <settings>
        <setting name="cacheEnabled" value="true"/>
        <setting name="lazyLoadingEnabled" value="false"/>
        <setting name="mapUnderscoreToCamelCase" value="true"/>
        <setting name="useGeneratedKeys" value="true"/>
        <setting name="defaultExecutorType" value="SIMPLE"/>
        <setting name="defaultStatementTimeout" value="30"/>
        <setting name="logImpl" value="SLF4J"/>
    </settings>
    
    <!-- 类型别名 -->
    <typeAliases>
        <package name="com.example.entity"/>
    </typeAliases>
    
    <!-- 类型处理器 -->
    <typeHandlers>
        <package name="com.example.handler"/>
    </typeHandlers>
    
    <!-- 插件 -->
    <plugins>
        <plugin interceptor="com.github.pagehelper.PageInterceptor">
            <property name="helperDialect" value="mysql"/>
        </plugin>
    </plugins>
    
    <!-- 环境配置 -->
    <environments default="development">
        <environment id="development">
            <transactionManager type="JDBC"/>
            <dataSource type="POOLED">
                <property name="driver" value="${driver}"/>
                <property name="url" value="${url}"/>
                <property name="username" value="${username}"/>
                <property name="password" value="${password}"/>
                <property name="poolMaximumActiveConnections" value="20"/>
                <property name="poolMaximumIdleConnections" value="10"/>
            </dataSource>
        </environment>
    </environments>
    
    <!-- 映射器 -->
    <mappers>
        <package name="com.example.mapper"/>
    </mappers>
</configuration>
```

## 相关链接

- [核心概念](/docs/mybatis/core-concepts) - 了解 MyBatis 架构
- [XML 映射](/docs/mybatis/xml-mapping) - 学习 Mapper XML 编写
- [Spring 集成](/docs/mybatis/spring-integration) - Spring Boot 中的配置方式

---

**最后更新**: 2025 年 12 月
