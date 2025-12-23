---
sidebar_position: 6
title: MyBatis 注解映射
---

# MyBatis 注解映射

MyBatis 支持使用注解来定义 SQL 映射，无需编写 XML 文件。本章介绍常用注解及其使用方法。

## 注解概览

### CRUD 注解

| 注解 | 描述 |
|------|------|
| `@Select` | 查询语句 |
| `@Insert` | 插入语句 |
| `@Update` | 更新语句 |
| `@Delete` | 删除语句 |

### 结果映射注解

| 注解 | 描述 |
|------|------|
| `@Results` | 结果映射集合 |
| `@Result` | 单个字段映射 |
| `@One` | 一对一关联 |
| `@Many` | 一对多关联 |
| `@ResultMap` | 引用 XML 中的 resultMap |

### 动态 SQL 注解

| 注解 | 描述 |
|------|------|
| `@SelectProvider` | 动态查询 SQL |
| `@InsertProvider` | 动态插入 SQL |
| `@UpdateProvider` | 动态更新 SQL |
| `@DeleteProvider` | 动态删除 SQL |

### 其他注解

| 注解 | 描述 |
|------|------|
| `@Options` | 配置选项（主键、超时等） |
| `@Param` | 参数命名 |
| `@MapKey` | Map 结果的 key |
| `@Flush` | 刷新语句 |
| `@CacheNamespace` | 缓存配置 |

## 基本 CRUD 注解

### @Select 查询

```java
public interface UserMapper {
    
    // 简单查询
    @Select("SELECT * FROM user WHERE id = #{id}")
    User selectById(Long id);
    
    // 查询列表
    @Select("SELECT * FROM user")
    List<User> selectAll();
    
    // 条件查询
    @Select("SELECT * FROM user WHERE name = #{name} AND age = #{age}")
    List<User> selectByNameAndAge(@Param("name") String name, 
                                   @Param("age") Integer age);
    
    // 模糊查询
    @Select("SELECT * FROM user WHERE name LIKE CONCAT('%', #{name}, '%')")
    List<User> selectByNameLike(String name);
    
    // 返回 Map
    @Select("SELECT * FROM user WHERE id = #{id}")
    Map<String, Object> selectByIdAsMap(Long id);
    
    // 返回 Map 集合，以 id 为 key
    @Select("SELECT * FROM user")
    @MapKey("id")
    Map<Long, User> selectAllAsMap();
}
```

### @Insert 插入

```java
public interface UserMapper {
    
    // 简单插入
    @Insert("INSERT INTO user(name, email, age) VALUES(#{name}, #{email}, #{age})")
    int insert(User user);
    
    // 获取自增主键
    @Insert("INSERT INTO user(name, email, age) VALUES(#{name}, #{email}, #{age})")
    @Options(useGeneratedKeys = true, keyProperty = "id")
    int insertWithKey(User user);
    
    // 使用 SelectKey（Oracle 序列）
    @Insert("INSERT INTO user(id, name, email) VALUES(#{id}, #{name}, #{email})")
    @SelectKey(statement = "SELECT user_seq.NEXTVAL FROM DUAL", 
               keyProperty = "id", before = true, resultType = Long.class)
    int insertWithSequence(User user);
}
```

### @Update 更新

```java
public interface UserMapper {
    
    // 简单更新
    @Update("UPDATE user SET name = #{name}, email = #{email} WHERE id = #{id}")
    int update(User user);
    
    // 部分更新
    @Update("UPDATE user SET name = #{name} WHERE id = #{id}")
    int updateName(@Param("id") Long id, @Param("name") String name);
}
```

### @Delete 删除

```java
public interface UserMapper {
    
    // 简单删除
    @Delete("DELETE FROM user WHERE id = #{id}")
    int deleteById(Long id);
    
    // 条件删除
    @Delete("DELETE FROM user WHERE age < #{age}")
    int deleteByAgeLessThan(Integer age);
}
```

## @Options 配置选项

`@Options` 用于配置语句的各种选项。

```java
public interface UserMapper {
    
    @Insert("INSERT INTO user(name, email) VALUES(#{name}, #{email})")
    @Options(
        useGeneratedKeys = true,    // 使用自增主键
        keyProperty = "id",         // 主键属性名
        keyColumn = "id",           // 主键列名
        timeout = 30,               // 超时时间（秒）
        flushCache = Options.FlushCachePolicy.TRUE,  // 刷新缓存
        useCache = false,           // 不使用缓存
        fetchSize = 100,            // 每次获取记录数
        statementType = StatementType.PREPARED  // 语句类型
    )
    int insert(User user);
}
```

### Options 属性

| 属性 | 描述 | 默认值 |
|------|------|--------|
| `useGeneratedKeys` | 使用自增主键 | false |
| `keyProperty` | 主键属性名 | - |
| `keyColumn` | 主键列名 | - |
| `timeout` | 超时时间（秒） | -1（无限制） |
| `flushCache` | 刷新缓存策略 | DEFAULT |
| `useCache` | 使用二级缓存 | true |
| `fetchSize` | 每次获取记录数 | -1 |
| `statementType` | 语句类型 | PREPARED |

## @Results 结果映射

### 基本映射

```java
public interface UserMapper {
    
    @Select("SELECT id, user_name, user_email, user_age FROM user WHERE id = #{id}")
    @Results(id = "userResultMap", value = {
        @Result(property = "id", column = "id", id = true),
        @Result(property = "name", column = "user_name"),
        @Result(property = "email", column = "user_email"),
        @Result(property = "age", column = "user_age")
    })
    User selectById(Long id);
    
    // 复用 resultMap
    @Select("SELECT id, user_name, user_email, user_age FROM user")
    @ResultMap("userResultMap")
    List<User> selectAll();
}
```

### @Result 属性

| 属性 | 描述 |
|------|------|
| `property` | Java 属性名 |
| `column` | 数据库列名 |
| `id` | 是否为主键 |
| `javaType` | Java 类型 |
| `jdbcType` | JDBC 类型 |
| `typeHandler` | 类型处理器 |
| `one` | 一对一关联 |
| `many` | 一对多关联 |

## 关联映射

### @One 一对一

```java
public interface OrderMapper {
    
    // 嵌套查询方式
    @Select("SELECT * FROM orders WHERE id = #{id}")
    @Results({
        @Result(property = "id", column = "id", id = true),
        @Result(property = "orderNo", column = "order_no"),
        @Result(property = "amount", column = "amount"),
        @Result(property = "user", column = "user_id",
                one = @One(select = "com.example.mapper.UserMapper.selectById",
                          fetchType = FetchType.LAZY))
    })
    Order selectById(Long id);
}
```

### @Many 一对多

```java
public interface UserMapper {
    
    // 嵌套查询方式
    @Select("SELECT * FROM user WHERE id = #{id}")
    @Results({
        @Result(property = "id", column = "id", id = true),
        @Result(property = "name", column = "name"),
        @Result(property = "orders", column = "id",
                many = @Many(select = "com.example.mapper.OrderMapper.selectByUserId",
                            fetchType = FetchType.LAZY))
    })
    User selectWithOrders(Long id);
}

public interface OrderMapper {
    
    @Select("SELECT * FROM orders WHERE user_id = #{userId}")
    List<Order> selectByUserId(Long userId);
}
```

### FetchType 加载类型

| 类型 | 描述 |
|------|------|
| `FetchType.DEFAULT` | 使用全局配置 |
| `FetchType.LAZY` | 延迟加载 |
| `FetchType.EAGER` | 立即加载 |

## @Provider 动态 SQL

Provider 注解用于动态生成 SQL，适合复杂的动态查询。

### @SelectProvider

```java
public interface UserMapper {
    
    @SelectProvider(type = UserSqlProvider.class, method = "selectByCondition")
    List<User> selectByCondition(UserQuery query);
}

public class UserSqlProvider {
    
    public String selectByCondition(UserQuery query) {
        return new SQL() {{
            SELECT("*");
            FROM("user");
            if (query.getName() != null) {
                WHERE("name LIKE CONCAT('%', #{name}, '%')");
            }
            if (query.getEmail() != null) {
                WHERE("email = #{email}");
            }
            if (query.getMinAge() != null) {
                WHERE("age >= #{minAge}");
            }
            if (query.getMaxAge() != null) {
                WHERE("age <= #{maxAge}");
            }
            ORDER_BY("create_time DESC");
        }}.toString();
    }
}
```

### @InsertProvider

```java
public interface UserMapper {
    
    @InsertProvider(type = UserSqlProvider.class, method = "insertSelective")
    @Options(useGeneratedKeys = true, keyProperty = "id")
    int insertSelective(User user);
}

public class UserSqlProvider {
    
    public String insertSelective(User user) {
        return new SQL() {{
            INSERT_INTO("user");
            if (user.getName() != null) {
                VALUES("name", "#{name}");
            }
            if (user.getEmail() != null) {
                VALUES("email", "#{email}");
            }
            if (user.getAge() != null) {
                VALUES("age", "#{age}");
            }
            VALUES("create_time", "NOW()");
        }}.toString();
    }
}
```

### @UpdateProvider

```java
public interface UserMapper {
    
    @UpdateProvider(type = UserSqlProvider.class, method = "updateSelective")
    int updateSelective(User user);
}

public class UserSqlProvider {
    
    public String updateSelective(User user) {
        return new SQL() {{
            UPDATE("user");
            if (user.getName() != null) {
                SET("name = #{name}");
            }
            if (user.getEmail() != null) {
                SET("email = #{email}");
            }
            if (user.getAge() != null) {
                SET("age = #{age}");
            }
            SET("update_time = NOW()");
            WHERE("id = #{id}");
        }}.toString();
    }
}
```

### @DeleteProvider

```java
public interface UserMapper {
    
    @DeleteProvider(type = UserSqlProvider.class, method = "deleteByIds")
    int deleteByIds(@Param("ids") List<Long> ids);
}

public class UserSqlProvider {
    
    public String deleteByIds(@Param("ids") List<Long> ids) {
        StringBuilder sql = new StringBuilder("DELETE FROM user WHERE id IN (");
        for (int i = 0; i < ids.size(); i++) {
            if (i > 0) sql.append(",");
            sql.append("#{ids[").append(i).append("]}");
        }
        sql.append(")");
        return sql.toString();
    }
}
```

### SQL 构建器

MyBatis 提供 `SQL` 类来构建 SQL 语句：

```java
public class UserSqlProvider {
    
    public String buildSelectSql(UserQuery query) {
        SQL sql = new SQL();
        sql.SELECT("id", "name", "email", "age");
        sql.FROM("user");
        
        if (query.getName() != null) {
            sql.WHERE("name = #{name}");
        }
        if (query.getStatus() != null) {
            sql.WHERE("status = #{status}");
        }
        
        sql.ORDER_BY("create_time DESC");
        
        return sql.toString();
    }
    
    // 使用 Lambda 风格
    public String buildSelectSqlLambda(UserQuery query) {
        return new SQL() {{
            SELECT("*");
            FROM("user");
            WHERE("status = 1");
            if (query.getName() != null) {
                AND();
                WHERE("name = #{name}");
            }
            ORDER_BY("id DESC");
        }}.toString();
    }
}
```

### SQL 类方法

| 方法 | 描述 |
|------|------|
| `SELECT(String...)` | SELECT 子句 |
| `SELECT_DISTINCT(String...)` | SELECT DISTINCT |
| `FROM(String...)` | FROM 子句 |
| `JOIN(String)` | JOIN |
| `INNER_JOIN(String)` | INNER JOIN |
| `LEFT_OUTER_JOIN(String)` | LEFT JOIN |
| `RIGHT_OUTER_JOIN(String)` | RIGHT JOIN |
| `WHERE(String)` | WHERE 条件 |
| `OR()` | OR 连接 |
| `AND()` | AND 连接 |
| `GROUP_BY(String...)` | GROUP BY |
| `HAVING(String)` | HAVING |
| `ORDER_BY(String...)` | ORDER BY |
| `LIMIT(String)` | LIMIT |
| `OFFSET(String)` | OFFSET |
| `INSERT_INTO(String)` | INSERT INTO |
| `VALUES(String, String)` | VALUES |
| `INTO_COLUMNS(String...)` | 插入列 |
| `INTO_VALUES(String...)` | 插入值 |
| `UPDATE(String)` | UPDATE |
| `SET(String)` | SET |
| `DELETE_FROM(String)` | DELETE FROM |

## @Param 参数注解

`@Param` 用于给参数命名，在多参数时必须使用。

```java
public interface UserMapper {
    
    // 单参数可以不用 @Param
    @Select("SELECT * FROM user WHERE id = #{id}")
    User selectById(Long id);
    
    // 多参数必须使用 @Param
    @Select("SELECT * FROM user WHERE name = #{name} AND age = #{age}")
    List<User> selectByNameAndAge(@Param("name") String name, 
                                   @Param("age") Integer age);
    
    // 对象参数 + 其他参数
    @Update("UPDATE user SET name = #{user.name} WHERE id = #{id}")
    int updateName(@Param("id") Long id, @Param("user") User user);
    
    // 集合参数
    @Select("<script>" +
            "SELECT * FROM user WHERE id IN " +
            "<foreach collection='ids' item='id' open='(' separator=',' close=')'>" +
            "#{id}" +
            "</foreach>" +
            "</script>")
    List<User> selectByIds(@Param("ids") List<Long> ids);
}
```

## 脚本模式

使用 `<script>` 标签可以在注解中使用动态 SQL 标签。

```java
public interface UserMapper {
    
    // 动态条件查询
    @Select("<script>" +
            "SELECT * FROM user" +
            "<where>" +
            "  <if test='name != null'>AND name = #{name}</if>" +
            "  <if test='email != null'>AND email = #{email}</if>" +
            "  <if test='age != null'>AND age = #{age}</if>" +
            "</where>" +
            "</script>")
    List<User> selectByCondition(UserQuery query);
    
    // 动态更新
    @Update("<script>" +
            "UPDATE user" +
            "<set>" +
            "  <if test='name != null'>name = #{name},</if>" +
            "  <if test='email != null'>email = #{email},</if>" +
            "  <if test='age != null'>age = #{age},</if>" +
            "  update_time = NOW()," +
            "</set>" +
            "WHERE id = #{id}" +
            "</script>")
    int updateSelective(User user);
    
    // 批量插入
    @Insert("<script>" +
            "INSERT INTO user(name, email, age) VALUES" +
            "<foreach collection='list' item='user' separator=','>" +
            "  (#{user.name}, #{user.email}, #{user.age})" +
            "</foreach>" +
            "</script>")
    int batchInsert(@Param("list") List<User> users);
    
    // choose/when/otherwise
    @Select("<script>" +
            "SELECT * FROM user" +
            "<where>" +
            "  <choose>" +
            "    <when test='id != null'>id = #{id}</when>" +
            "    <when test='name != null'>name = #{name}</when>" +
            "    <otherwise>status = 1</otherwise>" +
            "  </choose>" +
            "</where>" +
            "</script>")
    List<User> selectByPriority(UserQuery query);
}
```

## 缓存注解

### @CacheNamespace

```java
@CacheNamespace(
    implementation = PerpetualCache.class,  // 缓存实现
    eviction = LruCache.class,              // 淘汰策略
    flushInterval = 60000,                  // 刷新间隔（毫秒）
    size = 512,                             // 缓存大小
    readWrite = true,                       // 读写缓存
    blocking = false                        // 阻塞
)
public interface UserMapper {
    // ...
}
```

### @CacheNamespaceRef

```java
// 引用其他 Mapper 的缓存
@CacheNamespaceRef(UserMapper.class)
public interface OrderMapper {
    // ...
}
```

## XML vs 注解对比

| 方面 | XML | 注解 |
|------|-----|------|
| 复杂 SQL | ✅ 更适合 | ❌ 可读性差 |
| 简单 CRUD | ❌ 繁琐 | ✅ 简洁 |
| 动态 SQL | ✅ 强大 | ⚠️ 需要 script 或 Provider |
| 结果映射 | ✅ 灵活 | ⚠️ 功能有限 |
| 维护性 | ✅ SQL 集中管理 | ⚠️ 分散在代码中 |
| 重构 | ❌ 需要手动同步 | ✅ IDE 支持 |
| 学习曲线 | 中等 | 低 |

### 使用建议

1. **简单 CRUD** - 使用注解，简洁高效
2. **复杂查询** - 使用 XML，可读性好
3. **动态 SQL** - 使用 XML 或 Provider
4. **混合使用** - 同一 Mapper 可以混合使用

```java
// 混合使用示例
public interface UserMapper {
    
    // 简单查询用注解
    @Select("SELECT * FROM user WHERE id = #{id}")
    User selectById(Long id);
    
    // 复杂查询在 XML 中定义
    List<User> selectByCondition(UserQuery query);
    
    // 动态 SQL 用 Provider
    @SelectProvider(type = UserSqlProvider.class, method = "buildDynamicSql")
    List<User> selectDynamic(Map<String, Object> params);
}
```

## 完整示例

```java
@CacheNamespace(flushInterval = 60000, size = 512)
public interface UserMapper {
    
    // 基本查询
    @Select("SELECT * FROM user WHERE id = #{id}")
    @Results(id = "baseResultMap", value = {
        @Result(property = "id", column = "id", id = true),
        @Result(property = "name", column = "name"),
        @Result(property = "email", column = "email"),
        @Result(property = "age", column = "age"),
        @Result(property = "createTime", column = "create_time")
    })
    User selectById(Long id);
    
    // 复用 ResultMap
    @Select("SELECT * FROM user")
    @ResultMap("baseResultMap")
    List<User> selectAll();
    
    // 带关联查询
    @Select("SELECT * FROM user WHERE id = #{id}")
    @Results({
        @Result(property = "id", column = "id", id = true),
        @Result(property = "name", column = "name"),
        @Result(property = "orders", column = "id",
                many = @Many(select = "com.example.mapper.OrderMapper.selectByUserId"))
    })
    User selectWithOrders(Long id);
    
    // 插入
    @Insert("INSERT INTO user(name, email, age, create_time) " +
            "VALUES(#{name}, #{email}, #{age}, NOW())")
    @Options(useGeneratedKeys = true, keyProperty = "id")
    int insert(User user);
    
    // 动态更新
    @UpdateProvider(type = UserSqlProvider.class, method = "updateSelective")
    int updateSelective(User user);
    
    // 删除
    @Delete("DELETE FROM user WHERE id = #{id}")
    int deleteById(Long id);
    
    // 动态查询
    @SelectProvider(type = UserSqlProvider.class, method = "selectByCondition")
    List<User> selectByCondition(UserQuery query);
    
    // 批量操作
    @Insert("<script>" +
            "INSERT INTO user(name, email, age) VALUES " +
            "<foreach collection='list' item='u' separator=','>" +
            "(#{u.name}, #{u.email}, #{u.age})" +
            "</foreach>" +
            "</script>")
    int batchInsert(@Param("list") List<User> users);
}
```

## 相关链接

- [XML 映射](/docs/mybatis/xml-mapping) - XML 方式详解
- [动态 SQL](/docs/mybatis/dynamic-sql) - 动态 SQL 标签
- [缓存机制](/docs/mybatis/caching) - 缓存配置详解

---

**最后更新**: 2025 年 12 月
