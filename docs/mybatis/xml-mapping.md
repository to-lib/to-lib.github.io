---
sidebar_position: 4
title: MyBatis XML 映射
---

# MyBatis XML 映射

本章详细介绍 MyBatis Mapper XML 文件的编写，包括 CRUD 操作、结果映射和参数处理。

## Mapper XML 结构

### 基本结构

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE mapper
  PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN"
  "https://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.example.mapper.UserMapper">
    
    <!-- 结果映射 -->
    <resultMap id="..." type="..."/>
    
    <!-- SQL 片段 -->
    <sql id="..."/>
    
    <!-- 查询语句 -->
    <select id="..." resultType="..."/>
    
    <!-- 插入语句 -->
    <insert id="..."/>
    
    <!-- 更新语句 -->
    <update id="..."/>
    
    <!-- 删除语句 -->
    <delete id="..."/>
    
</mapper>
```

### 命名空间

`namespace` 必须与 Mapper 接口的全限定名一致：

```xml
<!-- Mapper XML -->
<mapper namespace="com.example.mapper.UserMapper">
```

```java
// Mapper 接口
package com.example.mapper;

public interface UserMapper {
    User selectById(Long id);
}
```

## select 查询

### 基本查询

```xml
<select id="selectById" parameterType="long" resultType="User">
    SELECT id, name, email, age, create_time
    FROM user
    WHERE id = #{id}
</select>
```

### select 属性

| 属性 | 描述 |
|------|------|
| `id` | 语句的唯一标识，对应 Mapper 接口方法名 |
| `parameterType` | 参数类型（可选，MyBatis 可自动推断） |
| `resultType` | 返回结果类型 |
| `resultMap` | 引用外部 resultMap |
| `flushCache` | 是否清空缓存，默认 false |
| `useCache` | 是否使用二级缓存，默认 true |
| `timeout` | 超时时间（秒） |
| `fetchSize` | 每次获取的记录数 |
| `statementType` | STATEMENT/PREPARED/CALLABLE |
| `resultSetType` | FORWARD_ONLY/SCROLL_SENSITIVE/SCROLL_INSENSITIVE |

### 查询列表

```xml
<select id="selectAll" resultType="User">
    SELECT * FROM user
</select>

<select id="selectByAge" resultType="User">
    SELECT * FROM user WHERE age > #{minAge}
</select>
```

```java
public interface UserMapper {
    List<User> selectAll();
    List<User> selectByAge(int minAge);
}
```

### 查询 Map

```xml
<!-- 返回单条记录为 Map -->
<select id="selectByIdAsMap" resultType="map">
    SELECT * FROM user WHERE id = #{id}
</select>

<!-- 返回多条记录，以指定字段为 key -->
<select id="selectAllAsMap" resultType="User">
    SELECT * FROM user
</select>
```

```java
public interface UserMapper {
    Map<String, Object> selectByIdAsMap(Long id);
    
    @MapKey("id")  // 以 id 字段作为 Map 的 key
    Map<Long, User> selectAllAsMap();
}
```

## insert 插入

### 基本插入

```xml
<insert id="insert" parameterType="User">
    INSERT INTO user (name, email, age, create_time)
    VALUES (#{name}, #{email}, #{age}, #{createTime})
</insert>
```

### 获取自增主键

```xml
<!-- MySQL 自增主键 -->
<insert id="insert" parameterType="User" 
        useGeneratedKeys="true" keyProperty="id">
    INSERT INTO user (name, email, age)
    VALUES (#{name}, #{email}, #{age})
</insert>
```

```java
User user = new User();
user.setName("张三");
user.setEmail("zhangsan@example.com");
userMapper.insert(user);
System.out.println(user.getId()); // 获取自增 ID
```

### Oracle 序列

```xml
<insert id="insert" parameterType="User">
    <selectKey keyProperty="id" resultType="long" order="BEFORE">
        SELECT user_seq.NEXTVAL FROM DUAL
    </selectKey>
    INSERT INTO user (id, name, email)
    VALUES (#{id}, #{name}, #{email})
</insert>
```

### 批量插入

```xml
<insert id="batchInsert" parameterType="list">
    INSERT INTO user (name, email, age)
    VALUES
    <foreach collection="list" item="user" separator=",">
        (#{user.name}, #{user.email}, #{user.age})
    </foreach>
</insert>
```

```java
public interface UserMapper {
    int batchInsert(List<User> users);
}
```

## update 更新

### 基本更新

```xml
<update id="update" parameterType="User">
    UPDATE user
    SET name = #{name},
        email = #{email},
        age = #{age},
        update_time = NOW()
    WHERE id = #{id}
</update>
```

### 动态更新

```xml
<update id="updateSelective" parameterType="User">
    UPDATE user
    <set>
        <if test="name != null">name = #{name},</if>
        <if test="email != null">email = #{email},</if>
        <if test="age != null">age = #{age},</if>
        update_time = NOW()
    </set>
    WHERE id = #{id}
</update>
```

### 批量更新

```xml
<!-- MySQL 批量更新 -->
<update id="batchUpdate" parameterType="list">
    <foreach collection="list" item="user" separator=";">
        UPDATE user
        SET name = #{user.name}, email = #{user.email}
        WHERE id = #{user.id}
    </foreach>
</update>
```

> [!NOTE]
> MySQL 批量更新需要在连接 URL 中添加 `allowMultiQueries=true`

## delete 删除

### 基本删除

```xml
<delete id="deleteById" parameterType="long">
    DELETE FROM user WHERE id = #{id}
</delete>
```

### 批量删除

```xml
<delete id="batchDelete" parameterType="list">
    DELETE FROM user
    WHERE id IN
    <foreach collection="list" item="id" open="(" separator="," close=")">
        #{id}
    </foreach>
</delete>
```

```java
public interface UserMapper {
    int batchDelete(List<Long> ids);
}
```

## 参数映射

### 单个参数

```xml
<select id="selectById" resultType="User">
    SELECT * FROM user WHERE id = #{id}
</select>
```

```java
User selectById(Long id);
```

### 多个参数

#### 使用 @Param 注解

```xml
<select id="selectByNameAndAge" resultType="User">
    SELECT * FROM user 
    WHERE name = #{name} AND age = #{age}
</select>
```

```java
List<User> selectByNameAndAge(@Param("name") String name, @Param("age") Integer age);
```

#### 使用 Map

```xml
<select id="selectByMap" resultType="User">
    SELECT * FROM user 
    WHERE name = #{name} AND age = #{age}
</select>
```

```java
List<User> selectByMap(Map<String, Object> params);
```

#### 使用对象

```xml
<select id="selectByCondition" resultType="User">
    SELECT * FROM user 
    WHERE name = #{name} AND age = #{age}
</select>
```

```java
List<User> selectByCondition(UserQuery query);
```

### #{} vs ${}

| 语法 | 描述 | 安全性 | 使用场景 |
|------|------|--------|----------|
| `#{}` | 预编译参数，使用 `?` 占位符 | 安全，防 SQL 注入 | 参数值 |
| `${}` | 字符串替换，直接拼接 | 不安全，有 SQL 注入风险 | 表名、列名、排序 |

```xml
<!-- #{} 预编译，安全 -->
<select id="selectById" resultType="User">
    SELECT * FROM user WHERE id = #{id}
</select>
<!-- 生成: SELECT * FROM user WHERE id = ? -->

<!-- ${} 字符串替换，用于动态表名/列名 -->
<select id="selectByColumn" resultType="User">
    SELECT * FROM user ORDER BY ${orderColumn} ${orderDir}
</select>
<!-- 生成: SELECT * FROM user ORDER BY create_time DESC -->
```

> [!WARNING]
> 使用 `${}` 时必须对参数进行校验，防止 SQL 注入攻击。

### 参数类型处理

```xml
<!-- 指定 JDBC 类型（处理 null 值） -->
<insert id="insert">
    INSERT INTO user (name, email, age)
    VALUES (
        #{name},
        #{email, jdbcType=VARCHAR},
        #{age, jdbcType=INTEGER}
    )
</insert>

<!-- 指定类型处理器 -->
<insert id="insert">
    INSERT INTO user (status)
    VALUES (#{status, typeHandler=com.example.handler.StatusTypeHandler})
</insert>
```

## resultMap 结果映射

### 基本映射

```xml
<resultMap id="userResultMap" type="User">
    <id property="id" column="id"/>
    <result property="userName" column="user_name"/>
    <result property="email" column="email"/>
    <result property="age" column="age"/>
    <result property="createTime" column="create_time"/>
</resultMap>

<select id="selectById" resultMap="userResultMap">
    SELECT id, user_name, email, age, create_time
    FROM user
    WHERE id = #{id}
</select>
```

### resultMap 元素

| 元素 | 描述 |
|------|------|
| `id` | 主键映射，可优化性能 |
| `result` | 普通字段映射 |
| `association` | 一对一关联映射 |
| `collection` | 一对多关联映射 |
| `discriminator` | 鉴别器，根据值选择不同映射 |
| `constructor` | 构造器注入 |

### 构造器映射

```xml
<resultMap id="userResultMap" type="User">
    <constructor>
        <idArg column="id" javaType="long"/>
        <arg column="user_name" javaType="String"/>
    </constructor>
    <result property="email" column="email"/>
</resultMap>
```

```java
public class User {
    public User(Long id, String userName) {
        this.id = id;
        this.userName = userName;
    }
}
```

## 关联映射

### 一对一关联 (association)

#### 嵌套查询

```xml
<resultMap id="orderResultMap" type="Order">
    <id property="id" column="id"/>
    <result property="orderNo" column="order_no"/>
    <result property="amount" column="amount"/>
    <!-- 嵌套查询：执行额外 SQL -->
    <association property="user" column="user_id" 
                 select="com.example.mapper.UserMapper.selectById"/>
</resultMap>

<select id="selectById" resultMap="orderResultMap">
    SELECT * FROM orders WHERE id = #{id}
</select>
```

#### 嵌套结果

```xml
<resultMap id="orderResultMap" type="Order">
    <id property="id" column="id"/>
    <result property="orderNo" column="order_no"/>
    <result property="amount" column="amount"/>
    <!-- 嵌套结果：从联表查询结果映射 -->
    <association property="user" javaType="User">
        <id property="id" column="user_id"/>
        <result property="name" column="user_name"/>
        <result property="email" column="user_email"/>
    </association>
</resultMap>

<select id="selectWithUser" resultMap="orderResultMap">
    SELECT o.id, o.order_no, o.amount,
           u.id as user_id, u.name as user_name, u.email as user_email
    FROM orders o
    LEFT JOIN user u ON o.user_id = u.id
    WHERE o.id = #{id}
</select>
```

### 一对多关联 (collection)

#### 嵌套查询

```xml
<resultMap id="userWithOrdersMap" type="User">
    <id property="id" column="id"/>
    <result property="name" column="name"/>
    <!-- 嵌套查询 -->
    <collection property="orders" column="id"
                select="com.example.mapper.OrderMapper.selectByUserId"/>
</resultMap>

<select id="selectWithOrders" resultMap="userWithOrdersMap">
    SELECT * FROM user WHERE id = #{id}
</select>
```

#### 嵌套结果

```xml
<resultMap id="userWithOrdersMap" type="User">
    <id property="id" column="id"/>
    <result property="name" column="name"/>
    <result property="email" column="email"/>
    <!-- 嵌套结果 -->
    <collection property="orders" ofType="Order">
        <id property="id" column="order_id"/>
        <result property="orderNo" column="order_no"/>
        <result property="amount" column="amount"/>
    </collection>
</resultMap>

<select id="selectWithOrders" resultMap="userWithOrdersMap">
    SELECT u.id, u.name, u.email,
           o.id as order_id, o.order_no, o.amount
    FROM user u
    LEFT JOIN orders o ON u.id = o.user_id
    WHERE u.id = #{id}
</select>
```

### 多对多关联

```xml
<!-- 用户-角色多对多 -->
<resultMap id="userWithRolesMap" type="User">
    <id property="id" column="id"/>
    <result property="name" column="name"/>
    <collection property="roles" ofType="Role">
        <id property="id" column="role_id"/>
        <result property="name" column="role_name"/>
        <result property="code" column="role_code"/>
    </collection>
</resultMap>

<select id="selectWithRoles" resultMap="userWithRolesMap">
    SELECT u.id, u.name,
           r.id as role_id, r.name as role_name, r.code as role_code
    FROM user u
    LEFT JOIN user_role ur ON u.id = ur.user_id
    LEFT JOIN role r ON ur.role_id = r.id
    WHERE u.id = #{id}
</select>
```

### 延迟加载

```xml
<!-- 全局配置 -->
<settings>
    <setting name="lazyLoadingEnabled" value="true"/>
    <setting name="aggressiveLazyLoading" value="false"/>
</settings>

<!-- 单独配置 -->
<association property="user" column="user_id" 
             select="selectUser" fetchType="lazy"/>

<collection property="orders" column="id"
            select="selectOrders" fetchType="eager"/>
```

## sql 片段

### 定义和引用

```xml
<!-- 定义 SQL 片段 -->
<sql id="baseColumns">
    id, name, email, age, create_time, update_time
</sql>

<sql id="whereCondition">
    <where>
        <if test="name != null">AND name LIKE #{name}</if>
        <if test="age != null">AND age = #{age}</if>
    </where>
</sql>

<!-- 引用 SQL 片段 -->
<select id="selectAll" resultType="User">
    SELECT <include refid="baseColumns"/>
    FROM user
    <include refid="whereCondition"/>
</select>
```

### 带参数的 SQL 片段

```xml
<sql id="selectColumns">
    ${alias}.id, ${alias}.name, ${alias}.email
</sql>

<select id="selectUsers" resultType="User">
    SELECT
    <include refid="selectColumns">
        <property name="alias" value="u"/>
    </include>
    FROM user u
</select>
```

## 完整示例

### 实体类

```java
@Data
public class User {
    private Long id;
    private String name;
    private String email;
    private Integer age;
    private Integer status;
    private Date createTime;
    private Date updateTime;
    private List<Order> orders;
}

@Data
public class Order {
    private Long id;
    private String orderNo;
    private BigDecimal amount;
    private Long userId;
    private User user;
}
```

### Mapper 接口

```java
public interface UserMapper {
    User selectById(Long id);
    List<User> selectByCondition(UserQuery query);
    User selectWithOrders(Long id);
    int insert(User user);
    int update(User user);
    int deleteById(Long id);
    int batchInsert(List<User> users);
}
```

### Mapper XML

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN"
  "https://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.example.mapper.UserMapper">

    <!-- SQL 片段 -->
    <sql id="baseColumns">
        id, name, email, age, status, create_time, update_time
    </sql>

    <!-- 基础结果映射 -->
    <resultMap id="baseResultMap" type="User">
        <id property="id" column="id"/>
        <result property="name" column="name"/>
        <result property="email" column="email"/>
        <result property="age" column="age"/>
        <result property="status" column="status"/>
        <result property="createTime" column="create_time"/>
        <result property="updateTime" column="update_time"/>
    </resultMap>

    <!-- 带订单的结果映射 -->
    <resultMap id="withOrdersMap" type="User" extends="baseResultMap">
        <collection property="orders" ofType="Order">
            <id property="id" column="order_id"/>
            <result property="orderNo" column="order_no"/>
            <result property="amount" column="amount"/>
        </collection>
    </resultMap>

    <!-- 根据 ID 查询 -->
    <select id="selectById" resultMap="baseResultMap">
        SELECT <include refid="baseColumns"/>
        FROM user
        WHERE id = #{id}
    </select>

    <!-- 条件查询 -->
    <select id="selectByCondition" resultMap="baseResultMap">
        SELECT <include refid="baseColumns"/>
        FROM user
        <where>
            <if test="name != null and name != ''">
                AND name LIKE CONCAT('%', #{name}, '%')
            </if>
            <if test="email != null and email != ''">
                AND email = #{email}
            </if>
            <if test="minAge != null">
                AND age >= #{minAge}
            </if>
            <if test="maxAge != null">
                AND age &lt;= #{maxAge}
            </if>
            <if test="status != null">
                AND status = #{status}
            </if>
        </where>
        ORDER BY create_time DESC
    </select>

    <!-- 查询用户及其订单 -->
    <select id="selectWithOrders" resultMap="withOrdersMap">
        SELECT u.id, u.name, u.email, u.age, u.status,
               u.create_time, u.update_time,
               o.id as order_id, o.order_no, o.amount
        FROM user u
        LEFT JOIN orders o ON u.id = o.user_id
        WHERE u.id = #{id}
    </select>

    <!-- 插入 -->
    <insert id="insert" parameterType="User" 
            useGeneratedKeys="true" keyProperty="id">
        INSERT INTO user (name, email, age, status, create_time)
        VALUES (#{name}, #{email}, #{age}, #{status}, NOW())
    </insert>

    <!-- 更新 -->
    <update id="update" parameterType="User">
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

    <!-- 删除 -->
    <delete id="deleteById">
        DELETE FROM user WHERE id = #{id}
    </delete>

    <!-- 批量插入 -->
    <insert id="batchInsert" parameterType="list">
        INSERT INTO user (name, email, age, status, create_time)
        VALUES
        <foreach collection="list" item="user" separator=",">
            (#{user.name}, #{user.email}, #{user.age}, 
             #{user.status}, NOW())
        </foreach>
    </insert>

</mapper>
```

## 相关链接

- [动态 SQL](/docs/mybatis/dynamic-sql) - 深入学习动态 SQL
- [注解映射](/docs/mybatis/annotations) - 使用注解方式
- [缓存机制](/docs/mybatis/caching) - 了解 MyBatis 缓存

---

**最后更新**: 2025 年 12 月
