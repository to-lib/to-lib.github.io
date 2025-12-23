---
sidebar_position: 11
title: MyBatis 快速参考
---

# MyBatis 快速参考

本页提供 MyBatis 常用配置、标签和注解的快速参考。

## 配置项速查

### mybatis-config.xml 结构

```xml
<configuration>
    <properties/>      <!-- 属性 -->
    <settings/>        <!-- 设置 -->
    <typeAliases/>     <!-- 类型别名 -->
    <typeHandlers/>    <!-- 类型处理器 -->
    <objectFactory/>   <!-- 对象工厂 -->
    <plugins/>         <!-- 插件 -->
    <environments/>    <!-- 环境 -->
    <databaseIdProvider/>  <!-- 数据库厂商 -->
    <mappers/>         <!-- 映射器 -->
</configuration>
```

### 常用 settings

| 设置 | 描述 | 默认值 |
|------|------|--------|
| `cacheEnabled` | 二级缓存开关 | true |
| `lazyLoadingEnabled` | 延迟加载 | false |
| `mapUnderscoreToCamelCase` | 驼峰命名映射 | false |
| `useGeneratedKeys` | 自增主键 | false |
| `defaultExecutorType` | 执行器类型 | SIMPLE |
| `defaultStatementTimeout` | SQL 超时（秒） | null |
| `logImpl` | 日志实现 | 未设置 |
| `localCacheScope` | 一级缓存作用域 | SESSION |

### Spring Boot 配置

```yaml
mybatis:
  mapper-locations: classpath:mapper/*.xml
  type-aliases-package: com.example.entity
  configuration:
    map-underscore-to-camel-case: true
    cache-enabled: true
    log-impl: org.apache.ibatis.logging.slf4j.Slf4jImpl
```

## XML 标签速查

### CRUD 标签

| 标签 | 属性 | 描述 |
|------|------|------|
| `<select>` | id, parameterType, resultType, resultMap | 查询 |
| `<insert>` | id, parameterType, useGeneratedKeys, keyProperty | 插入 |
| `<update>` | id, parameterType | 更新 |
| `<delete>` | id, parameterType | 删除 |

### select 属性

```xml
<select
    id="selectById"
    parameterType="long"
    resultType="User"
    resultMap="userResultMap"
    useCache="true"
    flushCache="false"
    timeout="30"
    fetchSize="100"
    statementType="PREPARED">
</select>
```

### insert 属性

```xml
<insert
    id="insert"
    parameterType="User"
    useGeneratedKeys="true"
    keyProperty="id"
    keyColumn="id"
    timeout="30">
</insert>
```

### resultMap 结构

```xml
<resultMap id="userMap" type="User" autoMapping="true">
    <constructor>
        <idArg column="id" javaType="long"/>
        <arg column="name" javaType="String"/>
    </constructor>
    <id property="id" column="id"/>
    <result property="name" column="name"/>
    <association property="dept" javaType="Dept"/>
    <collection property="orders" ofType="Order"/>
    <discriminator javaType="int" column="type">
        <case value="1" resultType="Admin"/>
    </discriminator>
</resultMap>
```

### 关联映射

```xml
<!-- 一对一 -->
<association property="user" javaType="User"
    column="user_id" select="selectUser"
    fetchType="lazy"/>

<!-- 一对多 -->
<collection property="orders" ofType="Order"
    column="id" select="selectOrders"
    fetchType="lazy"/>
```

## 动态 SQL 速查

### 条件标签

```xml
<!-- if -->
<if test="name != null">AND name = #{name}</if>

<!-- choose/when/otherwise -->
<choose>
    <when test="id != null">id = #{id}</when>
    <when test="name != null">name = #{name}</when>
    <otherwise>status = 1</otherwise>
</choose>
```

### 格式化标签

```xml
<!-- where：自动去除开头 AND/OR -->
<where>
    <if test="name != null">AND name = #{name}</if>
</where>

<!-- set：自动去除末尾逗号 -->
<set>
    <if test="name != null">name = #{name},</if>
</set>

<!-- trim：自定义前后缀 -->
<trim prefix="WHERE" prefixOverrides="AND |OR ">
    ...
</trim>
```

### foreach

```xml
<foreach
    collection="list"
    item="item"
    index="index"
    open="("
    separator=","
    close=")">
    #{item}
</foreach>
```

| collection 值 | 参数类型 |
|---------------|----------|
| `list` | List 参数 |
| `array` | 数组参数 |
| `collection` | Collection 参数 |
| 自定义名 | @Param 指定 |

### SQL 片段

```xml
<!-- 定义 -->
<sql id="columns">id, name, email</sql>

<!-- 引用 -->
<include refid="columns"/>

<!-- 带参数 -->
<sql id="cols">${alias}.id, ${alias}.name</sql>
<include refid="cols">
    <property name="alias" value="u"/>
</include>
```

### bind 变量

```xml
<bind name="pattern" value="'%' + name + '%'"/>
SELECT * FROM user WHERE name LIKE #{pattern}
```

## 注解速查

### CRUD 注解

```java
@Select("SELECT * FROM user WHERE id = #{id}")
User selectById(Long id);

@Insert("INSERT INTO user(name) VALUES(#{name})")
@Options(useGeneratedKeys = true, keyProperty = "id")
int insert(User user);

@Update("UPDATE user SET name = #{name} WHERE id = #{id}")
int update(User user);

@Delete("DELETE FROM user WHERE id = #{id}")
int deleteById(Long id);
```

### 结果映射注解

```java
@Results(id = "userMap", value = {
    @Result(property = "id", column = "id", id = true),
    @Result(property = "name", column = "user_name"),
    @Result(property = "dept", column = "dept_id",
            one = @One(select = "selectDept")),
    @Result(property = "orders", column = "id",
            many = @Many(select = "selectOrders"))
})
@Select("SELECT * FROM user WHERE id = #{id}")
User selectById(Long id);

@ResultMap("userMap")
@Select("SELECT * FROM user")
List<User> selectAll();
```

### Provider 注解

```java
@SelectProvider(type = UserSqlProvider.class, method = "selectByCondition")
List<User> selectByCondition(UserQuery query);

@InsertProvider(type = UserSqlProvider.class, method = "insert")
int insert(User user);

@UpdateProvider(type = UserSqlProvider.class, method = "update")
int update(User user);

@DeleteProvider(type = UserSqlProvider.class, method = "delete")
int delete(Long id);
```

### 其他注解

```java
@Param("name")           // 参数命名
@MapKey("id")            // Map 结果的 key
@Flush                   // 刷新语句
@CacheNamespace          // 缓存配置
@CacheNamespaceRef       // 缓存引用
```

## 常用代码片段

### 基础 CRUD

```xml
<!-- 查询 -->
<select id="selectById" resultType="User">
    SELECT id, name, email, age, create_time
    FROM user WHERE id = #{id}
</select>

<!-- 插入 -->
<insert id="insert" useGeneratedKeys="true" keyProperty="id">
    INSERT INTO user (name, email, age, create_time)
    VALUES (#{name}, #{email}, #{age}, NOW())
</insert>

<!-- 更新 -->
<update id="update">
    UPDATE user
    SET name = #{name}, email = #{email}, update_time = NOW()
    WHERE id = #{id}
</update>

<!-- 删除 -->
<delete id="deleteById">
    DELETE FROM user WHERE id = #{id}
</delete>
```

### 动态条件查询

```xml
<select id="selectByCondition" resultType="User">
    SELECT * FROM user
    <where>
        <if test="name != null and name != ''">
            AND name LIKE CONCAT('%', #{name}, '%')
        </if>
        <if test="email != null">
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
```

### 动态更新

```xml
<update id="updateSelective">
    UPDATE user
    <set>
        <if test="name != null">name = #{name},</if>
        <if test="email != null">email = #{email},</if>
        <if test="age != null">age = #{age},</if>
        update_time = NOW(),
    </set>
    WHERE id = #{id}
</update>
```

### 批量插入

```xml
<insert id="batchInsert">
    INSERT INTO user (name, email, age)
    VALUES
    <foreach collection="list" item="user" separator=",">
        (#{user.name}, #{user.email}, #{user.age})
    </foreach>
</insert>
```

### 批量删除

```xml
<delete id="batchDelete">
    DELETE FROM user
    WHERE id IN
    <foreach collection="ids" item="id" open="(" separator="," close=")">
        #{id}
    </foreach>
</delete>
```

### 分页查询

```xml
<select id="selectPage" resultType="User">
    SELECT * FROM user
    <include refid="whereCondition"/>
    ORDER BY create_time DESC
    LIMIT #{offset}, #{pageSize}
</select>
```

### 联表查询

```xml
<resultMap id="userWithOrders" type="User">
    <id property="id" column="id"/>
    <result property="name" column="name"/>
    <collection property="orders" ofType="Order">
        <id property="id" column="order_id"/>
        <result property="orderNo" column="order_no"/>
    </collection>
</resultMap>

<select id="selectWithOrders" resultMap="userWithOrders">
    SELECT u.id, u.name, o.id as order_id, o.order_no
    FROM user u
    LEFT JOIN orders o ON u.id = o.user_id
    WHERE u.id = #{id}
</select>
```

## 特殊字符转义

| 字符 | 转义 | 描述 |
|------|------|------|
| `<` | `&lt;` | 小于 |
| `>` | `&gt;` | 大于 |
| `&` | `&amp;` | 与 |
| `'` | `&apos;` | 单引号 |
| `"` | `&quot;` | 双引号 |

```xml
<!-- 使用转义 -->
<if test="age &gt;= 18">...</if>

<!-- 使用 CDATA -->
<![CDATA[AND age >= 18]]>
```

## 类型别名

| 别名 | Java 类型 |
|------|-----------|
| `string` | String |
| `int`/`integer` | Integer |
| `long` | Long |
| `double` | Double |
| `boolean` | Boolean |
| `date` | Date |
| `map` | Map |
| `list` | List |
| `arraylist` | ArrayList |
| `hashmap` | HashMap |

## 相关链接

- [配置详解](/docs/mybatis/configuration)
- [XML 映射](/docs/mybatis/xml-mapping)
- [动态 SQL](/docs/mybatis/dynamic-sql)
- [注解映射](/docs/mybatis/annotations)

---

**最后更新**: 2025 年 12 月
