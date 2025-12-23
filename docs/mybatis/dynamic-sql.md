---
sidebar_position: 5
title: MyBatis 动态 SQL
---

# MyBatis 动态 SQL

动态 SQL 是 MyBatis 的强大特性之一，它允许根据不同条件动态生成 SQL 语句，避免手动拼接 SQL 字符串的繁琐和错误。

## 动态 SQL 元素

MyBatis 提供以下动态 SQL 元素：

| 元素 | 描述 |
|------|------|
| `if` | 条件判断 |
| `choose/when/otherwise` | 多条件分支（类似 switch） |
| `where` | 智能处理 WHERE 子句 |
| `set` | 智能处理 UPDATE SET 子句 |
| `trim` | 自定义前缀/后缀处理 |
| `foreach` | 遍历集合 |
| `sql` | 可重用的 SQL 片段 |
| `bind` | 创建变量并绑定到上下文 |

## if 条件判断

`if` 是最常用的动态 SQL 元素，用于条件判断。

### 基本用法

```xml
<select id="selectByCondition" resultType="User">
    SELECT * FROM user
    WHERE status = 1
    <if test="name != null and name != ''">
        AND name LIKE CONCAT('%', #{name}, '%')
    </if>
    <if test="email != null">
        AND email = #{email}
    </if>
    <if test="age != null">
        AND age = #{age}
    </if>
</select>
```

### test 表达式

`test` 属性使用 OGNL 表达式：

```xml
<!-- 字符串判断 -->
<if test="name != null and name != ''">...</if>
<if test="name != null and name.trim() != ''">...</if>

<!-- 数值判断 -->
<if test="age != null and age > 0">...</if>
<if test="age != null and age >= 18 and age <= 60">...</if>

<!-- 布尔判断 -->
<if test="enabled">...</if>
<if test="enabled == true">...</if>
<if test="!disabled">...</if>

<!-- 集合判断 -->
<if test="ids != null and ids.size() > 0">...</if>
<if test="list != null and !list.isEmpty()">...</if>

<!-- 枚举判断 -->
<if test="status != null and status.name() == 'ACTIVE'">...</if>

<!-- 字符串比较（注意单引号） -->
<if test="type == 'admin'">...</if>
<if test='type == "admin"'>...</if>
<if test="type == 'admin'.toString()">...</if>
```

### 常见陷阱

```xml
<!-- ❌ 错误：单字符比较 -->
<if test="type == 'A'">...</if>

<!-- ✅ 正确：转为字符串 -->
<if test="type == 'A'.toString()">...</if>
<if test='type == "A"'>...</if>

<!-- ❌ 错误：直接比较字符串 -->
<if test="status == ACTIVE">...</if>

<!-- ✅ 正确：加引号 -->
<if test="status == 'ACTIVE'">...</if>
```

## choose/when/otherwise

类似 Java 的 `switch` 语句，只会选择一个分支执行。

### 基本用法

```xml
<select id="selectByCondition" resultType="User">
    SELECT * FROM user
    WHERE status = 1
    <choose>
        <when test="id != null">
            AND id = #{id}
        </when>
        <when test="name != null">
            AND name = #{name}
        </when>
        <when test="email != null">
            AND email = #{email}
        </when>
        <otherwise>
            AND create_time > DATE_SUB(NOW(), INTERVAL 7 DAY)
        </otherwise>
    </choose>
</select>
```

### 多条件优先级

```xml
<select id="selectUsers" resultType="User">
    SELECT * FROM user
    <where>
        <choose>
            <!-- 优先按 ID 查询 -->
            <when test="id != null">
                id = #{id}
            </when>
            <!-- 其次按用户名精确查询 -->
            <when test="username != null and exact">
                username = #{username}
            </when>
            <!-- 再次按用户名模糊查询 -->
            <when test="username != null">
                username LIKE CONCAT('%', #{username}, '%')
            </when>
            <!-- 默认查询活跃用户 -->
            <otherwise>
                status = 'ACTIVE'
            </otherwise>
        </choose>
    </where>
</select>
```

## where 元素

`where` 元素智能处理 WHERE 子句，自动去除多余的 AND/OR 前缀。

### 问题场景

```xml
<!-- ❌ 问题：如果所有条件都为空，会产生 "WHERE" 语法错误 -->
<!-- ❌ 问题：如果第一个条件为空，会产生 "WHERE AND" 语法错误 -->
<select id="selectByCondition" resultType="User">
    SELECT * FROM user
    WHERE
    <if test="name != null">
        name = #{name}
    </if>
    <if test="email != null">
        AND email = #{email}
    </if>
</select>
```

### 使用 where 解决

```xml
<!-- ✅ 正确：使用 where 元素 -->
<select id="selectByCondition" resultType="User">
    SELECT * FROM user
    <where>
        <if test="name != null">
            AND name = #{name}
        </if>
        <if test="email != null">
            AND email = #{email}
        </if>
        <if test="age != null">
            AND age = #{age}
        </if>
    </where>
</select>
```

`where` 元素的行为：
- 只有在子元素有内容时才插入 WHERE
- 自动去除内容开头的 AND 或 OR

## set 元素

`set` 元素用于 UPDATE 语句，智能处理 SET 子句，自动去除多余的逗号。

### 问题场景

```xml
<!-- ❌ 问题：最后一个逗号会导致语法错误 -->
<update id="update">
    UPDATE user SET
    <if test="name != null">name = #{name},</if>
    <if test="email != null">email = #{email},</if>
    <if test="age != null">age = #{age},</if>
    WHERE id = #{id}
</update>
```

### 使用 set 解决

```xml
<!-- ✅ 正确：使用 set 元素 -->
<update id="update" parameterType="User">
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

`set` 元素的行为：
- 自动添加 SET 关键字
- 自动去除内容末尾的逗号

## trim 元素

`trim` 是最灵活的元素，可以自定义前缀、后缀以及要去除的内容。

### 属性说明

| 属性 | 描述 |
|------|------|
| `prefix` | 添加的前缀 |
| `suffix` | 添加的后缀 |
| `prefixOverrides` | 要去除的前缀（多个用 `|` 分隔） |
| `suffixOverrides` | 要去除的后缀（多个用 `|` 分隔） |

### 实现 where

```xml
<!-- 等价于 <where> -->
<trim prefix="WHERE" prefixOverrides="AND |OR ">
    <if test="name != null">AND name = #{name}</if>
    <if test="email != null">AND email = #{email}</if>
</trim>
```

### 实现 set

```xml
<!-- 等价于 <set> -->
<trim prefix="SET" suffixOverrides=",">
    <if test="name != null">name = #{name},</if>
    <if test="email != null">email = #{email},</if>
</trim>
```

### 自定义用法

```xml
<!-- 动态 IN 子句 -->
<select id="selectByIds" resultType="User">
    SELECT * FROM user
    WHERE id IN
    <trim prefix="(" suffix=")" suffixOverrides=",">
        <foreach collection="ids" item="id">
            #{id},
        </foreach>
    </trim>
</select>

<!-- 动态插入列 -->
<insert id="insertSelective">
    INSERT INTO user
    <trim prefix="(" suffix=")" suffixOverrides=",">
        <if test="name != null">name,</if>
        <if test="email != null">email,</if>
        <if test="age != null">age,</if>
    </trim>
    <trim prefix="VALUES (" suffix=")" suffixOverrides=",">
        <if test="name != null">#{name},</if>
        <if test="email != null">#{email},</if>
        <if test="age != null">#{age},</if>
    </trim>
</insert>
```

## foreach 遍历

`foreach` 用于遍历集合，常用于 IN 查询和批量操作。

### 属性说明

| 属性 | 描述 |
|------|------|
| `collection` | 要遍历的集合 |
| `item` | 当前元素的变量名 |
| `index` | 当前索引（List）或键（Map） |
| `open` | 开始符号 |
| `close` | 结束符号 |
| `separator` | 元素分隔符 |

### collection 取值

```java
// 1. List 参数
List<Long> selectByIds(List<Long> ids);
// collection="list" 或 collection="collection" 或 @Param("ids") 后用 "ids"

// 2. Array 参数
List<Long> selectByIds(Long[] ids);
// collection="array" 或 @Param("ids") 后用 "ids"

// 3. @Param 注解
List<Long> selectByIds(@Param("ids") List<Long> ids);
// collection="ids"

// 4. Map 参数
List<Long> selectByMap(Map<String, Object> params);
// collection="ids"（Map 中的 key）
```

### IN 查询

```xml
<select id="selectByIds" resultType="User">
    SELECT * FROM user
    WHERE id IN
    <foreach collection="ids" item="id" open="(" separator="," close=")">
        #{id}
    </foreach>
</select>
```

生成 SQL：`SELECT * FROM user WHERE id IN (1, 2, 3)`

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

生成 SQL：
```sql
INSERT INTO user (name, email, age)
VALUES ('张三', 'a@test.com', 20), ('李四', 'b@test.com', 25)
```

### 批量更新

```xml
<!-- 方式一：多条 UPDATE（需要 allowMultiQueries=true） -->
<update id="batchUpdate">
    <foreach collection="list" item="user" separator=";">
        UPDATE user
        SET name = #{user.name}, email = #{user.email}
        WHERE id = #{user.id}
    </foreach>
</update>

<!-- 方式二：CASE WHEN -->
<update id="batchUpdate">
    UPDATE user
    SET name = CASE id
        <foreach collection="list" item="user">
            WHEN #{user.id} THEN #{user.name}
        </foreach>
        END,
        email = CASE id
        <foreach collection="list" item="user">
            WHEN #{user.id} THEN #{user.email}
        </foreach>
        END
    WHERE id IN
    <foreach collection="list" item="user" open="(" separator="," close=")">
        #{user.id}
    </foreach>
</update>
```

### 遍历 Map

```xml
<select id="selectByMap" resultType="User">
    SELECT * FROM user
    <where>
        <foreach collection="conditions" index="key" item="value">
            AND ${key} = #{value}
        </foreach>
    </where>
</select>
```

```java
Map<String, Object> conditions = new HashMap<>();
conditions.put("name", "张三");
conditions.put("age", 20);
userMapper.selectByMap(conditions);
```

## sql 片段

`sql` 元素用于定义可重用的 SQL 片段。

### 基本用法

```xml
<!-- 定义 -->
<sql id="baseColumns">
    id, name, email, age, status, create_time, update_time
</sql>

<sql id="activeCondition">
    AND status = 1 AND deleted = 0
</sql>

<!-- 引用 -->
<select id="selectAll" resultType="User">
    SELECT <include refid="baseColumns"/>
    FROM user
    WHERE 1=1
    <include refid="activeCondition"/>
</select>
```

### 带参数的片段

```xml
<sql id="selectColumns">
    ${alias}.id, ${alias}.name, ${alias}.email
</sql>

<select id="selectWithAlias" resultType="User">
    SELECT
    <include refid="selectColumns">
        <property name="alias" value="u"/>
    </include>
    FROM user u
</select>
```

### 嵌套引用

```xml
<sql id="simpleColumns">id, name</sql>

<sql id="allColumns">
    <include refid="simpleColumns"/>, email, age, create_time
</sql>

<select id="selectAll" resultType="User">
    SELECT <include refid="allColumns"/>
    FROM user
</select>
```

## bind 变量绑定

`bind` 元素用于创建变量并绑定到上下文，常用于模糊查询。

### 基本用法

```xml
<select id="selectByName" resultType="User">
    <bind name="pattern" value="'%' + name + '%'"/>
    SELECT * FROM user
    WHERE name LIKE #{pattern}
</select>
```

### 多个绑定

```xml
<select id="selectByCondition" resultType="User">
    <bind name="namePattern" value="'%' + name + '%'"/>
    <bind name="emailPattern" value="email + '%'"/>
    SELECT * FROM user
    <where>
        <if test="name != null">
            AND name LIKE #{namePattern}
        </if>
        <if test="email != null">
            AND email LIKE #{emailPattern}
        </if>
    </where>
</select>
```

### 与 CONCAT 对比

```xml
<!-- 使用 bind（推荐，数据库无关） -->
<select id="selectByName" resultType="User">
    <bind name="pattern" value="'%' + name + '%'"/>
    SELECT * FROM user WHERE name LIKE #{pattern}
</select>

<!-- 使用 CONCAT（MySQL 特定） -->
<select id="selectByName" resultType="User">
    SELECT * FROM user WHERE name LIKE CONCAT('%', #{name}, '%')
</select>

<!-- 使用 ||（Oracle/PostgreSQL） -->
<select id="selectByName" resultType="User">
    SELECT * FROM user WHERE name LIKE '%' || #{name} || '%'
</select>
```

## 最佳实践

### 1. 通用查询模板

```xml
<select id="selectByCondition" resultType="User">
    SELECT <include refid="baseColumns"/>
    FROM user
    <where>
        <if test="id != null">
            AND id = #{id}
        </if>
        <if test="name != null and name != ''">
            <bind name="namePattern" value="'%' + name + '%'"/>
            AND name LIKE #{namePattern}
        </if>
        <if test="email != null and email != ''">
            AND email = #{email}
        </if>
        <if test="minAge != null">
            AND age &gt;= #{minAge}
        </if>
        <if test="maxAge != null">
            AND age &lt;= #{maxAge}
        </if>
        <if test="status != null">
            AND status = #{status}
        </if>
        <if test="ids != null and ids.size() > 0">
            AND id IN
            <foreach collection="ids" item="id" open="(" separator="," close=")">
                #{id}
            </foreach>
        </if>
        <if test="startTime != null">
            AND create_time &gt;= #{startTime}
        </if>
        <if test="endTime != null">
            AND create_time &lt;= #{endTime}
        </if>
    </where>
    <choose>
        <when test="orderBy != null and orderBy != ''">
            ORDER BY ${orderBy}
            <if test="orderDir != null">${orderDir}</if>
        </when>
        <otherwise>
            ORDER BY create_time DESC
        </otherwise>
    </choose>
    <if test="limit != null">
        LIMIT #{limit}
        <if test="offset != null">OFFSET #{offset}</if>
    </if>
</select>
```

### 2. 动态插入

```xml
<insert id="insertSelective" parameterType="User"
        useGeneratedKeys="true" keyProperty="id">
    INSERT INTO user
    <trim prefix="(" suffix=")" suffixOverrides=",">
        <if test="name != null">name,</if>
        <if test="email != null">email,</if>
        <if test="age != null">age,</if>
        <if test="status != null">status,</if>
        create_time,
    </trim>
    <trim prefix="VALUES (" suffix=")" suffixOverrides=",">
        <if test="name != null">#{name},</if>
        <if test="email != null">#{email},</if>
        <if test="age != null">#{age},</if>
        <if test="status != null">#{status},</if>
        NOW(),
    </trim>
</insert>
```

### 3. 动态更新

```xml
<update id="updateSelective" parameterType="User">
    UPDATE user
    <set>
        <if test="name != null">name = #{name},</if>
        <if test="email != null">email = #{email},</if>
        <if test="age != null">age = #{age},</if>
        <if test="status != null">status = #{status},</if>
        update_time = NOW(),
    </set>
    WHERE id = #{id}
</update>
```

### 4. 安全的动态排序

```xml
<!-- 使用白名单验证排序字段 -->
<select id="selectWithSort" resultType="User">
    SELECT * FROM user
    <where>
        status = 1
    </where>
    ORDER BY
    <choose>
        <when test="orderBy == 'name'">name</when>
        <when test="orderBy == 'age'">age</when>
        <when test="orderBy == 'createTime'">create_time</when>
        <otherwise>id</otherwise>
    </choose>
    <choose>
        <when test="orderDir == 'asc'">ASC</when>
        <otherwise>DESC</otherwise>
    </choose>
</select>
```

## 常见问题

### 1. XML 特殊字符

在 XML 中，某些字符需要转义：

| 字符 | 转义 | 描述 |
|------|------|------|
| `<` | `&lt;` | 小于 |
| `>` | `&gt;` | 大于 |
| `&` | `&amp;` | 与 |
| `'` | `&apos;` | 单引号 |
| `"` | `&quot;` | 双引号 |

```xml
<!-- 使用转义 -->
<if test="age &gt;= 18 and age &lt;= 60">...</if>

<!-- 使用 CDATA -->
<if test="age != null">
    <![CDATA[AND age >= #{minAge} AND age <= #{maxAge}]]>
</if>
```

### 2. 空字符串判断

```xml
<!-- 同时判断 null 和空字符串 -->
<if test="name != null and name != ''">
    AND name = #{name}
</if>

<!-- 使用 trim() 去除空格 -->
<if test="name != null and name.trim() != ''">
    AND name = #{name}
</if>
```

### 3. 集合判空

```xml
<!-- 判断集合不为空 -->
<if test="ids != null and ids.size() > 0">
    AND id IN
    <foreach collection="ids" item="id" open="(" separator="," close=")">
        #{id}
    </foreach>
</if>

<!-- 或使用 isEmpty() -->
<if test="ids != null and !ids.isEmpty()">
    ...
</if>
```

## 相关链接

- [XML 映射](/docs/mybatis/xml-mapping) - Mapper XML 基础
- [注解映射](/docs/mybatis/annotations) - 注解方式的动态 SQL
- [最佳实践](/docs/mybatis/best-practices) - SQL 编写规范

---

**最后更新**: 2025 年 12 月
