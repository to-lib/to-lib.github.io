---
sidebar_position: 25
title: MyBatis 面试题
---

# MyBatis 面试题

本页汇总 MyBatis 常见面试题，涵盖基础概念、缓存机制、动态 SQL 和源码原理。

## 基础概念

### Q1: 什么是 MyBatis？它有什么优点？

**答案**：

MyBatis 是一款优秀的持久层框架，它支持自定义 SQL、存储过程以及高级映射。

**优点**：
- **SQL 灵活**：直接编写原生 SQL，便于优化和调试
- **学习成本低**：相比 Hibernate，更容易上手
- **与 Spring 集成好**：MyBatis-Spring 提供无缝集成
- **动态 SQL 强大**：提供丰富的动态 SQL 标签
- **映射灵活**：支持复杂的结果集映射

**缺点**：
- SQL 编写工作量大
- 数据库移植性差
- 需要手动处理关联关系

---

### Q2: MyBatis 的核心组件有哪些？

**答案**：

| 组件 | 描述 |
|------|------|
| `SqlSessionFactoryBuilder` | 构建 SqlSessionFactory |
| `SqlSessionFactory` | 创建 SqlSession 的工厂 |
| `SqlSession` | 执行 SQL 的核心接口 |
| `Mapper` | 数据访问接口 |
| `Configuration` | 核心配置类 |
| `Executor` | SQL 执行器 |
| `StatementHandler` | 语句处理器 |
| `ParameterHandler` | 参数处理器 |
| `ResultSetHandler` | 结果集处理器 |

---

### Q3: #{} 和 ${} 的区别？

**答案**：

| 特性 | #{} | ${} |
|------|-----|-----|
| 处理方式 | 预编译，使用 ? 占位符 | 字符串替换 |
| 安全性 | 防止 SQL 注入 | 有 SQL 注入风险 |
| 使用场景 | 参数值 | 表名、列名、排序字段 |

```xml
<!-- #{} 生成: WHERE id = ? -->
WHERE id = #{id}

<!-- ${} 生成: ORDER BY create_time -->
ORDER BY ${orderColumn}
```

**使用建议**：优先使用 `#{}`，`${}` 只用于动态表名/列名，且需要白名单校验。

---

### Q4: MyBatis 如何获取自增主键？

**答案**：

**方式一：useGeneratedKeys（推荐）**

```xml
<insert id="insert" useGeneratedKeys="true" keyProperty="id">
    INSERT INTO user (name) VALUES (#{name})
</insert>
```

**方式二：selectKey**

```xml
<insert id="insert">
    <selectKey keyProperty="id" resultType="long" order="AFTER">
        SELECT LAST_INSERT_ID()
    </selectKey>
    INSERT INTO user (name) VALUES (#{name})
</insert>
```

---

### Q5: MyBatis 的 Mapper 接口如何与 XML 绑定？

**答案**：

1. **namespace 绑定**：XML 的 namespace 必须是 Mapper 接口的全限定名
2. **方法绑定**：XML 中的 id 必须与接口方法名一致
3. **参数绑定**：参数类型自动匹配或使用 @Param 注解

```java
package com.example.mapper;
public interface UserMapper {
    User selectById(Long id);
}
```

```xml
<mapper namespace="com.example.mapper.UserMapper">
    <select id="selectById" resultType="User">
        SELECT * FROM user WHERE id = #{id}
    </select>
</mapper>
```

---

### Q6: resultType 和 resultMap 的区别？

**答案**：

| 特性 | resultType | resultMap |
|------|------------|-----------|
| 使用场景 | 简单映射 | 复杂映射 |
| 字段映射 | 自动（需开启驼峰映射） | 手动配置 |
| 关联映射 | 不支持 | 支持（association/collection） |
| 性能 | 略高 | 略低 |

```xml
<!-- resultType：简单映射 -->
<select id="selectById" resultType="User">
    SELECT id, name, email FROM user WHERE id = #{id}
</select>

<!-- resultMap：复杂映射 -->
<resultMap id="userMap" type="User">
    <id property="id" column="id"/>
    <result property="userName" column="user_name"/>
    <collection property="orders" ofType="Order">
        <id property="id" column="order_id"/>
    </collection>
</resultMap>
```

## 缓存机制

### Q7: MyBatis 的一级缓存和二级缓存？

**答案**：

**一级缓存（SqlSession 级别）**：
- 默认开启
- 作用域：SqlSession
- 生命周期：SqlSession 创建到关闭
- 失效条件：增删改、clearCache()、commit()

**二级缓存（Mapper 级别）**：
- 默认关闭，需要手动开启
- 作用域：Mapper（namespace）
- 生命周期：应用级别
- 需要实体类实现 Serializable

```xml
<!-- 开启二级缓存 -->
<cache eviction="LRU" flushInterval="60000" size="512"/>
```

---

### Q8: 一级缓存在什么情况下会失效？

**答案**：

1. **不同 SqlSession**：一级缓存是 SqlSession 级别的
2. **执行增删改操作**：会清空当前 SqlSession 的缓存
3. **调用 clearCache()**：手动清空缓存
4. **调用 commit() 或 rollback()**：提交或回滚事务
5. **配置 localCacheScope=STATEMENT**：每次查询后清空

---

### Q9: 二级缓存的使用注意事项？

**答案**：

1. **实体类必须实现 Serializable**
2. **SqlSession 提交后才生效**
3. **多表关联查询可能导致脏数据**
4. **分布式环境需要使用第三方缓存**

**不适合使用二级缓存的场景**：
- 数据变化频繁
- 对实时性要求高
- 多表关联查询
- 财务、库存等敏感数据

---

### Q10: MyBatis 缓存的查询顺序？

**答案**：

```
查询请求 → 二级缓存 → 一级缓存 → 数据库
```

1. 先查二级缓存
2. 二级缓存未命中，查一级缓存
3. 一级缓存未命中，查数据库
4. 结果放入一级缓存
5. SqlSession 关闭后，放入二级缓存

## 动态 SQL

### Q11: MyBatis 有哪些动态 SQL 标签？

**答案**：

| 标签 | 描述 |
|------|------|
| `<if>` | 条件判断 |
| `<choose>/<when>/<otherwise>` | 多条件分支 |
| `<where>` | 智能处理 WHERE |
| `<set>` | 智能处理 SET |
| `<trim>` | 自定义前后缀 |
| `<foreach>` | 遍历集合 |
| `<sql>` | SQL 片段 |
| `<bind>` | 变量绑定 |

---

### Q12: where 标签的作用？

**答案**：

`<where>` 标签会：
1. 只有在子元素有内容时才插入 WHERE
2. 自动去除内容开头的 AND 或 OR

```xml
<select id="selectByCondition" resultType="User">
    SELECT * FROM user
    <where>
        <if test="name != null">AND name = #{name}</if>
        <if test="age != null">AND age = #{age}</if>
    </where>
</select>
```

---

### Q13: foreach 标签的属性有哪些？

**答案**：

| 属性 | 描述 |
|------|------|
| `collection` | 要遍历的集合 |
| `item` | 当前元素变量名 |
| `index` | 当前索引 |
| `open` | 开始符号 |
| `close` | 结束符号 |
| `separator` | 分隔符 |

```xml
<foreach collection="ids" item="id" open="(" separator="," close=")">
    #{id}
</foreach>
```

---

### Q14: 如何实现批量插入？

**答案**：

**方式一：foreach**

```xml
<insert id="batchInsert">
    INSERT INTO user (name, email) VALUES
    <foreach collection="list" item="user" separator=",">
        (#{user.name}, #{user.email})
    </foreach>
</insert>
```

**方式二：BATCH 执行器**

```java
try (SqlSession session = sqlSessionFactory.openSession(ExecutorType.BATCH)) {
    UserMapper mapper = session.getMapper(UserMapper.class);
    for (User user : users) {
        mapper.insert(user);
    }
    session.flushStatements();
    session.commit();
}
```

## 源码原理

### Q15: MyBatis 的执行流程？

**答案**：

1. **加载配置**：解析 mybatis-config.xml 和 Mapper XML
2. **创建 SqlSessionFactory**：构建 Configuration 对象
3. **获取 SqlSession**：从工厂获取 SqlSession
4. **获取 Mapper 代理**：通过 JDK 动态代理生成 Mapper 实现
5. **执行 SQL**：
   - MapperProxy 拦截方法调用
   - 找到对应的 MappedStatement
   - Executor 执行 SQL
   - StatementHandler 处理语句
   - ParameterHandler 设置参数
   - ResultSetHandler 映射结果
6. **返回结果**

---

### Q16: MyBatis 的 Mapper 接口没有实现类，为什么能调用？

**答案**：

MyBatis 使用 **JDK 动态代理** 为 Mapper 接口生成代理对象。

```java
// MapperProxy 实现 InvocationHandler
public class MapperProxy<T> implements InvocationHandler {
    @Override
    public Object invoke(Object proxy, Method method, Object[] args) {
        // 根据方法名找到 MappedStatement
        // 调用 SqlSession 执行 SQL
        return sqlSession.selectOne(statement, args);
    }
}
```

---

### Q17: MyBatis 的插件原理？

**答案**：

MyBatis 插件基于 **责任链模式** 和 **JDK 动态代理**。

**可拦截的四大对象**：
- Executor
- StatementHandler
- ParameterHandler
- ResultSetHandler

**原理**：
1. 插件实现 Interceptor 接口
2. 使用 @Intercepts 注解指定拦截点
3. MyBatis 为目标对象生成代理
4. 调用时按责任链顺序执行

```java
@Intercepts({
    @Signature(type = Executor.class, method = "query", 
               args = {MappedStatement.class, Object.class, 
                       RowBounds.class, ResultHandler.class})
})
public class MyPlugin implements Interceptor {
    @Override
    public Object intercept(Invocation invocation) throws Throwable {
        // 前置处理
        Object result = invocation.proceed();
        // 后置处理
        return result;
    }
}
```

---

### Q18: MyBatis 有哪些 Executor 类型？

**答案**：

| 类型 | 描述 |
|------|------|
| `SimpleExecutor` | 默认，每次执行创建新 Statement |
| `ReuseExecutor` | 重用 Statement |
| `BatchExecutor` | 批量执行，适合大量 insert/update |

```java
SqlSession session = sqlSessionFactory.openSession(ExecutorType.BATCH);
```

---

### Q19: MyBatis 如何防止 SQL 注入？

**答案**：

1. **使用 #{}**：预编译参数，使用 PreparedStatement
2. **避免 ${}**：或对 ${} 参数进行白名单校验
3. **使用动态 SQL 标签**：避免手动拼接 SQL

```xml
<!-- 安全：预编译 -->
WHERE name = #{name}

<!-- 危险：字符串拼接 -->
WHERE name = '${name}'
```

---

### Q20: MyBatis 延迟加载的原理？

**答案**：

MyBatis 延迟加载基于 **动态代理**（CGLIB 或 Javassist）。

**原理**：
1. 查询主对象时，关联对象返回代理对象
2. 访问关联对象属性时，触发代理方法
3. 代理方法执行关联查询 SQL
4. 将结果填充到关联对象

**配置**：

```xml
<settings>
    <setting name="lazyLoadingEnabled" value="true"/>
    <setting name="aggressiveLazyLoading" value="false"/>
</settings>
```

**触发加载的方法**：equals、clone、hashCode、toString

## Spring 集成

### Q21: MyBatis 与 Spring 集成的核心类？

**答案**：

| 类 | 描述 |
|------|------|
| `SqlSessionFactoryBean` | 创建 SqlSessionFactory |
| `MapperScannerConfigurer` | 扫描 Mapper 接口 |
| `SqlSessionTemplate` | 线程安全的 SqlSession |
| `MapperFactoryBean` | 创建 Mapper 代理 |

---

### Q22: Spring 中 MyBatis 的一级缓存为什么不生效？

**答案**：

Spring 默认每次调用 Mapper 方法都创建新的 SqlSession，因此一级缓存不生效。

**解决方案**：在事务中执行

```java
@Transactional
public void test() {
    // 同一事务中使用同一个 SqlSession
    User user1 = userMapper.selectById(1L);  // 查询
    User user2 = userMapper.selectById(1L);  // 从缓存获取
}
```

---

### Q23: @MapperScan 和 @Mapper 的区别？

**答案**：

| 注解 | 位置 | 作用 |
|------|------|------|
| `@Mapper` | Mapper 接口 | 标记单个接口为 Mapper |
| `@MapperScan` | 配置类 | 批量扫描包下所有 Mapper |

```java
// @Mapper：每个接口都要加
@Mapper
public interface UserMapper { }

// @MapperScan：一次配置，扫描整个包
@MapperScan("com.example.mapper")
public class Application { }
```

## 相关链接

- [MyBatis 学习指南](/docs/mybatis)
- [核心概念](/docs/mybatis/core-concepts)
- [缓存机制](/docs/mybatis/caching)
- [动态 SQL](/docs/mybatis/dynamic-sql)
- [Spring 集成](/docs/mybatis/spring-integration)

---

**最后更新**: 2025 年 12 月
