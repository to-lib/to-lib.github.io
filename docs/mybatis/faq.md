---
sidebar_position: 12
title: MyBatis 常见问题
---

# MyBatis 常见问题

本页汇总 MyBatis 开发中的常见问题和解决方案。

## 配置相关问题

### Q: Mapper 接口找不到，报 `Invalid bound statement`

**原因**：Mapper XML 未正确加载或 namespace 不匹配。

**解决方案**：

1. 检查 XML 文件位置和配置：
```yaml
mybatis:
  mapper-locations: classpath:mapper/*.xml
```

2. 确保 namespace 与接口全限定名一致：
```xml
<mapper namespace="com.example.mapper.UserMapper">
```

3. 检查方法 id 与接口方法名一致：
```xml
<select id="selectById" ...>  <!-- 对应 UserMapper.selectById() -->
```

---

### Q: 驼峰命名映射不生效

**原因**：未开启驼峰命名自动映射。

**解决方案**：

```yaml
mybatis:
  configuration:
    map-underscore-to-camel-case: true
```

或在 mybatis-config.xml 中：
```xml
<settings>
    <setting name="mapUnderscoreToCamelCase" value="true"/>
</settings>
```

---

### Q: 配置文件加载顺序问题

**原因**：同时使用 mybatis-config.xml 和 application.yml 配置。

**解决方案**：

- 优先使用 application.yml 配置
- 如需使用 mybatis-config.xml，确保不重复配置：

```yaml
mybatis:
  config-location: classpath:mybatis-config.xml
  # 不要同时配置 configuration 属性
```

---

### Q: 日志不输出 SQL

**解决方案**：

1. 配置日志实现：
```yaml
mybatis:
  configuration:
    log-impl: org.apache.ibatis.logging.slf4j.Slf4jImpl
```

2. 配置日志级别（logback）：
```xml
<logger name="com.example.mapper" level="DEBUG"/>
```

3. 或使用 application.yml：
```yaml
logging:
  level:
    com.example.mapper: debug
```

## 映射相关问题

### Q: 查询结果为 null，但数据库有数据

**可能原因**：

1. 字段名与属性名不匹配
2. 未开启驼峰映射
3. resultType 类型错误

**解决方案**：

```xml
<!-- 方案一：使用 resultMap -->
<resultMap id="userMap" type="User">
    <result property="userName" column="user_name"/>
</resultMap>

<!-- 方案二：使用别名 -->
<select id="selectById" resultType="User">
    SELECT user_name AS userName FROM user WHERE id = #{id}
</select>

<!-- 方案三：开启驼峰映射 -->
```

---

### Q: 插入后获取不到自增 ID

**解决方案**：

```xml
<insert id="insert" useGeneratedKeys="true" keyProperty="id">
    INSERT INTO user (name) VALUES (#{name})
</insert>
```

```java
// 插入后从对象获取 ID
userMapper.insert(user);
Long id = user.getId();  // 自增 ID 已回填
```

---

### Q: 参数传递问题，报 `Parameter 'xxx' not found`

**原因**：多参数未使用 @Param 注解。

**解决方案**：

```java
// ❌ 错误
List<User> select(String name, Integer age);

// ✅ 正确
List<User> select(@Param("name") String name, @Param("age") Integer age);
```

---

### Q: #{} 和 ${} 的区别

| 语法 | 特点 | 使用场景 |
|------|------|----------|
| `#{}` | 预编译，防 SQL 注入 | 参数值 |
| `${}` | 字符串替换，有注入风险 | 表名、列名、排序 |

```xml
<!-- 参数值用 #{} -->
WHERE name = #{name}

<!-- 动态表名/列名用 ${} -->
ORDER BY ${orderColumn}
```

---

### Q: 模糊查询写法

```xml
<!-- 方式一：CONCAT（推荐） -->
WHERE name LIKE CONCAT('%', #{name}, '%')

<!-- 方式二：bind -->
<bind name="pattern" value="'%' + name + '%'"/>
WHERE name LIKE #{pattern}

<!-- ❌ 错误：直接拼接 -->
WHERE name LIKE '%${name}%'
```

---

### Q: 一对多查询结果只有一条

**原因**：主键映射缺失，MyBatis 无法区分不同记录。

**解决方案**：

```xml
<resultMap id="userWithOrders" type="User">
    <id property="id" column="id"/>  <!-- 必须有 id 映射 -->
    <result property="name" column="name"/>
    <collection property="orders" ofType="Order">
        <id property="id" column="order_id"/>  <!-- 必须有 id 映射 -->
        <result property="orderNo" column="order_no"/>
    </collection>
</resultMap>
```

---

### Q: 枚举类型映射

**方式一：使用枚举名称（默认）**

```java
public enum Status { ACTIVE, INACTIVE }
```

数据库存储：`ACTIVE`、`INACTIVE`

**方式二：使用枚举值**

```java
public enum Status {
    ACTIVE(1), INACTIVE(0);
    private final int code;
}

// 自定义 TypeHandler
@MappedTypes(Status.class)
public class StatusTypeHandler extends BaseTypeHandler<Status> {
    @Override
    public void setNonNullParameter(PreparedStatement ps, int i, 
            Status parameter, JdbcType jdbcType) throws SQLException {
        ps.setInt(i, parameter.getCode());
    }
    // ...
}
```

## 性能相关问题

### Q: N+1 查询问题

**问题**：嵌套查询导致大量 SQL 执行。

```xml
<!-- 问题代码：每个用户都会执行一次 selectOrders -->
<collection property="orders" column="id" select="selectOrders"/>
```

**解决方案**：

1. 使用联表查询：
```xml
<select id="selectWithOrders" resultMap="userWithOrders">
    SELECT u.*, o.id as order_id, o.order_no
    FROM user u LEFT JOIN orders o ON u.id = o.user_id
</select>
```

2. 使用延迟加载：
```xml
<collection property="orders" fetchType="lazy" .../>
```

---

### Q: 批量插入性能差

**解决方案**：

1. 使用 foreach 批量插入：
```xml
<insert id="batchInsert">
    INSERT INTO user (name, email) VALUES
    <foreach collection="list" item="u" separator=",">
        (#{u.name}, #{u.email})
    </foreach>
</insert>
```

2. 使用 BATCH 执行器：
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

3. 控制批量大小（建议 500-1000 条）

---

### Q: 大数据量查询内存溢出

**解决方案**：使用流式查询

```java
@Select("SELECT * FROM user")
@Options(resultSetType = ResultSetType.FORWARD_ONLY, fetchSize = 1000)
void selectAll(ResultHandler<User> handler);

// 使用
userMapper.selectAll(context -> {
    User user = context.getResultObject();
    processUser(user);
});
```

---

### Q: 缓存不生效

**一级缓存不生效原因**：
- 不同 SqlSession
- 执行了增删改操作
- 调用了 clearCache()

**二级缓存不生效原因**：
- 未开启二级缓存
- 实体类未实现 Serializable
- SqlSession 未提交/关闭
- 配置了 useCache="false"

## Spring 集成问题

### Q: Spring 中一级缓存不生效

**原因**：Spring 默认每次调用创建新的 SqlSession。

**解决方案**：在事务中执行

```java
@Transactional
public void test() {
    User user1 = userMapper.selectById(1L);  // 查询
    User user2 = userMapper.selectById(1L);  // 从缓存获取
}
```

---

### Q: @Transactional 不生效

**可能原因**：

1. 方法不是 public
2. 同类内部调用
3. 异常被捕获
4. 未启用事务管理

**解决方案**：

```java
// 1. 确保方法是 public
@Transactional
public void save() { ... }

// 2. 避免同类调用，或注入自身
@Autowired
private UserService self;

public void outer() {
    self.inner();  // 通过代理调用
}

// 3. 正确抛出异常
@Transactional(rollbackFor = Exception.class)
public void save() throws Exception {
    // 不要 catch 异常后不抛出
}

// 4. 启用事务管理
@EnableTransactionManagement
```

---

### Q: 多数据源事务问题

**问题**：跨数据源事务无法保证一致性。

**解决方案**：

1. 使用分布式事务（Seata）
2. 使用本地消息表
3. 业务上允许最终一致性

---

### Q: Mapper 注入为 null

**解决方案**：

1. 确保 Mapper 被扫描：
```java
@MapperScan("com.example.mapper")
```

2. 或使用 @Mapper 注解：
```java
@Mapper
public interface UserMapper { }
```

3. 检查包路径是否正确

## 动态 SQL 问题

### Q: if 判断字符串相等

```xml
<!-- ❌ 错误 -->
<if test="type == 'A'">...</if>

<!-- ✅ 正确 -->
<if test="type == 'A'.toString()">...</if>
<if test='type == "A"'>...</if>
```

---

### Q: 特殊字符报错

```xml
<!-- ❌ 错误：< > 是 XML 特殊字符 -->
<if test="age > 18">...</if>

<!-- ✅ 正确：使用转义 -->
<if test="age &gt; 18">...</if>

<!-- ✅ 正确：使用 CDATA -->
<![CDATA[AND age > 18]]>
```

---

### Q: foreach collection 取值

| 参数类型 | collection 值 |
|----------|---------------|
| List | `list` 或 `collection` |
| Array | `array` |
| @Param("ids") | `ids` |
| Map 中的 key | 对应的 key 名 |

---

### Q: where 标签不去除 AND

**原因**：AND 前有空格或换行。

```xml
<!-- ❌ 问题 -->
<where>
    <if test="name != null">
        AND name = #{name}  <!-- AND 前有换行 -->
    </if>
</where>

<!-- ✅ 正确 -->
<where>
    <if test="name != null">AND name = #{name}</if>
</where>
```

## 其他问题

### Q: 时区问题

**解决方案**：

```properties
# 数据库连接添加时区
spring.datasource.url=jdbc:mysql://localhost:3306/db?serverTimezone=Asia/Shanghai
```

---

### Q: 中文乱码

**解决方案**：

```properties
# 数据库连接添加编码
spring.datasource.url=jdbc:mysql://localhost:3306/db?characterEncoding=utf8
```

---

### Q: 连接池耗尽

**解决方案**：

1. 增加连接池大小
2. 检查是否有连接泄漏
3. 减少长事务

```yaml
spring:
  datasource:
    hikari:
      maximum-pool-size: 20
      minimum-idle: 5
      connection-timeout: 30000
```

## 相关链接

- [配置详解](/docs/mybatis/configuration)
- [最佳实践](/docs/mybatis/best-practices)
- [Spring 集成](/docs/mybatis/spring-integration)

---

**最后更新**: 2025 年 12 月
