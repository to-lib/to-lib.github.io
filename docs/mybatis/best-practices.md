---
sidebar_position: 10
title: MyBatis 最佳实践
---

# MyBatis 最佳实践

本章总结 MyBatis 开发中的最佳实践，包括 Mapper 设计规范、SQL 编写规范、性能优化和项目结构建议。

## Mapper 接口设计规范

### 命名规范

```java
// ✅ 推荐：使用 Mapper 后缀
public interface UserMapper {}
public interface OrderMapper {}

// ❌ 不推荐
public interface UserDao {}
public interface UserRepository {}
```

### 方法命名

```java
public interface UserMapper {
    
    // 查询单条
    User selectById(Long id);
    User selectByUsername(String username);
    
    // 查询列表
    List<User> selectAll();
    List<User> selectByStatus(Integer status);
    List<User> selectByCondition(UserQuery query);
    
    // 查询数量
    int countByStatus(Integer status);
    long countAll();
    
    // 插入
    int insert(User user);
    int insertSelective(User user);
    int batchInsert(List<User> users);
    
    // 更新
    int update(User user);
    int updateSelective(User user);
    int updateStatus(Long id, Integer status);
    
    // 删除
    int deleteById(Long id);
    int deleteByIds(List<Long> ids);
    int deleteByCondition(UserQuery query);
    
    // 存在性检查
    boolean existsById(Long id);
    boolean existsByUsername(String username);
}
```

### 参数设计

```java
// ✅ 单参数：直接使用
User selectById(Long id);

// ✅ 多参数：使用 @Param
User selectByNameAndAge(@Param("name") String name, @Param("age") Integer age);

// ✅ 复杂查询：使用查询对象
List<User> selectByCondition(UserQuery query);

// ✅ 分页查询：使用分页对象
List<User> selectPage(@Param("query") UserQuery query, @Param("page") PageParam page);

// ❌ 避免：过多参数
User select(String name, Integer age, String email, Integer status, Date startTime);
```

### 查询对象设计

```java
@Data
public class UserQuery {
    private String name;
    private String email;
    private Integer status;
    private Integer minAge;
    private Integer maxAge;
    private Date startTime;
    private Date endTime;
    private List<Long> ids;
    
    // 排序
    private String orderBy;
    private String orderDir;
}

@Data
public class PageParam {
    private Integer pageNum = 1;
    private Integer pageSize = 10;
    
    public Integer getOffset() {
        return (pageNum - 1) * pageSize;
    }
}
```

## SQL 编写规范

### 基本规范

```xml
<!-- ✅ 推荐：使用别名，字段明确 -->
<select id="selectById" resultType="User">
    SELECT 
        u.id,
        u.name,
        u.email,
        u.age,
        u.create_time
    FROM user u
    WHERE u.id = #{id}
</select>

<!-- ❌ 避免：SELECT * -->
<select id="selectById" resultType="User">
    SELECT * FROM user WHERE id = #{id}
</select>
```

### SQL 片段复用

```xml
<!-- 定义通用字段 -->
<sql id="baseColumns">
    id, name, email, age, status, create_time, update_time
</sql>

<!-- 定义通用条件 -->
<sql id="whereCondition">
    <where>
        <if test="name != null and name != ''">
            AND name LIKE CONCAT('%', #{name}, '%')
        </if>
        <if test="status != null">
            AND status = #{status}
        </if>
        <if test="minAge != null">
            AND age >= #{minAge}
        </if>
        <if test="maxAge != null">
            AND age &lt;= #{maxAge}
        </if>
    </where>
</sql>

<!-- 复用 -->
<select id="selectByCondition" resultType="User">
    SELECT <include refid="baseColumns"/>
    FROM user
    <include refid="whereCondition"/>
    ORDER BY create_time DESC
</select>
```

### 防止 SQL 注入

```xml
<!-- ✅ 使用 #{} 预编译参数 -->
<select id="selectByName" resultType="User">
    SELECT * FROM user WHERE name = #{name}
</select>

<!-- ⚠️ ${} 只用于表名、列名等，需要白名单校验 -->
<select id="selectWithSort" resultType="User">
    SELECT * FROM user
    ORDER BY
    <choose>
        <when test="orderBy == 'name'">name</when>
        <when test="orderBy == 'age'">age</when>
        <otherwise>id</otherwise>
    </choose>
    <choose>
        <when test="orderDir == 'asc'">ASC</when>
        <otherwise>DESC</otherwise>
    </choose>
</select>

<!-- ❌ 危险：直接拼接用户输入 -->
<select id="selectByName" resultType="User">
    SELECT * FROM user WHERE name = '${name}'
</select>
```

### 批量操作优化

```xml
<!-- ✅ 批量插入 -->
<insert id="batchInsert">
    INSERT INTO user (name, email, age)
    VALUES
    <foreach collection="list" item="user" separator=",">
        (#{user.name}, #{user.email}, #{user.age})
    </foreach>
</insert>

<!-- ✅ 批量更新（CASE WHEN） -->
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

<!-- ⚠️ 控制批量大小，避免 SQL 过长 -->
```

### 分页查询

```xml
<!-- MySQL 分页 -->
<select id="selectPage" resultType="User">
    SELECT <include refid="baseColumns"/>
    FROM user
    <include refid="whereCondition"/>
    ORDER BY create_time DESC
    LIMIT #{page.offset}, #{page.pageSize}
</select>

<!-- 推荐：使用 PageHelper 插件 -->
```

## 性能优化

### 1. 避免 N+1 查询

```xml
<!-- ❌ N+1 问题：嵌套查询 -->
<resultMap id="userWithOrders" type="User">
    <id property="id" column="id"/>
    <collection property="orders" column="id"
                select="com.example.mapper.OrderMapper.selectByUserId"/>
</resultMap>

<!-- ✅ 推荐：联表查询 -->
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

### 2. 合理使用缓存

```xml
<!-- 适合缓存：查询频繁、数据变化少 -->
<cache eviction="LRU" flushInterval="300000" size="1024"/>

<!-- 不适合缓存的查询 -->
<select id="selectRealtime" useCache="false" flushCache="true">
    SELECT balance FROM account WHERE id = #{id}
</select>
```

### 3. 延迟加载

```xml
<!-- 全局配置 -->
<settings>
    <setting name="lazyLoadingEnabled" value="true"/>
    <setting name="aggressiveLazyLoading" value="false"/>
</settings>

<!-- 单独配置 -->
<association property="user" fetchType="lazy" .../>
<collection property="orders" fetchType="lazy" .../>
```

### 4. 批量执行

```java
// 使用 BATCH 执行器
try (SqlSession session = sqlSessionFactory.openSession(ExecutorType.BATCH)) {
    UserMapper mapper = session.getMapper(UserMapper.class);
    
    for (int i = 0; i < users.size(); i++) {
        mapper.insert(users.get(i));
        
        // 每 500 条刷新一次
        if (i % 500 == 0) {
            session.flushStatements();
        }
    }
    
    session.flushStatements();
    session.commit();
}
```

### 5. 流式查询

```java
// 大数据量查询，避免内存溢出
@Select("SELECT * FROM user")
@Options(resultSetType = ResultSetType.FORWARD_ONLY, fetchSize = 1000)
void selectLargeData(ResultHandler<User> handler);

// 使用
userMapper.selectLargeData(context -> {
    User user = context.getResultObject();
    // 处理每条记录
    processUser(user);
});
```

### 6. 索引优化

```sql
-- 为常用查询条件创建索引
CREATE INDEX idx_user_name ON user(name);
CREATE INDEX idx_user_status_create_time ON user(status, create_time);

-- 避免索引失效
-- ❌ 函数操作
WHERE DATE(create_time) = '2024-01-01'
-- ✅ 范围查询
WHERE create_time >= '2024-01-01' AND create_time < '2024-01-02'

-- ❌ 前缀模糊
WHERE name LIKE '%张'
-- ✅ 后缀模糊
WHERE name LIKE '张%'
```

## 项目结构建议

### 标准项目结构

```
src/main/java
├── com.example
│   ├── config
│   │   └── MyBatisConfig.java
│   ├── entity
│   │   ├── User.java
│   │   └── Order.java
│   ├── mapper
│   │   ├── UserMapper.java
│   │   └── OrderMapper.java
│   ├── service
│   │   ├── UserService.java
│   │   └── impl
│   │       └── UserServiceImpl.java
│   ├── controller
│   │   └── UserController.java
│   └── dto
│       ├── UserQuery.java
│       └── UserDTO.java
│
src/main/resources
├── application.yml
├── mapper
│   ├── UserMapper.xml
│   └── OrderMapper.xml
└── mybatis-config.xml (可选)
```

### 多模块项目结构

```
project
├── project-common          # 公共模块
│   └── src/main/java
│       └── com.example.common
│           ├── entity      # 实体类
│           └── dto         # 数据传输对象
│
├── project-dao             # 数据访问模块
│   └── src/main
│       ├── java
│       │   └── com.example.dao
│       │       └── mapper  # Mapper 接口
│       └── resources
│           └── mapper      # Mapper XML
│
├── project-service         # 业务逻辑模块
│   └── src/main/java
│       └── com.example.service
│
└── project-web             # Web 模块
    └── src/main/java
        └── com.example.web
            └── controller
```

### Mapper XML 与接口同包

```
src/main
├── java
│   └── com.example.mapper
│       └── UserMapper.java
└── resources
    └── com/example/mapper    # 与接口同包路径
        └── UserMapper.xml
```

配置：

```yaml
mybatis:
  mapper-locations: classpath*:com/example/mapper/*.xml
```

## 常见问题处理

### 1. 驼峰命名映射

```yaml
mybatis:
  configuration:
    map-underscore-to-camel-case: true
```

### 2. 空值处理

```xml
<!-- 插入时处理 null -->
<insert id="insert">
    INSERT INTO user (name, email)
    VALUES (#{name, jdbcType=VARCHAR}, #{email, jdbcType=VARCHAR})
</insert>

<!-- 配置空值调用 setter -->
<settings>
    <setting name="callSettersOnNulls" value="true"/>
</settings>
```

### 3. 枚举处理

```java
// 方式一：使用枚举名称
public enum Status {
    ACTIVE, INACTIVE
}

// 方式二：使用枚举值
public enum Status {
    ACTIVE(1), INACTIVE(0);
    
    private final int value;
    
    @EnumValue  // MyBatis-Plus
    public int getValue() {
        return value;
    }
}

// 自定义类型处理器
@MappedTypes(Status.class)
public class StatusTypeHandler extends BaseTypeHandler<Status> {
    // ...
}
```

### 4. 日期处理

```java
// 使用 Java 8 日期类型
private LocalDateTime createTime;
private LocalDate birthDate;

// MyBatis 3.4.5+ 自动支持
```

### 5. JSON 字段处理

```java
// 自定义类型处理器
@MappedTypes(JsonObject.class)
public class JsonTypeHandler extends BaseTypeHandler<JsonObject> {
    private final ObjectMapper mapper = new ObjectMapper();
    
    @Override
    public void setNonNullParameter(PreparedStatement ps, int i, 
            JsonObject parameter, JdbcType jdbcType) throws SQLException {
        ps.setString(i, mapper.writeValueAsString(parameter));
    }
    
    @Override
    public JsonObject getNullableResult(ResultSet rs, String columnName) 
            throws SQLException {
        String json = rs.getString(columnName);
        return json == null ? null : mapper.readValue(json, JsonObject.class);
    }
    // ...
}
```

## 代码规范检查清单

### Mapper 接口

- [ ] 使用 `@Mapper` 或 `@MapperScan` 注解
- [ ] 方法命名清晰，遵循命名规范
- [ ] 多参数使用 `@Param` 注解
- [ ] 复杂查询使用查询对象

### Mapper XML

- [ ] namespace 与接口全限定名一致
- [ ] 避免使用 `SELECT *`
- [ ] 使用 `#{}` 防止 SQL 注入
- [ ] 复用 SQL 片段
- [ ] 动态 SQL 条件完整

### 性能

- [ ] 避免 N+1 查询
- [ ] 合理使用缓存
- [ ] 批量操作控制大小
- [ ] 大数据量使用流式查询

### 事务

- [ ] Service 层使用 `@Transactional`
- [ ] 只读操作使用 `readOnly = true`
- [ ] 指定回滚异常类型

## 相关链接

- [配置详解](/docs/mybatis/configuration) - MyBatis 配置
- [动态 SQL](/docs/mybatis/dynamic-sql) - 动态 SQL 编写
- [缓存机制](/docs/mybatis/caching) - 缓存优化
- [Spring 集成](/docs/mybatis/spring-integration) - Spring 集成

---

**最后更新**: 2025 年 12 月
