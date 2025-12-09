---
sidebar_position: 7
---

# 数据访问

> [!TIP]
> **数据持久化核心**: 本章介绍 Spring Boot 集成JPA、MyBatis等数据访问技术。掌握事务管理和连接池配置是生产环境的关键。

## Spring Data JPA

### 依赖配置

```xml
<!-- Spring Data JPA -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-jpa</artifactId>
</dependency>

<!-- MySQL Driver -->
<dependency>
    <groupId>com.mysql</groupId>
    <artifactId>mysql-connector-java</artifactId>
    <scope>runtime</scope>
</dependency>
```

### 配置文件

```yaml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/mydb?useSSL=false&serverTimezone=UTC
    username: root
    password: password
    driver-class-name: com.mysql.cj.jdbc.Driver
    hikari:
      maximum-pool-size: 20
      minimum-idle: 5
  
  jpa:
    hibernate:
      ddl-auto: update  # create, create-drop, update, validate
    show-sql: false
    properties:
      hibernate:
        dialect: org.hibernate.dialect.MySQL8Dialect
        format_sql: true
        use_sql_comments: true
```

## 实体类（Entity）

### 基本实体

```java
import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;

@Entity
@Table(name = "users")
@Data
public class User {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(name = "username", nullable = false, length = 50, unique = true)
    private String username;
    
    @Column(name = "email", nullable = false, unique = true)
    private String email;
    
    @Column(name = "age")
    private Integer age;
    
    @Column(name = "status", nullable = false, columnDefinition = "varchar(20) default 'ACTIVE'")
    private String status;
    
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;
    
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;
    
    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }
    
    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }
}
```

### 关联关系

#### 一对多关系

```java
// 用户表
@Entity
@Table(name = "users")
@Data
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    private String username;
    
    @OneToMany(mappedBy = "user", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private List<Post> posts = new ArrayList<>();
}

// 文章表
@Entity
@Table(name = "posts")
@Data
public class Post {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    private String title;
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;
}
```

#### 多对多关系

```java
// 用户角色（多对多）
@Entity
@Table(name = "users")
@Data
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    private String username;
    
    @ManyToMany(fetch = FetchType.LAZY)
    @JoinTable(
            name = "user_role",
            joinColumns = @JoinColumn(name = "user_id"),
            inverseJoinColumns = @JoinColumn(name = "role_id")
    )
    private Set<Role> roles = new HashSet<>();
}

@Entity
@Table(name = "roles")
@Data
public class Role {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    private String name;
    
    @ManyToMany(mappedBy = "roles")
    private Set<User> users = new HashSet<>();
}
```

## Repository 接口

### 基本 CRUD 操作

```java
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.Optional;
import java.util.List;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    
    // JpaRepository 已提供的基本方法：
    // - save(User user)
    // - saveAll(List<User> users)
    // - findById(Long id)
    // - findAll()
    // - count()
    // - delete(User user)
    // - deleteById(Long id)
    // - deleteAll()
    // - exists(Long id)
}
```

### 自定义查询方法

```java
@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    
    // 按属性查询
    User findByUsername(String username);
    
    Optional<User> findByEmail(String email);
    
    List<User> findByAgeGreaterThan(Integer age);
    
    // 多条件查询
    List<User> findByUsernameAndEmail(String username, String email);
    
    // 模糊查询
    List<User> findByUsernameLike(String username);
    
    // 排序查询
    List<User> findByAgeGreaterThanOrderByUsernameAsc(Integer age);
    
    // 分页查询
    Page<User> findByAgeGreaterThan(Integer age, Pageable pageable);
}
```

### @Query 自定义 SQL

```java
@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    
    // JPQL 查询
    @Query("SELECT u FROM User u WHERE u.username = ?1")
    User findByUsernameJpql(String username);
    
    // JPQL 查询（命名参数）
    @Query("SELECT u FROM User u WHERE u.username = :username AND u.status = :status")
    List<User> findByUsernameAndStatus(
            @Param("username") String username,
            @Param("status") String status);
    
    // 原生 SQL 查询
    @Query(value = "SELECT * FROM users WHERE age > ?1", nativeQuery = true)
    List<User> findByAgeGreaterThanNative(Integer age);
    
    // 修改查询
    @Modifying
    @Transactional
    @Query("UPDATE User u SET u.status = 'INACTIVE' WHERE u.id = ?1")
    void inactivateUser(Long userId);
    
    // 删除查询
    @Modifying
    @Transactional
    @Query("DELETE FROM User u WHERE u.age > ?1")
    void deleteByAgeGreaterThan(Integer age);
}
```

## Service 层

```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@Transactional
public class UserService {
    
    @Autowired
    private UserRepository userRepository;
    
    // 创建用户
    public User createUser(User user) {
        return userRepository.save(user);
    }
    
    // 获取用户
    public User getUserById(Long id) {
        return userRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("User not found"));
    }
    
    // 获取所有用户
    @Transactional(readOnly = true)
    public List<User> getAllUsers() {
        return userRepository.findAll();
    }
    
    // 分页查询
    @Transactional(readOnly = true)
    public Page<User> getUsersByAge(Integer age, Pageable pageable) {
        return userRepository.findByAgeGreaterThan(age, pageable);
    }
    
    // 更新用户
    public User updateUser(Long id, User userDetails) {
        User user = getUserById(id);
        user.setUsername(userDetails.getUsername());
        user.setEmail(userDetails.getEmail());
        user.setAge(userDetails.getAge());
        return userRepository.save(user);
    }
    
    // 删除用户
    public void deleteUser(Long id) {
        userRepository.deleteById(id);
    }
    
    // 批量操作
    public List<User> saveAll(List<User> users) {
        return userRepository.saveAll(users);
    }
}
```

## 查询示例

### 分页查询

```java
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;

@RestController
@RequestMapping("/api/users")
public class UserController {
    
    @Autowired
    private UserService userService;
    
    @GetMapping
    public ResponseEntity<Page<User>> getUsers(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size,
            @RequestParam(defaultValue = "id") String sortBy,
            @RequestParam(defaultValue = "ASC") Sort.Direction direction) {
        
        Pageable pageable = PageRequest.of(page, size, Sort.by(direction, sortBy));
        Page<User> users = userService.getAllUsers(pageable);
        
        return ResponseEntity.ok(users);
    }
}
```

### 条件查询

```java
import org.springframework.data.jpa.domain.Specification;
import jakarta.persistence.criteria.CriteriaBuilder;
import jakarta.persistence.criteria.CriteriaQuery;
import jakarta.persistence.criteria.Predicate;
import jakarta.persistence.criteria.Root;

@Service
public class UserService {
    
    @Autowired
    private UserRepository userRepository;
    
    public List<User> searchUsers(String username, Integer minAge, Integer maxAge) {
        Specification<User> spec = (root, query, cb) -> {
            List<Predicate> predicates = new ArrayList<>();
            
            if (username != null && !username.isEmpty()) {
                predicates.add(cb.like(root.get("username"), "%" + username + "%"));
            }
            
            if (minAge != null) {
                predicates.add(cb.greaterThanOrEqualTo(root.get("age"), minAge));
            }
            
            if (maxAge != null) {
                predicates.add(cb.lessThanOrEqualTo(root.get("age"), maxAge));
            }
            
            return cb.and(predicates.toArray(new Predicate[0]));
        };
        
        return userRepository.findAll(spec);
    }
}
```

## MyBatis 集成

### 依赖配置

```xml
<dependency>
    <groupId>org.mybatis.spring.boot</groupId>
    <artifactId>mybatis-spring-boot-starter</artifactId>
    <version>3.0.0</version>
</dependency>
```

### 配置文件

```yaml
mybatis:
  mapper-locations: classpath:mapper/*.xml
  type-aliases-package: com.example.entity
  configuration:
    map-underscore-to-camel-case: true
    cache-enabled: true
    default-executor-type: reuse
```

### Mapper 接口

```java
import org.apache.ibatis.annotations.*;

@Mapper
public interface UserMapper {
    
    @Select("SELECT * FROM users WHERE id = #{id}")
    @Results({
            @Result(property = "id", column = "id"),
            @Result(property = "username", column = "username")
    })
    User findById(@Param("id") Long id);
    
    @Select("SELECT * FROM users")
    List<User> findAll();
    
    @Insert("INSERT INTO users(username, email, age) VALUES(#{username}, #{email}, #{age})")
    @Options(useGeneratedKeys = true, keyProperty = "id")
    int insert(User user);
    
    @Update("UPDATE users SET username = #{username}, email = #{email} WHERE id = #{id}")
    int update(User user);
    
    @Delete("DELETE FROM users WHERE id = #{id}")
    int delete(@Param("id") Long id);
}
```

### XML 映射文件

`src/main/resources/mapper/UserMapper.xml`:

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" 
        "http://mybatis.org/dtd/mybatis-3-mapper.dtd">

<mapper namespace="com.example.mapper.UserMapper">
    
    <sql id="userColumns">
        id, username, email, age, created_at, updated_at
    </sql>
    
    <select id="findById" resultType="User">
        SELECT <include refid="userColumns"/>
        FROM users
        WHERE id = #{id}
    </select>
    
    <select id="findAll" resultType="User">
        SELECT <include refid="userColumns"/>
        FROM users
    </select>
    
    <select id="searchUsers" resultType="User">
        SELECT <include refid="userColumns"/>
        FROM users
        WHERE 1=1
        <if test="username != null and username != ''">
            AND username LIKE CONCAT('%', #{username}, '%')
        </if>
        <if test="minAge != null">
            AND age >= #{minAge}
        </if>
    </select>
    
    <insert id="insert" useGeneratedKeys="true" keyProperty="id">
        INSERT INTO users(username, email, age, created_at, updated_at)
        VALUES(#{username}, #{email}, #{age}, NOW(), NOW())
    </insert>
    
    <update id="update">
        UPDATE users
        <set>
            <if test="username != null">username = #{username},</if>
            <if test="email != null">email = #{email},</if>
            <if test="age != null">age = #{age},</if>
            updated_at = NOW()
        </set>
        WHERE id = #{id}
    </update>
    
    <delete id="delete">
        DELETE FROM users WHERE id = #{id}
    </delete>
    
</mapper>
```

## 事务管理

```java
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Propagation;

@Service
public class UserService {
    
    // 基本事务
    @Transactional
    public void transferMoney(Long fromId, Long toId, BigDecimal amount) {
        User from = userRepository.findById(fromId).orElseThrow();
        User to = userRepository.findById(toId).orElseThrow();
        
        from.setBalance(from.getBalance().subtract(amount));
        to.setBalance(to.getBalance().add(amount));
        
        userRepository.save(from);
        userRepository.save(to);
    }
    
    // 只读事务
    @Transactional(readOnly = true)
    public User getUser(Long id) {
        return userRepository.findById(id).orElseThrow();
    }
    
    // 指定隔离级别
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void updateUser(Long id, User userDetails) {
        User user = userRepository.findById(id).orElseThrow();
        user.setUsername(userDetails.getUsername());
        userRepository.save(user);
    }
    
    // 异常回滚配置
    @Transactional(rollbackFor = Exception.class, noRollbackFor = {ValidationException.class})
    public void createUserWithValidation(User user) {
        // 代码...
    }
}
```

## 总结

- **JPA** - 使用 Spring Data JPA 进行 ORM 映射
- **Repository** - 使用 Repository 接口进行数据访问
- **查询** - 支持自定义查询方法和 @Query 注解
- **MyBatis** - 对于复杂 SQL 使用 MyBatis
- **事务** - 使用 @Transactional 管理事务

下一步学习 [缓存管理](./cache-management.md)。
