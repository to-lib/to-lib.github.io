---
id: spring-data
title: Spring Data
sidebar_label: Spring Data
sidebar_position: 8
---

# Spring Data

> [!IMPORTANT] > **Spring Data 核心**: Spring Data 简化了数据访问层的开发，通过 Repository 接口自动生成实现。掌握查询方法命名规则和自定义查询是关键。

## 1. Spring Data 概述

**Spring Data** 是一个用于简化数据访问的项目集合，支持关系型和非关系型数据库。

### 1.1 主要模块

| 模块                    | 说明                | 应用场景             |
| ----------------------- | ------------------- | -------------------- |
| **Spring Data JPA**     | 基于 JPA 的数据访问 | 关系型数据库         |
| **Spring Data MongoDB** | MongoDB 数据访问    | 文档型数据库         |
| **Spring Data Redis**   | Redis 数据访问      | 缓存、会话存储       |
| **Spring Data JDBC**    | 轻量级 JDBC 封装    | 简单的关系型数据访问 |

### 1.2 核心特性

- **Repository 抽象** - 统一的数据访问接口
- **查询方法** - 通过方法名自动生成查询
- **自定义查询** - 使用 @Query 注解
- **分页和排序** - 内置分页支持
- **审计功能** - 自动记录创建和修改时间

## 2. Repository 接口

### 2.1 Repository 层次结构

```java
Repository                     // 标记接口
    ├── CrudRepository        // 基本 CRUD 操作
    ├── PagingAndSortingRepository  // 分页和排序
    └── JpaRepository         // JPA 特定功能
```

### 2.2 CrudRepository

提供基本的 CRUD 操作：

```java
public interface UserRepository extends CrudRepository<User, Long> {
    // 继承的方法：
    // save(S entity)
    // saveAll(Iterable<S> entities)
    // findById(ID id)
    // existsById(ID id)
    // findAll()
    // findAllById(Iterable<ID> ids)
    // count()
    // deleteById(ID id)
    // delete(T entity)
    // deleteAll()
}

// 使用示例
@Service
public class UserService {
    @Autowired
    private UserRepository userRepository;

    public void example() {
        // 保存
        User user = new User("John", "john@example.com");
        userRepository.save(user);

        // 查询
        Optional<User> found = userRepository.findById(1L);

        // 删除
        userRepository.deleteById(1L);

        // 统计
        long count = userRepository.count();
    }
}
```

### 2.3 JpaRepository

扩展了 CrudRepository，提供 JPA 特定功能：

```java
public interface UserRepository extends JpaRepository<User, Long> {
    // 继承的额外方法：
    // flush()
    // saveAndFlush(S entity)
    // deleteInBatch(Iterable<T> entities)
    // deleteAllInBatch()
    // getOne(ID id)  // 返回引用，延迟加载
}

// 使用示例
@Service
public class UserService {
    @Autowired
    private UserRepository userRepository;

    public void batchOperations() {
        // 批量删除（更高效）
        List<User> users = userRepository.findAll();
        userRepository.deleteInBatch(users);

        // 保存并立即刷新到数据库
        User user = new User("Jane", "jane@example.com");
        userRepository.saveAndFlush(user);
    }
}
```

## 3. 查询方法命名规则

### 3.1 基本规则

Spring Data 根据方法名自动生成查询：

```java
public interface UserRepository extends JpaRepository<User, Long> {

    // 根据单个属性查询
    User findByName(String name);
    User findByEmail(String email);

    // 多个条件 - AND
    User findByNameAndEmail(String name, String email);

    // 多个条件 - OR
    List<User> findByNameOrEmail(String name, String email);

    // 模糊查询
    List<User> findByNameContaining(String keyword);
    List<User> findByNameStartingWith(String prefix);
    List<User> findByNameEndingWith(String suffix);

    // 范围查询
    List<User> findByAgeGreaterThan(int age);
    List<User> findByAgeLessThan(int age);
    List<User> findByAgeBetween(int start, int end);

    // NULL 判断
    List<User> findByEmailIsNull();
    List<User> findByEmailIsNotNull();

    // 集合查询
    List<User> findByNameIn(Collection<String> names);
    List<User> findByNameNotIn(Collection<String> names);

    // 排序
    List<User> findByAgeOrderByNameAsc(int age);
    List<User> findByAgeOrderByNameDesc(int age);

    // 限制结果数量
    User findFirstByOrderByNameAsc();
    List<User> findTop3ByOrderByAgeDesc();

    // 去重
    List<User> findDistinctByName(String name);
}
```

### 3.2 关键字对照表

| 关键字           | 示例                      | JPQL                         |
| ---------------- | ------------------------- | ---------------------------- |
| And              | findByNameAndEmail        | where name = ? and email = ? |
| Or               | findByNameOrEmail         | where name = ? or email = ?  |
| Is, Equals       | findByName                | where name = ?               |
| Between          | findByAgeBetween          | where age between ? and ?    |
| LessThan         | findByAgeLessThan         | where age &lt; ?             |
| LessThanEqual    | findByAgeLessThanEqual    | where age &lt;= ?            |
| GreaterThan      | findByAgeGreaterThan      | where age &gt; ?             |
| GreaterThanEqual | findByAgeGreaterThanEqual | where age &gt;= ?            |
| After            | findByDateAfter           | where date &gt; ?            |
| Before           | findByDateBefore          | where date &lt; ?            |
| IsNull           | findByEmailIsNull         | where email is null          |
| IsNotNull        | findByEmailIsNotNull      | where email is not null      |
| Like             | findByNameLike            | where name like ?            |
| NotLike          | findByNameNotLike         | where name not like ?        |
| StartingWith     | findByNameStartingWith    | where name like '?%'         |
| EndingWith       | findByNameEndingWith      | where name like '%?'         |
| Containing       | findByNameContaining      | where name like '%?%'        |
| OrderBy          | findByAgeOrderByNameDesc  | order by name desc           |
| Not              | findByNameNot             | where name &lt;&gt; ?        |
| In               | findByAgeIn               | where age in (?)             |
| NotIn            | findByAgeNotIn            | where age not in (?)         |
| True             | findByActiveTrue          | where active = true          |
| False            | findByActiveFalse         | where active = false         |

## 4. 自定义查询

### 4.1 使用 @Query

当查询方法名过于复杂时，使用 @Query 注解：

```java
public interface UserRepository extends JpaRepository<User, Long> {

    // JPQL 查询
    @Query("SELECT u FROM User u WHERE u.email = ?1")
    User findByEmailAddress(String email);

    // 命名参数
    @Query("SELECT u FROM User u WHERE u.name = :name AND u.age > :age")
    List<User> findByNameAndAgeGreaterThan(
        @Param("name") String name,
        @Param("age") int age
    );

    // 原生 SQL 查询
    @Query(value = "SELECT * FROM users WHERE email = ?1", nativeQuery = true)
    User findByEmailNative(String email);

    // 更新查询
    @Modifying
    @Query("UPDATE User u SET u.status = :status WHERE u.id = :id")
    int updateUserStatus(@Param("id") Long id, @Param("status") String status);

    // 删除查询
    @Modifying
    @Query("DELETE FROM User u WHERE u.status = :status")
    int deleteByStatus(@Param("status") String status);

    // 聚合查询
    @Query("SELECT COUNT(u) FROM User u WHERE u.age > :age")
    long countByAgeGreaterThan(@Param("age") int age);

    // 投影查询（只查询部分字段）
    @Query("SELECT u.name, u.email FROM User u WHERE u.id = :id")
    Object[] findNameAndEmailById(@Param("id") Long id);
}
```

### 4.2 @Modifying 注意事项

```java
@Service
public class UserService {
    @Autowired
    private UserRepository userRepository;

    @Transactional  // 必须在事务中
    public int updateStatus(Long id, String status) {
        return userRepository.updateUserStatus(id, status);
    }
}
```

## 5. 分页和排序

### 5.1 基本分页

```java
public interface UserRepository extends JpaRepository<User, Long> {
    Page<User> findByAge(int age, Pageable pageable);
}

@Service
public class UserService {
    @Autowired
    private UserRepository userRepository;

    public Page<User> getUsersByAge(int age, int page, int size) {
        // 创建分页请求
        Pageable pageable = PageRequest.of(page, size);
        return userRepository.findByAge(age, pageable);
    }

    public void printPageInfo(Page<User> page) {
        System.out.println("总页数: " + page.getTotalPages());
        System.out.println("总记录数: " + page.getTotalElements());
        System.out.println("当前页: " + page.getNumber());
        System.out.println("每页大小: " + page.getSize());
        System.out.println("当前页记录数: " + page.getNumberOfElements());
        System.out.println("是否第一页: " + page.isFirst());
        System.out.println("是否最后一页: " + page.isLast());

        List<User> users = page.getContent();
        users.forEach(System.out::println);
    }
}
```

### 5.2 排序

```java
@Service
public class UserService {
    @Autowired
    private UserRepository userRepository;

    public List<User> getUsersSorted() {
        // 单字段排序
        Sort sort = Sort.by(Sort.Direction.DESC, "age");
        return userRepository.findAll(sort);
    }

    public List<User> getUsersMultiSort() {
        // 多字段排序
        Sort sort = Sort.by(
            Sort.Order.desc("age"),
            Sort.Order.asc("name")
        );
        return userRepository.findAll(sort);
    }

    public Page<User> getUsersPagedAndSorted(int page, int size) {
        // 分页 + 排序
        Sort sort = Sort.by(Sort.Direction.DESC, "age");
        Pageable pageable = PageRequest.of(page, size, sort);
        return userRepository.findAll(pageable);
    }
}
```

## 6. 实体关系映射

### 6.1 一对一关系

```java
@Entity
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String name;

    @OneToOne(cascade = CascadeType.ALL)
    @JoinColumn(name = "profile_id")
    private UserProfile profile;
}

@Entity
public class UserProfile {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String bio;
    private String avatar;

    @OneToOne(mappedBy = "profile")
    private User user;
}
```

### 6.2 一对多关系

```java
@Entity
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String name;

    @OneToMany(mappedBy = "user", cascade = CascadeType.ALL)
    private List<Order> orders = new ArrayList<>();
}

@Entity
public class Order {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String orderNumber;

    @ManyToOne
    @JoinColumn(name = "user_id")
    private User user;
}

// Repository 查询
public interface UserRepository extends JpaRepository<User, Long> {
    // 查询有订单的用户
    @Query("SELECT DISTINCT u FROM User u JOIN FETCH u.orders")
    List<User> findUsersWithOrders();
}
```

### 6.3 多对多关系

```java
@Entity
public class Student {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String name;

    @ManyToMany
    @JoinTable(
        name = "student_course",
        joinColumns = @JoinColumn(name = "student_id"),
        inverseJoinColumns = @JoinColumn(name = "course_id")
    )
    private Set<Course> courses = new HashSet<>();
}

@Entity
public class Course {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String name;

    @ManyToMany(mappedBy = "courses")
    private Set<Student> students = new HashSet<>();
}
```

## 7. 审计功能

### 7.1 启用审计

```java
@Configuration
@EnableJpaAuditing
public class JpaConfig {

    @Bean
    public AuditorAware<String> auditorProvider() {
        return () -> Optional.of("system"); // 实际项目中从安全上下文获取
    }
}
```

### 7.2 审计实体

```java
@Entity
@EntityListeners(AuditingEntityListener.class)
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String name;

    @CreatedDate
    private LocalDateTime createdDate;

    @LastModifiedDate
    private LocalDateTime lastModifiedDate;

    @CreatedBy
    private String createdBy;

    @LastModifiedBy
    private String lastModifiedBy;
}
```

### 7.3 使用审计基类

```java
@MappedSuperclass
@EntityListeners(AuditingEntityListener.class)
public abstract class Auditable {

    @CreatedDate
    @Column(updatable = false)
    private LocalDateTime createdDate;

    @LastModifiedDate
    private LocalDateTime lastModifiedDate;

    @CreatedBy
    @Column(updatable = false)
    private String createdBy;

    @LastModifiedBy
    private String lastModifiedBy;
}

@Entity
public class User extends Auditable {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String name;
    private String email;

    // 自动继承审计字段
}
```

## 8. 最佳实践

### 8.1 优先使用查询方法

```java
// ✅ 推荐：简单查询使用方法名
public interface UserRepository extends JpaRepository<User, Long> {
    User findByEmail(String email);
    List<User> findByAgeGreaterThan(int age);
}

// ❌ 避免：简单查询也用 @Query
public interface UserRepository extends JpaRepository<User, Long> {
    @Query("SELECT u FROM User u WHERE u.email = :email")
    User findByEmail(@Param("email") String email);
}
```

### 8.2 复杂查询使用 @Query

```java
// ✅ 推荐：复杂查询使用 @Query
public interface UserRepository extends JpaRepository<User, Long> {
    @Query("SELECT u FROM User u WHERE " +
           "(u.name LIKE %:keyword% OR u.email LIKE %:keyword%) " +
           "AND u.age BETWEEN :minAge AND :maxAge " +
           "AND u.status = :status")
    List<User> searchUsers(
        @Param("keyword") String keyword,
        @Param("minAge") int minAge,
        @Param("maxAge") int maxAge,
        @Param("status") String status
    );
}
```

### 8.3 避免 N+1 查询问题

```java
// ❌ 错误：会产生 N+1 查询
@Query("SELECT u FROM User u WHERE u.status = :status")
List<User> findByStatus(@Param("status") String status);
// 访问 user.getOrders() 会触发额外查询

// ✅ 正确：使用 JOIN FETCH
@Query("SELECT u FROM User u JOIN FETCH u.orders WHERE u.status = :status")
List<User> findByStatusWithOrders(@Param("status") String status);
```

### 8.4 使用 DTO 投影

```java
// 定义 DTO
public class UserDTO {
    private String name;
    private String email;

    public UserDTO(String name, String email) {
        this.name = name;
        this.email = email;
    }

    // getters
}

// Repository
public interface UserRepository extends JpaRepository<User, Long> {
    @Query("SELECT new com.example.dto.UserDTO(u.name, u.email) FROM User u")
    List<UserDTO> findAllUserDTOs();
}
```

## 9. 总结

| 概念           | 说明                  |
| -------------- | --------------------- |
| Repository     | 数据访问接口          |
| CrudRepository | 基本 CRUD 操作        |
| JpaRepository  | JPA 特定功能          |
| 查询方法       | 根据方法名生成查询    |
| @Query         | 自定义 JPQL/SQL 查询  |
| Pageable       | 分页参数              |
| Sort           | 排序参数              |
| 审计           | 自动记录创建/修改信息 |

---

**关键要点**：

- 优先使用查询方法命名规则
- 复杂查询使用 @Query
- 注意避免 N+1 查询问题
- 合理使用分页和排序
- 利用审计功能追踪变更

**下一步**：学习 [Spring 事务管理](./transactions)
