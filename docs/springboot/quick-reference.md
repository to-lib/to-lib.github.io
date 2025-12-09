---
sidebar_position: 10
---

# 快速参考

> [!TIP]
> **快速查找**: 使用 `Ctrl+F` (或 `Cmd+F`) 快速定位所需的注解、配置或代码片段。建议收藏此页面以便随时查阅！

快速查找常用注解、配置和代码片段。

## 常用注解速查表

### 启动和配置

| 注解 | 说明 | 示例 |
|------|------|------|
| `@SpringBootApplication` | 启动类 | `public class Application {}` |
| `@Configuration` | 配置类 | `@Configuration public class Config {}` |
| `@Bean` | 定义 Bean | `@Bean public User user() {}` |
| `@ComponentScan` | 组件扫描 | `@ComponentScan("com.example")` |
| `@EnableAutoConfiguration` | 启用自动配置 | 已包含在 @SpringBootApplication |

### 依赖注入

| 注解 | 说明 | 示例 |
|------|------|------|
| `@Component` | 通用组件 | `@Component public class MyClass {}` |
| `@Service` | 业务服务 | `@Service public class UserService {}` |
| `@Repository` | 数据访问 | `@Repository public interface UserRepo {}` |
| `@Controller` | Web 控制器 | `@Controller public class UserController {}` |
| `@RestController` | RESTful 控制器 | `@RestController public class ApiController {}` |
| `@Autowired` | 自动注入 | `@Autowired private UserService service;` |
| `@Qualifier` | 指定 Bean | `@Autowired @Qualifier("primary")` |
| `@Resource` | 按名称注入 | `@Resource(name = "userService")` |
| `@Inject` | 注入（JSR-330） | `@Inject private UserService service;` |

### 请求处理

| 注解 | 说明 | 示例 |
|------|------|------|
| `@RequestMapping` | 请求映射 | `@RequestMapping("/users")` |
| `@GetMapping` | GET 请求 | `@GetMapping("/{id}")` |
| `@PostMapping` | POST 请求 | `@PostMapping` |
| `@PutMapping` | PUT 请求 | `@PutMapping("/{id}")` |
| `@DeleteMapping` | DELETE 请求 | `@DeleteMapping("/{id}")` |
| `@PatchMapping` | PATCH 请求 | `@PatchMapping("/{id}")` |
| `@PathVariable` | 路径参数 | `@PathVariable Long id` |
| `@RequestParam` | 查询参数 | `@RequestParam String name` |
| `@RequestBody` | 请求体 | `@RequestBody User user` |
| `@RequestHeader` | 请求头 | `@RequestHeader String token` |
| `@CookieValue` | Cookie 值 | `@CookieValue String sessionId` |
| `@ResponseBody` | 返回 JSON | `@ResponseBody` |

### 数据验证

| 注解 | 说明 | 示例 |
|------|------|------|
| `@Valid` | 启用验证 | `@Valid @RequestBody User user` |
| `@Validated` | 验证器 | `@Validated public class MyService {}` |
| `@NotNull` | 非空 | `@NotNull private String name;` |
| `@NotBlank` | 非空字符串 | `@NotBlank private String email;` |
| `@NotEmpty` | 非空集合 | `@NotEmpty private List<String> items;` |
| `@Size` | 大小范围 | `@Size(min=3, max=20)` |
| `@Min` | 最小值 | `@Min(18)` |
| `@Max` | 最大值 | `@Max(100)` |
| `@Email` | 邮箱格式 | `@Email private String email;` |
| `@Pattern` | 正则表达式 | `@Pattern(regexp="^1[3-9]\\d{9}$")` |

### 数据库

| 注解 | 说明 | 示例 |
|------|------|------|
| `@Entity` | 实体类 | `@Entity public class User {}` |
| `@Table` | 表映射 | `@Table(name="users")` |
| `@Id` | 主键 | `@Id private Long id;` |
| `@GeneratedValue` | 自动生成 | `@GeneratedValue(strategy=AUTO)` |
| `@Column` | 列映射 | `@Column(name="user_name")` |
| `@OneToMany` | 一对多 | `@OneToMany(mappedBy="user")` |
| `@ManyToOne` | 多对一 | `@ManyToOne @JoinColumn(name="user_id")` |
| `@ManyToMany` | 多对多 | `@ManyToMany @JoinTable(...)` |
| `@Transient` | 非持久化 | `@Transient private String temp;` |
| `@CreationTimestamp` | 创建时间 | `@CreationTimestamp private LocalDateTime created;` |
| `@UpdateTimestamp` | 更新时间 | `@UpdateTimestamp private LocalDateTime updated;` |

### 配置属性

| 注解 | 说明 | 示例 |
|------|------|------|
| `@Value` | 注入单个属性 | `@Value("${app.name}")` |
| `@ConfigurationProperties` | 配置类 | `@ConfigurationProperties(prefix="app")` |
| `@EnableConfigurationProperties` | 启用配置属性 | 在 @Configuration 上使用 |
| `@Profile` | 环境隔离 | `@Profile("dev")` |

### 缓存

| 注解 | 说明 | 示例 |
|------|------|------|
| `@EnableCaching` | 启用缓存 | 在 @Configuration 上使用 |
| `@Cacheable` | 读缓存 | `@Cacheable("users")` |
| `@CachePut` | 写缓存 | `@CachePut("users")` |
| `@CacheEvict` | 清除缓存 | `@CacheEvict("users")` |

### 调度和异步

| 注解 | 说明 | 示例 |
|------|------|------|
| `@EnableScheduling` | 启用定时任务 | 在 @Configuration 上使用 |
| `@Scheduled` | 定时执行 | `@Scheduled(cron="0 0 * * * *")` |
| `@EnableAsync` | 启用异步 | 在 @Configuration 上使用 |
| `@Async` | 异步执行 | `@Async public void asyncMethod() {}` |

### 事务

| 注解 | 说明 | 示例 |
|------|------|------|
| `@Transactional` | 事务管理 | `@Transactional public void method() {}` |
| `@EnableTransactionManagement` | 启用事务 | 在 @Configuration 上使用 |

## 常用配置速查表

### application.yml

```yaml
# 服务器
server:
  port: 8080
  servlet:
    context-path: /api

# 数据库
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/mydb
    username: root
    password: password

# JPA
  jpa:
    hibernate:
      ddl-auto: update
    show-sql: false

# Redis
  redis:
    host: localhost
    port: 6379

# 日志
logging:
  level:
    root: INFO
```

## 常用代码片段

### 快速创建 REST API

```java
@RestController
@RequestMapping("/api/users")
public class UserController {
    @Autowired
    private UserService userService;
    
    @GetMapping
    public List<User> list() { return userService.list(); }
    
    @GetMapping("/{id}")
    public User get(@PathVariable Long id) { return userService.get(id); }
    
    @PostMapping
    public User create(@RequestBody User user) { return userService.save(user); }
    
    @PutMapping("/{id}")
    public User update(@PathVariable Long id, @RequestBody User user) 
    { return userService.update(id, user); }
    
    @DeleteMapping("/{id}")
    public void delete(@PathVariable Long id) { userService.delete(id); }
}
```

### 快速创建 Service

```java
@Service
@Transactional
public class UserService {
    @Autowired
    private UserRepository userRepository;
    
    public List<User> list() { return userRepository.findAll(); }
    public User get(Long id) { 
        return userRepository.findById(id).orElseThrow();
    }
    public User save(User user) { return userRepository.save(user); }
    public void delete(Long id) { userRepository.deleteById(id); }
}
```

### 快速创建 Repository

```java
@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    User findByUsername(String username);
    List<User> findByAgeGreaterThan(Integer age);
}
```

### 快速创建实体类

```java
@Entity
@Table(name = "users")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(nullable = false)
    private String username;
    
    @Column(unique = true)
    private String email;
    
    private Integer age;
}
```

### 异常处理

```java
@RestControllerAdvice
public class GlobalExceptionHandler {
    
    @ExceptionHandler(ResourceNotFoundException.class)
    public ResponseEntity<ErrorResponse> handle(ResourceNotFoundException e) {
        return ResponseEntity.status(404).body(new ErrorResponse(e.getMessage()));
    }
}
```

### 配置类

```java
@Configuration
public class AppConfig {
    
    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
    
    @Bean
    public CacheManager cacheManager() {
        return new ConcurrentMapCacheManager("users", "posts");
    }
}
```

## 常用 Maven 命令

```bash
# 创建项目
mvn archetype:generate

# 编译
mvn clean compile

# 测试
mvn test

# 打包
mvn clean package

# 跳过测试打包
mvn clean package -DskipTests

# 运行
mvn spring-boot:run

# 安装依赖
mvn clean install

# 清理
mvn clean

# 查看依赖树
mvn dependency:tree
```

## 常用 gradle 命令

```bash
# 编译
gradle build

# 运行
gradle bootRun

# 跳过测试打包
gradle build -x test

# 清理
gradle clean

# 查看依赖
gradle dependencies
```

## 常用环境变量

```bash
# 激活 Profile
export SPRING_PROFILES_ACTIVE=prod

# 修改端口
export SERVER_PORT=9000

# 修改数据库
export SPRING_DATASOURCE_URL=jdbc:mysql://db:3306/mydb
export SPRING_DATASOURCE_USERNAME=root
export SPRING_DATASOURCE_PASSWORD=password

# 日志级别
export LOGGING_LEVEL_ROOT=DEBUG
```

## 性能优化建议

> [!IMPORTANT]
> **生产环境最佳实践:**
>
> 1. **使用连接池** - HikariCP（默认），配置合理的连接池大小
> 2. **启用缓存** - 使用 Redis 或 Caffeine 缓存热点数据
> 3. **异步处理** - 使用 @Async 处理耗时操作
> 4. **定时清理** - 使用 @Scheduled 定时清理过期数据
> 5. **查询优化** - 使用 Pageable 分页查询
>
6. **监控指标** - 使用 Actuator 监控应用

## 调试技巧

```bash
# 启用调试输出
java -jar app.jar --debug

# 启用特定日志
-Dlogging.level.com.example=DEBUG

# 远程调试
-agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=5005

# 查看 Bean
ApplicationContext context = ...;
String[] beans = context.getBeanDefinitionNames();
```
