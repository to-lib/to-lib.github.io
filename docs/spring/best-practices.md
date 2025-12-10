---
id: best-practices
title: 最佳实践
sidebar_label: 最佳实践
sidebar_position: 98
---

# Spring Framework 最佳实践

> [!IMPORTANT]
> **Spring 最佳实践**: 优先使用构造器注入、合理使用作用域、注意事务边界。遵循这些原则可提升代码质量和可维护性。

### 1.1 优先使用构造函数注入

```java
// ✅ 最佳实践：构造函数注入
@Service
public class UserService {
    private final UserRepository userRepository;
    private final EmailService emailService;
    
    public UserService(UserRepository userRepository, EmailService emailService) {
        this.userRepository = userRepository;
        this.emailService = emailService;
    }
}

// ❌ 避免：字段注入
@Service
public class UserService {
    @Autowired
    private UserRepository userRepository;
    
    @Autowired
    private EmailService emailService;
}

// ⚠️ 避免：Setter注入（除非是可选依赖）
@Service
public class UserService {
    private UserRepository userRepository;
    
    @Autowired
    public void setUserRepository(UserRepository userRepository) {
        this.userRepository = userRepository;
    }
}
```

### 1.2 使用final修饰依赖

```java
// ✅ 最佳实践
@Service
public class UserService {
    private final UserRepository userRepository;  // final修饰
    
    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }
}

// ❌ 避免
@Service
public class UserService {
    private UserRepository userRepository;  // 可被修改
}
```

### 1.3 避免循环依赖

```java
// ❌ 错误：循环依赖
@Service
public class ServiceA {
    private final ServiceB serviceB;
    
    public ServiceA(ServiceB serviceB) {
        this.serviceB = serviceB;
    }
}

@Service
public class ServiceB {
    private final ServiceA serviceA;
    
    public ServiceB(ServiceA serviceA) {
        this.serviceA = serviceA;
    }
}

// ✅ 解决：重构设计，提取公共逻辑
@Service
public class CommonService {
    // 公共逻辑
}

@Service
public class ServiceA {
    private final CommonService commonService;
    
    public ServiceA(CommonService commonService) {
        this.commonService = commonService;
    }
}

@Service
public class ServiceB {
    private final CommonService commonService;
    
    public ServiceB(CommonService commonService) {
        this.commonService = commonService;
    }
}
```

## 2. 组件设计最佳实践

### 2.1 单一职责原则

```java
// ❌ 坏的设计：一个类干多种事情
@Service
public class UserService {
    public void createUser(User user) { }
    public void sendEmail(String email) { }
    public void saveToDatabase(User user) { }
    public void validateInput(User user) { }
}

// ✅ 好的设计：每个类只负责一件事
@Service
public class UserService {
    private final UserRepository repository;
    private final EmailService emailService;
    private final UserValidator validator;
    
    public void createUser(User user) {
        validator.validate(user);
        repository.save(user);
        emailService.sendWelcome(user);
    }
}

@Service
public class EmailService {
    public void sendWelcome(User user) { }
}

@Component
public class UserValidator {
    public void validate(User user) { }
}

@Repository
public class UserRepository {
    public void save(User user) { }
}
```

### 2.2 使用接口

```java
// ✅ 最佳实践：针对接口编程
public interface UserRepository {
    void save(User user);
    User findById(Long id);
}

@Repository
public class JpaUserRepository implements UserRepository {
    // JPA实现
}

@Repository
public class MongoUserRepository implements UserRepository {
    // MongoDB实现
}

@Service
public class UserService {
    private final UserRepository repository;
    
    public UserService(UserRepository repository) {
        this.repository = repository;
    }
}
```

### 2.3 避免静态方法

```java
// ❌ 避免：静态方法难以测试
public class UserUtils {
    public static boolean isValidEmail(String email) {
        return email.contains("@");
    }
}

// ✅ 最佳实践：使用非静态方法
@Component
public class UserValidator {
    public boolean isValidEmail(String email) {
        return email.contains("@");
    }
}

@Service
public class UserService {
    private final UserValidator validator;
    
    public UserService(UserValidator validator) {
        this.validator = validator;
    }
}
```

## 3. 数据库操作最佳实践

### 3.1 事务管理

```java
// ✅ 最佳实践：在Service层使用@Transactional
@Service
public class OrderService {
    @Autowired
    private OrderRepository orderRepository;
    
    @Autowired
    private InventoryService inventoryService;
    
    @Transactional(rollbackFor = Exception.class)
    public void createOrder(Order order) {
        orderRepository.save(order);
        
        for (OrderItem item : order.getItems()) {
            inventoryService.decreaseStock(item.getProductId(), item.getQuantity());
        }
    }
}

// ❌ 避免：在Repository层使用@Transactional
@Repository
public class OrderRepository {
    @Transactional  // 不推荐
    public void save(Order order) {
    }
}
```

### 3.2 使用Repository模式

```java
// ✅ 最佳实践：继承Spring Data Repository
public interface UserRepository extends JpaRepository<User, Long> {
    User findByEmail(String email);
    
    List<User> findByAgeGreaterThan(int age);
    
    @Query("SELECT u FROM User u WHERE u.status = :status")
    List<User> findActiveUsers(@Param("status") String status);
}

@Service
public class UserService {
    @Autowired
    private UserRepository userRepository;
    
    public void createUser(User user) {
        userRepository.save(user);
    }
}

// ❌ 避免：直接使用EntityManager
@Service
public class UserService {
    @PersistenceContext
    private EntityManager entityManager;
    
    public void createUser(User user) {
        entityManager.persist(user);  // 不推荐
    }
}
```

### 3.3 延迟加载

```java
// ✅ 最佳实践：合理使用延迟加载
@Entity
public class User {
    @Id
    private Long id;
    
    private String name;
    
    @OneToMany(mappedBy = "user", fetch = FetchType.LAZY)
    private List<Order> orders;
}

// 需要时才加载
@Service
public class UserService {
    public User getUserWithOrders(Long id) {
        User user = userRepository.findById(id).orElse(null);
        // 访问orders时才会加载
        user.getOrders().size();
        return user;
    }
}

// ❌ 避免：N+1查询问题
@Service
public class UserService {
    public List<Order> getUserOrders(Long userId) {
        User user = userRepository.findById(userId).orElse(null);
        return user.getOrders();  // 这会导致额外的查询
    }
}
```

## 4. Web层最佳实践

### 4.1 RESTful设计

```java
// ✅ 最佳实践：遵循RESTful原则
@RestController
@RequestMapping("/api/users")
public class UserController {
    @Autowired
    private UserService userService;
    
    // GET - 获取资源列表
    @GetMapping
    public ResponseEntity<List<User>> getAllUsers() {
        return ResponseEntity.ok(userService.getAllUsers());
    }
    
    // GET - 获取单个资源
    @GetMapping("/{id}")
    public ResponseEntity<User> getUser(@PathVariable Long id) {
        return ResponseEntity.ok(userService.getUserById(id));
    }
    
    // POST - 创建资源
    @PostMapping
    public ResponseEntity<User> createUser(@Valid @RequestBody User user) {
        User created = userService.saveUser(user);
        return ResponseEntity
            .created(URI.create("/api/users/" + created.getId()))
            .body(created);
    }
    
    // PUT - 更新资源
    @PutMapping("/{id}")
    public ResponseEntity<User> updateUser(
        @PathVariable Long id,
        @Valid @RequestBody User user) {
        user.setId(id);
        return ResponseEntity.ok(userService.updateUser(user));
    }
    
    // DELETE - 删除资源
    @DeleteMapping("/{id}")
    public ResponseEntity<?> deleteUser(@PathVariable Long id) {
        userService.deleteUser(id);
        return ResponseEntity.noContent().build();
    }
}

// ❌ 避免：RPC风格的URL
@RestController
@RequestMapping("/api")
public class BadUserController {
    @GetMapping("/getUserById")  // 不推荐
    public User getUserById(Long id) {
        return null;
    }
    
    @PostMapping("/createUser")  // 不推荐
    public User createUser(User user) {
        return null;
    }
}
```

### 4.2 参数验证

```java
// ✅ 最佳实践：使用Bean Validation
@Data
public class UserDTO {
    @NotNull(message = "ID不能为空")
    private Long id;
    
    @NotBlank(message = "名称不能为空")
    @Length(min = 2, max = 50, message = "名称长度必须在2-50之间")
    private String name;
    
    @NotBlank(message = "邮箱不能为空")
    @Email(message = "邮箱格式不正确")
    private String email;
    
    @Min(value = 0, message = "年龄不能为负数")
    @Max(value = 150, message = "年龄不能超过150")
    private Integer age;
}

@RestController
@RequestMapping("/api/users")
public class UserController {
    
    @PostMapping
    public ResponseEntity<User> createUser(@Valid @RequestBody UserDTO userDTO) {
        // 如果验证失败，会自动返回400
        return ResponseEntity.ok(convertToEntity(userDTO));
    }
}

// ❌ 避免：在Controller中手动验证
@PostMapping
public ResponseEntity<User> createUser(UserDTO userDTO) {
    if (userDTO.getName() == null || userDTO.getName().isEmpty()) {
        // 手动验证
    }
}
```

### 4.3 异常处理

```java
// ✅ 最佳实践：全局异常处理
@RestControllerAdvice
public class GlobalExceptionHandler {
    
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ErrorResponse> handleValidationException(
        MethodArgumentNotValidException ex) {
        
        List<String> errors = ex.getBindingResult()
            .getAllErrors()
            .stream()
            .map(ObjectError::getDefaultMessage)
            .collect(Collectors.toList());
        
        ErrorResponse response = new ErrorResponse("Validation failed", errors);
        return ResponseEntity.badRequest().body(response);
    }
    
    @ExceptionHandler(ResourceNotFoundException.class)
    public ResponseEntity<ErrorResponse> handleNotFoundException(
        ResourceNotFoundException ex) {
        
        ErrorResponse response = new ErrorResponse("Resource not found", ex.getMessage());
        return ResponseEntity.status(HttpStatus.NOT_FOUND).body(response);
    }
    
    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleException(Exception ex) {
        ErrorResponse response = new ErrorResponse("Internal server error", 
            ex.getMessage());
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
            .body(response);
    }
}

// ❌ 避免：在每个Controller中处理异常
@RestController
public class UserController {
    @GetMapping("/{id}")
    public User getUser(@PathVariable Long id) {
        try {
            return userService.getUserById(id);
        } catch (Exception ex) {
            // 在每个方法中处理
        }
    }
}
```

## 5. 配置管理最佳实践

### 5.1 使用外部配置

```yaml
# application.yml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/mydb
    username: root
    password: password
  jpa:
    hibernate:
      ddl-auto: update
    properties:
      hibernate:
        dialect: org.hibernate.dialect.MySQL5InnoDBDialect
    show-sql: false

server:
  port: 8080
  servlet:
    context-path: /api

app:
  name: MyApplication
  version: 1.0.0
  features:
    enableCache: true
    enableAudit: true
```

### 5.2 配置属性绑定

```java
// ✅ 最佳实践：使用@ConfigurationProperties
@Configuration
@ConfigurationProperties(prefix = "app")
@Data
public class AppProperties {
    private String name;
    private String version;
    
    @Nested
    private Features features;
    
    @Data
    public static class Features {
        private boolean enableCache;
        private boolean enableAudit;
    }
}

@Service
public class AppService {
    @Autowired
    private AppProperties appProperties;
    
    public void printConfig() {
        System.out.println("App: " + appProperties.getName());
        System.out.println("Cache enabled: " + appProperties.getFeatures().isEnableCache());
    }
}

// ❌ 避免：使用@Value注入多个属性
@Service
public class AppService {
    @Value("${app.name}")
    private String name;
    
    @Value("${app.version}")
    private String version;
    
    @Value("${app.features.enableCache}")
    private boolean enableCache;
    
    @Value("${app.features.enableAudit}")
    private boolean enableAudit;
    // 太多@Value注解
}
```

## 6. 测试最佳实践

### 6.1 单元测试

```java
// ✅ 最佳实践：使用Mockito进行单元测试
@ExtendWith(MockitoExtension.class)
class UserServiceTest {
    
    @Mock
    private UserRepository userRepository;
    
    @InjectMocks
    private UserService userService;
    
    @Test
    void testSaveUser() {
        // Arrange
        User user = new User(1L, "John", "john@example.com");
        when(userRepository.save(user)).thenReturn(user);
        
        // Act
        User result = userService.saveUser(user);
        
        // Assert
        assertEquals("John", result.getName());
        verify(userRepository, times(1)).save(user);
    }
}
```

### 6.2 集成测试

```java
// ✅ 最佳实践：使用@SpringBootTest进行集成测试
@SpringBootTest
@ActiveProfiles("test")
class UserControllerTest {
    
    @Autowired
    private TestRestTemplate restTemplate;
    
    @Test
    void testGetUser() {
        ResponseEntity<User> response = restTemplate
            .getForEntity("/api/users/1", User.class);
        
        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals("John", response.getBody().getName());
    }
}
```

## 7. 日志最佳实践

### 7.1 使用SLF4J

```java
// ✅ 最佳实践：使用SLF4J和Logback
@Service
@Slf4j  // 使用Lombok的@Slf4j注解
public class UserService {
    
    public void saveUser(User user) {
        log.debug("Saving user: {}", user.getName());
        
        try {
            // 业务逻辑
            log.info("User {} saved successfully", user.getName());
        } catch (Exception ex) {
            log.error("Failed to save user {}", user.getName(), ex);
            throw ex;
        }
    }
}

// ❌ 避免：使用System.out.println
@Service
public class UserService {
    public void saveUser(User user) {
        System.out.println("Saving user: " + user.getName());  // 不推荐
    }
}
```

### 7.2 日志级别

```java
@Service
@Slf4j
public class UserService {
    
    public void process(User user) {
        // DEBUG - 开发调试信息
        log.debug("Processing user with ID: {}", user.getId());
        
        // INFO - 重要业务流程
        log.info("User {} started processing", user.getName());
        
        // WARN - 警告信息
        if (user.getAge() > 100) {
            log.warn("User age seems unusual: {}", user.getAge());
        }
        
        // ERROR - 错误信息
        try {
            // 业务逻辑
        } catch (Exception ex) {
            log.error("Failed to process user {}", user.getName(), ex);
        }
    }
}
```

## 8. 性能优化最佳实践

### 8.1 缓存

```java
// ✅ 最佳实践：使用Spring Cache
@Configuration
@EnableCaching
public class CacheConfig {
    
    @Bean
    public CacheManager cacheManager() {
        return new ConcurrentMapCacheManager("users", "products");
    }
}

@Service
public class UserService {
    
    @Cacheable(value = "users", key = "#id")
    public User getUserById(Long id) {
        // 只在缓存中不存在时执行
        return userRepository.findById(id).orElse(null);
    }
    
    @CacheEvict(value = "users", key = "#user.id")
    public void updateUser(User user) {
        userRepository.save(user);
    }
    
    @CacheEvict(value = "users", allEntries = true)
    public void clearCache() {
    }
}
```

### 8.2 分页和排序

```java
// ✅ 最佳实践：使用分页避免一次加载大量数据
@RestController
@RequestMapping("/api/users")
public class UserController {
    
    @Autowired
    private UserRepository userRepository;
    
    @GetMapping
    public ResponseEntity<Page<User>> getUsers(
        @RequestParam(defaultValue = "0") int page,
        @RequestParam(defaultValue = "10") int size,
        @RequestParam(defaultValue = "id,desc") String[] sort) {
        
        Sort.Order order = new Sort.Order(Sort.Direction.DESC, sort[0]);
        Pageable pageable = PageRequest.of(page, size, Sort.by(order));
        
        Page<User> users = userRepository.findAll(pageable);
        return ResponseEntity.ok(users);
    }
}

// ❌ 避免：一次加载所有数据
@GetMapping
public ResponseEntity<List<User>> getAllUsers() {
    return ResponseEntity.ok(userRepository.findAll());
}
```

## 9. 总结表

| 方面 | 最佳实践 | 避免做法 |
|------|--------|--------|
| 依赖注入 | 构造函数注入 | 字段注入 |
| 组件设计 | 单一职责原则 | 一个类干多件事 |
| 接口使用 | 针对接口编程 | 直接依赖实现类 |
| 事务管理 | Service层 | Repository层 |
| REST设计 | RESTful风格 | RPC风格 |
| 参数验证 | Bean Validation | 手动验证 |
| 异常处理 | 全局异常处理 | 每个方法都处理 |
| 配置管理 | 外部配置文件 | 硬编码配置值 |
| 日志记录 | SLF4J + Logback | System.out.println |

---

**关键原则**：

1. 遵循SOLID设计原则
2. 使用Spring提供的抽象而不是底层API
3. 充分利用Spring的自动化特性
4. 编写可测试的代码
5. 保持代码简洁清晰

**下一步**：参考[快速参考](/docs/spring/quick-reference)查看常用代码片段
