---
sidebar_position: 12
---

# 最佳实践

> [!IMPORTANT]
> **生产环境关键**: 本文档总结了在生产环境中应遵循的最佳实践,包括代码组织、性能优化、安全性等关键方面。建议所有开发者在开始项目前认真阅读。

## 代码组织

### 项目结构

遵循标准的分层架构：

```
src/main/java/com/example/app/
├── controller/          # Web 控制器
│   ├── UserController.java
│   └── PostController.java
├── service/             # 业务逻辑
│   ├── UserService.java
│   └── impl/
│       └── UserServiceImpl.java
├── repository/          # 数据访问
│   ├── UserRepository.java
│   └── PostRepository.java
├── entity/              # 实体类
│   └── User.java
├── dto/                 # 数据传输对象
│   └── UserDTO.java
├── exception/           # 异常处理
│   ├── GlobalExceptionHandler.java
│   └── ResourceNotFoundException.java
├── config/              # 配置类
│   ├── WebConfig.java
│   └── SecurityConfig.java
├── util/                # 工具类
│   └── CommonUtils.java
└── Application.java     # 启动类
```

### 命名规范

```java
// ✅ 好的命名
class UserService {}
class UserRepository {}
class CreateUserRequest {}
class UserResponse {}

// ❌ 不好的命名
class UserBiz {}
class UserDAO {}
class User_Create_Req {}
class UserResp {}
```

## 依赖注入

### 优先使用构造器注入

> [!TIP]
> **为什么推荐构造器注入？**
>
> - ✅ 依赖关系明确,便于测试
> - ✅ 对象创建时就是完全初始化的状态
> - ✅ 可以使用 `final` 关键字,保证不可变性
> - ✅ 避免循环依赖问题

```java
// ✅ 推荐
@Service
public class UserService {
    private final UserRepository userRepository;
    
    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }
}

// ❌ 避免
@Service
public class UserService {
    @Autowired
    private UserRepository userRepository;  // 难以测试
}
```

### 使用 @RequiredArgsConstructor

```java
// ✅ 最简洁
@Service
@RequiredArgsConstructor
public class UserService {
    private final UserRepository userRepository;
    private final UserMapper userMapper;
    // 自动生成构造器
}
```

## 数据访问

### Repository 方法命名

```java
@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    
    // ✅ 清晰的方法名
    User findByUsername(String username);
    Optional<User> findByEmail(String email);
    List<User> findByStatusAndAgeGreaterThan(String status, Integer age);
    
    // ❌ 不清晰的名称
    User getUser(String name);
    User selectUser(String email);
}
```

### 使用 Specification 进行复杂查询

```java
@Service
@RequiredArgsConstructor
public class UserService {
    private final UserRepository userRepository;
    
    public List<User> search(UserSearchRequest request) {
        Specification<User> spec = (root, query, cb) -> {
            List<Predicate> predicates = new ArrayList<>();
            
            if (request.getUsername() != null) {
                predicates.add(cb.like(root.get("username"), 
                    "%" + request.getUsername() + "%"));
            }
            
            if (request.getMinAge() != null) {
                predicates.add(cb.greaterThanOrEqualTo(root.get("age"), 
                    request.getMinAge()));
            }
            
            return cb.and(predicates.toArray(new Predicate[0]));
        };
        
        return userRepository.findAll(spec);
    }
}
```

### 避免 N+1 查询问题

> [!WARNING]
> **N+1 查询性能陷阱**: 这是最常见的性能问题之一！当查询1条父记录后,又分别查误N条子记录时,导致执行1+N次SQL。在数据量大时会严重影响性能。

```java
// ❌ N+1 查询
@Transactional
public List<UserDTO> getUsersWithPosts() {
    List<User> users = userRepository.findAll();  // 1 次查询
    return users.stream()
        .map(user -> {
            List<Post> posts = postRepository.findByUserId(user.getId());  // N 次查询
            return toDTO(user, posts);
        })
        .collect(Collectors.toList());
}

// ✅ 解决方案 1：使用 JOIN FETCH
@Query("SELECT u FROM User u LEFT JOIN FETCH u.posts WHERE u.status = :status")
List<User> findUsersWithPosts(@Param("status") String status);

// ✅ 解决方案 2：设置 FetchType.EAGER
@Entity
public class User {
    @OneToMany(mappedBy = "user", fetch = FetchType.EAGER)
    private List<Post> posts;
}
```

## 业务逻辑

### Service 层职责

```java
// ✅ 好的 Service 层
@Service
@Transactional
@RequiredArgsConstructor
public class UserService {
    private final UserRepository userRepository;
    private final UserMapper userMapper;
    private final MailService mailService;
    
    // 业务逻辑集中在 Service 层
    public UserDTO createUser(CreateUserRequest request) {
        // 验证
        validateCreateUserRequest(request);
        
        // 转换
        User user = userMapper.toEntity(request);
        
        // 保存
        User savedUser = userRepository.save(user);
        
        // 附加操作
        mailService.sendWelcomeEmail(savedUser.getEmail());
        
        return userMapper.toDTO(savedUser);
    }
    
    // 业务规则封装
    public void updateUserStatus(Long userId, String newStatus) {
        User user = userRepository.findById(userId)
            .orElseThrow(() -> new UserNotFoundException(userId));
        
        // 检查状态转换是否合法
        if (!user.getStatus().canTransitionTo(newStatus)) {
            throw new InvalidStatusTransitionException(user.getStatus(), newStatus);
        }
        
        user.setStatus(newStatus);
        userRepository.save(user);
    }
}

// ❌ 避免：业务逻辑放在 Controller
@PostMapping
public UserDTO create(@RequestBody CreateUserRequest request) {
    // 不应该在 Controller 中做复杂的业务逻辑
    if (userRepository.findByEmail(request.getEmail()).isPresent()) {
        throw new DuplicateEmailException();
    }
    // ...
}
```

## 异常处理

### 自定义异常

```java
// ✅ 创建自定义异常
public abstract class ApplicationException extends RuntimeException {
    private final String errorCode;
    
    public ApplicationException(String errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }
    
    public String getErrorCode() {
        return errorCode;
    }
}

public class ResourceNotFoundException extends ApplicationException {
    public ResourceNotFoundException(String resource, Long id) {
        super("RESOURCE_NOT_FOUND", 
            String.format("%s not found with id: %d", resource, id));
    }
}

public class DuplicateEmailException extends ApplicationException {
    public DuplicateEmailException(String email) {
        super("DUPLICATE_EMAIL", 
            String.format("Email already exists: %s", email));
    }
}
```

### 统一异常处理

```java
@RestControllerAdvice
public class GlobalExceptionHandler {
    
    @ExceptionHandler(ResourceNotFoundException.class)
    public ResponseEntity<ErrorResponse> handleNotFound(
            ResourceNotFoundException ex,
            WebRequest request) {
        ErrorResponse error = ErrorResponse.builder()
            .timestamp(LocalDateTime.now())
            .status(HttpStatus.NOT_FOUND.value())
            .errorCode(ex.getErrorCode())
            .message(ex.getMessage())
            .path(request.getDescription(false).replace("uri=", ""))
            .build();
        return new ResponseEntity<>(error, HttpStatus.NOT_FOUND);
    }
    
    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGenericException(
            Exception ex,
            WebRequest request) {
        ErrorResponse error = ErrorResponse.builder()
            .timestamp(LocalDateTime.now())
            .status(HttpStatus.INTERNAL_SERVER_ERROR.value())
            .errorCode("INTERNAL_SERVER_ERROR")
            .message("An unexpected error occurred")
            .path(request.getDescription(false).replace("uri=", ""))
            .build();
        return new ResponseEntity<>(error, HttpStatus.INTERNAL_SERVER_ERROR);
    }
}
```

## 配置管理

### 使用 @ConfigurationProperties

```java
// ✅ 使用配置属性类
@Configuration
@ConfigurationProperties(prefix = "app")
@Data
@Validated
public class AppProperties {
    
    @NotBlank
    private String name;
    
    @NotBlank
    private String version;
    
    private Mail mail = new Mail();
    
    @Data
    public static class Mail {
        @NotBlank
        private String from;
        
        @NotBlank
        private String host;
        
        @Min(1)
        @Max(65535)
        private int port = 25;
    }
}

// ❌ 避免使用 @Value 注解
@Component
public class OldWayConfig {
    @Value("${app.name}")
    private String appName;
    
    @Value("${app.mail.from}")
    private String mailFrom;
    // ... 很多 @Value 注解
}
```

## 日志记录

### 正确的日志级别

```java
@Service
@RequiredArgsConstructor
public class UserService {
    private static final Logger logger = LoggerFactory.getLogger(UserService.class);
    
    public UserDTO getUserById(Long id) {
        // DEBUG：开发调试信息
        logger.debug("Fetching user with id: {}", id);
        
        User user = userRepository.findById(id)
            .orElseThrow(() -> {
                // WARN：可恢复的异常
                logger.warn("User not found with id: {}", id);
                return new ResourceNotFoundException("User", id);
            });
        
        // INFO：重要的业务事件
        logger.info("User {} loaded successfully", id);
        
        return toDTO(user);
    }
    
    public void updateUser(Long id, UpdateUserRequest request) {
        try {
            User user = getUserById(id);
            user.setEmail(request.getEmail());
            userRepository.save(user);
            logger.info("User {} updated successfully", id);
        } catch (Exception e) {
            // ERROR：需要关注的错误
            logger.error("Failed to update user {}", id, e);
            throw e;
        }
    }
}
```

### 使用 Lombok 简化日志

```java
// ✅ 使用 Lombok @Slf4j
@Service
@RequiredArgsConstructor
@Slf4j
public class UserService {
    
    public UserDTO getUserById(Long id) {
        log.debug("Fetching user with id: {}", id);
        // ... 代码
    }
}
```

## 数据验证

### 验证规范

```java
// ✅ 好的验证
@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
public class UserController {
    
    @PostMapping
    public ResponseEntity<UserDTO> createUser(
            @Valid @RequestBody CreateUserRequest request) {
        // @Valid 会自动触发验证，验证失败返回 400
        return ResponseEntity.ok(userService.createUser(request));
    }
}

// 请求对象
@Data
public class CreateUserRequest {
    
    @NotBlank(message = "Username is required")
    @Size(min = 3, max = 50, message = "Username must be between 3 and 50 characters")
    private String username;
    
    @NotBlank(message = "Email is required")
    @Email(message = "Email format is invalid")
    private String email;
    
    @Min(value = 18, message = "Age must be at least 18")
    @Max(value = 150, message = "Age is invalid")
    private Integer age;
}
```

## 性能优化

### 使用缓存

```java
@Service
@RequiredArgsConstructor
@Slf4j
public class UserService {
    private final UserRepository userRepository;
    
    // 读取操作使用缓存
    @Cacheable(value = "users", key = "#id")
    public User getUserById(Long id) {
        log.debug("Fetching user from database with id: {}", id);
        return userRepository.findById(id)
            .orElseThrow(() -> new ResourceNotFoundException("User", id));
    }
    
    // 修改操作更新缓存
    @CachePut(value = "users", key = "#result.id")
    public User updateUser(Long id, UpdateUserRequest request) {
        User user = getUserById(id);
        user.setEmail(request.getEmail());
        return userRepository.save(user);
    }
    
    // 删除操作清除缓存
    @CacheEvict(value = "users", key = "#id")
    public void deleteUser(Long id) {
        userRepository.deleteById(id);
    }
}
```

### 使用分页和排序

```java
@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
public class UserController {
    
    private final UserService userService;
    
    @GetMapping
    public ResponseEntity<Page<UserDTO>> listUsers(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            @RequestParam(defaultValue = "id") String sortBy,
            @RequestParam(defaultValue = "ASC") Sort.Direction direction) {
        
        Pageable pageable = PageRequest.of(page, size, Sort.by(direction, sortBy));
        Page<UserDTO> users = userService.listUsers(pageable);
        return ResponseEntity.ok(users);
    }
}
```

## 测试

> [!TIP]
> **测试金字塔**: 单元测试 (70%) > 集成测试 (20%) > E2E测试 (10%)。保证代码覆盖率达到80%以上,核心业务逻辑达到90%以上。

### 单元测试

```java
@SpringBootTest
public class UserServiceTest {
    
    @MockBean
    private UserRepository userRepository;
    
    @InjectMocks
    private UserService userService;
    
    @Test
    public void testGetUserById_Success() {
        // Arrange
        Long userId = 1L;
        User user = new User(userId, "john", "john@example.com");
        when(userRepository.findById(userId)).thenReturn(Optional.of(user));
        
        // Act
        User result = userService.getUserById(userId);
        
        // Assert
        assertNotNull(result);
        assertEquals("john", result.getUsername());
        verify(userRepository, times(1)).findById(userId);
    }
    
    @Test
    public void testGetUserById_NotFound() {
        // Arrange
        Long userId = 999L;
        when(userRepository.findById(userId)).thenReturn(Optional.empty());
        
        // Act & Assert
        assertThrows(ResourceNotFoundException.class, 
            () -> userService.getUserById(userId));
    }
}
```

## API 设计

### RESTful 规范

```java
// ✅ 好的 API 设计
@RestController
@RequestMapping("/api/v1/users")
@RequiredArgsConstructor
public class UserController {
    
    // 获取所有用户
    @GetMapping
    public ResponseEntity<Page<UserDTO>> list(
            @ParameterObject Pageable pageable) {
        return ResponseEntity.ok(userService.list(pageable));
    }
    
    // 获取单个用户
    @GetMapping("/{id}")
    public ResponseEntity<UserDTO> get(@PathVariable Long id) {
        return ResponseEntity.ok(userService.getById(id));
    }
    
    // 创建用户
    @PostMapping
    public ResponseEntity<UserDTO> create(@Valid @RequestBody CreateUserRequest request) {
        UserDTO user = userService.create(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(user);
    }
    
    // 更新用户
    @PutMapping("/{id}")
    public ResponseEntity<UserDTO> update(
            @PathVariable Long id,
            @Valid @RequestBody UpdateUserRequest request) {
        return ResponseEntity.ok(userService.update(id, request));
    }
    
    // 删除用户
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        userService.delete(id);
        return ResponseEntity.noContent().build();
    }
}
```

## 安全性

### 敏感信息处理

```java
@RestController
@RequestMapping("/api/users")
public class UserController {
    
    // ❌ 不应该返回密码等敏感信息
    @GetMapping("/{id}")
    public ResponseEntity<User> getUser(@PathVariable Long id) {
        User user = userService.getUser(id);
        // user 对象包含密码！
        return ResponseEntity.ok(user);
    }
    
    // ✅ 使用 DTO 只返回需要的字段
    @GetMapping("/{id}")
    public ResponseEntity<UserDTO> getUser(@PathVariable Long id) {
        User user = userService.getUser(id);
        UserDTO dto = UserMapper.toDTO(user);  // 不包含密码
        return ResponseEntity.ok(dto);
    }
}

// DTO 中不包含敏感字段
@Data
public class UserDTO {
    private Long id;
    private String username;
    private String email;
    // 没有 password 字段
}
```

## 总结

> [!IMPORTANT]
> **最佳实践核心原则:**
>
> 1. ✅ **提高代码质量和可维护性** - 使用标准化的项目结构和命名规范
> 2. ✅ **减少 bug 和性能问题** - 注意 N+1 查询、缓存使用、分页查询
> 3. ✅ **便于团队协作和代码审查** - 统一的编码风格和最佳实践
> 4. ✅ **更好地应对扩展和变化** - Service 层封装业务逻辑，DTO 分离层次
> 5. ✅ **提升应用的安全性和可靠性** - 统一异常处理，敏感信息保护

下一步学习 [部署上线](./deployment)。
