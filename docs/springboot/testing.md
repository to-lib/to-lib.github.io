---
sidebar_position: 14
---

# Spring Boot 测试

> [!TIP]
> **测试金字塔原则**: 单元测试（70%）> 集成测试（20%）> 端到端测试（10%）。好的测试覆盖率是高质量代码的保证！

## 测试依赖

Spring Boot 提供了 `spring-boot-starter-test` 依赖，包含了常用的测试库：

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-test</artifactId>
    <scope>test</scope>
</dependency>
```

这个 Starter 包含：

- **JUnit 5** - 单元测试框架
- **Spring Test** - Spring 集成测试支持
- **AssertJ** - 流式断言库
- **Hamcrest** - 匹配器库
- **Mockito** - Mock 框架
- **JSONassert** - JSON 断言
- **JsonPath** - JSON 路径表达式

## 单元测试

### 基本单元测试

```java
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.assertj.core.api.Assertions.assertThat;

@DisplayName("用户服务测试")
class UserServiceTest {
    
    @Test
    @DisplayName("应该正确创建用户")
    void shouldCreateUser() {
        // Given - 准备测试数据
        String username = "john";
        String email = "john@example.com";
        
        // When - 执行测试逻辑
        User user = new User(username, email);
        
        // Then - 验证结果
        assertThat(user.getUsername()).isEqualTo(username);
        assertThat(user.getEmail()).isEqualTo(email);
        assertThat(user.getId()).isNull();
    }
    
    @Test
    @DisplayName("空用户名应该抛出异常")
    void shouldThrowExceptionForNullUsername() {
        assertThatThrownBy(() -> new User(null, "test@example.com"))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("Username cannot be null");
    }
}
```

### 使用 Mockito

```java
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import static org.mockito.Mockito.*;
import static org.assertj.core.api.Assertions.assertThat;

@ExtendWith(MockitoExtension.class)
class UserServiceTest {
    
    @Mock
    private UserRepository userRepository;
    
    @InjectMocks
    private UserService userService;
    
    private User testUser;
    
    @BeforeEach
    void setUp() {
        testUser = new User("john", "john@example.com");
        testUser.setId(1L);
    }
    
    @Test
    @DisplayName("应该通过ID查找用户")
    void shouldFindUserById() {
        // Given
        when(userRepository.findById(1L))
            .thenReturn(Optional.of(testUser));
        
        // When
        User found = userService.findById(1L);
        
        // Then
        assertThat(found).isNotNull();
        assertThat(found.getUsername()).isEqualTo("john");
        verify(userRepository, times(1)).findById(1L);
    }
    
    @Test
    @DisplayName("应该保存用户")
    void shouldSaveUser() {
        // Given
        User newUser = new User("jane", "jane@example.com");
        when(userRepository.save(any(User.class)))
            .thenReturn(newUser);
        
        // When
        User saved = userService.save(newUser);
        
        // Then
        assertThat(saved).isNotNull();
        assertThat(saved.getUsername()).isEqualTo("jane");
        verify(userRepository).save(newUser);
    }
}
```

## Spring Boot 集成测试

### @SpringBootTest

```java
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

@SpringBootTest
@ActiveProfiles("test")
class UserServiceIntegrationTest {
    
    @Autowired
    private UserService userService;
    
    @Autowired
    private UserRepository userRepository;
    
    @Test
    void shouldSaveAndRetrieveUser() {
        // Given
        User user = new User("john", "john@example.com");
        
        // When
        User saved = userService.save(user);
        User found = userService.findById(saved.getId());
        
        // Then
        assertThat(found).isNotNull();
        assertThat(found.getUsername()).isEqualTo("john");
        assertThat(found.getEmail()).isEqualTo("john@example.com");
    }
}
```

### WebEnvironment 配置

```java
// 1. Mock 环境（默认）- 不启动真实服务器
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.MOCK)

// 2. 随机端口 - 启动真实服务器，使用随机端口
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)

// 3. 定义端口 - 启动真实服务器，使用指定端口
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.DEFINED_PORT)

// 4. 不启动 Web 环境
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE)
```

## Web 层测试

### MockMvc 测试

```java
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;
import static org.hamcrest.Matchers.*;

@WebMvcTest(UserController.class)
class UserControllerTest {
    
    @Autowired
    private MockMvc mockMvc;
    
    @MockBean
    private UserService userService;
    
    @Test
    @DisplayName("GET /api/users/{id} 应该返回用户")
    void shouldReturnUser() throws Exception {
        // Given
        User user = new User("john", "john@example.com");
        user.setId(1L);
        when(userService.findById(1L)).thenReturn(user);
        
        // When & Then
        mockMvc.perform(get("/api/users/1"))
            .andExpect(status().isOk())
            .andExpect(content().contentType(MediaType.APPLICATION_JSON))
            .andExpect(jsonPath("$.id", is(1)))
            .andExpect(jsonPath("$.username", is("john")))
            .andExpect(jsonPath("$.email", is("john@example.com")));
    }
    
    @Test
    @DisplayName("POST /api/users 应该创建用户")
    void shouldCreateUser() throws Exception {
        // Given
        User user = new User("jane", "jane@example.com");
        user.setId(2L);
        when(userService.save(any(User.class))).thenReturn(user);
        
        String userJson = """
            {
                "username": "jane",
                "email": "jane@example.com"
            }
            """;
        
        // When & Then
        mockMvc.perform(post("/api/users")
                .contentType(MediaType.APPLICATION_JSON)
                .content(userJson))
            .andExpect(status().isCreated())
            .andExpect(jsonPath("$.id", is(2)))
            .andExpect(jsonPath("$.username", is("jane")));
    }
    
    @Test
    @DisplayName("GET /api/users/{id} 用户不存在应该返回404")
    void shouldReturn404WhenUserNotFound() throws Exception {
        // Given
        when(userService.findById(999L)).thenReturn(null);
        
        // When & Then
        mockMvc.perform(get("/api/users/999"))
            .andExpect(status().isNotFound());
    }
}
```

### TestRestTemplate 测试

```java
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class UserControllerIntegrationTest {
    
    @Autowired
    private TestRestTemplate restTemplate;
    
    @Test
    void shouldCreateAndRetrieveUser() {
        // Given
        User newUser = new User("john", "john@example.com");
        
        // When - 创建用户
        ResponseEntity<User> createResponse = restTemplate
            .postForEntity("/api/users", newUser, User.class);
        
        // Then - 验证创建
        assertThat(createResponse.getStatusCode()).isEqualTo(HttpStatus.CREATED);
        User createdUser = createResponse.getBody();
        assertThat(createdUser.getId()).isNotNull();
        
        // When - 获取用户
        ResponseEntity<User> getResponse = restTemplate
            .getForEntity("/api/users/" + createdUser.getId(), User.class);
        
        // Then - 验证获取
        assertThat(getResponse.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(getResponse.getBody().getUsername()).isEqualTo("john");
    }
}
```

### WebTestClient 测试（响应式）

```java
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.reactive.AutoConfigureWebTestClient;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.reactive.server.WebTestClient;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@AutoConfigureWebTestClient
class UserControllerWebFluxTest {
    
    @Autowired
    private WebTestClient webTestClient;
    
    @Test
    void shouldGetUser() {
        webTestClient.get()
            .uri("/api/users/1")
            .exchange()
            .expectStatus().isOk()
            .expectBody()
            .jsonPath("$.username").isEqualTo("john")
            .jsonPath("$.email").isEqualTo("john@example.com");
    }
}
```

## 数据层测试

### @DataJpaTest

```java
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.boot.test.autoconfigure.orm.jpa.TestEntityManager;

@DataJpaTest
class UserRepositoryTest {
    
    @Autowired
    private TestEntityManager entityManager;
    
    @Autowired
    private UserRepository userRepository;
    
    @Test
    void shouldFindUserByUsername() {
        // Given
        User user = new User("john", "john@example.com");
        entityManager.persist(user);
        entityManager.flush();
        
        // When
        User found = userRepository.findByUsername("john");
        
        // Then
        assertThat(found).isNotNull();
        assertThat(found.getUsername()).isEqualTo("john");
        assertThat(found.getEmail()).isEqualTo("john@example.com");
    }
    
    @Test
    void shouldSaveUser() {
        // Given
        User user = new User("jane", "jane@example.com");
        
        // When
        User saved = userRepository.save(user);
        
        // Then
        assertThat(saved.getId()).isNotNull();
        assertThat(userRepository.findById(saved.getId())).isPresent();
    }
}
```

## 测试配置

### 测试配置文件

创建 `src/test/resources/application-test.yml`:

```yaml
spring:
  datasource:
    url: jdbc:h2:mem:testdb
    driver-class-name: org.h2.Driver
    username: sa
    password: 
  
  jpa:
    hibernate:
      ddl-auto: create-drop
    show-sql: true
    properties:
      hibernate:
        format_sql: true
  
  h2:
    console:
      enabled: true

logging:
  level:
    org.springframework: INFO
    com.example: DEBUG
```

### 自定义测试配置

```java
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Primary;

@TestConfiguration
public class TestConfig {
    
    @Bean
    @Primary
    public UserService testUserService() {
        return new UserService() {
            // 测试专用实现
        };
    }
}

// 在测试类中使用
@SpringBootTest
@Import(TestConfig.class)
class MyTest {
    // ...
}
```

## Testcontainers

> [!IMPORTANT]
> **真实环境测试**: Testcontainers 允许在 Docker 容器中运行真实的数据库、消息队列等服务，提供更接近生产环境的测试。

### 依赖

```xml
<dependency>
    <groupId>org.testcontainers</groupId>
    <artifactId>testcontainers</artifactId>
    <version>1.19.3</version>
    <scope>test</scope>
</dependency>
<dependency>
    <groupId>org.testcontainers</groupId>
    <artifactId>mysql</artifactId>
    <version>1.19.3</version>
    <scope>test</scope>
</dependency>
<dependency>
    <groupId>org.testcontainers</groupId>
    <artifactId>junit-jupiter</artifactId>
    <version>1.19.3</version>
    <scope>test</scope>
</dependency>
```

### MySQL Testcontainer

```java
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

@SpringBootTest
@Testcontainers
class UserRepositoryTestcontainersTest {
    
    @Container
    static MySQLContainer<?> mysql = new MySQLContainer<>("mysql:8.0")
        .withDatabaseName("testdb")
        .withUsername("test")
        .withPassword("test");
    
    @DynamicPropertySource
    static void properties(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", mysql::getJdbcUrl);
        registry.add("spring.datasource.username", mysql::getUsername);
        registry.add("spring.datasource.password", mysql::getPassword);
    }
    
    @Autowired
    private UserRepository userRepository;
    
    @Test
    void shouldSaveToRealDatabase() {
        // Given
        User user = new User("john", "john@example.com");
        
        // When
        User saved = userRepository.save(user);
        
        // Then
        assertThat(saved.getId()).isNotNull();
        assertThat(userRepository.findById(saved.getId())).isPresent();
    }
}
```

## 测试切片

Spring Boot 提供了多种测试切片注解，只加载必要的组件：

| 注解 | 用途 | 加载的组件 |
|------|------|-----------|
| `@WebMvcTest` | Web 层测试 | Controller, Filter, Advice |
| `@DataJpaTest` | JPA 测试 | Repository, EntityManager |
| `@DataMongoTest` | MongoDB 测试 | MongoDB Repository |
| `@DataRedisTest` | Redis 测试 | Redis Repository |
| `@RestClientTest` | REST Client 测试 | RestTemplate, WebClient |
| `@JsonTest` | JSON 序列化测试 | JSON 映射器 |

### @JsonTest 示例

```java
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.json.JsonTest;
import org.springframework.boot.test.json.JacksonTester;

@JsonTest
class UserJsonTest {
    
    @Autowired
    private JacksonTester<User> json;
    
    @Test
    void shouldSerialize() throws Exception {
        // Given
        User user = new User("john", "john@example.com");
        user.setId(1L);
        
        // When & Then
        assertThat(json.write(user))
            .extractingJsonPathNumberValue("$.id").isEqualTo(1);
        assertThat(json.write(user))
            .extractingJsonPathStringValue("$.username").isEqualTo("john");
    }
    
    @Test
    void shouldDeserialize() throws Exception {
        // Given
        String content = """
            {
                "id": 1,
                "username": "john",
                "email": "john@example.com"
            }
            """;
        
        // When
        User user = json.parse(content).getObject();
        
        // Then
        assertThat(user.getId()).isEqualTo(1L);
        assertThat(user.getUsername()).isEqualTo("john");
    }
}
```

## 性能测试

### JMH 基准测试

```xml
<dependency>
    <groupId>org.openjdk.jmh</groupId>
    <artifactId>jmh-core</artifactId>
    <version>1.37</version>
    <scope>test</scope>
</dependency>
<dependency>
    <groupId>org.openjdk.jmh</groupId>
    <artifactId>jmh-generator-annprocess</artifactId>
    <version>1.37</version>
    <scope>test</scope>
</dependency>
```

```java
import org.openjdk.jmh.annotations.*;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, warmups = 1)
@Warmup(iterations = 3)
@Measurement(iterations = 5)
public class UserServiceBenchmark {
    
    private UserService userService;
    
    @Setup
    public void setup() {
        userService = new UserService();
    }
    
    @Benchmark
    public void testFindUser() {
        userService.findById(1L);
    }
}
```

## 最佳实践

> [!TIP]
> **测试最佳实践**：
>
> 1. **遵循 AAA 模式** - Arrange（准备）、Act（执行）、Assert（断言）
> 2. **测试命名清晰** - 使用 @DisplayName 或 should/when 命名法
> 3. **隔离测试** - 每个测试独立，不依赖执行顺序
> 4. **合理使用 Mock** - 只 Mock 外部依赖
> 5. **测试边界条件** - 空值、异常、边界值
> 6. **持续集成** - 在 CI/CD 中自动运行测试

### 测试命名规范

```java
// 方法1：should/when 命名法
@Test
void shouldReturnUserWhenIdExists() { }

@Test
void shouldThrowExceptionWhenUsernameIsNull() { }

// 方法2：given/when/then 命名法
@Test
void givenValidUser_whenSaving_thenSuccess() { }

// 方法3：使用 @DisplayName
@Test
@DisplayName("用户名为空时应该抛出异常")
void test1() { }
```

### 测试数据准备

```java
// 使用 @BeforeEach 准备通用数据
@BeforeEach
void setUp() {
    testUser = new User("john", "john@example.com");
}

// 使用 Builder 模式
User user = User.builder()
    .username("john")
    .email("john@example.com")
    .age(30)
    .build();

// 使用测试数据工厂
public class UserTestFactory {
    public static User createDefaultUser() {
        return new User("john", "john@example.com");
    }
    
    public static User createUserWithId(Long id) {
        User user = createDefaultUser();
        user.setId(id);
        return user;
    }
}
```

## 测试覆盖率

### JaCoCo 配置

```xml
<plugin>
    <groupId>org.jacoco</groupId>
    <artifactId>jacoco-maven-plugin</artifactId>
    <version>0.8.11</version>
    <executions>
        <execution>
            <goals>
                <goal>prepare-agent</goal>
            </goals>
        </execution>
        <execution>
            <id>report</id>
            <phase>test</phase>
            <goals>
                <goal>report</goal>
            </goals>
        </execution>
    </executions>
</plugin>
```

运行测试并生成报告：

```bash
mvn clean test
mvn jacoco:report

# 报告位置：target/site/jacoco/index.html
```

## 常见问题

**Q: 单元测试和集成测试的区别？**  
A: 单元测试只测试单个类/方法，使用 Mock 隔离依赖；集成测试测试多个组件的协作，使用真实或接近真实的依赖。

**Q: 何时使用 @SpringBootTest？**  
A: 需要完整的 Spring 上下文时使用，但会影响测试速度。优先使用更轻量的测试切片（如 @WebMvcTest）。

**Q: 如何加速测试？**  
A:

1. 使用测试切片而非完整的 @SpringBootTest
2. 合理使用 Mock
3. 使用内存数据库（H2）
4. 并行运行测试

## 总结

- **单元测试** - JUnit 5 + Mockito，测试单个组件
- **集成测试** - @SpringBootTest，测试组件协作
- **Web 测试** - MockMvc/WebTestClient，测试 API
- **数据测试** - @DataJpaTest，测试数据层
- **Testcontainers** - 真实环境测试
- **测试覆盖率** - JaCoCo 生成报告

下一步学习 [AOP 面向切面编程](/docs/springboot/aop)。
