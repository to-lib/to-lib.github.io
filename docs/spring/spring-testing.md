---
title: Spring 测试
sidebar_label: 测试
sidebar_position: 9
---

# Spring 测试

> [!IMPORTANT] > **测试驱动开发**: Spring TestContext Framework 提供了强大的测试支持。理解不同测试注解的使用场景是编写高质量测试的关键。

## 1. 测试概述

Spring 提供了完整的测试支持，可以编写单元测试和集成测试。

### 1.1 测试类型

| 测试类型       | 说明             | 工具                      |
| -------------- | ---------------- | ------------------------- |
| **单元测试**   | 测试单个类或方法 | JUnit + Mockito           |
| **集成测试**   | 测试多个组件协作 | @SpringBootTest           |
| **切片测试**   | 测试特定层       | @WebMvcTest, @DataJpaTest |
| **端到端测试** | 测试完整流程     | TestRestTemplate          |

### 1.2 依赖配置

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-test</artifactId>
    <scope>test</scope>
</dependency>
```

自动包含：

- JUnit 5
- Mockito
- AssertJ
- Hamcrest
- Spring Test

## 2. 单元测试

### 2.1 基本单元测试

```java
@ExtendWith(MockitoExtension.class)
class UserServiceTest {

    @Mock
    private UserRepository userRepository;

    @InjectMocks
    private UserService userService;

    @Test
    void testFindUserById() {
        // Arrange (准备)
        User mockUser = new User(1L, "John", "john@example.com");
        when(userRepository.findById(1L)).thenReturn(Optional.of(mockUser));

        // Act (执行)
        User result = userService.findById(1L);

        // Assert (断言)
        assertNotNull(result);
        assertEquals("John", result.getName());
        assertEquals("john@example.com", result.getEmail());

        // Verify (验证)
        verify(userRepository, times(1)).findById(1L);
    }

    @Test
    void testSaveUser() {
        // Arrange
        User newUser = new User(null, "Jane", "jane@example.com");
        User savedUser = new User(2L, "Jane", "jane@example.com");
        when(userRepository.save(any(User.class))).thenReturn(savedUser);

        // Act
        User result = userService.save(newUser);

        // Assert
        assertNotNull(result.getId());
        assertEquals(2L, result.getId());
        verify(userRepository).save(newUser);
    }
}
```

### 2.2 Mockito 常用方法

```java
@ExtendWith(MockitoExtension.class)
class MockitoExamplesTest {

    @Mock
    private UserRepository userRepository;

    @Test
    void mockitoExamples() {
        // 返回特定值
        when(userRepository.findById(1L))
            .thenReturn(Optional.of(new User(1L, "John", "john@example.com")));

        // 返回多个值（多次调用）
        when(userRepository.count())
            .thenReturn(1L)
            .thenReturn(2L)
            .thenReturn(3L);

        // 抛出异常
        when(userRepository.findById(999L))
            .thenThrow(new ResourceNotFoundException("User not found"));

        // 参数匹配器
        when(userRepository.findById(anyLong()))
            .thenReturn(Optional.of(new User()));

        // 验证调用次数
        verify(userRepository, times(1)).findById(1L);
        verify(userRepository, never()).delete(any());
        verify(userRepository, atLeast(1)).count();
        verify(userRepository, atMost(3)).findAll();

        // 验证参数
        ArgumentCaptor<User> userCaptor = ArgumentCaptor.forClass(User.class);
        verify(userRepository).save(userCaptor.capture());
        User capturedUser = userCaptor.getValue();
        assertEquals("John", capturedUser.getName());
    }
}
```

## 3. Spring 集成测试

### 3.1 @SpringBootTest

加载完整的应用上下文：

```java
@SpringBootTest
class UserServiceIntegrationTest {

    @Autowired
    private UserService userService;

    @Autowired
    private UserRepository userRepository;

    @BeforeEach
    void setUp() {
        userRepository.deleteAll();
    }

    @Test
    void testUserRegistration() {
        // 测试完整的用户注册流程
        User user = new User(null, "John", "john@example.com");
        User saved = userService.registerUser(user);

        assertNotNull(saved.getId());

        // 验证数据库中确实保存了
        Optional<User> found = userRepository.findById(saved.getId());
        assertTrue(found.isPresent());
        assertEquals("John", found.get().getName());
    }
}
```

### 3.2 配置测试环境

```java
// 使用特定配置类
@SpringBootTest(classes = TestConfig.class)
class CustomConfigTest {
}

// 设置环境属性
@SpringBootTest(properties = {
    "spring.datasource.url=jdbc:h2:mem:testdb",
    "logging.level.root=WARN"
})
class PropertyTest {
}

// 使用随机端口
@SpringBootTest(webEnvironment = WebEnvironment.RANDOM_PORT)
class WebTest {

    @LocalServerPort
    private int port;

    @Test
    void testPort() {
        System.out.println("Server port: " + port);
    }
}
```

### 3.3 @ActiveProfiles

使用特定的配置文件：

```java
@SpringBootTest
@ActiveProfiles("test")
class ProfileTest {
    // 会加载 application-test.yml 或 application-test.properties
}
```

```yaml
# application-test.yml
spring:
  datasource:
    url: jdbc:h2:mem:testdb
  jpa:
    hibernate:
      ddl-auto: create-drop
```

## 4. Web 层测试

### 4.1 @WebMvcTest

只加载 Web 层组件：

```java
@WebMvcTest(UserController.class)
class UserControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private UserService userService;

    @Test
    void testGetUser() throws Exception {
        // Arrange
        User mockUser = new User(1L, "John", "john@example.com");
        when(userService.findById(1L)).thenReturn(mockUser);

        // Act & Assert
        mockMvc.perform(get("/api/users/1"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.id").value(1))
            .andExpect(jsonPath("$.name").value("John"))
            .andExpect(jsonPath("$.email").value("john@example.com"));

        verify(userService).findById(1L);
    }

    @Test
    void testCreateUser() throws Exception {
        // Arrange
        User newUser = new User(null, "Jane", "jane@example.com");
        User savedUser = new User(2L, "Jane", "jane@example.com");
        when(userService.save(any(User.class))).thenReturn(savedUser);

        // Act & Assert
        mockMvc.perform(post("/api/users")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"name\":\"Jane\",\"email\":\"jane@example.com\"}"))
            .andExpect(status().isCreated())
            .andExpect(jsonPath("$.id").value(2))
            .andExpect(jsonPath("$.name").value("Jane"));
    }

    @Test
    void testDeleteUser() throws Exception {
        mockMvc.perform(delete("/api/users/1"))
            .andExpect(status().isNoContent());

        verify(userService).deleteById(1L);
    }
}
```

### 4.2 MockMvc 详解

```java
@WebMvcTest
class MockMvcExamplesTest {

    @Autowired
    private MockMvc mockMvc;

    @Test
    void mockMvcExamples() throws Exception {
        mockMvc.perform(
            get("/api/users")
                .param("page", "0")
                .param("size", "10")
                .header("Authorization", "Bearer token")
                .accept(MediaType.APPLICATION_JSON)
        )
        .andExpect(status().isOk())
        .andExpect(content().contentType(MediaType.APPLICATION_JSON))
        .andExpect(jsonPath("$.length()").value(10))
        .andDo(print());  // 打印请求和响应详情
    }
}
```

## 5. 数据层测试

### 5.1 @DataJpaTest

只加载 JPA 组件：

```java
@DataJpaTest
class UserRepositoryTest {

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private TestEntityManager entityManager;

    @Test
    void testFindByEmail() {
        // Arrange
        User user = new User(null, "John", "john@example.com");
        entityManager.persist(user);
        entityManager.flush();

        // Act
        User found = userRepository.findByEmail("john@example.com");

        // Assert
        assertNotNull(found);
        assertEquals("John", found.getName());
    }

    @Test
    void testFindByAgeGreaterThan() {
        // Arrange
        entityManager.persist(new User(null, "John", "john@example.com", 25));
        entityManager.persist(new User(null, "Jane", "jane@example.com", 30));
        entityManager.persist(new User(null, "Bob", "bob@example.com", 20));
        entityManager.flush();

        // Act
        List<User> users = userRepository.findByAgeGreaterThan(22);

        // Assert
        assertEquals(2, users.size());
    }
}
```

### 5.2 使用内存数据库

```xml
<!-- H2 内存数据库 -->
<dependency>
    <groupId>com.h2database</groupId>
    <artifactId>h2</artifactId>
    <scope>test</scope>
</dependency>
```

```yaml
# application-test.yml
spring:
  datasource:
    url: jdbc:h2:mem:testdb
    driver-class-name: org.h2.Driver
  jpa:
    hibernate:
      ddl-auto: create-drop
    show-sql: true
```

## 6. 事务测试

### 6.1 @Transactional

```java
@SpringBootTest
@Transactional  // 测试后自动回滚
class TransactionalTest {

    @Autowired
    private UserRepository userRepository;

    @Test
    void testSaveUser() {
        User user = new User(null, "John", "john@example.com");
        userRepository.save(user);

        // 测试内可以查到
        assertEquals(1, userRepository.count());

        // 测试结束后会自动回滚，不会真正保存到数据库
    }
}
```

### 6.2 禁用自动回滚

```java
@SpringBootTest
class NoRollbackTest {

    @Autowired
    private UserRepository userRepository;

    @Test
    @Transactional
    @Rollback(false)  // 不回滚，保留数据
    void testSaveUserPermanently() {
        User user = new User(null, "John", "john@example.com");
        userRepository.save(user);
    }

    @AfterEach
    void cleanup() {
        // 手动清理数据
        userRepository.deleteAll();
    }
}
```

## 7. REST 测试

### 7.1 TestRestTemplate

```java
@SpringBootTest(webEnvironment = WebEnvironment.RANDOM_PORT)
class UserRestTest {

    @Autowired
    private TestRestTemplate restTemplate;

    @Test
    void testGetUser() {
        ResponseEntity<User> response = restTemplate.getForEntity(
            "/api/users/1",
            User.class
        );

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertNotNull(response.getBody());
        assertEquals("John", response.getBody().getName());
    }

    @Test
    void testCreateUser() {
        User newUser = new User(null, "Jane", "jane@example.com");

        ResponseEntity<User> response = restTemplate.postForEntity(
            "/api/users",
            newUser,
            User.class
        );

        assertEquals(HttpStatus.CREATED, response.getStatusCode());
        assertNotNull(response.getBody().getId());
    }

    @Test
    void testUpdateUser() {
        User updateUser = new User(1L, "John Updated", "john@example.com");

        restTemplate.put("/api/users/1", updateUser);

        // 验证更新
        ResponseEntity<User> response = restTemplate.getForEntity(
            "/api/users/1",
            User.class
        );
        assertEquals("John Updated", response.getBody().getName());
    }

    @Test
    void testDeleteUser() {
        restTemplate.delete("/api/users/1");

        // 验证删除
        ResponseEntity<User> response = restTemplate.getForEntity(
            "/api/users/1",
            User.class
        );
        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
    }
}
```

## 8. 测试配置

### 8.1 @TestConfiguration

```java
@TestConfiguration
public class TestConfig {

    @Bean
    public DataSource dataSource() {
        return new EmbeddedDatabaseBuilder()
            .setType(EmbeddedDatabaseType.H2)
            .build();
    }

    @Bean
    public EmailService emailService() {
        return new MockEmailService();  // 测试用的 Mock 实现
    }
}

@SpringBootTest
@Import(TestConfig.class)
class ConfigTest {
    // 使用测试配置
}
```

### 8.2 @MockBean vs @Mock

```java
@SpringBootTest
class MockBeanExampleTest {

    // @MockBean - Spring 容器中的 Mock
    // 会替换容器中的真实 Bean
    @MockBean
    private EmailService emailService;

    @Autowired
    private UserService userService;  // 会注入上面的 MockBean

    @Test
    void test() {
        when(emailService.send(anyString())).thenReturn(true);
        userService.registerUser(new User());
        verify(emailService).send(anyString());
    }
}

@ExtendWith(MockitoExtension.class)
class MockExampleTest {

    // @Mock - 纯 Mockito Mock
    // 不在 Spring 容器中
    @Mock
    private EmailService emailService;

    @InjectMocks
    private UserService userService;  // 手动注入 Mock

    @Test
    void test() {
        when(emailService.send(anyString())).thenReturn(true);
        userService.registerUser(new User());
    }
}
```

## 9. 断言库

### 9.1 JUnit 断言

```java
@Test
void junitAssertions() {
    User user = new User(1L, "John", "john@example.com");

    assertEquals(1L, user.getId());
    assertEquals("John", user.getName());
    assertNotNull(user.getEmail());
    assertTrue(user.getEmail().contains("@"));
    assertFalse(user.getName().isEmpty());

    assertThrows(IllegalArgumentException.class, () -> {
        user.setAge(-1);  // 假设会抛异常
    });
}
```

### 9.2 AssertJ 断言

```java
@Test
void assertJAssertions() {
    User user = new User(1L, "John", "john@example.com");

    assertThat(user.getId()).isEqualTo(1L);
    assertThat(user.getName()).isEqualTo("John").startsWith("J");
    assertThat(user.getEmail()).isNotNull().contains("@");

    List<User> users = Arrays.asList(
        new User(1L, "John", "john@example.com"),
        new User(2L, "Jane", "jane@example.com")
    );

    assertThat(users)
        .hasSize(2)
        .extracting(User::getName)
        .containsExactly("John", "Jane");
}
```

## 10. 最佳实践

### 10.1 遵循 AAA 模式

```java
@Test
void testExample() {
    // Arrange - 准备测试数据
    User user = new User(null, "John", "john@example.com");
    when(userRepository.save(any())).thenReturn(user);

    // Act - 执行被测试的方法
    User result = userService.save(user);

    // Assert - 验证结果
    assertNotNull(result);
    assertEquals("John", result.getName());
}
```

### 10.2 使用有意义的测试名称

```java
// ✅ 好的命名
@Test
void shouldReturnUserWhenUserExists() { }

@Test
void shouldThrowExceptionWhenUserNotFound() { }

// ❌ 差的命名
@Test
void test1() { }

@Test
void testUser() { }
```

### 10.3 一个测试只测一个场景

```java
// ✅ 推荐：每个测试方法只测一个场景
@Test
void shouldReturnUserWhenIdExists() {
    // 测试正常情况
}

@Test
void shouldThrowExceptionWhenIdNotExists() {
    // 测试异常情况
}

// ❌ 避免：一个测试方法测多个场景
@Test
void testFindUser() {
    // 测试多种情况...
}
```

### 10.4 避免测试私有方法

```java
// ✅ 推荐：测试公共接口
@Test
void testPublicMethod() {
    userService.registerUser(user);  // 测试公共方法
}

// ❌ 避免：测试私有方法
// 私有方法应该通过公共方法间接测试
```

## 11. 总结

| 注解            | 用途         | 加载内容           |
| --------------- | ------------ | ------------------ |
| @SpringBootTest | 完整集成测试 | 完整应用上下文     |
| @WebMvcTest     | Web 层测试   | 仅 Web 层          |
| @DataJpaTest    | 数据层测试   | 仅 JPA 组件        |
| @MockBean       | Spring Mock  | 容器中的 Mock Bean |
| @Mock           | Mockito Mock | 纯 Mock 对象       |

---

**关键要点**：

- 单元测试用 Mockito，集成测试用 @SpringBootTest
- 使用切片测试提高测试速度
- 遵循 AAA 模式编写清晰的测试
- 使用内存数据库进行数据层测试
- 测试应该快速、独立、可重复

**下一步**：学习 [最佳实践](/docs/spring/best-practices)
