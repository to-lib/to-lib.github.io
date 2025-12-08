---
sidebar_position: 6
---

# Web 开发

## RESTful API 基础

### REST 原则

1. **资源** - 一切皆资源（User、Post、Comment 等）
2. **方法** - 使用 HTTP 方法操作资源（GET、POST、PUT、DELETE）
3. **无状态** - 每个请求都是独立的
4. **表现层** - 资源的多种表现形式（JSON、XML）

### HTTP 方法对应操作

| HTTP 方法 | CRUD 操作 | 说明 |
|----------|----------|------|
| GET | Read | 获取资源 |
| POST | Create | 创建资源 |
| PUT | Update | 更新资源（全量） |
| PATCH | Update | 更新资源（部分） |
| DELETE | Delete | 删除资源 |

## 创建 RESTful API

### 基本控制器

```java
package com.example.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import com.example.entity.User;
import com.example.service.UserService;
import java.util.List;

@RestController
@RequestMapping("/api/users")
public class UserController {
    
    @Autowired
    private UserService userService;
    
    // GET /api/users - 获取所有用户
    @GetMapping
    public ResponseEntity<List<User>> getAllUsers() {
        List<User> users = userService.findAll();
        return ResponseEntity.ok(users);
    }
    
    // GET /api/users/{id} - 获取单个用户
    @GetMapping("/{id}")
    public ResponseEntity<User> getUserById(@PathVariable Long id) {
        User user = userService.findById(id);
        if (user != null) {
            return ResponseEntity.ok(user);
        } else {
            return ResponseEntity.notFound().build();
        }
    }
    
    // POST /api/users - 创建新用户
    @PostMapping
    public ResponseEntity<User> createUser(@RequestBody User user) {
        User savedUser = userService.save(user);
        return ResponseEntity.status(HttpStatus.CREATED).body(savedUser);
    }
    
    // PUT /api/users/{id} - 更新用户
    @PutMapping("/{id}")
    public ResponseEntity<User> updateUser(
            @PathVariable Long id,
            @RequestBody User user) {
        User updatedUser = userService.update(id, user);
        if (updatedUser != null) {
            return ResponseEntity.ok(updatedUser);
        } else {
            return ResponseEntity.notFound().build();
        }
    }
    
    // DELETE /api/users/{id} - 删除用户
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteUser(@PathVariable Long id) {
        userService.delete(id);
        return ResponseEntity.noContent().build();
    }
}
```

## 参数处理

### @PathVariable - 路径参数

```java
@GetMapping("/users/{id}/posts/{postId}")
public ResponseEntity<Post> getUserPost(
        @PathVariable Long id,
        @PathVariable Long postId) {
    // id = 1, postId = 5
    Post post = postService.getPost(id, postId);
    return ResponseEntity.ok(post);
}
```

访问：`/users/1/posts/5`

### @RequestParam - 查询参数

```java
@GetMapping("/users")
public ResponseEntity<List<User>> searchUsers(
        @RequestParam(required = false) String name,
        @RequestParam(required = false) Integer age,
        @RequestParam(defaultValue = "1") Integer page,
        @RequestParam(defaultValue = "10") Integer size) {
    List<User> users = userService.search(name, age, page, size);
    return ResponseEntity.ok(users);
}
```

访问：`/users?name=John&age=30&page=2&size=20`

### @RequestBody - 请求体

```java
@PostMapping("/users")
public ResponseEntity<User> createUser(@RequestBody User user) {
    // 自动将 JSON 反序列化为 User 对象
    User savedUser = userService.save(user);
    return ResponseEntity.status(HttpStatus.CREATED).body(savedUser);
}

// 请求示例
// POST /users
// Content-Type: application/json
// {
//   "name": "John Doe",
//   "email": "john@example.com",
//   "age": 30
// }
```

### @RequestHeader - 请求头

```java
@PostMapping("/users")
public ResponseEntity<User> createUser(
        @RequestBody User user,
        @RequestHeader("Authorization") String token,
        @RequestHeader(value = "X-API-Key", required = false) String apiKey) {
    // 获取请求头中的值
    return ResponseEntity.ok(userService.save(user));
}
```

### @CookieValue - Cookie

```java
@GetMapping("/profile")
public ResponseEntity<User> getProfile(
        @CookieValue("sessionId") String sessionId) {
    User user = userService.getUserBySession(sessionId);
    return ResponseEntity.ok(user);
}
```

## 数据验证

### Bean Validation

```java
import jakarta.validation.constraints.*;

public class User {
    
    @NotNull(message = "用户名不能为空")
    @Size(min = 3, max = 20, message = "用户名长度在3-20之间")
    private String username;
    
    @NotNull(message = "邮箱不能为空")
    @Email(message = "邮箱格式不正确")
    private String email;
    
    @Min(value = 18, message = "年龄不能小于18岁")
    @Max(value = 100, message = "年龄不能大于100岁")
    private Integer age;
    
    @NotBlank(message = "电话号码不能为空")
    @Pattern(regexp = "^1[3-9]\\d{9}$", message = "电话号码格式不正确")
    private String phone;
}
```

### 控制器中使用验证

```java
@PostMapping("/users")
public ResponseEntity<User> createUser(
        @Valid @RequestBody User user,
        BindingResult bindingResult) {
    
    if (bindingResult.hasErrors()) {
        // 处理验证错误
        String errorMessage = bindingResult.getFieldError()
                .getDefaultMessage();
        return ResponseEntity.badRequest().build();
    }
    
    User savedUser = userService.save(user);
    return ResponseEntity.status(HttpStatus.CREATED).body(savedUser);
}
```

### 自定义验证注解

```java
import jakarta.validation.Constraint;
import jakarta.validation.ConstraintValidator;
import jakarta.validation.ConstraintValidatorContext;
import java.lang.annotation.*;

@Target(ElementType.FIELD)
@Retention(RetentionPolicy.RUNTIME)
@Constraint(validatedBy = PhoneValidator.class)
public @interface ValidPhone {
    String message() default "Invalid phone number";
    Class<?>[] groups() default {};
    Class<? extends Payload>[] payload() default {};
}

public class PhoneValidator implements ConstraintValidator<ValidPhone, String> {
    
    @Override
    public boolean isValid(String value, ConstraintValidatorContext context) {
        if (value == null) {
            return true;
        }
        return value.matches("^1[3-9]\\d{9}$");
    }
}

// 使用
public class User {
    @ValidPhone
    private String phone;
}
```

## 异常处理

### 全局异常处理

```java
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.context.request.WebRequest;
import java.time.LocalDateTime;

@RestControllerAdvice
public class GlobalExceptionHandler {
    
    @ExceptionHandler(ResourceNotFoundException.class)
    public ResponseEntity<ErrorResponse> handleResourceNotFound(
            ResourceNotFoundException ex,
            WebRequest request) {
        
        ErrorResponse errorResponse = ErrorResponse.builder()
                .timestamp(LocalDateTime.now())
                .status(HttpStatus.NOT_FOUND.value())
                .error("Not Found")
                .message(ex.getMessage())
                .path(request.getDescription(false).replace("uri=", ""))
                .build();
        
        return new ResponseEntity<>(errorResponse, HttpStatus.NOT_FOUND);
    }
    
    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ErrorResponse> handleIllegalArgument(
            IllegalArgumentException ex,
            WebRequest request) {
        
        ErrorResponse errorResponse = ErrorResponse.builder()
                .timestamp(LocalDateTime.now())
                .status(HttpStatus.BAD_REQUEST.value())
                .error("Bad Request")
                .message(ex.getMessage())
                .path(request.getDescription(false).replace("uri=", ""))
                .build();
        
        return new ResponseEntity<>(errorResponse, HttpStatus.BAD_REQUEST);
    }
    
    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGenericException(
            Exception ex,
            WebRequest request) {
        
        ErrorResponse errorResponse = ErrorResponse.builder()
                .timestamp(LocalDateTime.now())
                .status(HttpStatus.INTERNAL_SERVER_ERROR.value())
                .error("Internal Server Error")
                .message("An unexpected error occurred")
                .path(request.getDescription(false).replace("uri=", ""))
                .build();
        
        return new ResponseEntity<>(errorResponse, HttpStatus.INTERNAL_SERVER_ERROR);
    }
}

// 错误响应对象
@Data
@Builder
public class ErrorResponse {
    private LocalDateTime timestamp;
    private int status;
    private String error;
    private String message;
    private String path;
}

// 自定义异常
public class ResourceNotFoundException extends RuntimeException {
    public ResourceNotFoundException(String message) {
        super(message);
    }
}
```

## 内容协商

### 返回不同格式

```java
@GetMapping(value = "/users/{id}", produces = {
        MediaType.APPLICATION_JSON_VALUE,
        MediaType.APPLICATION_XML_VALUE
})
public ResponseEntity<User> getUserById(@PathVariable Long id) {
    User user = userService.findById(id);
    return ResponseEntity.ok(user);
}
```

根据 Accept 请求头返回对应格式：
- `Accept: application/json` → JSON
- `Accept: application/xml` → XML

## CORS 配置

```java
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class CorsConfig implements WebMvcConfigurer {
    
    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/api/**")
                .allowedOrigins("http://localhost:3000", "https://example.com")
                .allowedMethods("GET", "POST", "PUT", "DELETE", "PATCH")
                .allowedHeaders("*")
                .exposedHeaders("Authorization", "Content-Type")
                .allowCredentials(true)
                .maxAge(3600);
    }
}
```

或使用注解：

```java
@RestController
@RequestMapping("/api/users")
@CrossOrigin(origins = "http://localhost:3000", maxAge = 3600)
public class UserController {
    // ...
}
```

## API 文档（Swagger/OpenAPI）

### 依赖

```xml
<dependency>
    <groupId>org.springdoc</groupId>
    <artifactId>springdoc-openapi-starter-webmvc-ui</artifactId>
    <version>2.0.4</version>
</dependency>
```

### 配置

```yaml
springdoc:
  swagger-ui:
    path: /swagger-ui.html
    enabled: true
  api-docs:
    path: /v3/api-docs
```

### 使用注解

```java
@RestController
@RequestMapping("/api/users")
@Tag(name = "用户管理", description = "用户相关接口")
public class UserController {
    
    @GetMapping("/{id}")
    @Operation(summary = "获取用户", description = "根据 ID 获取用户详情")
    @ApiResponse(responseCode = "200", description = "成功获取用户")
    @ApiResponse(responseCode = "404", description = "用户不存在")
    public ResponseEntity<User> getUserById(
            @PathVariable 
            @Parameter(description = "用户 ID") 
            Long id) {
        return ResponseEntity.ok(userService.findById(id));
    }
}
```

访问文档：`http://localhost:8080/swagger-ui.html`

## 总结

- **RESTful API** - 遵循 REST 原则设计 API
- **参数处理** - 使用 @PathVariable、@RequestParam、@RequestBody 等
- **数据验证** - 使用 Bean Validation 验证输入
- **异常处理** - 全局异常处理器统一处理异常
- **API 文档** - 使用 Swagger/OpenAPI 自动生成文档

下一步学习 [数据访问](./data-access.md)。
