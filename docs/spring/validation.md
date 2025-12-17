---
id: validation
title: 参数校验与 Bean Validation
sidebar_label: 参数校验
sidebar_position: 13
---

# 参数校验与 Bean Validation

在 Spring（特别是 Web 场景）中，参数校验通常基于 Bean Validation（Jakarta Validation / Hibernate Validator）。

典型收益：

- 统一的校验声明方式（注解）
- 自动返回 400（Web 场景）并携带详细错误
- 支持分组、级联校验、自定义约束

## 1. 常用注解速查

- `@NotNull` / `@NotBlank` / `@NotEmpty`
- `@Min` / `@Max`
- `@Size`
- `@Email`
- `@Pattern`

## 2. Web 请求体校验：`@Valid`

```java
public class CreateUserRequest {

    @NotBlank
    private String username;

    @Email
    private String email;
}

@RestController
@RequestMapping("/api/users")
public class UserController {

    @PostMapping
    public void create(@Valid @RequestBody CreateUserRequest req) {
    }
}
```

校验失败时，常见异常：

- `MethodArgumentNotValidException`（`@RequestBody`）
- `BindException`（表单/Query 参数绑定）

## 3. 方法参数校验：`@Validated`

如果你希望对 Service 方法参数做校验（不依赖 Web 层），需要开启方法级校验：

```java
@Service
@Validated
public class UserService {

    public void updateEmail(@NotNull Long userId, @Email String email) {
    }
}
```

校验失败时，常见异常：

- `ConstraintViolationException`

## 4. 分组校验（Groups）

```java
public interface Create {}
public interface Update {}

public class UserDTO {

    @NotNull(groups = Update.class)
    private Long id;

    @NotBlank(groups = {Create.class, Update.class})
    private String name;
}

@PostMapping
public void create(@Validated(Create.class) @RequestBody UserDTO dto) {
}
```

## 5. 自定义约束

### 5.1 定义注解

```java
@Target({ElementType.FIELD})
@Retention(RetentionPolicy.RUNTIME)
@Constraint(validatedBy = UsernameValidator.class)
public @interface Username {
    String message() default "invalid username";
    Class<?>[] groups() default {};
    Class<? extends Payload>[] payload() default {};
}
```

### 5.2 实现校验器

```java
public class UsernameValidator implements ConstraintValidator<Username, String> {

    @Override
    public boolean isValid(String value, ConstraintValidatorContext context) {
        if (value == null) {
            return true;
        }
        return value.matches("[a-zA-Z0-9_]{3,20}");
    }
}
```

## 6. 错误返回的统一处理（Web）

实践中建议用 `@RestControllerAdvice` 做统一的错误结构封装。

---

下一步建议：

- [Spring MVC](/docs/spring/spring-mvc)
- [最佳实践](/docs/spring/best-practices)
