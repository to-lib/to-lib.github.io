---
id: spring-mvc
title: Spring MVC
sidebar_label: Spring MVC
sidebar_position: 7
---

# Spring MVC

## 1. Spring MVC概述

**Spring MVC**是Spring框架的Web模块，实现了Model-View-Controller（MVC）设计模式。

### 1.1 MVC架构

```
请求 → DispatcherServlet → HandlerMapping → Controller → 
Model → ViewResolver → View → 响应
```

### 1.2 核心组件

| 组件 | 说明 |
|------|------|
| **DispatcherServlet** | 前端控制器，处理所有请求 |
| **HandlerMapping** | 请求处理器映射，将URL映射到Controller |
| **Controller** | 业务处理器，处理请求逻辑 |
| **Model** | 数据模型，用于传递数据到视图 |
| **ViewResolver** | 视图解析器，根据视图名查找实际视图 |
| **View** | 视图，最终呈现给用户的内容 |

## 2. 请求处理

### 2.1 基本Controller

```java
@Controller
@RequestMapping("/users")
public class UserController {
    
    @Autowired
    private UserService userService;
    
    // 处理GET请求
    @GetMapping("/{id}")
    public String getUser(@PathVariable Long id, Model model) {
        User user = userService.getUserById(id);
        model.addAttribute("user", user);
        return "user/detail";  // 视图名称
    }
    
    // 处理POST请求
    @PostMapping
    public String saveUser(@ModelAttribute User user) {
        userService.saveUser(user);
        return "redirect:/users/" + user.getId();
    }
    
    // 处理PUT请求
    @PutMapping("/{id}")
    public String updateUser(@PathVariable Long id, @RequestBody User user) {
        user.setId(id);
        userService.updateUser(user);
        return "redirect:/users/" + id;
    }
    
    // 处理DELETE请求
    @DeleteMapping("/{id}")
    public String deleteUser(@PathVariable Long id) {
        userService.deleteUser(id);
        return "redirect:/users";
    }
}
```

### 2.2 REST Controller

```java
@RestController  // 返回JSON而不是视图
@RequestMapping("/api/users")
public class UserRestController {
    
    @Autowired
    private UserService userService;
    
    @GetMapping("/{id}")
    public ResponseEntity<User> getUser(@PathVariable Long id) {
        User user = userService.getUserById(id);
        if (user == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(user);
    }
    
    @PostMapping
    public ResponseEntity<User> createUser(@RequestBody User user) {
        User created = userService.saveUser(user);
        return ResponseEntity.created(
            URI.create("/api/users/" + created.getId())
        ).body(created);
    }
    
    @PutMapping("/{id}")
    public ResponseEntity<User> updateUser(@PathVariable Long id, @RequestBody User user) {
        user.setId(id);
        User updated = userService.updateUser(user);
        return ResponseEntity.ok(updated);
    }
    
    @DeleteMapping("/{id}")
    public ResponseEntity<?> deleteUser(@PathVariable Long id) {
        userService.deleteUser(id);
        return ResponseEntity.noContent().build();
    }
}
```

## 3. 请求参数处理

### 3.1 路径参数（@PathVariable）

```java
@GetMapping("/users/{id}")
public String getUser(@PathVariable Long id) {
    return "User ID: " + id;
}

// 可选的路径参数
@GetMapping("/users/{id:\\d+}")
public String getUserWithValidation(@PathVariable Long id) {
    return "User ID: " + id;
}

// 多个路径参数
@GetMapping("/users/{userId}/posts/{postId}")
public String getUserPost(
    @PathVariable Long userId,
    @PathVariable Long postId) {
    return String.format("User: %d, Post: %d", userId, postId);
}
```

### 3.2 查询参数（@RequestParam）

```java
@GetMapping("/search")
public String search(
    @RequestParam String keyword,
    @RequestParam(required = false) Integer page,
    @RequestParam(defaultValue = "10") Integer pageSize) {
    
    if (page == null) {
        page = 1;
    }
    
    return String.format("Searching: %s (page: %d, size: %d)", 
        keyword, page, pageSize);
}

// 获取所有参数
@GetMapping("/params")
public String getAllParams(@RequestParam Map<String, String> params) {
    params.forEach((key, value) -> System.out.println(key + ": " + value));
    return "OK";
}
```

### 3.3 请求头（@RequestHeader）

```java
@GetMapping("/check-auth")
public String checkAuth(
    @RequestHeader String authorization,
    @RequestHeader(value = "User-Agent", required = false) String userAgent) {
    
    return String.format("Auth: %s, UserAgent: %s", authorization, userAgent);
}

// 获取所有请求头
@GetMapping("/headers")
public String getHeaders(@RequestHeader HttpHeaders headers) {
    headers.forEach((name, values) -> 
        System.out.println(name + ": " + values)
    );
    return "OK";
}
```

### 3.4 请求体（@RequestBody）

```java
@PostMapping("/users")
public String createUser(@RequestBody User user) {
    System.out.println("Name: " + user.getName());
    System.out.println("Email: " + user.getEmail());
    return "User created";
}

// 接收JSON列表
@PostMapping("/users/batch")
public String createUsers(@RequestBody List<User> users) {
    users.forEach(user -> System.out.println(user.getName()));
    return "Users created";
}

// 接收任意JSON
@PostMapping("/data")
public String processData(@RequestBody JsonNode jsonData) {
    String name = jsonData.get("name").asText();
    int age = jsonData.get("age").asInt();
    return String.format("Name: %s, Age: %d", name, age);
}
```

### 3.5 请求对象绑定（@ModelAttribute）

```java
// 自动将请求参数绑定到对象
@PostMapping("/users/form")
public String createUserFromForm(@ModelAttribute User user) {
    System.out.println("Name: " + user.getName());
    return "redirect:/users/" + user.getId();
}

// 对应HTML表单
/*
<form method="post" action="/users/form">
    <input type="text" name="name" />
    <input type="email" name="email" />
    <input type="submit" value="Create" />
</form>
*/

// 隐式使用@ModelAttribute
@PostMapping("/users")
public String createUser(User user) {  // 自动使用@ModelAttribute
    return "redirect:/users/" + user.getId();
}
```

## 4. 响应处理

### 4.1 返回视图

```java
@Controller
@RequestMapping("/")
public class ViewController {
    
    @GetMapping
    public String index() {
        return "index";  // 返回index.html或index.jsp
    }
    
    @GetMapping("/users/{id}")
    public String getUserPage(@PathVariable Long id, Model model) {
        User user = new User(id, "John", "john@example.com");
        model.addAttribute("user", user);
        return "users/detail";  // 返回users/detail.html
    }
}
```

### 4.2 返回JSON

```java
@RestController
@RequestMapping("/api")
public class ApiController {
    
    // 返回单个对象
    @GetMapping("/user/{id}")
    public User getUser(@PathVariable Long id) {
        return new User(id, "John", "john@example.com");
    }
    
    // 返回列表
    @GetMapping("/users")
    public List<User> getUsers() {
        return Arrays.asList(
            new User(1L, "John", "john@example.com"),
            new User(2L, "Jane", "jane@example.com")
        );
    }
    
    // 返回自定义响应
    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> getStatus() {
        Map<String, Object> response = new HashMap<>();
        response.put("status", "OK");
        response.put("timestamp", System.currentTimeMillis());
        
        return ResponseEntity.ok(response);
    }
}
```

### 4.3 响应状态码

```java
@RestController
@RequestMapping("/api")
public class ApiController {
    
    // 200 OK（默认）
    @GetMapping("/users/{id}")
    public User getUser(@PathVariable Long id) {
        return userService.getUserById(id);
    }
    
    // 201 Created
    @PostMapping("/users")
    public ResponseEntity<User> createUser(@RequestBody User user) {
        User created = userService.saveUser(user);
        return ResponseEntity
            .created(URI.create("/api/users/" + created.getId()))
            .body(created);
    }
    
    // 204 No Content
    @DeleteMapping("/users/{id}")
    public ResponseEntity<?> deleteUser(@PathVariable Long id) {
        userService.deleteUser(id);
        return ResponseEntity.noContent().build();
    }
    
    // 400 Bad Request
    @PostMapping("/validate")
    public ResponseEntity<?> validate(@RequestBody User user) {
        if (user.getName() == null || user.getName().isEmpty()) {
            return ResponseEntity.badRequest().body("Name is required");
        }
        return ResponseEntity.ok("Valid");
    }
    
    // 404 Not Found
    @GetMapping("/users/{id}")
    public ResponseEntity<User> getUserOrNotFound(@PathVariable Long id) {
        User user = userService.getUserById(id);
        if (user == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(user);
    }
    
    // 500 Internal Server Error
    @GetMapping("/error")
    public ResponseEntity<?> error() {
        return ResponseEntity.internalServerError().body("Server error");
    }
}
```

## 5. 数据绑定和验证

### 5.1 参数验证

```java
@Data
public class User {
    @NotNull(message = "ID不能为空")
    private Long id;
    
    @NotBlank(message = "名称不能为空")
    private String name;
    
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
    public ResponseEntity<?> createUser(@Valid @RequestBody User user) {
        // 如果验证失败，会自动返回400
        return ResponseEntity.ok(user);
    }
}
```

### 5.2 异常处理

```java
@RestController
@RequestMapping("/api")
public class UserController {
    
    @PostMapping("/users")
    public ResponseEntity<?> createUser(@Valid @RequestBody User user,
                                       BindingResult bindingResult) {
        // 手动检查验证结果
        if (bindingResult.hasErrors()) {
            List<String> errors = bindingResult.getAllErrors()
                .stream()
                .map(ObjectError::getDefaultMessage)
                .collect(Collectors.toList());
            
            return ResponseEntity.badRequest().body(errors);
        }
        
        return ResponseEntity.ok(user);
    }
}

// 全局异常处理
@RestControllerAdvice
public class GlobalExceptionHandler {
    
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<?> handleValidationException(MethodArgumentNotValidException ex) {
        List<String> errors = ex.getBindingResult()
            .getAllErrors()
            .stream()
            .map(ObjectError::getDefaultMessage)
            .collect(Collectors.toList());
        
        return ResponseEntity.badRequest().body(errors);
    }
    
    @ExceptionHandler(Exception.class)
    public ResponseEntity<?> handleException(Exception ex) {
        return ResponseEntity.internalServerError()
            .body("Internal server error: " + ex.getMessage());
    }
}
```

## 6. 会话管理

### 6.1 使用HttpSession

```java
@Controller
@RequestMapping("/session")
public class SessionController {
    
    @PostMapping("/login")
    public String login(String username, HttpSession session) {
        // 保存到会话
        session.setAttribute("username", username);
        return "redirect:/session/home";
    }
    
    @GetMapping("/home")
    public String home(HttpSession session, Model model) {
        // 从会话获取
        String username = (String) session.getAttribute("username");
        model.addAttribute("username", username);
        return "home";
    }
    
    @PostMapping("/logout")
    public String logout(HttpSession session) {
        session.invalidate();  // 销毁会话
        return "redirect:/";
    }
}
```

### 6.2 使用Cookie

```java
@Controller
@RequestMapping("/cookie")
public class CookieController {
    
    @PostMapping("/set")
    public String setCookie(HttpServletResponse response) {
        Cookie cookie = new Cookie("theme", "dark");
        cookie.setMaxAge(3600);  // 1小时
        cookie.setPath("/");
        response.addCookie(cookie);
        return "redirect:/";
    }
    
    @GetMapping("/get")
    public String getCookie(@CookieValue(value = "theme", defaultValue = "light") String theme) {
        System.out.println("Theme: " + theme);
        return "OK";
    }
}
```

## 7. 拦截器

### 7.1 实现拦截器

```java
@Component
public class LoggingInterceptor implements HandlerInterceptor {
    
    @Override
    public boolean preHandle(HttpServletRequest request, 
                             HttpServletResponse response, 
                             Object handler) throws Exception {
        System.out.println("Request: " + request.getMethod() + " " + request.getRequestURI());
        return true;  // 继续处理请求
    }
    
    @Override
    public void postHandle(HttpServletRequest request, 
                           HttpServletResponse response, 
                           Object handler, 
                           ModelAndView modelAndView) throws Exception {
        System.out.println("Response status: " + response.getStatus());
    }
    
    @Override
    public void afterCompletion(HttpServletRequest request, 
                                HttpServletResponse response, 
                                Object handler, 
                                Exception ex) throws Exception {
        System.out.println("Request completed");
    }
}

// 注册拦截器
@Configuration
public class WebConfig implements WebMvcConfigurer {
    
    @Autowired
    private LoggingInterceptor loggingInterceptor;
    
    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(loggingInterceptor)
            .addPathPatterns("/**")  // 拦截所有请求
            .excludePathPatterns("/public/**");  // 排除某些路径
    }
}
```

## 8. 文件上传

### 8.1 单文件上传

```java
@Controller
@RequestMapping("/upload")
public class FileUploadController {
    
    @PostMapping
    public String uploadFile(@RequestParam("file") MultipartFile file) {
        if (!file.isEmpty()) {
            try {
                String filename = file.getOriginalFilename();
                String path = "/uploads/" + filename;
                file.transferTo(new File(path));
                return "redirect:/upload/success";
            } catch (IOException ex) {
                ex.printStackTrace();
                return "redirect:/upload/error";
            }
        }
        return "redirect:/upload/error";
    }
}
```

### 8.2 多文件上传

```java
@PostMapping("/multiple")
public String uploadFiles(@RequestParam("files") MultipartFile[] files) {
    for (MultipartFile file : files) {
        if (!file.isEmpty()) {
            try {
                String filename = file.getOriginalFilename();
                file.transferTo(new File("/uploads/" + filename));
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }
    return "redirect:/upload/success";
}
```

## 9. 总结

| 概念 | 说明 |
|------|------|
| Controller | 处理请求的类 |
| RequestMapping | URL映射 |
| RequestParam | 查询参数 |
| PathVariable | 路径参数 |
| RequestBody | 请求体 |
| ResponseEntity | 响应对象 |
| Model | 数据模型 |
| View | 视图 |

---

**下一步**：学习[REST API开发](./rest-api.md)
