---
id: security-basics
title: Spring Security 基础
sidebar_label: 安全基础
sidebar_position: 10
---

# Spring Security 基础

> [!IMPORTANT]
> **安全第一**: Spring Security 提供身份认证和授权功能。理解认证(Authentication)和授权(Authorization)的区别是使用 Spring Security 的基础。

## 1. Spring Security 概述

**Spring Security** 是一个强大的安全框架，为 Spring 应用提供认证、授权和其他安全功能。

### 1.1 核心概念

| 概念 | 说明 |
|------|------|
| **Authentication** | 认证 - 验证用户身份（你是谁？） |
| **Authorization** | 授权 - 验证用户权限（你能做什么？） |
| **Principal** | 当前用户 |
| **GrantedAuthority** | 权限/角色 |
| **SecurityContext** | 安全上下文，存储认证信息 |

### 1.2 工作流程

```
请求 → Security过滤器链 → 认证 → 授权 → 资源访问
```

## 2. 基本配置

### 2.1 添加依赖

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-security</artifactId>
</dependency>
```

### 2.2 默认行为

添加依赖后，Spring Security 会：

- 保护所有端点
- 生成默认用户 `user`
- 在控制台打印随机密码
- 启用表单登录和 HTTP Basic 认证

### 2.3 自定义配置

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {
    
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/public/**").permitAll()  // 公开访问
                .requestMatchers("/admin/**").hasRole("ADMIN")  // 需要 ADMIN 角色
                .anyRequest().authenticated()  // 其他请求需要认证
            )
            .formLogin(form -> form
                .loginPage("/login")  // 自定义登录页
                .defaultSuccessUrl("/home")  // 登录成功后跳转
                .permitAll()
            )
            .logout(logout -> logout
                .logoutSuccessUrl("/login?logout")  // 登出成功跳转
                .permitAll()
            );
        
        return http.build();
    }
}
```

## 3. 用户认证

### 3.1 内存用户

```java
@Configuration
public class SecurityConfig {
    
    @Bean
    public UserDetailsService userDetailsService() {
        UserDetails user = User.builder()
            .username("user")
            .password(passwordEncoder().encode("password"))
            .roles("USER")
            .build();
        
        UserDetails admin = User.builder()
            .username("admin")
            .password(passwordEncoder().encode("admin123"))
            .roles("ADMIN", "USER")
            .build();
        
        return new InMemoryUserDetailsManager(user, admin);
    }
    
    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}
```

### 3.2 数据库用户

```java
@Entity
@Table(name = "users")
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(unique = true)
    private String username;
    
    private String password;
    
    private boolean enabled = true;
    
    @ElementCollection(fetch = FetchType.EAGER)
    private Set<String> roles = new HashSet<>();
    
    // getters and setters
}

@Service
public class CustomUserDetailsService implements UserDetailsService {
    
    @Autowired
    private UserRepository userRepository;
    
    @Override
    public UserDetails loadUserByUsername(String username) 
            throws UsernameNotFoundException {
        
        User user = userRepository.findByUsername(username)
            .orElseThrow(() -> 
                new UsernameNotFoundException("User not found: " + username));
        
        return org.springframework.security.core.userdetails.User.builder()
            .username(user.getUsername())
            .password(user.getPassword())
            .disabled(!user.isEnabled())
            .authorities(user.getRoles().stream()
                .map(role -> new SimpleGrantedAuthority("ROLE_" + role))
                .collect(Collectors.toList()))
            .build();
    }
}
```

## 4. 密码加密

### 4.1 BCrypt 加密

```java
@Service
public class UserService {
    
    @Autowired
    private PasswordEncoder passwordEncoder;
    
    @Autowired
    private UserRepository userRepository;
    
    public User registerUser(String username, String rawPassword) {
        User user = new User();
        user.setUsername(username);
        
        // 加密密码
        String encodedPassword = passwordEncoder.encode(rawPassword);
        user.setPassword(encodedPassword);
        
        return userRepository.save(user);
    }
    
    public boolean checkPassword(String rawPassword, String encodedPassword) {
        // 验证密码
        return passwordEncoder.matches(rawPassword, encodedPassword);
    }
}
```

### 4.2 密码编码器

```java
@Configuration
public class PasswordConfig {
    
    @Bean
    public PasswordEncoder passwordEncoder() {
        // BCrypt (推荐)
        return new BCryptPasswordEncoder();
        
        // 或使用其他编码器
        // return new Argon2PasswordEncoder();
        // return new SCryptPasswordEncoder();
        // return NoOpPasswordEncoder.getInstance(); // 不加密（仅用于测试）
    }
}
```

## 5. 授权

### 5.1 基于角色的授权

```java
@Configuration
public class SecurityConfig {
    
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(auth -> auth
                // 公开访问
                .requestMatchers("/", "/home", "/public/**").permitAll()
                
                // 需要 USER 角色
                .requestMatchers("/user/**").hasRole("USER")
                
                // 需要 ADMIN 角色
                .requestMatchers("/admin/**").hasRole("ADMIN")
                
                // 需要任一角色
                .requestMatchers("/protected/**").hasAnyRole("USER", "ADMIN")
                
                // 其他请求需要认证
                .anyRequest().authenticated()
            );
        
        return http.build();
    }
}
```

### 5.2 基于权限的授权

```java
http.authorizeHttpRequests(auth -> auth
    // 需要特定权限
    .requestMatchers("/users/delete").hasAuthority("DELETE_USER")
    .requestMatchers("/users/edit").hasAuthority("EDIT_USER")
    
    // 需要任一权限
    .requestMatchers("/users/**").hasAnyAuthority("READ_USER", "WRITE_USER")
);
```

### 5.3 方法级别授权

```java
@Configuration
@EnableMethodSecurity
public class MethodSecurityConfig {
}

@Service
public class UserService {
    
    // 需要 ADMIN 角色
    @PreAuthorize("hasRole('ADMIN')")
    public void deleteUser(Long id) {
        userRepository.deleteById(id);
    }
    
    // 需要特定权限
    @PreAuthorize("hasAuthority('WRITE_USER')")
    public User updateUser(User user) {
        return userRepository.save(user);
    }
    
    // 只能访问自己的数据
    @PreAuthorize("#username == authentication.principal.username")
    public User getUserProfile(String username) {
        return userRepository.findByUsername(username);
    }
    
    // 执行后检查
    @PostAuthorize("returnObject.username == authentication.principal.username")
    public User getUser(Long id) {
        return userRepository.findById(id).orElse(null);
    }
}
```

## 6. 获取当前用户

### 6.1 在 Controller 中

```java
@RestController
public class UserController {
    
    // 方式1：使用 Authentication
    @GetMapping("/current-user")
    public String getCurrentUser(Authentication authentication) {
        String username = authentication.getName();
        return "Current user: " + username;
    }
    
    // 方式2：使用 @AuthenticationPrincipal
    @GetMapping("/user-details")
    public UserDetails getUserDetails(@AuthenticationPrincipal UserDetails userDetails) {
        return userDetails;
    }
    
    // 方式3：使用 Principal
    @GetMapping("/username")
    public String getUsername(Principal principal) {
        return principal.getName();
    }
}
```

### 6.2 在 Service 中

```java
@Service
public class UserService {
    
    public String getCurrentUsername() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        
        if (authentication == null || !authentication.isAuthenticated()) {
            return null;
        }
        
        return authentication.getName();
    }
    
    public UserDetails getCurrentUserDetails() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        
        if (authentication != null && authentication.getPrincipal() instanceof UserDetails) {
            return (UserDetails) authentication.getPrincipal();
        }
        
        return null;
    }
}
```

## 7. 常用注解

### 7.1 @Secured

```java
@Service
public class AdminService {
    
    @Secured("ROLE_ADMIN")
    public void adminOperation() {
        // 只有 ADMIN 可以访问
    }
    
    @Secured({"ROLE_ADMIN", "ROLE_MANAGER"})
    public void managerOperation() {
        // ADMIN 或 MANAGER 可以访问
    }
}
```

### 7.2 @PreAuthorize 和 @PostAuthorize

```java
@Service
public class OrderService {
    
    // 执行前检查
    @PreAuthorize("hasRole('USER')")
    public Order createOrder(Order order) {
        return orderRepository.save(order);
    }
    
    // 复杂表达式
    @PreAuthorize("hasRole('ADMIN') or #order.userId == principal.userId")
    public Order updateOrder(Order order) {
        return orderRepository.save(order);
    }
    
    // 执行后检查
    @PostAuthorize("returnObject.userId == principal.userId")
    public Order getOrder(Long orderId) {
        return orderRepository.findById(orderId).orElse(null);
    }
}
```

## 8. CSRF 保护

### 8.1 启用 CSRF

```java
@Configuration
public class SecurityConfig {
    
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .csrf(csrf -> csrf.csrfTokenRepository(CookieCsrfTokenRepository.withHttpOnlyFalse()))
            // CSRF 默认启用
            .authorizeHttpRequests(auth -> auth.anyRequest().authenticated());
        
        return http.build();
    }
}
```

### 8.2 禁用 CSRF（仅用于 REST API）

```java
http.csrf(csrf -> csrf.disable());
```

### 8.3 在表单中使用 CSRF Token

```html
<form method="post" action="/submit">
    <input type="hidden" name="${_csrf.parameterName}" value="${_csrf.token}"/>
    <!-- 其他表单字段 -->
    <button type="submit">提交</button>
</form>
```

## 9. 最佳实践

### 9.1 永远不要明文存储密码

```java
// ✅ 正确
String encodedPassword = passwordEncoder.encode(rawPassword);
user.setPassword(encodedPassword);

// ❌ 错误
user.setPassword(rawPassword);  // 明文存储
```

### 9.2 使用 HTTPS

```yaml
# application.yml
server:
  ssl:
    enabled: true
    key-store: classpath:keystore.p12
    key-store-password: password
    key-store-type: PKCS12
```

### 9.3 最小权限原则

```java
// ✅ 正确：只给必需的权限
@PreAuthorize("hasAuthority('READ_USER')")
public User getUser(Long id) { }

// ❌ 错误：给过多权限
@PreAuthorize("hasRole('ADMIN')")  // 太宽泛
public User getUser(Long id) { }
```

### 9.4 保护敏感端点

```java
http.authorizeHttpRequests(auth -> auth
    // 健康检查端点应该保护
    .requestMatchers("/actuator/health").permitAll()
    .requestMatchers("/actuator/**").hasRole("ADMIN")
    
    // API 文档可以公开或保护
    .requestMatchers("/swagger-ui/**", "/api-docs/**").permitAll()
);
```

## 10. 总结

| 概念 | 说明 |
|------|------|
| Authentication | 认证，验证用户身份 |
| Authorization | 授权，验证用户权限 |
| @PreAuthorize | 方法执行前检查权限 |
| @Secured | 基于角色的方法安全 |
| PasswordEncoder | 密码加密 |
| CSRF | 跨站请求伪造保护 |

---

**关键要点**：

- 区分认证和授权
- 永远加密密码，使用 BCrypt
- 使用方法级别安全控制细粒度权限
- REST API 可以禁用 CSRF
- 遵循最小权限原则

**下一步**：学习 Spring Boot Security 获取更多高级特性
