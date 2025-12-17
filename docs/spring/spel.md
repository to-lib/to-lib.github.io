---
id: spel
title: Spring Expression Language (SpEL)
sidebar_label: SpEL
sidebar_position: 12
---

# Spring Expression Language (SpEL)

SpEL（Spring Expression Language）是 Spring 生态中贯穿多处的表达式语言，常见于：

- `@Value` 注入表达式
- 缓存注解的 `key/condition/unless`
- `@EventListener(condition = "...")`
- Spring Security 的方法鉴权表达式（表达式语法类似，但上下文不同）

## 1. 基本语法

### 1.1 字面量与运算

```text
'hello'
1 + 2
10 > 3
true and false
```

### 1.2 访问属性与方法

```text
user.name
user.getName()
```

### 1.3 安全导航与 Elvis 运算符

```text
user?.address?.city
user.nickname ?: user.name
```

## 2. 在 `@Value` 中使用 SpEL

### 2.1 读取系统属性与环境变量

```java
@Component
public class BuildInfo {

    @Value("#{systemProperties['user.timezone']}")
    private String timezone;
}
```

### 2.2 组合字符串

```java
@Component
public class Urls {

    @Value("#{'https://api.example.com/' + '${app.version}'}")
    private String apiUrl;
}
```

## 3. 集合：选择（selection）与投影（projection）

- 选择：筛选子集
- 投影：映射成新集合

```text
#users.?[age >= 18]      // 选择：成年人
#users.![name]          // 投影：提取 name 列表
```

## 4. 在 Spring 注解中使用 SpEL

### 4.1 `@EventListener(condition = "...")`

```java
@Component
public class OrderListener {

    @EventListener(condition = "#event.amount > 1000")
    public void onLargeOrder(OrderCreatedEvent event) {
    }
}
```

### 4.2 缓存注解里的 key/condition/unless

```java
@Cacheable(cacheNames = "users", key = "#id", unless = "#result == null")
public User findById(Long id) {
}
```

## 5. SpEL 上下文与变量

在不同注解里，SpEL 可用变量不完全一致。常见变量：

- `#root`：根对象（包含方法、目标对象等信息）
- `#this`：当前对象（视上下文而定）
- `#args`：参数数组
- `#p0/#a0`：第 0 个参数
- 直接用参数名：`#id`（前提是能获取到参数名）

## 6. 最佳实践

- **[表达式保持简单]**：复杂规则优先写 Java 代码，SpEL 做“轻量 glue”。
- **[避免重计算]**：在缓存 key 中避免复杂逻辑。
- **[参数命名可读]**：使用 `#id`、`#userId` 比 `#p0` 更易维护。

---

下一步建议：

- [缓存抽象](/docs/spring/caching)
- [配置与 Profiles](/docs/spring/configuration)
