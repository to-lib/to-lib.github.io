---
sidebar_position: 9
---

# 建造者模式 (Builder Pattern)

## 模式定义

**建造者模式**是一种创建型设计模式，它允许你分步骤构建复杂对象，无需在构造函数中列出所有可能的参数组合。

## 问题分析

当需要创建复杂对象（包含很多可选属性）时，传统方法的问题：

```java
// 不好的做法 - 构造函数多载爆炸
public class Person {
    public Person(String name) { }
    public Person(String name, int age) { }
    public Person(String name, int age, String email) { }
    public Person(String name, int age, String email, String phone) { }
    // ... 更多组合
}
```

**问题**：
- 构造函数过多
- 代码可读性差
- 维护困难

## 解决方案

使用Builder模式分步构建对象：

```
┌─────────────────────────┐
│  Product（复杂对象）     │
│ - name                  │
│ - age                   │
│ - email                 │
│ - phone                 │
└─────────────────────────┘
         △
         │ creates
┌─────────────────────────┐
│  Builder（构建器）       │
│ + setName()             │
│ + setAge()              │
│ + setEmail()            │
│ + setPhone()            │
│ + build()               │
└─────────────────────────┘
```

## 代码实现

### 1. 复杂对象类

```java
public class Person {
    // 必需字段
    private String name;
    private int age;
    
    // 可选字段
    private String email;
    private String phone;
    private String address;
    private String city;
    
    // 私有构造函数，防止直接实例化
    private Person(Builder builder) {
        this.name = builder.name;
        this.age = builder.age;
        this.email = builder.email;
        this.phone = builder.phone;
        this.address = builder.address;
        this.city = builder.city;
    }
    
    // Getter方法
    public String getName() { return name; }
    public int getAge() { return age; }
    public String getEmail() { return email; }
    public String getPhone() { return phone; }
    public String getAddress() { return address; }
    public String getCity() { return city; }
    
    // 静态Builder类
    public static class Builder {
        // 必需字段
        private final String name;
        private final int age;
        
        // 可选字段，初始化为默认值
        private String email = "";
        private String phone = "";
        private String address = "";
        private String city = "";
        
        // 构造函数接收必需参数
        public Builder(String name, int age) {
            this.name = name;
            this.age = age;
        }
        
        // 设置可选字段的方法，返回Builder以支持链式调用
        public Builder email(String email) {
            this.email = email;
            return this;
        }
        
        public Builder phone(String phone) {
            this.phone = phone;
            return this;
        }
        
        public Builder address(String address) {
            this.address = address;
            return this;
        }
        
        public Builder city(String city) {
            this.city = city;
            return this;
        }
        
        // 构建对象
        public Person build() {
            return new Person(this);
        }
    }
}
```

### 2. 客户端使用

```java
public class Demo {
    public static void main(String[] args) {
        // 创建Person对象，使用链式调用
        Person person = new Person.Builder("张三", 28)
            .email("zhangsan@example.com")
            .phone("13800138000")
            .address("中国")
            .city("北京")
            .build();
        
        System.out.println("姓名: " + person.getName());
        System.out.println("年龄: " + person.getAge());
        System.out.println("邮箱: " + person.getEmail());
        System.out.println("电话: " + person.getPhone());
    }
}
```

## 实际应用示例

### HTTP请求构建器

```java
public class HttpRequest {
    private String url;
    private String method;
    private Map<String, String> headers;
    private String body;
    private int timeout;
    
    private HttpRequest(Builder builder) {
        this.url = builder.url;
        this.method = builder.method;
        this.headers = builder.headers;
        this.body = builder.body;
        this.timeout = builder.timeout;
    }
    
    public static class Builder {
        private final String url;
        private String method = "GET";
        private Map<String, String> headers = new HashMap<>();
        private String body = "";
        private int timeout = 30;
        
        public Builder(String url) {
            this.url = url;
        }
        
        public Builder method(String method) {
            this.method = method;
            return this;
        }
        
        public Builder header(String key, String value) {
            headers.put(key, value);
            return this;
        }
        
        public Builder body(String body) {
            this.body = body;
            return this;
        }
        
        public Builder timeout(int seconds) {
            this.timeout = seconds;
            return this;
        }
        
        public HttpRequest build() {
            return new HttpRequest(this);
        }
    }
    
    public void send() {
        System.out.println("发送请求: " + method + " " + url);
        System.out.println("超时: " + timeout + "秒");
    }
}

// 使用
HttpRequest request = new HttpRequest.Builder("https://api.example.com/users")
    .method("POST")
    .header("Content-Type", "application/json")
    .header("Authorization", "Bearer token")
    .body("{\"name\":\"张三\"}")
    .timeout(60)
    .build();

request.send();
```

### SQL查询构建器

```java
public class SqlBuilder {
    private StringBuilder query = new StringBuilder();
    
    public SqlBuilder select(String... columns) {
        query.append("SELECT ");
        query.append(String.join(", ", columns));
        return this;
    }
    
    public SqlBuilder from(String table) {
        query.append(" FROM ").append(table);
        return this;
    }
    
    public SqlBuilder where(String condition) {
        query.append(" WHERE ").append(condition);
        return this;
    }
    
    public SqlBuilder orderBy(String column) {
        query.append(" ORDER BY ").append(column);
        return this;
    }
    
    public SqlBuilder limit(int n) {
        query.append(" LIMIT ").append(n);
        return this;
    }
    
    public String build() {
        return query.toString();
    }
}

// 使用
String sql = new SqlBuilder()
    .select("id", "name", "email")
    .from("users")
    .where("age > 18")
    .orderBy("id")
    .limit(10)
    .build();

System.out.println(sql);
// SELECT id, name, email FROM users WHERE age > 18 ORDER BY id LIMIT 10
```

### 对象配置构建器

```java
public class DatabaseConfig {
    private String host;
    private int port;
    private String database;
    private String username;
    private String password;
    private int maxConnections;
    private boolean useSSL;
    
    private DatabaseConfig(Builder builder) {
        this.host = builder.host;
        this.port = builder.port;
        this.database = builder.database;
        this.username = builder.username;
        this.password = builder.password;
        this.maxConnections = builder.maxConnections;
        this.useSSL = builder.useSSL;
    }
    
    public static class Builder {
        private final String host;
        private final String database;
        
        private int port = 3306;
        private String username = "root";
        private String password = "";
        private int maxConnections = 10;
        private boolean useSSL = true;
        
        public Builder(String host, String database) {
            this.host = host;
            this.database = database;
        }
        
        public Builder port(int port) {
            this.port = port;
            return this;
        }
        
        public Builder credentials(String username, String password) {
            this.username = username;
            this.password = password;
            return this;
        }
        
        public Builder maxConnections(int max) {
            this.maxConnections = max;
            return this;
        }
        
        public Builder ssl(boolean useSSL) {
            this.useSSL = useSSL;
            return this;
        }
        
        public DatabaseConfig build() {
            return new DatabaseConfig(this);
        }
    }
}

// 使用
DatabaseConfig config = new DatabaseConfig.Builder("localhost", "myapp")
    .port(3306)
    .credentials("admin", "password123")
    .maxConnections(20)
    .ssl(true)
    .build();
```

## Java内置Builder

```java
// StringBuilder
String result = new StringBuilder()
    .append("Hello")
    .append(" ")
    .append("World")
    .toString();

// Stream API
List<String> result = list.stream()
    .filter(s -> s.length() > 5)
    .map(String::toUpperCase)
    .collect(Collectors.toList());
```

## 建造者模式 vs 工厂模式

| 特性 | 建造者 | 工厂 |
|------|-------|------|
| 对象复杂度 | 高 | 低 |
| 构造步骤 | 多步 | 一步 |
| 灵活性 | 高 | 低 |
| 可选参数 | 支持 | 不支持 |

## 优缺点

### 优点
- ✅ 避免构造函数参数过多
- ✅ 代码可读性好
- ✅ 支持灵活的对象构建
- ✅ 易于维护和扩展

### 缺点
- ❌ 代码复杂度增加
- ❌ 需要创建Builder类
- ❌ 对于简单对象可能过度设计

## 适用场景

- ✓ 对象有很多可选属性
- ✓ 对象构建步骤复杂
- ✓ 需要构建不可变对象
- ✓ 需要链式调用API
- ✓ 配置对象的创建

## 最佳实践

1. **Builder作为内部类** - 紧密关联
2. **支持链式调用** - 返回Builder对象
3. **验证参数** - 在build()中验证
4. **不可变对象** - 构建后不可修改
5. **清晰的参数分类** - 区分必需和可选参数
