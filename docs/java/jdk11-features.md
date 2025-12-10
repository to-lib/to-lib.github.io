---
sidebar_position: 17
title: JDK 11 新特性
---

# JDK 11 新特性

JDK 11 是继 JDK 8 之后的第一个长期支持版本（LTS），引入了 HTTP Client、字符串增强、局部变量类型推断增强等实用特性。

## HTTP Client API

JDK 11 正式引入了新的 HTTP Client API，替代旧的 `HttpURLConnection`。

### 同步请求

```java
import java.net.URI;
import java.net.http.*;
import java.time.Duration;

public class HttpClientSync {
    public static void main(String[] args) throws Exception {
        // 创建 HttpClient
        HttpClient client = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_2)  // HTTP/2
            .connectTimeout(Duration.ofSeconds(10))
            .build();

        // 创建 GET 请求
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create("https://api.github.com/users/octocat"))
            .header("Accept", "application/json")
            .timeout(Duration.ofSeconds(10))
            .GET()
            .build();

        // 发送请求并获取响应
        HttpResponse<String> response = client.send(
            request,
            HttpResponse.BodyHandlers.ofString()
        );

        // 处理响应
        System.out.println("状态码: " + response.statusCode());
        System.out.println("响应头: " + response.headers().map());
        System.out.println("响应体: " + response.body());
    }
}
```

### 异步请求

```java
import java.net.URI;
import java.net.http.*;
import java.util.concurrent.CompletableFuture;

public class HttpClientAsync {
    public static void main(String[] args) {
        HttpClient client = HttpClient.newHttpClient();

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create("https://api.github.com/users/octocat"))
            .build();

        // 异步发送请求
        CompletableFuture<HttpResponse<String>> futureResponse =
            client.sendAsync(request, HttpResponse.BodyHandlers.ofString());

        // 链式处理响应
        futureResponse
            .thenApply(HttpResponse::body)
            .thenAccept(body -> System.out.println("响应: " + body))
            .exceptionally(ex -> {
                System.err.println("请求失败: " + ex.getMessage());
                return null;
            });

        // 等待完成
        futureResponse.join();
    }
}
```

### POST 请求

```java
import java.net.URI;
import java.net.http.*;

public class HttpClientPost {
    public static void main(String[] args) throws Exception {
        HttpClient client = HttpClient.newHttpClient();

        // JSON 请求体
        String json = """
            {
                "name": "张三",
                "email": "zhangsan@example.com",
                "age": 25
            }
            """;

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create("https://api.example.com/users"))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(json))
            .build();

        HttpResponse<String> response = client.send(
            request,
            HttpResponse.BodyHandlers.ofString()
        );

        System.out.println("状态码: " + response.statusCode());
        System.out.println("响应: " + response.body());
    }
}
```

### 文件上传下载

```java
import java.net.URI;
import java.net.http.*;
import java.nio.file.*;

public class HttpClientFile {
    public static void main(String[] args) throws Exception {
        HttpClient client = HttpClient.newHttpClient();

        // 下载文件
        HttpRequest downloadRequest = HttpRequest.newBuilder()
            .uri(URI.create("https://example.com/file.zip"))
            .build();

        Path downloadPath = Paths.get("downloaded-file.zip");
        HttpResponse<Path> downloadResponse = client.send(
            downloadRequest,
            HttpResponse.BodyHandlers.ofFile(downloadPath)
        );
        System.out.println("文件已下载到: " + downloadResponse.body());

        // 上传文件
        Path uploadPath = Paths.get("upload.txt");
        HttpRequest uploadRequest = HttpRequest.newBuilder()
            .uri(URI.create("https://api.example.com/upload"))
            .header("Content-Type", "text/plain")
            .POST(HttpRequest.BodyPublishers.ofFile(uploadPath))
            .build();

        HttpResponse<String> uploadResponse = client.send(
            uploadRequest,
            HttpResponse.BodyHandlers.ofString()
        );
        System.out.println("上传响应: " + uploadResponse.body());
    }
}
```

## String 新方法

JDK 11 为 String 类添加了多个实用方法。

```java
public class StringEnhancements {
    public static void main(String[] args) {
        // isBlank()：检查是否为空或只包含空白字符
        String str1 = "   ";
        System.out.println(str1.isBlank());  // true
        System.out.println(str1.isEmpty());  // false

        // lines()：按行分割为 Stream
        String multiline = "第一行\n第二行\n第三行";
        multiline.lines().forEach(System.out::println);

        // strip()、stripLeading()、stripTrailing()：去除空白
        String str2 = "  Hello World  ";
        System.out.println(str2.strip());         // "Hello World"
        System.out.println(str2.stripLeading());  // "Hello World  "
        System.out.println(str2.stripTrailing()); // "  Hello World"

        // repeat()：重复字符串
        String str3 = "Hi ";
        System.out.println(str3.repeat(3));  // "Hi Hi Hi "

        // 比较 strip() 和 trim()
        String unicode = "\u2000Hello\u2000";  // 包含 Unicode 空白
        System.out.println(unicode.trim());   // 不能去除
        System.out.println(unicode.strip());  // 可以去除
    }
}
```

## 文件操作增强

新的文件读写方法使操作更加简洁。

```java
import java.nio.file.*;

public class FileEnhancements {
    public static void main(String[] args) throws Exception {
        Path path = Paths.get("test.txt");

        // 写入文件
        String content = "Hello, JDK 11!";
        Files.writeString(path, content);

        // 读取文件
        String readContent = Files.readString(path);
        System.out.println(readContent);

        // 判断文件是否为空
        boolean isEmpty = Files.size(path) == 0;
    }
}
```

## var 局部变量类型推断增强

JDK 10 引入了 `var`，JDK 11 允许在 Lambda 参数中使用。

```java
import java.util.*;

public class VarEnhancements {
    public static void main(String[] args) {
        // JDK 10: var 用于局部变量
        var list = new ArrayList<String>();
        var str = "Hello";
        var number = 42;

        // JDK 11: Lambda 参数中使用 var（可以添加注解）
        list.add("a");
        list.add("b");

        list.forEach((var item) -> System.out.println(item));

        // 为 Lambda 参数添加注解
        list.forEach((@Deprecated var item) -> {
            System.out.println(item);
        });
    }
}
```

## 集合工厂方法

创建不可变集合更加方便。

```java
import java.util.*;

public class CollectionFactories {
    public static void main(String[] args) {
        // List.of()（JDK 9+）
        List<String> list = List.of("a", "b", "c");

        // Set.of()
        Set<Integer> set = Set.of(1, 2, 3);

        // Map.of()
        Map<String, Integer> map1 = Map.of(
            "one", 1,
            "two", 2,
            "three", 3
        );

        // Map.ofEntries()：更多键值对
        Map<String, Integer> map2 = Map.ofEntries(
            Map.entry("one", 1),
            Map.entry("two", 2),
            Map.entry("three", 3)
        );

        // 注意：这些集合都是不可变的
        // list.add("d");  // 抛出 UnsupportedOperationException
    }
}
```

## Predicate.not()

Predicate 接口新增了 `not()` 方法，用于否定条件。

```java
import java.util.*;
import java.util.function.Predicate;
import java.util.stream.Collectors;

public class PredicateNot {
    public static void main(String[] args) {
        List<String> names = Arrays.asList("张三", "李四", "", "王五", "  ");

        // 传统方式
        List<String> nonBlank1 = names.stream()
            .filter(s -> !s.isBlank())
            .collect(Collectors.toList());

        // 使用 Predicate.not()
        List<String> nonBlank2 = names.stream()
            .filter(Predicate.not(String::isBlank))
            .collect(Collectors.toList());

        System.out.println(nonBlank2);  // [张三, 李四, 王五]
    }
}
```

## Optional 增强

Optional 类新增了 `isEmpty()` 方法。

```java
import java.util.Optional;

public class OptionalEnhancements {
    public static void main(String[] args) {
        Optional<String> empty = Optional.empty();
        Optional<String> nonEmpty = Optional.of("Hello");

        // isEmpty()：JDK 11 新增
        System.out.println(empty.isEmpty());     // true
        System.out.println(nonEmpty.isEmpty());  // false

        // 对比 isPresent()
        System.out.println(empty.isPresent());  // false
    }
}
```

## 嵌套类访问控制

JDK 11 新增了嵌套类访问 API。

```java
public class NestedAccess {
    public static void main(String[] args) {
        // 判断是否为嵌套类
        boolean isNested = InnerClass.class.isNestmate();

        // 获取嵌套主类
        Class<?> nestHost = InnerClass.class.getNestHost();
        System.out.println("嵌套主类: " + nestHost.getName());

        // 获取所有嵌套成员
        Class<?>[] nestMembers = NestedAccess.class.getNestMembers();
        for (Class<?> member : nestMembers) {
            System.out.println("嵌套成员: " + member.getName());
        }
    }

    class InnerClass {
        // 内部类
    }
}
```

## 运行单文件程序

JDK 11 支持直接运行单个 Java 源文件，无需先编译。

```bash
# 直接运行 Java 文件
java Hello.java

# 不再需要
javac Hello.java
java Hello
```

```java
// Hello.java
public class Hello {
    public static void main(String[] args) {
        System.out.println("Hello, JDK 11!");
    }
}
```

## 移除和废弃的特性

### 移除的特性

- **Java EE 和 CORBA 模块**：需要单独引入依赖
- **JavaFX**：从 JDK 中分离，需单独下载
- **Nashorn JavaScript 引擎**：被标记为废弃

### 使用 Java EE 模块

```xml
<!-- 如果需要 JAXB，需要添加依赖 -->
<dependency>
    <groupId>javax.xml.bind</groupId>
    <artifactId>jaxb-api</artifactId>
    <version>2.3.1</version>
</dependency>
```

## 性能改进

### Epsilon GC

一个"无操作"的垃圾收集器，用于性能测试。

```bash
# 启用 Epsilon GC
java -XX:+UnlockExperimentalVMOptions -XX:+UseEpsilonGC MyApp
```

### ZGC（实验性）

可扩展的低延迟垃圾收集器。

```bash
# 启用 ZGC
java -XX:+UnlockExperimentalVMOptions -XX:+UseZGC MyApp
```

## 实战示例

### 使用新特性重构代码

```java
import java.net.URI;
import java.net.http.*;
import java.util.*;
import java.util.function.Predicate;
import java.util.stream.Collectors;

public class RefactoringExample {
    public static void main(String[] args) throws Exception {
        // 使用 var 简化声明
        var client = HttpClient.newHttpClient();

        // 使用 String.lines() 和 Stream API
        var text = """
            apple
            banana

            cherry
            """;

        var fruits = text.lines()
            .filter(Predicate.not(String::isBlank))  // 过滤空行
            .map(String::strip)                       // 去除空白
            .collect(Collectors.toList());

        System.out.println(fruits);  // [apple, banana, cherry]

        // 使用新的 HTTP Client
        var request = HttpRequest.newBuilder()
            .uri(URI.create("https://api.example.com/fruits"))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(
                fruits.stream()
                    .collect(Collectors.joining(",", "[\"", "\"]"))
            ))
            .build();

        var response = client.send(
            request,
            HttpResponse.BodyHandlers.ofString()
        );

        if (response.statusCode() == 200) {
            System.out.println("提交成功: " + response.body());
        }
    }
}
```

## 最佳实践

### 1. 使用新的 HTTP Client

```java
// ✅ 推荐：使用新的 HTTP Client
HttpClient client = HttpClient.newHttpClient();
HttpRequest request = HttpRequest.newBuilder()
    .uri(URI.create("https://api.example.com"))
    .build();
HttpResponse<String> response = client.send(
    request,
    HttpResponse.BodyHandlers.ofString()
);

// ❌ 不推荐：使用旧的 HttpURLConnection
URL url = new URL("https://api.example.com");
HttpURLConnection conn = (HttpURLConnection) url.openConnection();
// ...
```

### 2. 使用 String 新方法

```java
// ✅ 使用 isBlank() 检查空白字符串
if (str.isBlank()) {
    // 处理空白字符串
}

// ✅ 使用 strip() 去除 Unicode 空白
String cleaned = str.strip();

// ✅ 使用 lines() 处理多行文本
text.lines().forEach(System.out::println);
```

### 3. 简化文件操作

```java
// ✅ 推荐：使用新的文件方法
String content = Files.readString(path);
Files.writeString(path, "新内容");

// ❌ 不推荐：使用旧的方式
List<String> lines = Files.readAllLines(path);
String content = String.join("\n", lines);
```

## 总结

JDK 11 的主要新特性：

- **HTTP Client API**：现代化的 HTTP 通信 API
- **String 增强**：`isBlank()`、`lines()`、`strip()`、`repeat()`
- **文件操作**：`Files.readString()`、`Files.writeString()`
- **var 增强**：Lambda 参数支持 var
- **集合工厂**：`List.of()`、`Set.of()`、`Map.of()`
- **Predicate.not()**：简化否定条件
- **Optional.isEmpty()**：检查 Optional 是否为空
- **单文件运行**：直接运行 `.java` 文件
- **性能改进**：Epsilon GC、ZGC

作为 LTS 版本，JDK 11 是生产环境的推荐选择。
