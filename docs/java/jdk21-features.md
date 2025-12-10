---
sidebar_position: 16
title: JDK 21 新特性
---

# JDK 21 新特性

JDK 21 是继 JDK 17 之后的第二个长期支持版本(LTS),发布于 2023 年 9 月。本文介绍 JDK 21 的主要新特性和改进。

## 虚拟线程 (Virtual Threads)

虚拟线程是轻量级线程,可以大幅简化高并发应用的开发,特别适合 I/O 密集型任务。

### 基本概念

虚拟线程由 JVM 管理,而不是操作系统,因此可以创建数百万个虚拟线程而不会耗尽系统资源。

### 创建虚拟线程

```java
public class VirtualThreadExample {
    public static void main(String[] args) throws InterruptedException {
        // 方式 1: 使用 Thread.startVirtualThread()
        Thread vThread1 = Thread.startVirtualThread(() -> {
            System.out.println("虚拟线程 1: " + Thread.currentThread());
        });
        
        // 方式 2: 使用 Thread.ofVirtual()
        Thread vThread2 = Thread.ofVirtual().start(() -> {
            System.out.println("虚拟线程 2: " + Thread.currentThread());
        });
        
        // 方式 3: 使用 Executors
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            executor.submit(() -> {
                System.out.println("虚拟线程 3: " + Thread.currentThread());
                return "结果";
            });
        }
        
        vThread1.join();
        vThread2.join();
    }
}
```

### 虚拟线程 vs 平台线程

```java
import java.time.Duration;
import java.util.concurrent.*;

public class VirtualThreadComparison {
    public static void main(String[] args) throws InterruptedException {
        // 平台线程 - 创建 1000 个线程可能会有问题
        long start1 = System.currentTimeMillis();
        try (var executor = Executors.newFixedThreadPool(100)) {
            for (int i = 0; i < 1000; i++) {
                executor.submit(() -> {
                    try {
                        Thread.sleep(Duration.ofSeconds(1));
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                });
            }
        }
        long end1 = System.currentTimeMillis();
        System.out.println("平台线程耗时: " + (end1 - start1) + "ms");
        
        // 虚拟线程 - 可以轻松创建 10000+ 个线程
        long start2 = System.currentTimeMillis();
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            for (int i = 0; i < 10000; i++) {
                executor.submit(() -> {
                    try {
                        Thread.sleep(Duration.ofSeconds(1));
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                });
            }
        }
        long end2 = System.currentTimeMillis();
        System.out.println("虚拟线程耗时: " + (end2 - start2) + "ms");
    }
}
```

### 实际应用场景

```java
import java.net.http.*;
import java.net.URI;
import java.util.*;
import java.util.concurrent.*;

public class VirtualThreadWebCrawler {
    public static void main(String[] args) {
        List<String> urls = List.of(
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3"
            // ... 可以是成千上万个 URL
        );
        
        // 使用虚拟线程并发抓取
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            List<CompletableFuture<String>> futures = urls.stream()
                .map(url -> CompletableFuture.supplyAsync(() -> fetchUrl(url), executor))
                .toList();
            
            // 等待所有结果
            futures.forEach(future -> {
                try {
                    String content = future.get();
                    System.out.println("获取内容: " + content.length() + " 字节");
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
        }
    }
    
    private static String fetchUrl(String url) {
        try {
            HttpClient client = HttpClient.newHttpClient();
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .build();
            HttpResponse<String> response = client.send(request, 
                HttpResponse.BodyHandlers.ofString());
            return response.body();
        } catch (Exception e) {
            return "错误: " + e.getMessage();
        }
    }
}
```

## 序列化集合 (Sequenced Collections)

新增的接口,为有序集合提供统一的 API。

### SequencedCollection 接口

```java
import java.util.*;

public class SequencedCollectionExample {
    public static void main(String[] args) {
        // List 实现了 SequencedCollection
        List<String> list = new ArrayList<>(List.of("A", "B", "C", "D"));
        
        // 新方法: getFirst() 和 getLast()
        System.out.println("第一个元素: " + list.getFirst());  // A
        System.out.println("最后一个元素: " + list.getLast());   // D
        
        // 新方法: addFirst() 和 addLast()
        list.addFirst("Z");
        list.addLast("E");
        System.out.println(list);  // [Z, A, B, C, D, E]
        
        // 新方法: removeFirst() 和 removeLast()
        list.removeFirst();
        list.removeLast();
        System.out.println(list);  // [A, B, C, D]
        
        // 新方法: reversed() - 返回反向视图
        List<String> reversed = list.reversed();
        System.out.println(reversed);  // [D, C, B, A]
        
        // 修改反向视图会影响原列表
        reversed.addFirst("X");  // 相当于原列表的 addLast
        System.out.println(list);  // [A, B, C, D, X]
    }
}
```

### SequencedSet 接口

```java
import java.util.*;

public class SequencedSetExample {
    public static void main(String[] args) {
        // LinkedHashSet 实现了 SequencedSet
        SequencedSet<String> set = new LinkedHashSet<>(
            List.of("Apple", "Banana", "Cherry")
        );
        
        System.out.println("第一个: " + set.getFirst());   // Apple
        System.out.println("最后一个: " + set.getLast());   // Cherry
        
        // 添加元素
        set.addFirst("Apricot");  // 添加到开头
        set.addLast("Date");      // 添加到末尾
        System.out.println(set);  // [Apricot, Apple, Banana, Cherry, Date]
        
        // 反向视图
        SequencedSet<String> reversed = set.reversed();
        System.out.println(reversed);  // [Date, Cherry, Banana, Apple, Apricot]
    }
}
```

### SequencedMap 接口

```java
import java.util.*;

public class SequencedMapExample {
    public static void main(String[] args) {
        // LinkedHashMap 实现了 SequencedMap
        SequencedMap<String, Integer> map = new LinkedHashMap<>();
        map.put("Alice", 25);
        map.put("Bob", 30);
        map.put("Charlie", 35);
        
        // 获取第一个和最后一个条目
        Map.Entry<String, Integer> first = map.firstEntry();
        Map.Entry<String, Integer> last = map.lastEntry();
        System.out.println("第一个: " + first);  // Alice=25
        System.out.println("最后一个: " + last);  // Charlie=35
        
        // 添加到开头或末尾
        map.putFirst("Zoe", 20);
        map.putLast("David", 40);
        System.out.println(map);  // {Zoe=20, Alice=25, Bob=30, Charlie=35, David=40}
        
        // 获取反向视图
        SequencedMap<String, Integer> reversed = map.reversed();
        System.out.println(reversed);  // {David=40, Charlie=35, Bob=30, Alice=25, Zoe=20}
        
        // 序列化的键和值
        System.out.println("键(顺序): " + map.sequencedKeySet());
        System.out.println("值(顺序): " + map.sequencedValues());
    }
}
```

## Switch 模式匹配

Switch 现在支持模式匹配,包括类型模式、守卫条件等。

### 基本模式匹配

```java
public class SwitchPatternMatching {
    public static String formatValue(Object obj) {
        return switch (obj) {
            case null -> "空值";
            case String s -> "字符串: " + s;
            case Integer i -> "整数: " + i;
            case Double d -> "浮点数: " + d;
            case int[] arr -> "整数数组,长度: " + arr.length;
            default -> "其他类型: " + obj.getClass().getName();
        };
    }
    
    public static void main(String[] args) {
        System.out.println(formatValue("Hello"));      // 字符串: Hello
        System.out.println(formatValue(42));           // 整数: 42
        System.out.println(formatValue(3.14));         // 浮点数: 3.14
        System.out.println(formatValue(new int[5]));   // 整数数组,长度: 5
        System.out.println(formatValue(null));         // 空值
    }
}
```

### 守卫条件 (When 子句)

```java
public class SwitchGuardedPatterns {
    public static String classify(Object obj) {
        return switch (obj) {
            case String s when s.isEmpty() -> "空字符串";
            case String s when s.length() < 5 -> "短字符串: " + s;
            case String s -> "长字符串: " + s;
            
            case Integer i when i < 0 -> "负整数";
            case Integer i when i == 0 -> "零";
            case Integer i when i > 0 -> "正整数";
            
            case null -> "空值";
            default -> "其他类型";
        };
    }
    
    public static void main(String[] args) {
        System.out.println(classify(""));        // 空字符串
        System.out.println(classify("Hi"));      // 短字符串: Hi
        System.out.println(classify("Hello World"));  // 长字符串: Hello World
        System.out.println(classify(-5));        // 负整数
        System.out.println(classify(0));         // 零
        System.out.println(classify(10));        // 正整数
    }
}
```

### 与密封类结合

```java
sealed interface Shape permits Circle, Rectangle, Triangle {}
record Circle(double radius) implements Shape {}
record Rectangle(double width, double height) implements Shape {}
record Triangle(double base, double height) implements Shape {}

public class ShapePatternMatching {
    public static double calculateArea(Shape shape) {
        return switch (shape) {
            case Circle c -> Math.PI * c.radius() * c.radius();
            case Rectangle r -> r.width() * r.height();
            case Triangle t -> 0.5 * t.base() * t.height();
            // 不需要 default,编译器知道所有可能的类型
        };
    }
    
    public static String describe(Shape shape) {
        return switch (shape) {
            case Circle c when c.radius() > 10 -> "大圆";
            case Circle c -> "小圆";
            case Rectangle r when r.width() == r.height() -> "正方形";
            case Rectangle r -> "长方形";
            case Triangle t -> "三角形";
        };
    }
    
    public static void main(String[] args) {
        Shape circle = new Circle(5.0);
        Shape rect = new Rectangle(4.0, 6.0);
        
        System.out.println("圆形面积: " + calculateArea(circle));
        System.out.println("矩形面积: " + calculateArea(rect));
        System.out.println(describe(circle));  // 小圆
        System.out.println(describe(new Rectangle(5, 5)));  // 正方形
    }
}
```

## 记录模式 (Record Patterns)

记录模式允许解构记录类型,提取其组件。

### 基本用法

```java
record Point(int x, int y) {}
record Rectangle(Point topLeft, Point bottomRight) {}

public class RecordPatternExample {
    public static void printPoint(Object obj) {
        // 解构记录
        if (obj instanceof Point(int x, int y)) {
            System.out.println("坐标: (" + x + ", " + y + ")");
        }
    }
    
    public static void printRectangle(Object obj) {
        // 嵌套解构
        if (obj instanceof Rectangle(Point(int x1, int y1), Point(int x2, int y2))) {
            System.out.println("矩形: (" + x1 + "," + y1 + ") 到 (" + x2 + "," + y2 + ")");
            int width = x2 - x1;
            int height = y2 - y1;
            System.out.println("宽度: " + width + ", 高度: " + height);
        }
    }
    
    public static void main(String[] args) {
        Point p = new Point(3, 4);
        printPoint(p);  // 坐标: (3, 4)
        
        Rectangle rect = new Rectangle(new Point(0, 0), new Point(10, 5));
        printRectangle(rect);  
        // 矩形: (0,0) 到 (10,5)
        // 宽度: 10, 高度: 5
    }
}
```

### 在 Switch 中使用

```java
record Person(String name, int age) {}
record Employee(String name, int age, String department) {}

public class RecordPatternSwitch {
    public static String describe(Object obj) {
        return switch (obj) {
            case Person(String name, int age) when age < 18 -> 
                name + " 是未成年人";
            case Person(String name, int age) -> 
                name + " 是成年人,年龄 " + age;
            case Employee(String name, int age, String dept) -> 
                name + " 在 " + dept + " 部门工作";
            case null -> "空对象";
            default -> "未知类型";
        };
    }
    
    public static void main(String[] args) {
        System.out.println(describe(new Person("小明", 15)));  
        // 小明 是未成年人
        
        System.out.println(describe(new Person("张三", 25)));  
        // 张三 是成年人,年龄 25
        
        System.out.println(describe(new Employee("李四", 30, "技术")));  
        // 李四 在 技术 部门工作
    }
}
```

### 复杂解构

```java
record Order(String id, List<Item> items) {}
record Item(String name, int quantity, double price) {}

public class ComplexRecordPattern {
    public static void analyzeOrder(Object obj) {
        switch (obj) {
            case Order(String id, List<Item> items) when items.isEmpty() -> 
                System.out.println("订单 " + id + " 为空");
                
            case Order(String id, List<Item> items) when items.size() == 1 -> {
                Item item = items.get(0);
                System.out.println("订单 " + id + " 只有一个商品: " + item.name());
            }
            
            case Order(String id, List<Item> items) -> {
                double total = items.stream()
                    .mapToDouble(item -> item.quantity() * item.price())
                    .sum();
                System.out.println("订单 " + id + " 总金额: " + total);
            }
            
            default -> System.out.println("不是订单");
        }
    }
    
    public static void main(String[] args) {
        Order order1 = new Order("001", List.of());
        Order order2 = new Order("002", List.of(new Item("笔记本", 1, 5000)));
        Order order3 = new Order("003", List.of(
            new Item("鼠标", 2, 100),
            new Item("键盘", 1, 300)
        ));
        
        analyzeOrder(order1);  // 订单 001 为空
        analyzeOrder(order2);  // 订单 002 只有一个商品: 笔记本
        analyzeOrder(order3);  // 订单 003 总金额: 500.0
    }
}
```

## 字符串模板 (预览功能)

字符串模板提供了一种更安全、更方便的字符串插值方式。

> **注意**: 这是预览功能,需要使用 `--enable-preview` 标志。

### 基本用法

```java
public class StringTemplateExample {
    public static void main(String[] args) {
        String name = "张三";
        int age = 25;
        
        // 传统方式
        String msg1 = "姓名: " + name + ", 年龄: " + age;
        String msg2 = String.format("姓名: %s, 年龄: %d", name, age);
        
        // 字符串模板 (预览)
        // String msg3 = STR."姓名: \{name}, 年龄: \{age}";
        
        // System.out.println(msg3);
    }
}
```

## 未命名模式和变量

使用 `_` 表示未使用的变量或模式。

### 基本用法

```java
public class UnnamedPatternsExample {
    record Point(int x, int y, int z) {}
    
    public static void main(String[] args) {
        Point p = new Point(1, 2, 3);
        
        // 只关心 x 坐标
        if (p instanceof Point(int x, _, _)) {
            System.out.println("x 坐标: " + x);
        }
        
        // 在 switch 中使用
        String result = switch (p) {
            case Point(int x, _, _) when x > 0 -> "x 为正";
            case Point(int x, _, _) -> "x 为非正";
        };
        System.out.println(result);
    }
}
```

### 在 Lambda 中使用

```java
import java.util.*;

public class UnnamedVariablesLambda {
    public static void main(String[] args) {
        Map<String, Integer> map = Map.of("A", 1, "B", 2, "C", 3);
        
        // 只关心键,不关心值
        map.forEach((key, _) -> System.out.println("键: " + key));
        
        // 只关心值,不关心键
        map.forEach((_, value) -> System.out.println("值: " + value));
    }
}
```

## 作用域值 (Scoped Values - 预览)

作用域值是 ThreadLocal 的替代方案,提供更好的性能和安全性。

> **注意**: 这是预览功能,需要使用 `--enable-preview` 标志。

### 基本概念

```java
// import java.lang.ScopedValue;

public class ScopedValueExample {
    // 定义作用域值
    // private static final ScopedValue<String> USER = ScopedValue.newInstance();
    
    public static void main(String[] args) {
        // 在作用域内设置值
        // ScopedValue.where(USER, "张三").run(() -> {
        //     processRequest();
        // });
    }
    
    private static void processRequest() {
        // 获取作用域值
        // String user = USER.get();
        // System.out.println("当前用户: " + user);
    }
}
```

## 性能改进

### Generational ZGC

JDK 21 引入了分代 ZGC,进一步减少 GC 停顿时间。

```bash
# 启用分代 ZGC
java -XX:+UseZGC -XX:+ZGenerational MyApp
```

### 其他性能优化

- 改进的向量 API
- 更快的字符串操作
- 优化的集合框架性能

## 迁移建议

### 从 JDK 17 迁移到 JDK 21

```java
public class MigrationGuide {
    public static void main(String[] args) {
        // 1. 使用虚拟线程替代线程池(I/O 密集型场景)
        // 旧方式
        // ExecutorService executor = Executors.newFixedThreadPool(100);
        
        // 新方式
        // ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
        
        // 2. 使用序列化集合的新方法
        List<String> list = new ArrayList<>(List.of("A", "B", "C"));
        String first = list.getFirst();  // 替代 list.get(0)
        String last = list.getLast();    // 替代 list.get(list.size() - 1)
        
        // 3. 使用 switch 模式匹配简化代码
        Object obj = "Hello";
        String result = switch (obj) {
            case String s when s.length() > 5 -> "长字符串";
            case String s -> "短字符串";
            case Integer i -> "整数";
            default -> "其他";
        };
        
        // 4. 使用记录模式解构数据
        record Person(String name, int age) {}
        Person p = new Person("张三", 25);
        if (p instanceof Person(String name, int age)) {
            System.out.println(name + " 的年龄是 " + age);
        }
    }
}
```

## 最佳实践

### 1. 合理使用虚拟线程

```java
// 好的做法: I/O 密集型任务使用虚拟线程
try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
    for (String url : urls) {
        executor.submit(() -> fetchUrl(url));
    }
}

// 避免: CPU 密集型任务不适合虚拟线程
// 对于 CPU 密集型任务,仍然使用平台线程池
```

### 2. 使用序列化集合 API

```java
// 好的做法
List<String> list = new ArrayList<>();
String first = list.getFirst();  // 更清晰
String last = list.getLast();

// 避免
String firstOld = list.get(0);
String lastOld = list.get(list.size() - 1);
```

### 3. 利用模式匹配简化代码

```java
// 好的做法: 使用模式匹配
public static String process(Object obj) {
    return switch (obj) {
        case String s when s.isEmpty() -> "空";
        case String s -> s.toUpperCase();
        case Integer i -> String.valueOf(i * 2);
        case null -> "null";
        default -> "unknown";
    };
}

// 避免: 传统的 instanceof 链
```

## 总结

JDK 21 带来了许多革命性的特性:

- ✅ **虚拟线程** - 简化高并发编程
- ✅ **序列化集合** - 统一的有序集合 API
- ✅ **Switch 模式匹配** - 更强大的 switch 语句
- ✅ **记录模式** - 解构记录类型
- ✅ **字符串模板** (预览) - 安全的字符串插值
- ✅ **未命名模式** - 简化代码
- ✅ **作用域值** (预览) - ThreadLocal 的替代
- ✅ **性能改进** - 分代 ZGC、向量 API 等

这些特性进一步提升了 Java 的表达能力、性能和开发效率。建议在新项目中积极采用这些特性,特别是虚拟线程和模式匹配。

继续学习其他 Java 核心主题:

- [多线程编程](/docs/java/multithreading)
- [JVM 基础](/docs/java/jvm-basics)
- [性能优化](/docs/java/performance)
