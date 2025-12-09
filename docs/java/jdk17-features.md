---
sidebar_position: 15
title: JDK 17 新特性
---

# JDK 17 新特性

JDK 17 是一个长期支持版本(LTS),发布于 2021 年 9 月。本文介绍 JDK 17 的主要新特性和改进。

## 密封类 (Sealed Classes)

密封类允许你控制哪些类可以继承或实现它,提供了比 `final` 更灵活的继承控制。

### 基本语法

```java
// 定义密封类
public sealed class Shape permits Circle, Rectangle, Triangle {
    // 公共属性和方法
}

// 允许的子类必须是 final、sealed 或 non-sealed
public final class Circle extends Shape {
    private double radius;
    
    public Circle(double radius) {
        this.radius = radius;
    }
    
    public double area() {
        return Math.PI * radius * radius;
    }
}

public final class Rectangle extends Shape {
    private double width;
    private double height;
    
    public Rectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }
    
    public double area() {
        return width * height;
    }
}

public non-sealed class Triangle extends Shape {
    // non-sealed 允许其他类继续继承
    private double base;
    private double height;
    
    public Triangle(double base, double height) {
        this.base = base;
        this.height = height;
    }
    
    public double area() {
        return 0.5 * base * height;
    }
}
```

### 密封接口

```java
public sealed interface Payment permits CreditCardPayment, PayPalPayment, CashPayment {
    void processPayment(double amount);
}

public final class CreditCardPayment implements Payment {
    private String cardNumber;
    
    @Override
    public void processPayment(double amount) {
        System.out.println("处理信用卡支付: " + amount);
    }
}

public final class PayPalPayment implements Payment {
    private String email;
    
    @Override
    public void processPayment(double amount) {
        System.out.println("处理 PayPal 支付: " + amount);
    }
}

public final class CashPayment implements Payment {
    @Override
    public void processPayment(double amount) {
        System.out.println("处理现金支付: " + amount);
    }
}
```

### 使用场景

```java
public class PaymentProcessor {
    public static void process(Payment payment, double amount) {
        // 编译器知道所有可能的实现类
        switch (payment) {
            case CreditCardPayment cc -> {
                System.out.println("信用卡支付");
                cc.processPayment(amount);
            }
            case PayPalPayment pp -> {
                System.out.println("PayPal 支付");
                pp.processPayment(amount);
            }
            case CashPayment cash -> {
                System.out.println("现金支付");
                cash.processPayment(amount);
            }
            // 不需要 default,编译器知道已覆盖所有情况
        }
    }
}
```

## 记录类型 (Records)

记录类型是一种特殊的类,用于不可变数据的简洁表示。

### 基本用法

```java
// 传统方式
public class PersonOld {
    private final String name;
    private final int age;
    
    public PersonOld(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    public String getName() { return name; }
    public int getAge() { return age; }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        PersonOld person = (PersonOld) o;
        return age == person.age && Objects.equals(name, person.name);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(name, age);
    }
    
    @Override
    public String toString() {
        return "Person[name=" + name + ", age=" + age + "]";
    }
}

// 使用 Record
public record Person(String name, int age) {
    // 自动生成构造器、getter、equals、hashCode、toString
}

// 使用
public class RecordExample {
    public static void main(String[] args) {
        Person person = new Person("张三", 25);
        
        System.out.println(person.name());      // 张三
        System.out.println(person.age());       // 25
        System.out.println(person);             // Person[name=张三, age=25]
        
        Person person2 = new Person("张三", 25);
        System.out.println(person.equals(person2));  // true
    }
}
```

### 自定义构造器

```java
public record Point(int x, int y) {
    // 紧凑构造器 - 用于验证
    public Point {
        if (x < 0 || y < 0) {
            throw new IllegalArgumentException("坐标不能为负");
        }
    }
    
    // 额外的构造器
    public Point() {
        this(0, 0);
    }
}

// 使用
public class PointExample {
    public static void main(String[] args) {
        Point p1 = new Point(3, 4);
        Point p2 = new Point();  // (0, 0)
        
        try {
            Point p3 = new Point(-1, 5);  // 抛出异常
        } catch (IllegalArgumentException e) {
            System.out.println(e.getMessage());
        }
    }
}
```

### Record 的方法

```java
public record Employee(String name, String department, double salary) {
    // 可以添加静态方法
    public static Employee createIntern(String name, String department) {
        return new Employee(name, department, 3000.0);
    }
    
    // 可以添加实例方法
    public double annualSalary() {
        return salary * 12;
    }
    
    // 可以重写访问器方法
    @Override
    public String name() {
        return name.toUpperCase();
    }
}

// 使用
public class EmployeeExample {
    public static void main(String[] args) {
        Employee emp = new Employee("李四", "技术部", 15000);
        System.out.println(emp.name());           // 李四
        System.out.println(emp.annualSalary());   // 180000.0
        
        Employee intern = Employee.createIntern("王五", "市场部");
        System.out.println(intern);  // Employee[name=王五, department=市场部, salary=3000.0]
    }
}
```

## instanceof 模式匹配

简化了类型检查和转换的代码。

### 传统方式 vs 模式匹配

```java
public class PatternMatchingExample {
    // 传统方式
    public static void processOld(Object obj) {
        if (obj instanceof String) {
            String str = (String) obj;  // 需要显式转换
            System.out.println("字符串长度: " + str.length());
        } else if (obj instanceof Integer) {
            Integer num = (Integer) obj;
            System.out.println("数值: " + num);
        }
    }
    
    // 模式匹配
    public static void processNew(Object obj) {
        if (obj instanceof String str) {  // 自动转换并声明变量
            System.out.println("字符串长度: " + str.length());
        } else if (obj instanceof Integer num) {
            System.out.println("数值: " + num);
        }
    }
    
    // 在表达式中使用
    public static void processWithLogic(Object obj) {
        if (obj instanceof String str && str.length() > 5) {
            System.out.println("长字符串: " + str);
        }
        
        if (obj instanceof Integer num && num > 0) {
            System.out.println("正整数: " + num);
        }
    }
}
```

### 实际应用

```java
public class ShapeCalculator {
    public static double calculateArea(Object shape) {
        if (shape instanceof Circle c) {
            return Math.PI * c.radius() * c.radius();
        } else if (shape instanceof Rectangle r) {
            return r.width() * r.height();
        } else if (shape instanceof Triangle t) {
            return 0.5 * t.base() * t.height();
        }
        throw new IllegalArgumentException("未知图形类型");
    }
    
    public static void main(String[] args) {
        record Circle(double radius) {}
        record Rectangle(double width, double height) {}
        record Triangle(double base, double height) {}
        
        Object circle = new Circle(5.0);
        System.out.println("圆形面积: " + calculateArea(circle));
    }
}
```

## Switch 表达式

Switch 现在可以作为表达式使用,返回值。

### 基本用法

```java
public class SwitchExpressionExample {
    public static void main(String[] args) {
        // 传统 switch 语句
        String dayOld;
        int day = 3;
        switch (day) {
            case 1:
                dayOld = "星期一";
                break;
            case 2:
                dayOld = "星期二";
                break;
            case 3:
                dayOld = "星期三";
                break;
            default:
                dayOld = "其他";
        }
        
        // Switch 表达式
        String dayNew = switch (day) {
            case 1 -> "星期一";
            case 2 -> "星期二";
            case 3 -> "星期三";
            default -> "其他";
        };
        
        System.out.println(dayNew);  // 星期三
    }
}
```

### 多个 case 标签

```java
public class MultiCaseExample {
    public static String getSeasonChinese(int month) {
        return switch (month) {
            case 3, 4, 5 -> "春季";
            case 6, 7, 8 -> "夏季";
            case 9, 10, 11 -> "秋季";
            case 12, 1, 2 -> "冬季";
            default -> throw new IllegalArgumentException("无效月份: " + month);
        };
    }
    
    public static void main(String[] args) {
        System.out.println(getSeasonChinese(4));   // 春季
        System.out.println(getSeasonChinese(10));  // 秋季
    }
}
```

### yield 关键字

```java
public class YieldExample {
    public static int calculate(String operation, int a, int b) {
        return switch (operation) {
            case "add" -> a + b;
            case "subtract" -> a - b;
            case "multiply" -> a * b;
            case "divide" -> {
                if (b == 0) {
                    throw new ArithmeticException("除数不能为 0");
                }
                yield a / b;  // 使用 yield 返回值
            }
            default -> throw new IllegalArgumentException("未知操作: " + operation);
        };
    }
    
    public static void main(String[] args) {
        System.out.println(calculate("add", 10, 5));       // 15
        System.out.println(calculate("multiply", 10, 5));  // 50
        System.out.println(calculate("divide", 10, 5));    // 2
    }
}
```

## 文本块 (Text Blocks)

多行字符串的简洁表示方法。

### 基本用法

```java
public class TextBlockExample {
    public static void main(String[] args) {
        // 传统方式
        String htmlOld = "<html>\n" +
                        "  <body>\n" +
                        "    <p>Hello, World!</p>\n" +
                        "  </body>\n" +
                        "</html>";
        
        // 文本块
        String htmlNew = """
                <html>
                  <body>
                    <p>Hello, World!</p>
                  </body>
                </html>
                """;
        
        System.out.println(htmlNew);
    }
}
```

### JSON 和 SQL

```java
public class TextBlockUseCases {
    public static void main(String[] args) {
        // JSON
        String json = """
                {
                  "name": "张三",
                  "age": 25,
                  "city": "北京"
                }
                """;
        
        // SQL
        String sql = """
                SELECT id, name, email
                FROM users
                WHERE age > 18
                  AND city = '北京'
                ORDER BY name
                """;
        
        // 多语言代码
        String python = """
                def hello(name):
                    print(f"Hello, {name}!")
                    
                hello("World")
                """;
        
        System.out.println(json);
        System.out.println(sql);
        System.out.println(python);
    }
}
```

### 文本块转义

```java
public class TextBlockEscape {
    public static void main(String[] args) {
        // 行尾使用 \ 可以连接下一行
        String text1 = """
                这是一行很长的文本,\
                但实际上是连续的一行
                """;
        System.out.println(text1);  // 没有换行
        
        // 使用 \s 保留空格
        String text2 = """
                Line 1    \s
                Line 2    \s
                """;
        
        // 使用 """ 在文本中
        String quote = """
                他说: "这是引号"
                """;
    }
}
```

## 新增 API

### Stream.toList()

```java
import java.util.*;
import java.util.stream.*;

public class StreamToListExample {
    public static void main(String[] args) {
        List<String> names = List.of("Alice", "Bob", "Charlie", "David");
        
        // JDK 17 之前
        List<String> filteredOld = names.stream()
                .filter(name -> name.length() > 4)
                .collect(Collectors.toList());
        
        // JDK 17
        List<String> filteredNew = names.stream()
                .filter(name -> name.length() > 4)
                .toList();  // 更简洁,返回不可变列表
        
        System.out.println(filteredNew);  // [Alice, Charlie, David]
    }
}
```

### 随机数生成器

```java
import java.util.random.*;

public class RandomGeneratorExample {
    public static void main(String[] args) {
        // 使用新的 RandomGenerator 接口
        RandomGenerator random = RandomGenerator.of("L64X128MixRandom");
        
        // 生成随机数
        int randomInt = random.nextInt(100);
        double randomDouble = random.nextDouble();
        boolean randomBoolean = random.nextBoolean();
        
        System.out.println("随机整数: " + randomInt);
        System.out.println("随机浮点数: " + randomDouble);
        System.out.println("随机布尔值: " + randomBoolean);
        
        // 生成随机流
        random.ints(5, 0, 100)
                .forEach(n -> System.out.println("随机数: " + n));
        
        // 列出所有可用的随机数生成器算法
        RandomGeneratorFactory.all()
                .map(RandomGeneratorFactory::name)
                .forEach(System.out::println);
    }
}
```

## 性能改进

### G1 垃圾回收器改进

JDK 17 对 G1 垃圾回收器进行了多项优化:

- 改进的并发标记算法
- 更好的内存管理
- 减少停顿时间

```bash
# 使用 G1 垃圾回收器
java -XX:+UseG1GC -XX:MaxGCPauseMillis=200 MyApp
```

### ZGC 改进

```bash
# 使用 ZGC(低延迟垃圾回收器)
java -XX:+UseZGC MyApp
```

## 弃用和移除

### 移除的功能

- **Applet API** - 已弃用并标记为移除
- **RMI Activation** - 已移除
- **Security Manager** - 标记为弃用

### 强封装 JDK 内部 API

```java
// JDK 17 默认强封装内部 API
// 以下代码将失败(除非使用 --add-opens)
// sun.misc.Unsafe unsafe = sun.misc.Unsafe.getUnsafe();

// 解决方案:使用公共 API 或添加启动参数
// java --add-opens java.base/sun.misc=ALL-UNNAMED MyApp
```

## 迁移建议

### 从 JDK 11 迁移

```java
public class MigrationChecklist {
    /**
     * 1. 检查使用的第三方库是否支持 JDK 17
     * 2. 移除对已弃用 API 的使用
     * 3. 更新构建工具(Maven/Gradle)
     * 4. 运行测试套件
     * 5. 性能测试和调优
     */
    
    public static void main(String[] args) {
        // 使用新特性改进代码
        
        // 1. 使用 Records 替代数据类
        record User(String name, int age) {}
        
        // 2. 使用 Switch 表达式
        String dayType = switch (LocalDate.now().getDayOfWeek()) {
            case MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY -> "工作日";
            case SATURDAY, SUNDAY -> "周末";
        };
        
        // 3. 使用文本块
        String config = """
                {
                  "database": "mysql",
                  "host": "localhost"
                }
                """;
    }
}
```

## 最佳实践

### 1. 使用密封类控制继承

```java
// 好的做法:使用密封类限制层次结构
public sealed interface Result<T> permits Success, Error {
    // 接口定义
}

public record Success<T>(T value) implements Result<T> {}
public record Error<T>(String message) implements Result<T> {}

// 使用时编译器确保覆盖所有情况
public static <T> void handleResult(Result<T> result) {
    switch (result) {
        case Success<T> s -> System.out.println("成功: " + s.value());
        case Error<T> e -> System.out.println("错误: " + e.message());
    }
}
```

### 2. 优先使用 Record 表示数据

```java
// 好的做法:使用 Record
public record OrderItem(String productId, int quantity, double price) {
    public double totalPrice() {
        return quantity * price;
    }
}

// 而不是传统类(除非需要可变性)
```

### 3. 使用 instanceof 模式匹配

```java
// 好的做法
if (obj instanceof String str && !str.isEmpty()) {
    process(str);
}

// 避免
if (obj instanceof String) {
    String str = (String) obj;
    if (!str.isEmpty()) {
        process(str);
    }
}
```

## 总结

JDK 17 是一个重要的 LTS 版本,带来了许多实用的新特性:

- ✅ **密封类** - 更好的继承控制
- ✅ **记录类型** - 简洁的数据类
- ✅ **模式匹配** - 简化类型检查
- ✅ **Switch 表达式** - 更强大的 switch
- ✅ **文本块** - 多行字符串支持
- ✅ **新增 API** - Stream.toList()、随机数生成器等
- ✅ **性能改进** - G1/ZGC 优化

这些特性可以显著提高代码的可读性、简洁性和类型安全性。建议在新项目中积极采用这些特性。

下一步可以学习 [JDK 21 新特性](./jdk21-features),了解最新的 Java 功能。
