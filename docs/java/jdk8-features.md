---
sidebar_position: 16
title: JDK 8 新特性
---

# JDK 8 新特性

JDK 8 是 Java 历史上最重要的版本之一，引入了 Lambda 表达式、Stream API、Optional 等革命性特性。

## Lambda 表达式

Lambda 表达式是一种简洁的函数式编程语法，可以替代匿名内部类。

### 基本语法

```java
// 语法：(parameters) -> expression 或 (parameters) -> { statements; }

// 无参数
() -> System.out.println("Hello")

// 一个参数（可省略括号）
x -> x * x

// 多个参数
(x, y) -> x + y

// 多行代码块
(x, y) -> {
    int sum = x + y;
    return sum;
}
```

### 实际应用

```java
import java.util.*;

public class LambdaExamples {
    public static void main(String[] args) {
        List<String> names = Arrays.asList("张三", "李四", "王五");

        // 传统方式
        Collections.sort(names, new Comparator<String>() {
            @Override
            public int compare(String s1, String s2) {
                return s1.compareTo(s2);
            }
        });

        // Lambda 方式
        Collections.sort(names, (s1, s2) -> s1.compareTo(s2));

        // 更简洁的方法引用
        Collections.sort(names, String::compareTo);

        // forEach 遍历
        names.forEach(name -> System.out.println(name));
        names.forEach(System.out::println);  // 方法引用

        // Runnable
        new Thread(() -> System.out.println("线程执行")).start();

        // 事件处理
        // button.addActionListener(e -> System.out.println("按钮被点击"));
    }
}
```

## 函数式接口

函数式接口是只有一个抽象方法的接口，可以使用 `@FunctionalInterface` 注解标注。

### 常用函数式接口

```java
import java.util.function.*;

public class FunctionalInterfaceExamples {
    public static void main(String[] args) {
        // Predicate<T>：测试条件，返回 boolean
        Predicate<String> isEmpty = String::isEmpty;
        System.out.println(isEmpty.test(""));  // true

        // Function<T, R>：转换函数，接收 T 返回 R
        Function<String, Integer> strLength = String::length;
        System.out.println(strLength.apply("Hello"));  // 5

        // Consumer<T>：消费函数，接收 T 无返回值
        Consumer<String> printer = System.out::println;
        printer.accept("Hello");

        // Supplier<T>：供应函数，无参数返回 T
        Supplier<Double> random = Math::random;
        System.out.println(random.get());

        // BiFunction<T, U, R>：双参数函数
        BiFunction<Integer, Integer, Integer> add = (a, b) -> a + b;
        System.out.println(add.apply(3, 5));  // 8

        // UnaryOperator<T>：一元操作，T -> T
        UnaryOperator<Integer> square = x -> x * x;
        System.out.println(square.apply(5));  // 25

        // BinaryOperator<T>：二元操作，(T, T) -> T
        BinaryOperator<Integer> multiply = (a, b) -> a * b;
        System.out.println(multiply.apply(3, 4));  // 12
    }
}
```

### 自定义函数式接口

```java
@FunctionalInterface
public interface Calculator {
    int calculate(int a, int b);

    // 可以有默认方法
    default int add(int a, int b) {
        return a + b;
    }

    // 可以有静态方法
    static int subtract(int a, int b) {
        return a - b;
    }
}

// 使用
Calculator multiply = (a, b) -> a * b;
System.out.println(multiply.calculate(3, 4));  // 12
System.out.println(multiply.add(3, 4));        // 7
```

## 方法引用

方法引用是 Lambda 表达式的简化形式。

### 四种方法引用

```java
import java.util.*;
import java.util.function.*;

public class MethodReferenceExamples {
    public static void main(String[] args) {
        // 1. 静态方法引用：ClassName::staticMethod
        Function<String, Integer> parseInt = Integer::parseInt;
        System.out.println(parseInt.apply("123"));  // 123

        // 2. 实例方法引用：instance::instanceMethod
        String str = "Hello";
        Supplier<String> upperCase = str::toUpperCase;
        System.out.println(upperCase.get());  // HELLO

        // 3. 类的实例方法引用：ClassName::instanceMethod
        Function<String, Integer> length = String::length;
        System.out.println(length.apply("Hello"));  // 5

        // 4. 构造器引用：ClassName::new
        Supplier<List<String>> listSupplier = ArrayList::new;
        List<String> list = listSupplier.get();

        Function<String, StringBuilder> sbCreator = StringBuilder::new;
        StringBuilder sb = sbCreator.apply("Hello");
    }
}
```

## Stream API

Stream API 提供了声明式的数据处理方式。

### 创建 Stream

```java
import java.util.*;
import java.util.stream.*;

public class StreamCreation {
    public static void main(String[] args) {
        // 从集合创建
        List<String> list = Arrays.asList("a", "b", "c");
        Stream<String> stream1 = list.stream();

        // 从数组创建
        String[] array = {"a", "b", "c"};
        Stream<String> stream2 = Arrays.stream(array);

        // 使用 Stream.of()
        Stream<String> stream3 = Stream.of("a", "b", "c");

        // 无限流
        Stream<Integer> infiniteStream = Stream.iterate(0, n -> n + 1);
        Stream<Double> randomStream = Stream.generate(Math::random);

        // 空流
        Stream<String> emptyStream = Stream.empty();

        // IntStream, LongStream, DoubleStream
        IntStream intStream = IntStream.range(1, 10);  // 1-9
        IntStream intStream2 = IntStream.rangeClosed(1, 10);  // 1-10
    }
}
```

### 中间操作

```java
import java.util.*;
import java.util.stream.*;

public class StreamIntermediateOps {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

        // filter：过滤
        List<Integer> evens = numbers.stream()
            .filter(n -> n % 2 == 0)
            .collect(Collectors.toList());
        System.out.println(evens);  // [2, 4, 6, 8, 10]

        // map：映射
        List<Integer> squares = numbers.stream()
            .map(n -> n * n)
            .collect(Collectors.toList());
        System.out.println(squares);  // [1, 4, 9, 16, ...]

        // flatMap：扁平化
        List<List<Integer>> nested = Arrays.asList(
            Arrays.asList(1, 2),
            Arrays.asList(3, 4)
        );
        List<Integer> flattened = nested.stream()
            .flatMap(List::stream)
            .collect(Collectors.toList());
        System.out.println(flattened);  // [1, 2, 3, 4]

        // distinct：去重
        List<Integer> unique = Arrays.asList(1, 2, 2, 3, 3, 3)
            .stream()
            .distinct()
            .collect(Collectors.toList());
        System.out.println(unique);  // [1, 2, 3]

        // sorted：排序
        List<Integer> sorted = numbers.stream()
            .sorted(Comparator.reverseOrder())
            .collect(Collectors.toList());

        // limit：限制数量
        List<Integer> limited = numbers.stream()
            .limit(5)
            .collect(Collectors.toList());

        // skip：跳过
        List<Integer> skipped = numbers.stream()
            .skip(5)
            .collect(Collectors.toList());

        // peek：查看元素（用于调试）
        numbers.stream()
            .peek(n -> System.out.println("处理: " + n))
            .forEach(System.out::println);
    }
}
```

### 终端操作

```java
import java.util.*;
import java.util.stream.*;

public class StreamTerminalOps {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);

        // forEach：遍历
        numbers.stream().forEach(System.out::println);

        // count：计数
        long count = numbers.stream().count();
        System.out.println("元素个数: " + count);

        // collect：收集
        List<Integer> list = numbers.stream().collect(Collectors.toList());
        Set<Integer> set = numbers.stream().collect(Collectors.toSet());

        // reduce：归约
        Optional<Integer> sum = numbers.stream()
            .reduce((a, b) -> a + b);
        System.out.println("总和: " + sum.orElse(0));

        int sum2 = numbers.stream()
            .reduce(0, Integer::sum);
        System.out.println("总和: " + sum2);

        // min/max：最小/最大值
        Optional<Integer> min = numbers.stream().min(Integer::compareTo);
        Optional<Integer> max = numbers.stream().max(Integer::compareTo);

        // anyMatch/allMatch/noneMatch：匹配
        boolean hasEven = numbers.stream().anyMatch(n -> n % 2 == 0);
        boolean allPositive = numbers.stream().allMatch(n -> n > 0);
        boolean noneNegative = numbers.stream().noneMatch(n -> n < 0);

        // findFirst/findAny：查找
        Optional<Integer> first = numbers.stream().findFirst();
        Optional<Integer> any = numbers.stream().findAny();

        // toArray：转数组
        Integer[] array = numbers.stream().toArray(Integer[]::new);
    }
}
```

### 收集器（Collectors）

```java
import java.util.*;
import java.util.stream.*;

public class CollectorsExamples {
    public static void main(String[] args) {
        List<String> names = Arrays.asList("张三", "李四", "王五", "赵六");

        // toList/toSet/toCollection
        List<String> list = names.stream().collect(Collectors.toList());
        Set<String> set = names.stream().collect(Collectors.toSet());
        LinkedList<String> linkedList = names.stream()
            .collect(Collectors.toCollection(LinkedList::new));

        // joining：字符串连接
        String joined = names.stream()
            .collect(Collectors.joining(", "));
        System.out.println(joined);  // 张三, 李四, 王五, 赵六

        String joinedWithPrefix = names.stream()
            .collect(Collectors.joining(", ", "[", "]"));
        System.out.println(joinedWithPrefix);  // [张三, 李四, 王五, 赵六]

        // counting：计数
        long count = names.stream().collect(Collectors.counting());

        // summingInt/averagingInt：求和/平均值
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        int sum = numbers.stream().collect(Collectors.summingInt(Integer::intValue));
        double avg = numbers.stream().collect(Collectors.averagingInt(Integer::intValue));

        // maxBy/minBy：最大/最小值
        Optional<String> longest = names.stream()
            .collect(Collectors.maxBy(Comparator.comparing(String::length)));

        // groupingBy：分组
        Map<Integer, List<String>> byLength = names.stream()
            .collect(Collectors.groupingBy(String::length));
        System.out.println(byLength);

        // partitioningBy：分区（返回布尔值的分组）
        Map<Boolean, List<Integer>> partitioned = numbers.stream()
            .collect(Collectors.partitioningBy(n -> n % 2 == 0));
        System.out.println("偶数: " + partitioned.get(true));
        System.out.println("奇数: " + partitioned.get(false));
    }
}
```

## Optional 类

Optional 是一个容器对象，用于避免 null 引用。

### 创建 Optional

```java
import java.util.Optional;

public class OptionalCreation {
    public static void main(String[] args) {
        // 创建空 Optional
        Optional<String> empty = Optional.empty();

        // 创建非空 Optional（值不能为 null）
        Optional<String> nonEmpty = Optional.of("Hello");

        // 创建可能为空的 Optional
        Optional<String> nullable = Optional.ofNullable(null);
        Optional<String> nullable2 = Optional.ofNullable("World");
    }
}
```

### Optional 常用方法

```java
import java.util.Optional;

public class OptionalMethods {
    public static void main(String[] args) {
        Optional<String> optional = Optional.of("Hello");

        // isPresent/isEmpty：检查是否有值
        if (optional.isPresent()) {
            System.out.println("有值");
        }

        // get：获取值（不推荐，可能抛异常）
        String value = optional.get();

        // orElse：有值返回值，无值返回默认值
        String result1 = optional.orElse("Default");

        // orElseGet：有值返回值，无值调用 Supplier
        String result2 = optional.orElseGet(() -> "Generated Default");

        // orElseThrow：有值返回值，无值抛异常
        String result3 = optional.orElseThrow(() ->
            new RuntimeException("值不存在"));

        // ifPresent：有值时执行操作
        optional.ifPresent(s -> System.out.println("值: " + s));

        // ifPresentOrElse：有值执行操作1，无值执行操作2（JDK 9+）
        // optional.ifPresentOrElse(
        //     s -> System.out.println("有值: " + s),
        //     () -> System.out.println("无值")
        // );

        // map：转换值
        Optional<Integer> length = optional.map(String::length);
        System.out.println(length.orElse(0));  // 5

        // flatMap：转换值（返回 Optional）
        Optional<String> upper = optional.flatMap(s ->
            Optional.of(s.toUpperCase()));

        // filter：过滤值
        Optional<String> filtered = optional.filter(s -> s.length() > 3);
    }
}
```

### 实际应用

```java
import java.util.*;

public class OptionalPractical {
    // 避免 null 检查
    public static String getUserEmail(User user) {
        // 传统方式
        if (user != null && user.getEmail() != null) {
            return user.getEmail();
        }
        return "未知";

        // Optional 方式
        return Optional.ofNullable(user)
            .map(User::getEmail)
            .orElse("未知");
    }

    // 链式调用
    public static String getUserCity(User user) {
        return Optional.ofNullable(user)
            .map(User::getAddress)
            .map(Address::getCity)
            .orElse("未知城市");
    }

    static class User {
        private String email;
        private Address address;

        public String getEmail() { return email; }
        public Address getAddress() { return address; }
    }

    static class Address {
        private String city;
        public String getCity() { return city; }
    }
}
```

## 接口默认方法和静态方法

JDK 8 允许在接口中定义默认方法和静态方法。

```java
public interface Vehicle {
    // 抽象方法
    void start();

    // 默认方法
    default void stop() {
        System.out.println("车辆停止");
    }

    default void honk() {
        System.out.println("嘟嘟~");
    }

    // 静态方法
    static void checkLicense() {
        System.out.println("检查驾驶证");
    }
}

class Car implements Vehicle {
    @Override
    public void start() {
        System.out.println("汽车启动");
    }

    // 可以重写默认方法
    @Override
    public void stop() {
        System.out.println("汽车刹车停止");
    }
}

// 使用
Car car = new Car();
car.start();  // 汽车启动
car.stop();   // 汽车刹车停止
car.honk();   // 嘟嘟~（使用默认实现）
Vehicle.checkLicense();  // 检查驾驶证（调用静态方法）
```

## 新的日期时间 API

详见 [日期时间 API](./date-time) 文档。

## 其他新特性

### Base64 编码

```java
import java.util.Base64;

public class Base64Example {
    public static void main(String[] args) {
        String original = "Hello World";

        // 编码
        String encoded = Base64.getEncoder()
            .encodeToString(original.getBytes());
        System.out.println("编码: " + encoded);

        // 解码
        byte[] decoded = Base64.getDecoder().decode(encoded);
        String decodedStr = new String(decoded);
        System.out.println("解码: " + decodedStr);

        // URL 安全的编码
        String urlEncoded = Base64.getUrlEncoder()
            .encodeToString(original.getBytes());
    }
}
```

### 并行数组操作

```java
import java.util.Arrays;

public class ParallelArrays {
    public static void main(String[] args) {
        int[] array = new int[10000];

        // 并行填充
        Arrays.parallelSetAll(array, i -> i * 2);

        // 并行前缀操作
        Arrays.parallelPrefix(array, (a, b) -> a + b);

        // 并行排序
        Arrays.parallelSort(array);
    }
}
```

## 总结

JDK 8 的主要新特性：

- **Lambda 表达式**：简化代码，支持函数式编程
- **函数式接口**：配合 Lambda 使用的接口
- **方法引用**：Lambda 的简化形式
- **Stream API**：声明式数据处理
- **Optional**：优雅处理 null
- **接口默认方法**：扩展接口而不破坏兼容性
- **新日期时间 API**：替代旧的 Date 和 Calendar
- **Base64**：内置 Base64 编解码
- **并行数组**：提高数组操作性能

这些特性让 Java 代码更加简洁、优雅和高效。
