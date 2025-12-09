---
sidebar_position: 9
title: 函数式编程
---

# 函数式编程

Java 8 引入了 Lambda 表达式和 Stream API，支持函数式编程。本文介绍 Lambda、函数式接口、方法引用和 Stream 的使用。

## Lambda 表达式

### 什么是 Lambda

Lambda 表达式是一个匿名函数，可以作为参数传递。

**语法：**

```
(parameters) -> expression
或
(parameters) -> { statements; }
```

### 基本示例

```java
public class LambdaBasic {
    public static void main(String[] args) {
        // 传统方式：匿名内部类
        Runnable r1 = new Runnable() {
            @Override
            public void run() {
                System.out.println("传统方式");
            }
        };
        
        // Lambda 方式
        Runnable r2 = () -> System.out.println("Lambda 方式");
        
        r1.run();
        r2.run();
        
        // 有参数的 Lambda
        Comparator<String> cmp1 = (s1, s2) -> s1.compareTo(s2);
        
        // 多行 Lambda
        Comparator<String> cmp2 = (s1, s2) -> {
            System.out.println("比较 " + s1 + " 和 " + s2);
            return s1.compareTo(s2);
        };
    }
}
```

### Lambda 语法变体

```java
public class LambdaSyntax {
    public static void main(String[] args) {
        // 1. 无参数
        Runnable r = () -> System.out.println("无参数");
        
        // 2. 一个参数（可以省略括号）
        Consumer<String> c1 = (s) -> System.out.println(s);
        Consumer<String> c2 = s -> System.out.println(s);
        
        // 3. 多个参数
        BiFunction<Integer, Integer, Integer> add = (a, b) -> a + b;
        
        // 4. 指定参数类型
        BiFunction<Integer, Integer, Integer> multiply = 
            (Integer a, Integer b) -> a * b;
        
        // 5. 多行语句
        Function<String, String> process = s -> {
            String result = s.toUpperCase();
            result = result.trim();
            return result;
        };
        
        // 6. 返回值
        Supplier<Integer> random = () -> (int) (Math.random() * 100);
    }
}
```

## 函数式接口

函数式接口只有一个抽象方法的接口，可以使用 `@FunctionalInterface` 注解。

### 自定义函数式接口

```java
@FunctionalInterface
public interface MyFunction {
    int apply(int x);
    
    // 可以有默认方法
    default int applyTwice(int x) {
        return apply(apply(x));
    }
    
    // 可以有静态方法
    static MyFunction identity() {
        return x -> x;
    }
}

// 使用
public class CustomFunctionalInterface {
    public static void main(String[] args) {
        MyFunction square = x -> x * x;
        MyFunction increment = x -> x + 1;
        
        System.out.println(square.apply(5));          // 25
        System.out.println(square.applyTwice(3));     // 81
        System.out.println(increment.apply(10));      // 11
        
        MyFunction identity = MyFunction.identity();
        System.out.println(identity.apply(42));       // 42
    }
}
```

### 内置函数式接口

Java 8 在 `java.util.function` 包中提供了常用的函数式接口：

```java
import java.util.function.*;

public class BuiltInFunctionalInterfaces {
    public static void main(String[] args) {
        // 1. Function<T, R>: 接受一个参数，返回一个结果
        Function<String, Integer> length = s -> s.length();
        System.out.println(length.apply("Hello"));  // 5
        
        // 2. Consumer<T>: 接受一个参数，无返回值
        Consumer<String> print = s -> System.out.println(s);
        print.accept("Hello");  // Hello
        
        // 3. Supplier<T>: 无参数，返回一个结果
        Supplier<Double> random = () -> Math.random();
        System.out.println(random.get());
        
        // 4. Predicate<T>: 接受一个参数，返回 boolean
        Predicate<Integer> isEven = n -> n % 2 == 0;
        System.out.println(isEven.test(4));   // true
        System.out.println(isEven.test(5));   // false
        
        // 5. BiFunction<T, U, R>: 接受两个参数，返回一个结果
        BiFunction<Integer, Integer, Integer> add = (a, b) -> a + b;
        System.out.println(add.apply(3, 5));  // 8
        
        // 6. BiConsumer<T, U>: 接受两个参数，无返回值
        BiConsumer<String, Integer> printAge = 
            (name, age) -> System.out.println(name + " is " + age);
        printAge.accept("张三", 25);
        
        // 7. BiPredicate<T, U>: 接受两个参数，返回 boolean
        BiPredicate<String, String> equals = (s1, s2) -> s1.equals(s2);
        System.out.println(equals.test("abc", "abc"));  // true
        
        // 8. UnaryOperator<T>: Function<T, T> 的特殊情况
        UnaryOperator<Integer> square = x -> x * x;
        System.out.println(square.apply(5));  // 25
        
        // 9. BinaryOperator<T>: BiFunction<T, T, T> 的特殊情况
        BinaryOperator<Integer> max = (a, b) -> a > b ? a : b;
        System.out.println(max.apply(10, 20));  // 20
    }
}
```

### 函数组合

```java
public class FunctionComposition {
    public static void main(String[] args) {
        Function<Integer, Integer> multiply2 = x -> x * 2;
        Function<Integer, Integer> add3 = x -> x + 3;
        
        // andThen: 先执行第一个，再执行第二个
        Function<Integer, Integer> multiply2ThenAdd3 = multiply2.andThen(add3);
        System.out.println(multiply2ThenAdd3.apply(5));  // (5*2)+3 = 13
        
        // compose: 先执行参数函数，再执行当前函数
        Function<Integer, Integer> add3ThenMultiply2 = multiply2.compose(add3);
        System.out.println(add3ThenMultiply2.apply(5));  // (5+3)*2 = 16
        
        // Predicate 组合
        Predicate<Integer> isEven = n -> n % 2 == 0;
        Predicate<Integer> isPositive = n -> n > 0;
        
        // and
        Predicate<Integer> isPositiveEven = isEven.and(isPositive);
        System.out.println(isPositiveEven.test(4));   // true
        System.out.println(isPositiveEven.test(-4));  // false
        
        // or
        Predicate<Integer> isEvenOrNegative = isEven.or(n -> n < 0);
        System.out.println(isEvenOrNegative.test(3));   // false
        System.out.println(isEvenOrNegative.test(-3));  // true
        
        // negate
        Predicate<Integer> isOdd = isEven.negate();
        System.out.println(isOdd.test(5));  // true
    }
}
```

## 方法引用

方法引用是 Lambda 表达式的简写形式。

### 方法引用的类型

```java
import java.util.*;

public class MethodReferenceExample {
    public static void main(String[] args) {
        List<String> list = Arrays.asList("Apple", "Banana", "Cherry");
        
        // 1. 静态方法引用：类名::静态方法名
        list.forEach(System.out::println);
        
        Function<String, Integer> parseInt1 = s -> Integer.parseInt(s);
        Function<String, Integer> parseInt2 = Integer::parseInt;
        
        // 2. 实例方法引用：对象::实例方法名
        String prefix = "Item: ";
        Consumer<String> printer1 = s -> System.out.println(prefix + s);
        Consumer<String> printer2 = System.out::println;
        
        // 3. 类的实例方法引用：类名::实例方法名
        Function<String, Integer> length1 = s -> s.length();
        Function<String, Integer> length2 = String::length;
        
        BiPredicate<String, String> equals1 = (s1, s2) -> s1.equals(s2);
        BiPredicate<String, String> equals2 = String::equals;
        
        // 4. 构造方法引用：类名::new
        Supplier<List<String>> listCreator1 = () -> new ArrayList<>();
        Supplier<List<String>> listCreator2 = ArrayList::new;
        
        Function<Integer, List<Integer>> listWithSize = ArrayList::new;
        List<Integer> newList = listWithSize.apply(10);
    }
}
```

### 数组构造方法引用

```java
public class ArrayConstructorReference {
    public static void main(String[] args) {
        // 数组构造方法引用
        IntFunction<int[]> arrayCreator1 = size -> new int[size];
        IntFunction<int[]> arrayCreator2 = int[]::new;
        
        int[] array = arrayCreator2.apply(10);
        System.out.println("数组长度: " + array.length);  // 10
    }
}
```

## Stream API

Stream 提供了函数式的数据处理操作。

### 创建 Stream

```java
import java.util.*;
import java.util.stream.*;

public class CreateStream {
    public static void main(String[] args) {
        // 1. 从集合创建
        List<String> list = Arrays.asList("a", "b", "c");
        Stream<String> stream1 = list.stream();
        
        // 2. 从数组创建
        String[] array = {"x", "y", "z"};
        Stream<String> stream2 = Arrays.stream(array);
        
        // 3. 使用 Stream.of()
        Stream<String> stream3 = Stream.of("1", "2", "3");
        
        // 4. 无限流
        Stream<Integer> infiniteStream = Stream.iterate(0, n -> n + 1);
        Stream<Double> randomStream = Stream.generate(Math::random);
        
        // 5. 范围
        IntStream range1 = IntStream.range(1, 5);        // 1,2,3,4
        IntStream range2 = IntStream.rangeClosed(1, 5);  // 1,2,3,4,5
        
        // 6. 空流
        Stream<String> emptyStream = Stream.empty();
        
        // 7. 文件行流
        try (Stream<String> lines = Files.lines(Paths.get("file.txt"))) {
            lines.forEach(System.out::println);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

### 中间操作

中间操作返回一个新的 Stream，支持链式调用。

```java
import java.util.*;
import java.util.stream.*;

public class IntermediateOperations {
    public static void main(String[] args) {
        List<String> words = Arrays.asList("apple", "banana", "cherry", "date", "elderberry");
        
        // 1. filter：筛选
        words.stream()
            .filter(w -> w.length() > 5)
            .forEach(System.out::println);  // banana, cherry, elderberry
        
        // 2. map：转换
        words.stream()
            .map(String::toUpperCase)
            .forEach(System.out::println);
        
        // 3. flatMap：扁平化
        List<List<Integer>> lists = Arrays.asList(
            Arrays.asList(1, 2),
            Arrays.asList(3, 4),
            Arrays.asList(5, 6)
        );
        lists.stream()
            .flatMap(List::stream)
            .forEach(System.out::println);  // 1,2,3,4,5,6
        
        // 4. distinct：去重
        Arrays.asList(1, 2, 2, 3, 3, 3).stream()
            .distinct()
            .forEach(System.out::println);  // 1,2,3
        
        // 5. sorted：排序
        words.stream()
            .sorted()
            .forEach(System.out::println);
        
        words.stream()
            .sorted(Comparator.reverseOrder())
            .forEach(System.out::println);
        
        // 6. limit：限制数量
        Stream.iterate(1, n -> n + 1)
            .limit(5)
            .forEach(System.out::println);  // 1,2,3,4,5
        
        // 7. skip：跳过
        Arrays.asList(1, 2, 3, 4, 5).stream()
            .skip(2)
            .forEach(System.out::println);  // 3,4,5
        
        // 8. peek：查看元素（用于调试）
        words.stream()
            .peek(w -> System.out.println("处理: " + w))
            .map(String::toUpperCase)
            .forEach(System.out::println);
    }
}
```

### 终端操作

终端操作产生结果或副作用。

```java
import java.util.*;
import java.util.stream.*;

public class TerminalOperations {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        // 1. forEach：遍历
        numbers.stream().forEach(System.out::println);
        
        // 2. collect：收集到集合
        List<Integer> evenNumbers = numbers.stream()
            .filter(n -> n % 2 == 0)
            .collect(Collectors.toList());
        
        Set<Integer> set = numbers.stream()
            .collect(Collectors.toSet());
        
        String joined = Arrays.asList("a", "b", "c").stream()
            .collect(Collectors.joining(", "));  // "a, b, c"
        
        // 3. reduce：归约
        int sum = numbers.stream()
            .reduce(0, (a, b) -> a + b);  // 55
        
        int product = numbers.stream()
            .reduce(1, (a, b) -> a * b);
        
        Optional<Integer> max = numbers.stream()
            .reduce(Integer::max);
        
        // 4. count：计数
        long count = numbers.stream()
            .filter(n -> n > 5)
            .count();  // 5
        
        // 5. anyMatch, allMatch, noneMatch
        boolean hasEven = numbers.stream().anyMatch(n -> n % 2 == 0);   // true
        boolean allPositive = numbers.stream().allMatch(n -> n > 0);    // true
        boolean noneNegative = numbers.stream().noneMatch(n -> n < 0);  // true
        
        // 6. findFirst, findAny
        Optional<Integer> first = numbers.stream()
            .filter(n -> n > 5)
            .findFirst();  // 6
        
        Optional<Integer> any = numbers.stream()
            .filter(n -> n > 5)
            .findAny();
        
        // 7. min, max
        Optional<Integer> min = numbers.stream().min(Integer::compare);
        Optional<Integer> maximum = numbers.stream().max(Integer::compare);
        
        // 8. toArray
        Integer[] arr = numbers.stream().toArray(Integer[]::new);
    }
}
```

### Collectors 工具类

```java
import java.util.*;
import java.util.stream.*;

public class CollectorsExample {
    static class Person {
        String name;
        int age;
        String city;
        
        public Person(String name, int age, String city) {
            this.name = name;
            this.age = age;
            this.city = city;
        }
        
        // getters
        public String getName() { return name; }
        public int getAge() { return age; }
        public String getCity() { return city; }
    }
    
    public static void main(String[] args) {
        List<Person> people = Arrays.asList(
            new Person("Alice", 25, "北京"),
            new Person("Bob", 30, "上海"),
            new Person("Charlie", 25, "北京"),
            new Person("David", 35, "上海")
        );
        
        // 1. toList, toSet
        List<String> names = people.stream()
            .map(Person::getName)
            .collect(Collectors.toList());
        
        // 2. toMap
        Map<String, Integer> nameToAge = people.stream()
            .collect(Collectors.toMap(Person::getName, Person::getAge));
        
        // 3. groupingBy：分组
        Map<String, List<Person>> byCity = people.stream()
            .collect(Collectors.groupingBy(Person::getCity));
        
        Map<Integer, List<Person>> byAge = people.stream()
            .collect(Collectors.groupingBy(Person::getAge));
        
        // 4. partitioningBy：分区
        Map<Boolean, List<Person>> partition = people.stream()
            .collect(Collectors.partitioningBy(p -> p.getAge() >= 30));
        
        // 5. counting
        Map<String, Long> cityCount = people.stream()
            .collect(Collectors.groupingBy(Person::getCity, Collectors.counting()));
        
        // 6. summarizingInt：统计
        IntSummaryStatistics stats = people.stream()
            .collect(Collectors.summarizingInt(Person::getAge));
        System.out.println("平均年龄: " + stats.getAverage());
        System.out.println("最大年龄: " + stats.getMax());
        System.out.println("最小年龄: " + stats.getMin());
        
        // 7. joining
        String allNames = people.stream()
            .map(Person::getName)
            .collect(Collectors.joining(", ", "[", "]"));
        System.out.println(allNames);  // [Alice, Bob, Charlie, David]
    }
}
```

### 并行流

```java
public class ParallelStreamExample {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        // 串行流
        long start1 = System.currentTimeMillis();
        int sum1 = numbers.stream()
            .map(n -> n * 2)
            .reduce(0, Integer::sum);
        long end1 = System.currentTimeMillis();
        System.out.println("串行耗时: " + (end1 - start1) + "ms");
        
        // 并行流
        long start2 = System.currentTimeMillis();
        int sum2 = numbers.parallelStream()
            .map(n -> n * 2)
            .reduce(0, Integer::sum);
        long end2 = System.currentTimeMillis();
        System.out.println("并行耗时: " + (end2 - start2) + "ms");
        
        // 转换为并行流
        numbers.stream().parallel().forEach(System.out::println);
    }
}
```

## Optional 类

Optional 用于避免空指针异常。

```java
import java.util.Optional;

public class OptionalExample {
    public static void main(String[] args) {
        // 创建 Optional
        Optional<String> empty = Optional.empty();
        Optional<String> nonEmpty = Optional.of("Hello");
        Optional<String> nullable = Optional.ofNullable(null);
        
        // isPresent：判断是否有值
        if (nonEmpty.isPresent()) {
            System.out.println(nonEmpty.get());
        }
        
        // ifPresent：有值时执行操作
        nonEmpty.ifPresent(s -> System.out.println(s));
        
        // orElse：提供默认值
        String value1 = nullable.orElse("默认值");
        
        // orElseGet：延迟提供默认值
        String value2 = nullable.orElseGet(() -> "计算的默认值");
        
        // orElseThrow：抛出异常
        try {
            String value3 = nullable.orElseThrow(() -> 
                new RuntimeException("值为空"));
        } catch (RuntimeException e) {
            System.out.println(e.getMessage());
        }
        
        // map：转换值
        Optional<Integer> length = nonEmpty.map(String::length);
        System.out.println(length.get());  // 5
        
        // flatMap
        Optional<String> upperCase = nonEmpty.flatMap(s -> 
            Optional.of(s.toUpperCase()));
        
        // filter：过滤
        Optional<String> filtered = nonEmpty.filter(s -> s.length() > 3);
    }
}
```

## 最佳实践

### 1. 优先使用 Lambda 而非匿名类

```java
// 不好
list.forEach(new Consumer<String>() {
    @Override
    public void accept(String s) {
        System.out.println(s);
    }
});

// 好
list.forEach(s -> System.out.println(s));
// 更好
list.forEach(System.out::println);
```

### 2. 使用方法引用简化 Lambda

```java
// 不好
list.stream().map(s -> s.toUpperCase())

// 好
list.stream().map(String::toUpperCase)
```

### 3. 避免副作用

```java
// 不好：有副作用
List<String> results = new ArrayList<>();
stream.forEach(s -> results.add(s.toUpperCase()));

// 好：无副作用
List<String> results = stream
    .map(String::toUpperCase)
    .collect(Collectors.toList());
```

### 4. 合理使用并行流

```java
// 小数据集或简单操作：使用串行流
list.stream().map(String::toUpperCase).collect(Collectors.toList());

// 大数据集且计算密集：使用并行流
bigList.parallelStream().map(expensiveOperation).collect(Collectors.toList());
```

## 总结

本文介绍了 Java 函数式编程的核心内容：

- ✅ Lambda 表达式语法
- ✅ 函数式接口和方法引用
- ✅ Stream API 的中间和终端操作
- ✅ Collectors 工具类
- ✅ 并行流
- ✅ Optional 类

掌握函数式编程后，可以继续学习 [注解](./annotations) 和 [反射](./reflection)。
