---
sidebar_position: 21
title: Java 快速参考
---

# Java 快速参考

常用语法和代码片段的快速查阅手册。

## 基础语法

### 数据类型

| 类型    | 大小  | 范围              | 默认值   |
| ------- | ----- | ----------------- | -------- |
| byte    | 8 位  | -128 到 127       | 0        |
| short   | 16 位 | -32,768 到 32,767 | 0        |
| int     | 32 位 | -2^31 到 2^31-1   | 0        |
| long    | 64 位 | -2^63 到 2^63-1   | 0L       |
| float   | 32 位 | 约 ±3.4E+38       | 0.0f     |
| double  | 64 位 | 约 ±1.8E+308      | 0.0d     |
| char    | 16 位 | 0 到 65,535       | '\u0000' |
| boolean | -     | true 或 false     | false    |

### 运算符优先级

| 优先级 | 运算符            | 说明                 |
| ------ | ----------------- | -------------------- | ---- | ------ |
| 1      | `()` `[]` `.`     | 括号、数组、成员访问 |
| 2      | `++` `--` `!` `~` | 一元运算符           |
| 3      | `*` `/` `%`       | 乘除模               |
| 4      | `+` `-`           | 加减                 |
| 5      | `<<` `>>` `>>>`   | 位移                 |
| 6      | `<` `<=` `>` `>=` | 关系                 |
| 7      | `==` `!=`         | 相等                 |
| 8      | `&`               | 位与                 |
| 9      | `^`               | 位异或               |
| 10     | `                 | `                    | 位或 |
| 11     | `&&`              | 逻辑与               |
| 12     | `                 |                      | `    | 逻辑或 |
| 13     | `?:`              | 三元运算符           |
| 14     | `=` `+=` `-=` 等  | 赋值                 |

## 集合框架

### 集合选择指南

| 需求               | 推荐集合          | 特点         |
| ------------------ | ----------------- | ------------ |
| 有序、可重复       | ArrayList         | 快速随机访问 |
| 有序、频繁插入删除 | LinkedList        | 快速插入删除 |
| 无序、不重复       | HashSet           | 快速查找     |
| 有序、不重复       | TreeSet           | 自动排序     |
| 键值对、无序       | HashMap           | 快速查找     |
| 键值对、有序       | LinkedHashMap     | 保持插入顺序 |
| 键值对、排序       | TreeMap           | 自动排序     |
| 线程安全的 Map     | ConcurrentHashMap | 高并发性能   |

### 常用方法速查

```java
// List
list.add(element);              // 添加元素
list.get(index);                // 获取元素
list.remove(index);             // 删除元素
list.size();                    // 获取大小
list.contains(element);         // 是否包含
list.indexOf(element);          // 查找索引
list.clear();                   // 清空
Collections.sort(list);         // 排序

// Set
set.add(element);               // 添加元素
set.remove(element);            // 删除元素
set.contains(element);          // 是否包含
set.size();                     // 获取大小

// Map
map.put(key, value);            // 添加键值对
map.get(key);                   // 获取值
map.remove(key);                // 删除
map.containsKey(key);           // 是否包含键
map.containsValue(value);       // 是否包含值
map.keySet();                   // 获取所有键
map.values();                   // 获取所有值
map.entrySet();                 // 获取所有键值对
```

## String 常用方法

```java
// 长度和判空
str.length()                    // 字符串长度
str.isEmpty()                   // 是否为空
str.isBlank()                   // 是否为空白（JDK 11+）

// 查找
str.indexOf("sub")              // 查找子串位置
str.lastIndexOf("sub")          // 最后一次出现位置
str.contains("sub")             // 是否包含
str.startsWith("pre")           // 是否以...开头
str.endsWith("suf")             // 是否以...结尾

// 截取和拆分
str.substring(start, end)       // 截取子串
str.split(",")                  // 拆分字符串
str.trim()                      // 去除两端空白
str.strip()                     // 去除空白（JDK 11+）

// 转换
str.toUpperCase()               // 转大写
str.toLowerCase()               // 转小写
str.replace("old", "new")       // 替换
str.replaceAll("regex", "new")  // 正则替换
```

## Stream API 速查

```java
// 创建 Stream
Stream.of(1, 2, 3)
list.stream()
Arrays.stream(array)

// 中间操作
.filter(x -> x > 0)             // 过滤
.map(x -> x * 2)                // 映射
.flatMap(List::stream)          // 扁平化
.distinct()                     // 去重
.sorted()                       // 排序
.limit(10)                      // 限制数量
.skip(5)                        // 跳过
.peek(System.out::println)      // 查看

// 终端操作
.forEach(System.out::println)   // 遍历
.collect(Collectors.toList())   // 收集为 List
.collect(Collectors.toSet())    // 收集为 Set
.reduce((a, b) -> a + b)        // 归约
.count()                        // 计数
.anyMatch(x -> x > 0)           // 任意匹配
.allMatch(x -> x > 0)           // 全部匹配
.noneMatch(x -> x < 0)          // 无匹配
.findFirst()                    // 查找第一个
.findAny()                      // 查找任意一个
.min(Comparator.naturalOrder()) // 最小值
.max(Comparator.naturalOrder()) // 最大值
```

## Lambda 表达式

```java
// 无参数
() -> System.out.println("Hello")

// 一个参数
x -> x * x

// 多个参数
(x, y) -> x + y

// 代码块
(x, y) -> {
    int sum = x + y;
    return sum;
}

// 方法引用
System.out::println             // 实例方法引用
String::length                   // 类的实例方法引用
Integer::parseInt                // 静态方法引用
ArrayList::new                   // 构造器引用
```

## 常用函数式接口

```java
// Predicate<T>：T -> boolean
Predicate<String> isEmpty = String::isEmpty;

// Function<T, R>：T -> R
Function<String, Integer> length = String::length;

// Consumer<T>：T -> void
Consumer<String> print = System.out::println;

// Supplier<T>：() -> T
Supplier<Double> random = Math::random;

// BiFunction<T, U, R>：(T, U) -> R
BiFunction<Integer, Integer, Integer> add = (a, b) -> a + b;

// UnaryOperator<T>：T -> T
UnaryOperator<Integer> square = x -> x * x;

// BinaryOperator<T>：(T, T) -> T
BinaryOperator<Integer> multiply = (a, b) -> a * b;
```

## Optional

```java
// 创建
Optional.of(value)              // 非空
Optional.ofNullable(value)      // 可能为空
Optional.empty()                // 空

// 判断
optional.isPresent()            // 是否有值
optional.isEmpty()              // 是否为空（JDK 11+）

// 获取值
optional.get()                  // 获取（可能抛异常）
optional.orElse(defaultValue)   // 有值返回值，否则返回默认值
optional.orElseGet(() -> value) // 有值返回值，否则调用 Supplier
optional.orElseThrow()          // 有值返回值，否则抛异常

// 转换
optional.map(String::length)    // 映射
optional.flatMap(x -> Optional.of(x)) // 扁平映射
optional.filter(x -> x.length() > 5)  // 过滤

// 操作
optional.ifPresent(System.out::println) // 有值时执行
```

## 日期时间 API

```java
// 当前日期时间
LocalDate.now()                 // 当前日期
LocalTime.now()                 // 当前时间
LocalDateTime.now()             // 当前日期时间

// 创建
LocalDate.of(2024, 1, 1)       // 指定日期
LocalTime.of(12, 30, 0)        // 指定时间
LocalDateTime.of(2024, 1, 1, 12, 30) // 指定日期时间

// 解析
LocalDate.parse("2024-01-01")
LocalTime.parse("12:30:00")
LocalDateTime.parse("2024-01-01T12:30:00")

// 格式化
DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
dateTime.format(formatter)

// 操作
date.plusDays(1)               // 加天数
date.minusMonths(1)            // 减月数
date.withYear(2025)            // 设置年份

// 比较
date1.isBefore(date2)          // 是否在之前
date1.isAfter(date2)           // 是否在之后
date1.isEqual(date2)           // 是否相等
```

## 文件操作

```java
// 读取文件
String content = Files.readString(path);                    // JDK 11+
List<String> lines = Files.readAllLines(path);
Stream<String> lineStream = Files.lines(path);

// 写入文件
Files.writeString(path, content);                           // JDK 11+
Files.write(path, lines);

// 复制、移动、删除
Files.copy(source, target);
Files.move(source, target);
Files.delete(path);
Files.deleteIfExists(path);

// 判断
Files.exists(path);
Files.isDirectory(path);
Files.isRegularFile(path);
Files.isReadable(path);
Files.isWritable(path);
```

## 异常处理

```java
// try-catch
try {
    riskyOperation();
} catch (IOException e) {
    logger.error("IO错误", e);
} catch (Exception e) {
    logger.error("其他错误", e);
} finally {
    cleanup();
}

// try-with-resources
try (BufferedReader reader = new BufferedReader(new FileReader("file.txt"))) {
    return reader.readLine();
}

// 自定义异常
public class CustomException extends RuntimeException {
    public CustomException(String message) {
        super(message);
    }
}
```

## 多线程

```java
// 创建线程
Thread thread = new Thread(() -> doWork());
thread.start();

// 线程池
ExecutorService executor = Executors.newFixedThreadPool(10);
executor.submit(() -> doWork());
executor.shutdown();

// CompletableFuture
CompletableFuture.supplyAsync(() -> getValue())
    .thenApply(value -> transform(value))
    .thenAccept(result -> System.out.println(result));

// 同步
synchronized (lock) {
    // 临界区代码
}

// Lock
Lock lock = new ReentrantLock();
lock.lock();
try {
    // 临界区代码
} finally {
    lock.unlock();
}
```

## 正则表达式

```java
// 常用正则
"^\\d+$"                        // 数字
"^[a-zA-Z]+$"                   // 字母
"^[\\w-]+(\\.[\\w-]+)*@[\\w-]+(\\.[\\w-]+)+$"  // 邮箱
"^1[3-9]\\d{9}$"                // 手机号

// 使用
Pattern pattern = Pattern.compile("\\d+");
Matcher matcher = pattern.matcher("abc123def456");
while (matcher.find()) {
    System.out.println(matcher.group());
}

// 简单匹配
boolean matches = str.matches("\\d+");
String replaced = str.replaceAll("\\d+", "NUM");
String[] parts = str.split("\\s+");
```

## 注解

```java
// 常用注解
@Override                       // 重写方法
@Deprecated                     // 已过时
@SuppressWarnings("unchecked")  // 抑制警告
@FunctionalInterface            // 函数式接口

// 自定义注解
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface MyAnnotation {
    String value() default "";
}
```

## 泛型

```java
// 泛型类
public class Box<T> {
    private T value;
    public T getValue() { return value; }
}

// 泛型方法
public <T> T getFirst(List<T> list) {
    return list.get(0);
}

// 通配符
List<? extends Number> numbers;  // 上界
List<? super Integer> integers;  // 下界
List<?> unknowns;                // 无界
```

## JVM 参数

```bash
# 内存设置
-Xms512m                        # 初始堆大小
-Xmx2g                          # 最大堆大小
-Xss256k                        # 线程栈大小

# GC 相关
-XX:+UseG1GC                    # 使用 G1 垃圾收集器
-XX:+PrintGCDetails             # 打印 GC 详情
-XX:+HeapDumpOnOutOfMemoryError # OOM 时生成 dump

# 其他
-XX:+UnlockExperimentalVMOptions # 解锁实验特性
-XX:+UseZGC                     # 使用 ZGC
-verbose:gc                      # 打印 GC 信息
```

## Maven 常用命令

```bash
mvn clean                       # 清理
mvn compile                     # 编译
mvn test                        # 测试
mvn package                     # 打包
mvn install                     # 安装到本地仓库
mvn deploy                      # 部署到远程仓库
mvn clean package -DskipTests   # 跳过测试打包
mvn dependency:tree             # 查看依赖树
```

## 常用设计模式代码片段

```java
// 单例模式（枚举）
public enum Singleton {
    INSTANCE;
    public void doSomething() { }
}

// 工厂模式
public class Factory {
    public static Product create(String type) {
        switch (type) {
            case "A": return new ProductA();
            case "B": return new ProductB();
            default: throw new IllegalArgumentException();
        }
    }
}

// 建造者模式
User user = User.builder()
    .name("张三")
    .age(25)
    .build();

// 观察者模式
subject.addObserver(observer);
subject.notifyObservers();
```

## 总结

这份快速参考涵盖了 Java 编程中最常用的语法和 API，可以作为日常开发的速查手册。建议收藏此页面以便快速查阅。
