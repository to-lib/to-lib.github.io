---
sidebar_position: 4
title: Java 常用类库
---

# Java 常用类库

Java 提供了丰富的标准类库，掌握这些常用类是 Java 编程的基础。本文介绍 Object 类、包装类、Math、Random、System 等核心类库。

## Object 类

Object 是 Java 中所有类的根类，每个类都直接或间接继承自 Object。

### 核心方法

```java
public class ObjectBasics {
    public static void main(String[] args) {
        String str1 = new String("Hello");
        String str2 = new String("Hello");

        // equals()：比较对象内容
        System.out.println(str1.equals(str2));  // true

        // hashCode()：返回哈希码
        System.out.println(str1.hashCode());

        // toString()：返回字符串表示
        System.out.println(str1.toString());  // Hello

        // getClass()：返回运行时类
        System.out.println(str1.getClass().getName());  // java.lang.String
    }
}
```

### 重写 equals 和 hashCode

```java
public class Person {
    private String name;
    private int age;

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;

        Person person = (Person) obj;
        return age == person.age &&
               (name != null ? name.equals(person.name) : person.name == null);
    }

    @Override
    public int hashCode() {
        int result = name != null ? name.hashCode() : 0;
        result = 31 * result + age;
        return result;
    }

    @Override
    public String toString() {
        return "Person{name='" + name + "', age=" + age + "}";
    }
}
```

### clone() 方法

```java
public class Student implements Cloneable {
    private String name;
    private int age;

    public Student(String name, int age) {
        this.name = name;
        this.age = age;
    }

    @Override
    protected Object clone() throws CloneNotSupportedException {
        return super.clone();  // 浅拷贝
    }

    // 深拷贝示例
    public Student deepClone() {
        return new Student(this.name, this.age);
    }
}

// 使用
try {
    Student s1 = new Student("张三", 20);
    Student s2 = (Student) s1.clone();
    System.out.println(s1 == s2);  // false（不同对象）
} catch (CloneNotSupportedException e) {
    e.printStackTrace();
}
```

## 包装类

Java 为每个基本类型提供了对应的包装类。

### 基本类型与包装类对应关系

| 基本类型 | 包装类    | 大小  |
| -------- | --------- | ----- |
| byte     | Byte      | 8 位  |
| short    | Short     | 16 位 |
| int      | Integer   | 32 位 |
| long     | Long      | 64 位 |
| float    | Float     | 32 位 |
| double   | Double    | 64 位 |
| char     | Character | 16 位 |
| boolean  | Boolean   | -     |

### 自动装箱和拆箱

```java
public class WrapperExample {
    public static void main(String[] args) {
        // 自动装箱（基本类型 -> 包装类）
        Integer num1 = 100;  // 等价于 Integer.valueOf(100)

        // 自动拆箱（包装类 -> 基本类型）
        int num2 = num1;  // 等价于 num1.intValue()

        // 包装类的缓存机制（-128 到 127）
        Integer a = 127;
        Integer b = 127;
        System.out.println(a == b);  // true（来自缓存）

        Integer c = 128;
        Integer d = 128;
        System.out.println(c == d);  // false（不在缓存范围）
        System.out.println(c.equals(d));  // true（内容相同）
    }
}
```

### Integer 常用方法

```java
public class IntegerMethods {
    public static void main(String[] args) {
        // 字符串转整数
        int num1 = Integer.parseInt("123");
        Integer num2 = Integer.valueOf("456");

        // 整数转字符串
        String str1 = Integer.toString(123);
        String str2 = String.valueOf(456);

        // 进制转换
        String binary = Integer.toBinaryString(10);   // "1010"
        String octal = Integer.toOctalString(10);     // "12"
        String hex = Integer.toHexString(10);         // "a"

        // 从其他进制解析
        int fromBinary = Integer.parseInt("1010", 2);  // 10
        int fromHex = Integer.parseInt("A", 16);       // 10

        // 最大值和最小值
        System.out.println(Integer.MAX_VALUE);  // 2147483647
        System.out.println(Integer.MIN_VALUE);  // -2147483648

        // 比较
        System.out.println(Integer.compare(10, 20));  // -1
    }
}
```

### Double 和 Float

```java
public class DoubleExample {
    public static void main(String[] args) {
        // 字符串转浮点数
        double d1 = Double.parseDouble("3.14");

        // 特殊值
        System.out.println(Double.NaN);           // 非数字
        System.out.println(Double.POSITIVE_INFINITY);  // 正无穷
        System.out.println(Double.NEGATIVE_INFINITY);  // 负无穷

        // 判断特殊值
        double result = 0.0 / 0.0;
        System.out.println(Double.isNaN(result));      // true
        System.out.println(Double.isInfinite(1.0/0));  // true

        // 比较浮点数（使用 epsilon）
        double a = 0.1 + 0.2;
        double b = 0.3;
        double epsilon = 0.000001;
        System.out.println(Math.abs(a - b) < epsilon);  // true
    }
}
```

## Math 类

Math 类提供了常用的数学运算方法。

### 常用方法

```java
public class MathExample {
    public static void main(String[] args) {
        // 绝对值
        System.out.println(Math.abs(-10));  // 10

        // 最大值和最小值
        System.out.println(Math.max(10, 20));  // 20
        System.out.println(Math.min(10, 20));  // 10

        // 幂运算
        System.out.println(Math.pow(2, 3));  // 8.0
        System.out.println(Math.sqrt(16));   // 4.0（平方根）
        System.out.println(Math.cbrt(8));    // 2.0（立方根）

        // 四舍五入
        System.out.println(Math.round(3.5));   // 4
        System.out.println(Math.round(3.4));   // 3
        System.out.println(Math.ceil(3.1));    // 4.0（向上取整）
        System.out.println(Math.floor(3.9));   // 3.0（向下取整）

        // 三角函数
        System.out.println(Math.sin(Math.PI / 2));  // 1.0
        System.out.println(Math.cos(0));            // 1.0
        System.out.println(Math.tan(Math.PI / 4));  // 1.0

        // 对数
        System.out.println(Math.log(Math.E));    // 1.0（自然对数）
        System.out.println(Math.log10(100));     // 2.0（以10为底）

        // 常量
        System.out.println(Math.PI);  // 3.141592653589793
        System.out.println(Math.E);   // 2.718281828459045
    }
}
```

### 实用示例

```java
public class MathPractical {
    // 计算两点之间的距离
    public static double distance(double x1, double y1, double x2, double y2) {
        return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
    }

    // 将角度转换为弧度
    public static double degreesToRadians(double degrees) {
        return Math.toRadians(degrees);
    }

    // 生成指定范围的随机整数
    public static int randomInRange(int min, int max) {
        return (int) (Math.random() * (max - min + 1)) + min;
    }

    public static void main(String[] args) {
        System.out.println(distance(0, 0, 3, 4));  // 5.0
        System.out.println(degreesToRadians(180)); // 3.141592653589793
        System.out.println(randomInRange(1, 10));  // 1-10之间的随机数
    }
}
```

## Random 类

Random 类用于生成伪随机数。

### 基本用法

```java
import java.util.Random;

public class RandomExample {
    public static void main(String[] args) {
        Random random = new Random();

        // 生成随机整数
        int randomInt = random.nextInt();  // 任意整数
        int boundedInt = random.nextInt(100);  // 0-99

        // 生成随机长整数
        long randomLong = random.nextLong();

        // 生成随机浮点数
        float randomFloat = random.nextFloat();    // 0.0-1.0
        double randomDouble = random.nextDouble(); // 0.0-1.0

        // 生成随机布尔值
        boolean randomBoolean = random.nextBoolean();

        // 生成指定范围的随机数
        int min = 10, max = 50;
        int rangeRandom = random.nextInt(max - min + 1) + min;  // 10-50

        System.out.println("随机整数: " + randomInt);
        System.out.println("0-99: " + boundedInt);
        System.out.println("10-50: " + rangeRandom);
    }
}
```

### 使用种子

```java
public class RandomSeed {
    public static void main(String[] args) {
        // 使用相同的种子会生成相同的随机序列
        Random r1 = new Random(42);
        Random r2 = new Random(42);

        System.out.println(r1.nextInt());  // 相同的值
        System.out.println(r2.nextInt());  // 相同的值

        // 生成高斯分布的随机数
        Random random = new Random();
        double gaussian = random.nextGaussian();  // 均值0，标准差1
        System.out.println("高斯分布: " + gaussian);
    }
}
```

## System 类

System 类提供了与系统相关的方法和属性。

### 常用方法

```java
public class SystemExample {
    public static void main(String[] args) {
        // 获取当前时间（毫秒）
        long startTime = System.currentTimeMillis();

        // 纳秒级时间（用于性能测试）
        long nanoTime = System.nanoTime();

        // 执行一些操作
        for (int i = 0; i < 1000000; i++) {
            // do something
        }

        long endTime = System.currentTimeMillis();
        System.out.println("执行时间: " + (endTime - startTime) + "ms");

        // 数组复制
        int[] src = {1, 2, 3, 4, 5};
        int[] dest = new int[5];
        System.arraycopy(src, 0, dest, 0, 5);

        // 垃圾回收
        System.gc();  // 建议 JVM 执行垃圾回收

        // 退出程序
        // System.exit(0);  // 0表示正常退出
    }
}
```

### 系统属性

```java
public class SystemProperties {
    public static void main(String[] args) {
        // 获取系统属性
        System.out.println("Java版本: " + System.getProperty("java.version"));
        System.out.println("操作系统: " + System.getProperty("os.name"));
        System.out.println("用户目录: " + System.getProperty("user.home"));
        System.out.println("当前目录: " + System.getProperty("user.dir"));
        System.out.println("文件分隔符: " + System.getProperty("file.separator"));
        System.out.println("路径分隔符: " + System.getProperty("path.separator"));
        System.out.println("换行符: " + System.getProperty("line.separator"));

        // 环境变量
        String path = System.getenv("PATH");
        System.out.println("PATH: " + path);

        // 所有环境变量
        System.getenv().forEach((k, v) ->
            System.out.println(k + " = " + v)
        );
    }
}
```

## Arrays 工具类

Arrays 类提供了数组操作的实用方法。

```java
import java.util.Arrays;

public class ArraysExample {
    public static void main(String[] args) {
        int[] arr = {3, 1, 4, 1, 5, 9, 2, 6};

        // 排序
        Arrays.sort(arr);
        System.out.println(Arrays.toString(arr));

        // 二分查找（数组必须有序）
        int index = Arrays.binarySearch(arr, 5);
        System.out.println("5的位置: " + index);

        // 填充
        int[] filled = new int[5];
        Arrays.fill(filled, 7);
        System.out.println(Arrays.toString(filled));  // [7, 7, 7, 7, 7]

        // 复制
        int[] copy = Arrays.copyOf(arr, arr.length);
        int[] subCopy = Arrays.copyOfRange(arr, 0, 3);

        // 比较
        System.out.println(Arrays.equals(arr, copy));  // true

        // 转换为 List
        Integer[] boxed = {1, 2, 3};
        var list = Arrays.asList(boxed);

        // 并行排序（大数组性能更好）
        int[] largeArray = new int[1000000];
        Arrays.parallelSort(largeArray);
    }
}
```

## Collections 工具类

Collections 类提供了集合操作的实用方法。

```java
import java.util.*;

public class CollectionsExample {
    public static void main(String[] args) {
        List<Integer> list = new ArrayList<>(Arrays.asList(3, 1, 4, 1, 5));

        // 排序
        Collections.sort(list);
        System.out.println(list);  // [1, 1, 3, 4, 5]

        // 反转
        Collections.reverse(list);
        System.out.println(list);  // [5, 4, 3, 1, 1]

        // 打乱
        Collections.shuffle(list);
        System.out.println(list);  // 随机顺序

        // 最大值和最小值
        System.out.println("最大值: " + Collections.max(list));
        System.out.println("最小值: " + Collections.min(list));

        // 频率统计
        int frequency = Collections.frequency(list, 1);
        System.out.println("1出现次数: " + frequency);

        // 填充
        Collections.fill(list, 0);

        // 创建不可修改集合
        List<Integer> unmodifiable = Collections.unmodifiableList(list);

        // 创建同步集合
        List<Integer> syncList = Collections.synchronizedList(new ArrayList<>());

        // 创建单例集合
        Set<String> singleton = Collections.singleton("唯一元素");

        // 创建空集合
        List<String> empty = Collections.emptyList();
    }
}
```

## 最佳实践

### 1. 正确使用 equals 和 hashCode

```java
// ✅ 好的做法
@Override
public boolean equals(Object obj) {
    if (this == obj) return true;
    if (obj == null || getClass() != obj.getClass()) return false;

    MyClass other = (MyClass) obj;
    return Objects.equals(field1, other.field1) &&
           Objects.equals(field2, other.field2);
}

@Override
public int hashCode() {
    return Objects.hash(field1, field2);
}
```

### 2. 避免包装类的陷阱

```java
// ❌ 错误：可能导致 NullPointerException
Integer num = null;
int value = num;  // 抛出 NullPointerException

// ✅ 正确：先检查 null
Integer num = null;
int value = (num != null) ? num : 0;
```

### 3. 使用合适的随机数生成器

```java
// ❌ 不推荐：Math.random()（线程不安全）
double random = Math.random();

// ✅ 推荐：使用 Random 或 ThreadLocalRandom
Random random = new Random();
int value = random.nextInt(100);

// ✅ 更好：多线程环境使用 ThreadLocalRandom
import java.util.concurrent.ThreadLocalRandom;
int value = ThreadLocalRandom.current().nextInt(100);
```

### 4. 数组操作优先使用工具类

```java
// ❌ 手动实现
int[] arr = {1, 2, 3};
for (int i = 0; i < arr.length; i++) {
    System.out.print(arr[i] + " ");
}

// ✅ 使用工具类
System.out.println(Arrays.toString(arr));
```

## 总结

- **Object 类**：所有类的父类，重写 equals、hashCode、toString 是常见需求
- **包装类**：提供基本类型的对象表示，注意自动装箱/拆箱和缓存机制
- **Math 类**：提供丰富的数学运算方法
- **Random 类**：生成各种类型的随机数
- **System 类**：访问系统资源和属性
- **Arrays 工具类**：简化数组操作
- **Collections 工具类**：简化集合操作

掌握这些常用类库能够大大提高 Java 编程效率。
