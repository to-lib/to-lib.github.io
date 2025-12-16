---
sidebar_position: 2
title: Java 基础语法
---

# Java 基础语法

掌握 Java 基础语法是学习 Java 编程的第一步。本文涵盖数据类型、变量、运算符和流程控制等核心概念。

## 数据类型

### 基本数据类型

Java 提供了 8 种基本数据类型：

| 类型    | 大小  | 范围           | 默认值   | 示例                  |
| ------- | ----- | -------------- | -------- | --------------------- |
| byte    | 8 位  | -128 ~ 127     | 0        | `byte b = 100;`       |
| short   | 16 位 | -32768 ~ 32767 | 0        | `short s = 1000;`     |
| int     | 32 位 | -2³¹ ~ 2³¹-1   | 0        | `int i = 100000;`     |
| long    | 64 位 | -2⁶³ ~ 2⁶³-1   | 0L       | `long l = 100000L;`   |
| float   | 32 位 | IEEE 754       | 0.0f     | `float f = 3.14f;`    |
| double  | 64 位 | IEEE 754       | 0.0d     | `double d = 3.14159;` |
| char    | 16 位 | 0 ~ 65535      | '\u0000' | `char c = 'A';`       |
| boolean | 1 位  | true/false     | false    | `boolean b = true;`   |

```java
// 基本数据类型示例
public class DataTypeExample {
    public static void main(String[] args) {
        // 整数类型
        byte age = 25;
        short year = 2024;
        int population = 1400000000;
        long distance = 384400000L; // 地月距离（米）

        // 浮点类型
        float pi = 3.14f;
        double e = 2.718281828;

        // 字符和布尔
        char grade = 'A';
        boolean isPassed = true;

        System.out.println("年龄: " + age);
        System.out.println("圆周率: " + pi);
    }
}
```

### 引用数据类型

引用类型包括类、接口、数组等：

```java
// 引用数据类型示例
String name = "张三";           // 字符串
int[] numbers = {1, 2, 3, 4, 5}; // 数组
Object obj = new Object();       // 对象
```

## 变量

### 变量声明

```java
// 变量声明和初始化
int x;              // 声明
x = 10;            // 赋值
int y = 20;        // 声明并初始化
int a = 1, b = 2;  // 同时声明多个变量
```

### 变量类型

```java
public class VariableTypes {
    // 成员变量（实例变量）
    private String instanceVar = "实例变量";

    // 静态变量（类变量）
    private static String classVar = "类变量";

    public void method() {
        // 局部变量
        String localVar = "局部变量";
        System.out.println(localVar);
    }

    public void parameterExample(String parameter) {
        // parameter 是参数变量
        System.out.println(parameter);
    }
}
```

### 常量

使用 `final` 关键字定义常量：

```java
public class Constants {
    // 常量命名使用大写字母和下划线
    public static final double PI = 3.14159265359;
    public static final int MAX_SIZE = 100;
    public static final String COMPANY_NAME = "ABC公司";

    public static void main(String[] args) {
        // PI = 3.14; // 编译错误：无法修改常量
        System.out.println("圆周率: " + PI);
    }
}
```

## 运算符

### 算术运算符

```java
public class ArithmeticOperators {
    public static void main(String[] args) {
        int a = 10, b = 3;

        System.out.println("a + b = " + (a + b));  // 加法: 13
        System.out.println("a - b = " + (a - b));  // 减法: 7
        System.out.println("a * b = " + (a * b));  // 乘法: 30
        System.out.println("a / b = " + (a / b));  // 除法: 3
        System.out.println("a % b = " + (a % b));  // 取模: 1

        // 自增自减
        int x = 5;
        System.out.println("x++ = " + (x++));  // 5 (后增)
        System.out.println("x = " + x);        // 6
        System.out.println("++x = " + (++x));  // 7 (前增)
    }
}
```

### 关系运算符

```java
public class RelationalOperators {
    public static void main(String[] args) {
        int a = 10, b = 5;

        System.out.println("a == b: " + (a == b));  // false
        System.out.println("a != b: " + (a != b));  // true
        System.out.println("a > b: " + (a > b));    // true
        System.out.println("a < b: " + (a < b));    // false
        System.out.println("a >= b: " + (a >= b));  // true
        System.out.println("a <= b: " + (a <= b));  // false
    }
}
```

### 逻辑运算符

```java
public class LogicalOperators {
    public static void main(String[] args) {
        boolean x = true, y = false;

        System.out.println("x && y: " + (x && y));  // 逻辑与: false
        System.out.println("x || y: " + (x || y));  // 逻辑或: true
        System.out.println("!x: " + (!x));          // 逻辑非: false

        // 短路运算
        int a = 5, b = 0;
        if (b != 0 && a / b > 1) {  // b != 0 为 false，不执行 a / b
            System.out.println("不会执行");
        }
    }
}
```

### 位运算符

```java
public class BitwiseOperators {
    public static void main(String[] args) {
        int a = 5;  // 0101
        int b = 3;  // 0011

        System.out.println("a & b = " + (a & b));   // 按位与: 1 (0001)
        System.out.println("a | b = " + (a | b));   // 按位或: 7 (0111)
        System.out.println("a ^ b = " + (a ^ b));   // 按位异或: 6 (0110)
        System.out.println("~a = " + (~a));         // 按位取反: -6
        System.out.println("a << 1 = " + (a << 1)); // 左移: 10
        System.out.println("a >> 1 = " + (a >> 1)); // 右移: 2
    }
}
```

## 流程控制

### 条件语句

#### if-else 语句

```java
public class IfElseExample {
    public static void main(String[] args) {
        int score = 85;

        // 简单 if
        if (score >= 60) {
            System.out.println("及格");
        }

        // if-else
        if (score >= 60) {
            System.out.println("及格");
        } else {
            System.out.println("不及格");
        }

        // if-else if-else
        if (score >= 90) {
            System.out.println("优秀");
        } else if (score >= 80) {
            System.out.println("良好");
        } else if (score >= 70) {
            System.out.println("中等");
        } else if (score >= 60) {
            System.out.println("及格");
        } else {
            System.out.println("不及格");
        }

        // 三元运算符
        String result = score >= 60 ? "及格" : "不及格";
        System.out.println(result);
    }
}
```

#### switch 语句

```java
public class SwitchExample {
    public static void main(String[] args) {
        int dayOfWeek = 3;

        // 传统 switch
        switch (dayOfWeek) {
            case 1:
                System.out.println("星期一");
                break;
            case 2:
                System.out.println("星期二");
                break;
            case 3:
                System.out.println("星期三");
                break;
            default:
                System.out.println("其他");
                break;
        }

        // 使用 Map 替代多分支 switch（JDK 8 推荐）
        String[] days = {"", "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"};
        String day = (dayOfWeek >= 1 && dayOfWeek <= 7) ? days[dayOfWeek] : "无效";
        System.out.println(day);
    }
}
```

:::tip JDK 12+ 特性
如果你使用 JDK 12 或更高版本，可以使用增强的 switch 表达式语法：

```java
// JDK 12+ 增强 switch 表达式（需要 JDK 12+）
String day = switch (dayOfWeek) {
    case 1 -> "星期一";
    case 2 -> "星期二";
    case 3 -> "星期三";
    case 4 -> "星期四";
    case 5 -> "星期五";
    case 6 -> "星期六";
    case 7 -> "星期日";
    default -> "无效";
};
```

:::

### 循环语句

#### for 循环

```java
public class ForLoopExample {
    public static void main(String[] args) {
        // 基本 for 循环
        for (int i = 0; i < 5; i++) {
            System.out.println("i = " + i);
        }

        // 增强 for 循环（for-each）
        int[] numbers = {1, 2, 3, 4, 5};
        for (int num : numbers) {
            System.out.println("num = " + num);
        }

        // 嵌套循环
        for (int i = 1; i <= 3; i++) {
            for (int j = 1; j <= 3; j++) {
                System.out.println("i=" + i + ", j=" + j);
            }
        }
    }
}
```

#### while 循环

```java
public class WhileLoopExample {
    public static void main(String[] args) {
        // while 循环
        int i = 0;
        while (i < 5) {
            System.out.println("i = " + i);
            i++;
        }

        // do-while 循环（至少执行一次）
        int j = 0;
        do {
            System.out.println("j = " + j);
            j++;
        } while (j < 5);
    }
}
```

### 跳转语句

```java
public class JumpStatements {
    public static void main(String[] args) {
        // break：跳出循环
        for (int i = 0; i < 10; i++) {
            if (i == 5) {
                break;  // 当 i=5 时跳出循环
            }
            System.out.println("i = " + i);
        }

        // continue：跳过当前迭代
        for (int i = 0; i < 5; i++) {
            if (i == 2) {
                continue;  // 跳过 i=2
            }
            System.out.println("i = " + i);
        }

        // 标签配合 break/continue
        outer: for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == 1 && j == 1) {
                    break outer;  // 跳出外层循环
                }
                System.out.println("i=" + i + ", j=" + j);
            }
        }
    }
}
```

## 字符串详解

### 字符串的不可变性

Java 中的 String 是不可变的（Immutable）。这意味着一旦创建了 String 对象，就无法修改它。

```java
public class StringImmutability {
    public static void main(String[] args) {
        String str = "Hello";
        str = str + " World";  // 创建新的 String 对象

        System.out.println(str);  // Hello World

        // ❌ 不能这样做：没有 setChar 方法
        // str.setChar(0, 'h');

        // ✅ 字符串虽然不可变，但变量可以重新赋值
        str = "Goodbye";
    }
}
```

### String 常量池

Java 为了提高效率，设立了字符串常量池。相同的字符串字面量会共用一个对象。

```java
public class StringPoolExample {
    public static void main(String[] args) {
        // 都从常量池取同一个对象
        String str1 = "Hello";
        String str2 = "Hello";
        System.out.println(str1 == str2);  // true（同一对象）

        // 使用 new 创建新对象，不用常量池
        String str3 = new String("Hello");
        System.out.println(str1 == str3);      // false（不同对象）
        System.out.println(str1.equals(str3)); // true（内容相同）

        // intern() 方法：强制放入常量池
        String str4 = new String("Hello").intern();
        System.out.println(str1 == str4);  // true（同一对象）
    }
}
```

### 字符串创建和操作

```java
public class StringBasics {
    public static void main(String[] args) {
        // 字符串创建
        String str1 = "Hello";
        String str2 = new String("World");

        // 字符串连接
        String greeting = str1 + " " + str2;
        System.out.println(greeting);  // Hello World

        // 字符串长度
        System.out.println("长度: " + greeting.length());

        // 字符串比较
        System.out.println("相等: " + str1.equals("Hello"));
        System.out.println("相等(忽略大小写): " + str1.equalsIgnoreCase("hello"));

        // 字符串查找
        System.out.println("包含: " + greeting.contains("World"));
        System.out.println("起始位置: " + greeting.indexOf("World"));

        // 字符串截取
        System.out.println("子串: " + greeting.substring(0, 5));

        // 字符串替换
        System.out.println("替换: " + greeting.replace("World", "Java"));

        // 字符串分割
        String[] words = greeting.split(" ");
        for (String word : words) {
            System.out.println(word);
        }
    }
}
```

### StringBuilder 和 StringBuffer

对于频繁修改字符串的场景，应该使用 StringBuilder 而不是 String 连接。

```java
public class StringBuilderExample {
    public static void main(String[] args) {
        // StringBuilder：不同步，性能更好
        StringBuilder sb = new StringBuilder();

        // append：追加字符
        sb.append("Hello");
        sb.append(" ");
        sb.append("World");

        System.out.println(sb.toString());  // Hello World

        // insert：插入字符
        sb.insert(5, ",");  // 在位置 5 插入 ","
        System.out.println(sb);  // Hello, World

        // delete：删除字符
        sb.delete(5, 6);  // 删除位置 5-6 的字符
        System.out.println(sb);  // Hello World

        // reverse：反转
        sb.reverse();
        System.out.println(sb);  // dlroW olleH
    }
}
```

### 字符串性能对比

```java
public class StringPerformance {
    public static void main(String[] args) {
        // 性能测试：10000 次字符串操作

        // ❌ 使用 String 连接（低效）
        long startTime = System.currentTimeMillis();
        String str = "";
        for (int i = 0; i < 10000; i++) {
            str += "x";  // 每次都创建新 String 对象
        }
        long time1 = System.currentTimeMillis() - startTime;
        System.out.println("String 连接耗时: " + time1 + "ms");

        // ✅ 使用 StringBuilder（高效）
        startTime = System.currentTimeMillis();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 10000; i++) {
            sb.append("x");  // 在同一对象上操作
        }
        String result = sb.toString();
        long time2 = System.currentTimeMillis() - startTime;
        System.out.println("StringBuilder 耗时: " + time2 + "ms");

        System.out.println("性能提升: " + (time1 / time2) + " 倍");
    }
}
```

### String vs StringBuilder vs StringBuffer

| 特性     | String         | StringBuilder  | StringBuffer   |
| -------- | -------------- | -------------- | -------------- |
| 可变性   | 不可变         | 可变           | 可变           |
| 线程安全 | 是             | 否             | 是             |
| 性能     | 低（频繁修改） | 高             | 中等           |
| 使用场景 | 字符串不修改   | 单线程频繁修改 | 多线程频繁修改 |

```java
public class StringComparison {
    public static void main(String[] args) {
        // StringBuffer：线程安全但性能稍差
        StringBuffer sbuf = new StringBuffer();
        sbuf.append("Hello");  // 同步方法

        // StringBuilder：性能最好但非线程安全
        StringBuilder sbuilder = new StringBuilder();
        sbuilder.append("Hello");  // 非同步方法

        System.out.println(sbuf.toString());     // Hello
        System.out.println(sbuilder.toString()); // Hello
    }
}
```

### 常用 String 方法

```java
public class StringMethods {
    public static void main(String[] args) {
        String str = "  Hello Java World  ";

        // 大小写转换
        System.out.println(str.toUpperCase());   // "  HELLO JAVA WORLD  "
        System.out.println(str.toLowerCase());   // "  hello java world  "

        // 去掉首尾空格
        System.out.println(str.trim());  // "Hello Java World"

        // 开始和结尾判断
        System.out.println(str.startsWith("  Hello"));  // true
        System.out.println(str.endsWith("World  "));    // true

        // 字符替换
        System.out.println(str.replace(" ", ""));    // "HelloJavaWorld"
        System.out.println(str.replaceAll("\\s+", " "));  // " Hello Java World "

        // 转换成字符数组
        char[] chars = str.toCharArray();
        System.out.println("首字符: " + chars[2]);  // 'H'

        // 字符串拆分
        String[] words = str.trim().split(" ");
        for (String word : words) {
            System.out.println(word);
        }
    }
}
```

## 数组基础

### 数组声明和初始化

```java
public class ArrayBasics {
    public static void main(String[] args) {
        // 数组声明和初始化
        int[] arr1 = new int[5];           // 创建长度为5的数组
        int[] arr2 = {1, 2, 3, 4, 5};      // 声明并初始化
        int[] arr3 = new int[]{1, 2, 3};   // 另一种初始化方式

        // 访问数组元素
        arr1[0] = 10;
        System.out.println("第一个元素: " + arr1[0]);
        System.out.println("数组长度: " + arr1.length);

        // 遍历数组
        for (int i = 0; i < arr2.length; i++) {
            System.out.println("arr2[" + i + "] = " + arr2[i]);
        }

        // 使用增强 for 循环
        for (int num : arr2) {
            System.out.println(num);
        }

        // 多维数组
        int[][] matrix = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                System.out.print(matrix[i][j] + " ");
            }
            System.out.println();
        }
    }
}
```

## 最佳实践

### 命名规范

```java
// 类名：大驼峰（PascalCase）
public class MyClass {}

// 变量和方法名：小驼峰（camelCase）
int myVariable;
public void myMethod() {}

// 常量：全大写加下划线
public static final int MAX_VALUE = 100;

// 包名：全小写
package com.example.myapp;
```

### 代码风格

```java
public class CodeStyle {
    // 1. 使用有意义的变量名
    int age = 25;  // 好
    int a = 25;    // 不好

    // 2. 适当使用空格和缩进
    public void goodExample() {
        if (age > 18) {
            System.out.println("成年人");
        }
    }

    // 3. 一行代码不要过长
    String message = "这是一个很长的字符串，" +
                     "应该分成多行显示";

    // 4. 及时释放资源
    public void resourceExample() {
        // 使用 try-with-resources
        try (BufferedReader reader = new BufferedReader(new FileReader("file.txt"))) {
            String line = reader.readLine();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

## 常见问题

### Q1: == 和 equals() 的区别？

```java
String s1 = "Hello";
String s2 = "Hello";
String s3 = new String("Hello");

System.out.println(s1 == s2);      // true (同一个对象)
System.out.println(s1 == s3);      // false (不同对象)
System.out.println(s1.equals(s3)); // true (内容相同)
```

- `==` 比较对象引用（地址）
- `equals()` 比较对象内容

### Q2: 如何避免整数溢出？

```java
int a = Integer.MAX_VALUE;
int b = a + 1;  // 溢出，变成负数

// 使用 long 类型
long c = (long)a + 1;  // 正确

// 或使用 Math.addExact() 检测溢出
try {
    int d = Math.addExact(a, 1);
} catch (ArithmeticException e) {
    System.out.println("溢出了！");
}
```

### Q3: 为什么要避免使用浮点数进行精确计算？

```java
double d1 = 0.1 + 0.2;
System.out.println(d1);  // 0.30000000000000004

// 使用 BigDecimal 进行精确计算
BigDecimal bd1 = new BigDecimal("0.1");
BigDecimal bd2 = new BigDecimal("0.2");
BigDecimal result = bd1.add(bd2);
System.out.println(result);  // 0.3
```

## 总结

本文介绍了 Java 的基础语法，包括：

- ✅ 8 种基本数据类型和引用类型
- ✅ 变量、常量和运算符
- ✅ 条件语句和循环语句
- ✅ 字符串和数组基础操作
- ✅ 命名规范和代码风格

掌握这些基础知识后，可以继续学习 [面向对象编程](/docs/java/oop)。
