---
sidebar_position: 2
title: Java 基础语法
---

# Java 基础语法

掌握 Java 基础语法是学习 Java 编程的第一步。本文涵盖数据类型、变量、运算符和流程控制等核心概念。

## 数据类型

### 基本数据类型

Java 提供了 8 种基本数据类型：

| 类型 | 大小 | 范围 | 默认值 | 示例 |
|------|------|------|--------|------|
| byte | 8位 | -128 ~ 127 | 0 | `byte b = 100;` |
| short | 16位 | -32768 ~ 32767 | 0 | `short s = 1000;` |
| int | 32位 | -2³¹ ~ 2³¹-1 | 0 | `int i = 100000;` |
| long | 64位 | -2⁶³ ~ 2⁶³-1 | 0L | `long l = 100000L;` |
| float | 32位 | IEEE 754 | 0.0f | `float f = 3.14f;` |
| double | 64位 | IEEE 754 | 0.0d | `double d = 3.14159;` |
| char | 16位 | 0 ~ 65535 | '\u0000' | `char c = 'A';` |
| boolean | 1位 | true/false | false | `boolean b = true;` |

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
        
        // Java 12+ 增强 switch 表达式
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
        System.out.println(day);
    }
}
```

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

## 字符串基础

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

掌握这些基础知识后，可以继续学习 [面向对象编程](./oop)。
