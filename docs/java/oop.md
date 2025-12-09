---
sidebar_position: 3
title: 面向对象编程
---

# 面向对象编程

面向对象编程（OOP）是 Java 的核心特性。本文详细介绍类、对象、封装、继承、多态等核心概念。

## 类和对象

### 类的定义

类是对象的模板，定义了对象的属性和行为。

```java
public class Person {
    // 属性（成员变量）
    private String name;
    private int age;
    
    // 构造方法
    public Person() {
        this.name = "未知";
        this.age = 0;
    }
    
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    // 方法
    public void introduce() {
        System.out.println("我叫" + name + "，今年" + age + "岁");
    }
    
    // Getter 和 Setter
    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }
    
    public int getAge() {
        return age;
    }
    
    public void setAge(int age) {
        if (age > 0 && age < 150) {
            this.age = age;
        }
    }
}
```

### 对象的创建和使用

```java
public class ObjectExample {
    public static void main(String[] args) {
        // 创建对象
        Person person1 = new Person();
        Person person2 = new Person("张三", 25);
        
        // 使用对象
        person1.setName("李四");
        person1.setAge(30);
        
        person1.introduce();  // 我叫李四，今年30岁
        person2.introduce();  // 我叫张三，今年25岁
    }
}
```

## 封装

封装是将数据和操作数据的方法绑定在一起，隐藏内部实现细节。

### 访问修饰符

| 修饰符 | 类内部 | 同包 | 子类 | 其他包 |
|--------|--------|------|------|--------|
| private | ✓ | ✗ | ✗ | ✗ |
| default (无修饰符) | ✓ | ✓ | ✗ | ✗ |
| protected | ✓ | ✓ | ✓ | ✗ |
| public | ✓ | ✓ | ✓ | ✓ |

```java
public class BankAccount {
    // 私有属性
    private String accountNumber;
    private double balance;
    
    public BankAccount(String accountNumber, double initialBalance) {
        this.accountNumber = accountNumber;
        this.balance = initialBalance;
    }
    
    // 公共方法访问私有数据
    public void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            System.out.println("存款成功，当前余额: " + balance);
        } else {
            System.out.println("存款金额必须大于0");
        }
    }
    
    public void withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            System.out.println("取款成功，当前余额: " + balance);
        } else {
            System.out.println("余额不足或金额无效");
        }
    }
    
    public double getBalance() {
        return balance;
    }
}
```

## 继承

继承允许一个类继承另一个类的属性和方法，实现代码复用。

### 基本继承

```java
// 父类（超类、基类）
public class Animal {
    protected String name;
    protected int age;
    
    public Animal(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    public void eat() {
        System.out.println(name + " 正在吃东西");
    }
    
    public void sleep() {
        System.out.println(name + " 正在睡觉");
    }
}

// 子类（派生类）
public class Dog extends Animal {
    private String breed;
    
    public Dog(String name, int age, String breed) {
        super(name, age);  // 调用父类构造方法
        this.breed = breed;
    }
    
    // 子类特有方法
    public void bark() {
        System.out.println(name + " 正在叫: 汪汪汪！");
    }
    
    // 方法重写（Override）
    @Override
    public void eat() {
        System.out.println(name + " 正在吃狗粮");
    }
}

// 另一个子类
public class Cat extends Animal {
    public Cat(String name, int age) {
        super(name, age);
    }
    
    public void meow() {
        System.out.println(name + " 正在叫: 喵喵喵！");
    }
    
    @Override
    public void eat() {
        System.out.println(name + " 正在吃猫粮");
    }
}
```

### 继承的使用

```java
public class InheritanceExample {
    public static void main(String[] args) {
        Dog dog = new Dog("旺财", 3, "金毛");
        Cat cat = new Cat("咪咪", 2);
        
        dog.eat();    // 旺财 正在吃狗粮
        dog.sleep();  // 旺财 正在睡觉
        dog.bark();   // 旺财 正在叫: 汪汪汪！
        
        cat.eat();    // 咪咪 正在吃猫粮
        cat.meow();   // 咪咪 正在叫: 喵喵喵！
    }
}
```

### super 和 this 关键字

```java
public class Vehicle {
    protected String brand;
    
    public Vehicle(String brand) {
        this.brand = brand;
    }
    
    public void display() {
        System.out.println("品牌: " + brand);
    }
}

public class Car extends Vehicle {
    private int doors;
    
    public Car(String brand, int doors) {
        super(brand);      // 调用父类构造方法
        this.doors = doors; // this 引用当前对象
    }
    
    @Override
    public void display() {
        super.display();   // 调用父类方法
        System.out.println("门数: " + doors);
    }
    
    public void thiExample() {
        this.display();    // 调用当前类的方法
    }
}
```

## 多态

多态是指同一个行为具有多种不同表现形式。

### 方法重载（Overload）

同一个类中，方法名相同但参数不同。

```java
public class Calculator {
    // 两个整数相加
    public int add(int a, int b) {
        return a + b;
    }
    
    // 三个整数相加
    public int add(int a, int b, int c) {
        return a + b + c;
    }
    
    // 两个浮点数相加
    public double add(double a, double b) {
        return a + b;
    }
    
    // 数组求和
    public int add(int[] numbers) {
        int sum = 0;
        for (int num : numbers) {
            sum += num;
        }
        return sum;
    }
}
```

### 方法重写（Override）

子类重写父类的方法。

```java
public class Shape {
    public void draw() {
        System.out.println("绘制形状");
    }
    
    public double getArea() {
        return 0.0;
    }
}

public class Circle extends Shape {
    private double radius;
    
    public Circle(double radius) {
        this.radius = radius;
    }
    
    @Override
    public void draw() {
        System.out.println("绘制圆形");
    }
    
    @Override
    public double getArea() {
        return Math.PI * radius * radius;
    }
}

public class Rectangle extends Shape {
    private double width;
    private double height;
    
    public Rectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }
    
    @Override
    public void draw() {
        System.out.println("绘制矩形");
    }
    
    @Override
    public double getArea() {
        return width * height;
    }
}
```

### 多态的应用

```java
public class PolymorphismExample {
    public static void main(String[] args) {
        // 父类引用指向子类对象
        Shape shape1 = new Circle(5);
        Shape shape2 = new Rectangle(4, 6);
        
        // 多态调用
        printShapeInfo(shape1);  // 绘制圆形，面积: 78.54
        printShapeInfo(shape2);  // 绘制矩形，面积: 24.0
        
        // 使用数组实现多态
        Shape[] shapes = {
            new Circle(3),
            new Rectangle(5, 4),
            new Circle(7)
        };
        
        for (Shape shape : shapes) {
            shape.draw();
            System.out.println("面积: " + shape.getArea());
        }
    }
    
    // 利用多态，方法可以接受任何 Shape 子类对象
    public static void printShapeInfo(Shape shape) {
        shape.draw();
        System.out.println("面积: " + shape.getArea());
    }
}
```

## 抽象类

抽象类不能被实例化，可以包含抽象方法和具体方法。

```java
public abstract class Employee {
    protected String name;
    protected String id;
    protected double baseSalary;
    
    public Employee(String name, String id, double baseSalary) {
        this.name = name;
        this.id = id;
        this.baseSalary = baseSalary;
    }
    
    // 抽象方法（子类必须实现）
    public abstract double calculateSalary();
    
    // 具体方法
    public void displayInfo() {
        System.out.println("姓名: " + name + ", 工号: " + id);
    }
}

// 全职员工
public class FullTimeEmployee extends Employee {
    private double bonus;
    
    public FullTimeEmployee(String name, String id, double baseSalary, double bonus) {
        super(name, id, baseSalary);
        this.bonus = bonus;
    }
    
    @Override
    public double calculateSalary() {
        return baseSalary + bonus;
    }
}

// 兼职员工
public class PartTimeEmployee extends Employee {
    private int hoursWorked;
    private double hourlyRate;
    
    public PartTimeEmployee(String name, String id, int hoursWorked, double hourlyRate) {
        super(name, id, 0);
        this.hoursWorked = hoursWorked;
        this.hourlyRate = hourlyRate;
    }
    
    @Override
    public double calculateSalary() {
        return hoursWorked * hourlyRate;
    }
}
```

## 接口

接口是完全抽象的类型，定义了一组方法规范。

### 接口定义和实现

```java
// 接口定义
public interface Flyable {
    // 接口中的方法默认是 public abstract
    void fly();
    void land();
    
    // Java 8+ 可以有默认方法
    default void takeOff() {
        System.out.println("起飞中...");
    }
    
    // 静态方法
    static void checkWeather() {
        System.out.println("检查天气状况");
    }
}

public interface Swimmable {
    void swim();
}

// 实现接口
public class Bird implements Flyable {
    private String name;
    
    public Bird(String name) {
        this.name = name;
    }
    
    @Override
    public void fly() {
        System.out.println(name + " 在天空中飞翔");
    }
    
    @Override
    public void land() {
        System.out.println(name + " 降落了");
    }
}

// 实现多个接口
public class Duck implements Flyable, Swimmable {
    private String name;
    
    public Duck(String name) {
        this.name = name;
    }
    
    @Override
    public void fly() {
        System.out.println(name + " 飞起来了");
    }
    
    @Override
    public void land() {
        System.out.println(name + " 着陆了");
    }
    
    @Override
    public void swim() {
        System.out.println(name + " 在水中游泳");
    }
}
```

### 接口的使用

```java
public class InterfaceExample {
    public static void main(String[] args) {
        Bird bird = new Bird("麻雀");
        Duck duck = new Duck("鸭子");
        
        bird.takeOff();  // 使用默认方法
        bird.fly();
        bird.land();
        
        duck.fly();
        duck.swim();
        
        // 多态
        Flyable flyable = new Duck("野鸭");
        flyable.fly();
        
        // 静态方法调用
        Flyable.checkWeather();
    }
}
```

## 抽象类 vs 接口

| 特性 | 抽象类 | 接口 |
|------|--------|------|
| 实例化 | 不能实例化 | 不能实例化 |
| 构造方法 | 可以有 | 不能有 |
| 成员变量 | 可以有任意类型 | 只能有 public static final |
| 方法 | 可以有抽象和具体方法 | 主要是抽象方法，Java 8+ 可以有默认和静态方法 |
| 继承 | 单继承 | 多实现 |
| 使用场景 | 有共同实现的父类 | 定义规范和能力 |

```java
// 使用抽象类：有共同实现
public abstract class Vehicle {
    protected String brand;
    
    public Vehicle(String brand) {
        this.brand = brand;
    }
    
    // 具体方法
    public void start() {
        System.out.println(brand + " 启动了");
    }
    
    // 抽象方法
    public abstract void run();
}

// 使用接口：定义能力
public interface Chargeable {
    void charge();
}

// 电动车既继承又实现
public class ElectricCar extends Vehicle implements Chargeable {
    public ElectricCar(String brand) {
        super(brand);
    }
    
    @Override
    public void run() {
        System.out.println(brand + " 电动车正在行驶");
    }
    
    @Override
    public void charge() {
        System.out.println(brand + " 正在充电");
    }
}
```

## 内部类

### 成员内部类

```java
public class Outer {
    private String outerField = "外部类字段";
    
    // 成员内部类
    public class Inner {
        private String innerField = "内部类字段";
        
        public void display() {
            System.out.println(outerField);  // 可以访问外部类成员
            System.out.println(innerField);
        }
    }
    
    public void test() {
        Inner inner = new Inner();
        inner.display();
    }
}

// 使用
Outer outer = new Outer();
Outer.Inner inner = outer.new Inner();
inner.display();
```

### 静态内部类

```java
public class Outer {
    private static String staticField = "静态字段";
    
    // 静态内部类
    public static class StaticInner {
        public void display() {
            System.out.println(staticField);  // 只能访问外部类的静态成员
        }
    }
}

// 使用
Outer.StaticInner staticInner = new Outer.StaticInner();
staticInner.display();
```

### 局部内部类

```java
public class Outer {
    public void method() {
        final String localVar = "局部变量";
        
        // 局部内部类
        class LocalInner {
            public void display() {
                System.out.println(localVar);  // 可以访问 final 局部变量
            }
        }
        
        LocalInner inner = new LocalInner();
        inner.display();
    }
}
```

### 匿名内部类

```java
public class AnonymousExample {
    public static void main(String[] args) {
        // 匿名内部类实现接口
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                System.out.println("匿名内部类执行");
            }
        };
        runnable.run();
        
        // 使用 Lambda 表达式（Java 8+）
        Runnable runnable2 = () -> System.out.println("Lambda 表达式执行");
        runnable2.run();
    }
}
```

## 最佳实践

### 1. 优先使用组合而非继承

```java
// 不推荐：过度使用继承
public class Dog extends Animal {
    // ...
}

// 推荐：使用组合
public class Dog {
    private Animal animal;  // 组合
    
    public Dog(Animal animal) {
        this.animal = animal;
    }
}
```

### 2. 遵循单一职责原则

```java
// 不好：一个类做太多事情
public class UserManager {
    public void createUser() {}
    public void deleteUser() {}
    public void sendEmail() {}
    public void generateReport() {}
}

// 好：职责分离
public class UserService {
    public void createUser() {}
    public void deleteUser() {}
}

public class EmailService {
    public void sendEmail() {}
}

public class ReportService {
    public void generateReport() {}
}
```

### 3. 使用接口面向抽象编程

```java
// 定义接口
public interface UserRepository {
    User findById(Long id);
    void save(User user);
}

// 实现类
public class UserRepositoryImpl implements UserRepository {
    @Override
    public User findById(Long id) {
        // 实现
        return null;
    }
    
    @Override
    public void save(User user) {
        // 实现
    }
}

// 使用接口类型
public class UserService {
    private UserRepository userRepository;  // 依赖接口而非实现
    
    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }
}
```

## 总结

本文介绍了 Java 面向对象编程的核心概念：

- ✅ 类和对象的定义与使用
- ✅ 封装：隐藏实现细节
- ✅ 继承：代码复用和扩展
- ✅ 多态：方法重载和重写
- ✅ 抽象类和接口的使用
- ✅ 内部类的各种形式

掌握这些概念后，可以继续学习 [集合框架](./collections) 和 [异常处理](./exception-handling)。
