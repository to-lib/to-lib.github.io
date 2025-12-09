---
sidebar_position: 11
title: 反射与注解
---

# 反射与注解

反射（Reflection）是 Java 的一个强大特性，允许程序在运行时检查和操作类、方法、字段等。注解（Annotation）则为反射提供了元数据。本文介绍反射 API 和自定义注解的深度应用。

## 反射基础

### 获取 Class 对象

Class 对象代表了 Java 中的类。有三种获取方式：

```java
public class GetClassExample {
    public static void main(String[] args) throws ClassNotFoundException {
        // 方式1：通过 .class 获取（推荐）
        Class<?> clazz1 = String.class;
        
        // 方式2：通过 object.getClass() 获取
        String str = "Hello";
        Class<?> clazz2 = str.getClass();
        
        // 方式3：通过 Class.forName() 获取
        Class<?> clazz3 = Class.forName("java.lang.String");
        
        // 三个 Class 对象都是同一个
        System.out.println(clazz1 == clazz2);  // true
        System.out.println(clazz2 == clazz3);  // true
    }
}
```

### 获取类的信息

```java
import java.lang.reflect.*;

public class ClassInspection {
    public static void main(String[] args) {
        Class<?> clazz = String.class;
        
        // 获取类名
        System.out.println("类名: " + clazz.getSimpleName());
        System.out.println("全限定名: " + clazz.getName());
        
        // 获取父类
        Class<?> superClass = clazz.getSuperclass();
        System.out.println("父类: " + superClass.getSimpleName());
        
        // 获取实现的接口
        Class<?>[] interfaces = clazz.getInterfaces();
        for (Class<?> iface : interfaces) {
            System.out.println("实现的接口: " + iface.getSimpleName());
        }
        
        // 获取访问修饰符
        int modifiers = clazz.getModifiers();
        System.out.println("是否 public: " + Modifier.isPublic(modifiers));
        System.out.println("是否 final: " + Modifier.isFinal(modifiers));
    }
}
```

## 反射 Field（字段）

### 获取和设置字段值

```java
import java.lang.reflect.Field;

public class Person {
    private String name;
    private int age;
    
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    public String getName() {
        return name;
    }
    
    public int getAge() {
        return age;
    }
}

public class FieldReflectionExample {
    public static void main(String[] args) throws Exception {
        Person person = new Person("张三", 25);
        Class<?> clazz = Person.class;
        
        // 获取所有字段
        Field[] fields = clazz.getDeclaredFields();
        for (Field field : fields) {
            System.out.println("字段名: " + field.getName());
            System.out.println("字段类型: " + field.getType().getSimpleName());
        }
        
        // 获取特定字段
        Field nameField = clazz.getDeclaredField("name");
        
        // 设置 accessible（跳过 private 检查）
        nameField.setAccessible(true);
        
        // 读取字段值
        Object value = nameField.get(person);
        System.out.println("name 字段值: " + value);
        
        // 修改字段值
        nameField.set(person, "李四");
        System.out.println("修改后的名字: " + person.getName());
        
        // 获取 age 字段
        Field ageField = clazz.getDeclaredField("age");
        ageField.setAccessible(true);
        ageField.set(person, 30);
        System.out.println("修改后的年龄: " + person.getAge());
    }
}
```

## 反射 Method（方法）

### 获取和调用方法

```java
import java.lang.reflect.Method;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    public int subtract(int a, int b) {
        return a - b;
    }
    
    private int multiply(int a, int b) {
        return a * b;
    }
}

public class MethodReflectionExample {
    public static void main(String[] args) throws Exception {
        Calculator calculator = new Calculator();
        Class<?> clazz = Calculator.class;
        
        // 获取所有公共方法
        Method[] methods = clazz.getDeclaredMethods();
        for (Method method : methods) {
            System.out.println("方法: " + method.getName());
        }
        
        // 获取特定方法：add(int, int)
        Method addMethod = clazz.getMethod("add", int.class, int.class);
        System.out.println("方法名: " + addMethod.getName());
        System.out.println("返回类型: " + addMethod.getReturnType().getSimpleName());
        
        // 调用方法
        int result = (int) addMethod.invoke(calculator, 5, 3);
        System.out.println("5 + 3 = " + result);
        
        // 调用 subtract 方法
        Method subtractMethod = clazz.getMethod("subtract", int.class, int.class);
        result = (int) subtractMethod.invoke(calculator, 10, 4);
        System.out.println("10 - 4 = " + result);
        
        // 调用 private 方法
        Method multiplyMethod = clazz.getDeclaredMethod("multiply", int.class, int.class);
        multiplyMethod.setAccessible(true);
        result = (int) multiplyMethod.invoke(calculator, 6, 7);
        System.out.println("6 * 7 = " + result);
    }
}
```

## 反射 Constructor（构造方法）

### 创建对象实例

```java
import java.lang.reflect.Constructor;

public class User {
    private String username;
    private String password;
    
    // 默认构造方法
    public User() {
        this.username = "guest";
        this.password = "123456";
    }
    
    // 带参构造方法
    public User(String username, String password) {
        this.username = username;
        this.password = password;
    }
    
    @Override
    public String toString() {
        return "User{" + "username='" + username + '\'' + ", password='" + password + '\'' + '}';
    }
}

public class ConstructorReflectionExample {
    public static void main(String[] args) throws Exception {
        Class<?> clazz = User.class;
        
        // 获取所有构造方法
        Constructor<?>[] constructors = clazz.getDeclaredConstructors();
        for (Constructor<?> constructor : constructors) {
            System.out.println("构造方法: " + constructor);
        }
        
        // 获取无参构造方法并创建对象
        Constructor<?> noArgConstructor = clazz.getConstructor();
        Object user1 = noArgConstructor.newInstance();
        System.out.println("user1: " + user1);
        
        // 获取带参构造方法并创建对象
        Constructor<?> argConstructor = clazz.getConstructor(String.class, String.class);
        Object user2 = argConstructor.newInstance("admin", "password123");
        System.out.println("user2: " + user2);
        
        // 简便方式：直接 newInstance()
        User user3 = (User) clazz.getDeclaredConstructor().newInstance();
        System.out.println("user3: " + user3);
    }
}
```

## 动态代理

### JDK 动态代理

```java
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;

// 定义接口
public interface UserService {
    void addUser(String name);
    void deleteUser(int id);
    void updateUser(String name);
}

// 实现接口
public class UserServiceImpl implements UserService {
    @Override
    public void addUser(String name) {
        System.out.println("添加用户: " + name);
    }
    
    @Override
    public void deleteUser(int id) {
        System.out.println("删除用户: " + id);
    }
    
    @Override
    public void updateUser(String name) {
        System.out.println("更新用户: " + name);
    }
}

// 创建代理处理器
public class UserServiceProxy implements InvocationHandler {
    private UserService target;
    
    public UserServiceProxy(UserService target) {
        this.target = target;
    }
    
    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        // 调用前处理
        System.out.println("[前置处理] 执行方法: " + method.getName());
        long startTime = System.currentTimeMillis();
        
        // 调用实际方法
        Object result = method.invoke(target, args);
        
        // 调用后处理
        long endTime = System.currentTimeMillis();
        System.out.println("[后置处理] 执行耗时: " + (endTime - startTime) + "ms");
        
        return result;
    }
}

// 使用代理
public class ProxyExample {
    public static void main(String[] args) {
        UserService target = new UserServiceImpl();
        
        // 创建代理对象
        UserService proxy = (UserService) Proxy.newProxyInstance(
            UserService.class.getClassLoader(),
            new Class[]{UserService.class},
            new UserServiceProxy(target)
        );
        
        // 调用代理方法
        proxy.addUser("张三");
        proxy.deleteUser(1);
        proxy.updateUser("李四");
    }
}
```

## 自定义注解的深度应用

### 创建参数验证注解

```java
import java.lang.annotation.*;
import java.lang.reflect.Method;
import java.lang.reflect.Parameter;

// 定义非空验证注解
@Target(ElementType.PARAMETER)
@Retention(RetentionPolicy.RUNTIME)
public @interface NotNull {
    String message() default "参数不能为空";
}

// 定义长度验证注解
@Target(ElementType.PARAMETER)
@Retention(RetentionPolicy.RUNTIME)
public @interface Length {
    int min() default 0;
    int max() default Integer.MAX_VALUE;
    String message() default "参数长度不符合要求";
}

// 使用注解
public class UserValidator {
    public void register(@NotNull(message = "用户名不能为空") String username,
                       @Length(min = 6, max = 20, message = "密码长度必须在6-20之间") String password) {
        System.out.println("注册用户: " + username);
    }
}

// 注解处理器
public class ValidationProcessor {
    public static void validateParameters(Object obj, Method method, Object[] args) throws Exception {
        Parameter[] parameters = method.getParameters();
        
        for (int i = 0; i < parameters.length; i++) {
            Parameter param = parameters[i];
            Object arg = args[i];
            
            // 检查 @NotNull 注解
            NotNull notNullAnnotation = param.getAnnotation(NotNull.class);
            if (notNullAnnotation != null && arg == null) {
                throw new IllegalArgumentException(notNullAnnotation.message());
            }
            
            // 检查 @Length 注解
            Length lengthAnnotation = param.getAnnotation(Length.class);
            if (lengthAnnotation != null && arg instanceof String) {
                String str = (String) arg;
                if (str.length() < lengthAnnotation.min() || str.length() > lengthAnnotation.max()) {
                    throw new IllegalArgumentException(lengthAnnotation.message());
                }
            }
        }
    }
}
```

### 创建持久化注解

```java
import java.lang.annotation.*;
import java.lang.reflect.Field;

// 标记实体类
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface Entity {
    String tableName();
}

// 标记字段映射到表列
@Target(ElementType.FIELD)
@Retention(RetentionPolicy.RUNTIME)
public @interface Column {
    String name();
    boolean primaryKey() default false;
    boolean nullable() default true;
}

// 使用注解定义实体
@Entity(tableName = "users")
public class User {
    @Column(name = "id", primaryKey = true)
    private int id;
    
    @Column(name = "username", nullable = false)
    private String username;
    
    @Column(name = "email")
    private String email;
    
    // 省略 getter/setter
}

// SQL 生成器
public class SQLGenerator {
    public static String generateCreateTableSQL(Class<?> clazz) {
        Entity entity = clazz.getAnnotation(Entity.class);
        if (entity == null) {
            throw new IllegalArgumentException("类必须被 @Entity 注解标记");
        }
        
        StringBuilder sql = new StringBuilder();
        sql.append("CREATE TABLE ").append(entity.tableName()).append(" (\n");
        
        Field[] fields = clazz.getDeclaredFields();
        for (int i = 0; i < fields.length; i++) {
            Field field = fields[i];
            Column column = field.getAnnotation(Column.class);
            
            if (column != null) {
                sql.append("  ").append(column.name()).append(" ");
                
                // 根据 Java 类型推断 SQL 类型
                if (field.getType() == int.class) {
                    sql.append("INT");
                } else if (field.getType() == String.class) {
                    sql.append("VARCHAR(255)");
                }
                
                if (column.primaryKey()) {
                    sql.append(" PRIMARY KEY");
                }
                
                if (!column.nullable()) {
                    sql.append(" NOT NULL");
                }
                
                if (i < fields.length - 1) {
                    sql.append(",");
                }
                sql.append("\n");
            }
        }
        
        sql.append(");");
        return sql.toString();
    }
}

// 使用
public class ORM {
    public static void main(String[] args) {
        String sql = SQLGenerator.generateCreateTableSQL(User.class);
        System.out.println(sql);
        // 输出：
        // CREATE TABLE users (
        //   id INT PRIMARY KEY,
        //   username VARCHAR(255) NOT NULL,
        //   email VARCHAR(255)
        // );
    }
}
```

### 创建事件处理注解

```java
import java.lang.annotation.*;
import java.lang.reflect.Method;

// 事件注解
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface EventListener {
    Class<?> eventType();
}

// 定义事件
public class UserRegisteredEvent {
    private String username;
    private long timestamp;
    
    public UserRegisteredEvent(String username) {
        this.username = username;
        this.timestamp = System.currentTimeMillis();
    }
    
    public String getUsername() {
        return username;
    }
}

// 事件处理器
public class UserEventHandler {
    @EventListener(eventType = UserRegisteredEvent.class)
    public void onUserRegistered(UserRegisteredEvent event) {
        System.out.println("用户已注册: " + event.getUsername());
        // 发送欢迎邮件等操作
    }
    
    @EventListener(eventType = UserRegisteredEvent.class)
    public void logUserRegistration(UserRegisteredEvent event) {
        System.out.println("[日志] 用户注册时间: " + event.getUsername());
    }
}

// 事件总线
public class EventBus {
    private Map<Class<?>, List<Method>> eventHandlers = new HashMap<>();
    
    public void register(Object handler) {
        Class<?> handlerClass = handler.getClass();
        
        for (Method method : handlerClass.getDeclaredMethods()) {
            EventListener listener = method.getAnnotation(EventListener.class);
            if (listener != null) {
                Class<?> eventType = listener.eventType();
                eventHandlers.computeIfAbsent(eventType, k -> new ArrayList<>()).add(method);
            }
        }
    }
    
    public void publish(Object event) {
        Class<?> eventType = event.getClass();
        List<Method> handlers = eventHandlers.getOrDefault(eventType, new ArrayList<>());
        
        for (Method method : handlers) {
            try {
                method.invoke(null, event);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
```

## 反射性能优化

```java
public class ReflectionPerformance {
    private static final int ITERATIONS = 1000000;
    
    public static void main(String[] args) throws Exception {
        // 直接调用
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < ITERATIONS; i++) {
            String str = "test";
            int len = str.length();
        }
        long directTime = System.currentTimeMillis() - startTime;
        
        // 反射调用
        startTime = System.currentTimeMillis();
        Method lengthMethod = String.class.getMethod("length");
        String str = "test";
        for (int i = 0; i < ITERATIONS; i++) {
            lengthMethod.invoke(str);
        }
        long reflectionTime = System.currentTimeMillis() - startTime;
        
        System.out.println("直接调用耗时: " + directTime + "ms");
        System.out.println("反射调用耗时: " + reflectionTime + "ms");
        System.out.println("性能差异: " + (reflectionTime / directTime) + " 倍");
        
        // 性能优化建议：
        // 1. 缓存 Method、Field、Constructor 对象
        // 2. 使用 MethodHandle（Java 7+）
        // 3. 在关键循环中避免反射
    }
}
```

## 反射最佳实践

### 1. 缓存反射对象

```java
public class ReflectionCache {
    private static final Map<String, Method> methodCache = new HashMap<>();
    private static final Map<String, Field> fieldCache = new HashMap<>();
    
    public static Method getMethod(Class<?> clazz, String methodName, Class<?>... paramTypes) {
        String key = clazz.getName() + "." + methodName;
        return methodCache.computeIfAbsent(key, k -> {
            try {
                return clazz.getMethod(methodName, paramTypes);
            } catch (NoSuchMethodException e) {
                throw new RuntimeException(e);
            }
        });
    }
    
    public static Field getField(Class<?> clazz, String fieldName) {
        String key = clazz.getName() + "." + fieldName;
        return fieldCache.computeIfAbsent(key, k -> {
            try {
                Field field = clazz.getDeclaredField(fieldName);
                field.setAccessible(true);
                return field;
            } catch (NoSuchFieldException e) {
                throw new RuntimeException(e);
            }
        });
    }
}
```

### 2. 异常处理

```java
public class ReflectionExceptionHandling {
    public static void invokeMethod(Object obj, String methodName) {
        try {
            Method method = obj.getClass().getMethod(methodName);
            method.invoke(obj);
        } catch (NoSuchMethodException e) {
            System.err.println("方法不存在: " + methodName);
        } catch (IllegalAccessException e) {
            System.err.println("无权访问方法: " + methodName);
        } catch (InvocationTargetException e) {
            System.err.println("方法执行异常: " + e.getCause());
        }
    }
}
```

## 总结

本文介绍了 Java 反射和注解的深度应用：

- ✅ 反射 API：获取 Class、Field、Method、Constructor
- ✅ 动态代理：JDK 动态代理的实现和应用
- ✅ 自定义注解：参数验证、持久化、事件处理
- ✅ 性能优化：缓存反射对象、选择合适的工具
- ✅ 最佳实践：异常处理、资源管理

反射和注解是构建框架和实现 AOP 的基础，但要注意性能和安全问题。
