---
sidebar_position: 22
title: Java 常见问题
---

# Java 常见问题（FAQ）

本文汇总了 Java 编程中的常见问题和解决方案。

## String 相关

### Q1: String, StringBuilder, StringBuffer 的区别？

**答：**

- **String**：不可变，线程安全，适合不变的字符串
- **StringBuilder**：可变，非线程安全，性能最好，适合单线程字符串拼接
- **StringBuffer**：可变，线程安全，性能较 StringBuilder 差，适合多线程字符串拼接

```java
// String：每次拼接都创建新对象
String str = "Hello";
str += " World";  // 创建了新的 String 对象

// StringBuilder：在同一对象上修改
StringBuilder sb = new StringBuilder("Hello");
sb.append(" World");  // 不创建新对象
```

### Q2: `==` 和 `equals()` 的区别？

**答：**

- `==` 比较的是对象的引用（内存地址）
- `equals()` 比较的是对象的内容

```java
String s1 = new String("Hello");
String s2 = new String("Hello");
String s3 = "Hello";
String s4 = "Hello";

System.out.println(s1 == s2);       // false（不同对象）
System.out.println(s1.equals(s2));  // true（内容相同）
System.out.println(s3 == s4);       // true（字符串常量池）
```

### Q3: 为什么 String 是不可变的？

**答：**

1. **安全性**：字符串常用于参数传递，不可变保证了安全
2. **线程安全**：不可变对象天然线程安全
3. **效率**：可以缓存 hashCode，字符串池可以共享
4. **类加载**：类名是字符串，不可变保证类加载安全

### Q4: String 的 `intern()` 方法是什么？

**答：** `intern()` 方法将字符串放入字符串常量池，如果池中已存在，返回池中的引用。

```java
String s1 = new String("Hello");
String s2 = s1.intern();
String s3 = "Hello";

System.out.println(s2 == s3);  // true（都来自常量池）
```

## 集合框架

### Q5: ArrayList 和 LinkedList 的区别？

**答：**
| 特性 | ArrayList | LinkedList |
|--------------|------------------|--------------------|
| 底层实现 | 动态数组 | 双向链表 |
| 随机访问 | O(1) 快 | O(n) 慢 |
| 插入删除 | O(n) 慢 | O(1) 快（已定位） |
| 内存占用 | 较小 | 较大（存储指针） |
| 适用场景 | 频繁读取 | 频繁插入删除 |

```java
// ArrayList：适合随机访问
List<String> arrayList = new ArrayList<>();
String item = arrayList.get(100);  // 快速

// LinkedList：适合频繁插入删除
List<String> linkedList = new LinkedList<>();
linkedList.add(0, "first");  // 快速（在头部插入）
```

### Q6: HashMap 的工作原理？

**答：** HashMap 使用哈希表实现，通过 key 的 hashCode 计算存储位置。

1. 计算 key 的 hashCode
2. 通过哈希函数计算数组索引
3. 如果有冲突，使用链表或红黑树（JDK 8+）解决

```java
// 存储过程
int index = (n - 1) & hash(key);  // 计算索引
table[index] = new Node(key, value, next);

// JDK 8 优化：链表长度超过 8 转为红黑树
if (binCount >= TREEIFY_THRESHOLD - 1) {
    treeifyBin(tab, hash);
}
```

### Q7: HashMap 和 ConcurrentHashMap 的区别？

**答：**

- **HashMap**：非线程安全，性能好，适合单线程
- **ConcurrentHashMap**：线程安全，使用分段锁（JDK 7）或 CAS（JDK 8+），适合多线程

```java
// HashMap：线程不安全
Map<String, Integer> hashMap = new HashMap<>();

// ConcurrentHashMap：线程安全
Map<String, Integer> concurrentMap = new ConcurrentHashMap<>();
```

### Q8: HashSet 如何保证元素不重复？

**答：** HashSet 底层使用 HashMap 实现，将元素作为 HashMap 的 key，value 是固定的 Object。通过 key 的唯一性保证元素不重复。

```java
// HashSet 内部实现
private transient HashMap<E, Object> map;
private static final Object PRESENT = new Object();

public boolean add(E e) {
    return map.put(e, PRESENT) == null;
}
```

## 多线程

### Q9: 创建线程的几种方式？

**答：** 主要有 4 种方式：

```java
// 1. 继承 Thread 类
class MyThread extends Thread {
    public void run() {
        System.out.println("线程运行");
    }
}
new MyThread().start();

// 2. 实现 Runnable 接口
new Thread(() -> System.out.println("线程运行")).start();

// 3. 实现 Callable 接口（有返回值）
FutureTask<String> task = new FutureTask<>(() -> "结果");
new Thread(task).start();
String result = task.get();

// 4. 使用线程池
ExecutorService executor = Executors.newFixedThreadPool(10);
executor.submit(() -> System.out.println("线程运行"));
```

### Q10: `sleep()` 和 `wait()` 的区别？

**答：**
| 特性 | sleep() | wait() |
|--------------|------------------|--------------------|
| 所属类 | Thread | Object |
| 释放锁 | 不释放 | 释放 |
| 使用场景 | 暂停执行 | 线程间通信 |
| 唤醒方式 | 自动 | notify/notifyAll |

```java
// sleep：不释放锁
synchronized (lock) {
    Thread.sleep(1000);  // 持有锁睡眠
}

// wait：释放锁
synchronized (lock) {
    lock.wait();  // 释放锁，等待通知
}
```

### Q11: `synchronized` 和 `Lock` 的区别？

**答：**
| 特性 | synchronized | Lock |
|--------------|------------------|--------------------|
| 类型 | 关键字 | 接口 |
| 锁释放 | 自动 | 手动（finally） |
| 灵活性 | 低 | 高（可中断、超时） |
| 性能 | 较低 | 较高 |

```java
// synchronized：自动释放
synchronized (lock) {
    // 临界区
}

// Lock：手动释放
Lock lock = new ReentrantLock();
lock.lock();
try {
    // 临界区
} finally {
    lock.unlock();  // 必须手动释放
}
```

### Q12: volatile 关键字的作用？

**答：** volatile 保证了变量的可见性和有序性，但不保证原子性。

```java
// 可见性：修改对其他线程立即可见
private volatile boolean flag = false;

// 禁止指令重排序
private volatile Singleton instance;
```

## 内存和性能

### Q13: Java 内存泄漏的常见场景？

**答：**

1. **集合类**：长期存活的集合持有不再使用的对象
2. **监听器**：未注销的事件监听器
3. **内部类**：非静态内部类持有外部类引用
4. **ThreadLocal**：未清理的 ThreadLocal 变量

```java
// ❌ 内存泄漏示例
public class MemoryLeak {
    private static List<Object> list = new ArrayList<>();

    public void addObject() {
        list.add(new Object());  // 对象永远不会被移除
    }
}

// ✅ 及时清理
public void cleanup() {
    list.clear();
}
```

### Q14: 如何避免 NullPointerException？

**答：**

1. 使用 Optional
2. 添加 null 检查
3. 使用注解（@NonNull、@Nullable）
4. 使用默认值

```java
// 1. Optional
Optional<String> opt = Optional.ofNullable(getValue());
String result = opt.orElse("默认值");

// 2. null 检查
if (str != null && !str.isEmpty()) {
    // 使用 str
}

// 3. Objects 工具类
Objects.requireNonNull(value, "值不能为 null");

// 4. 默认值
String name = user != null ? user.getName() : "未知";
```

### Q15: Java 的垃圾回收算法有哪些？

**答：**

1. **标记-清除**：标记存活对象，清除未标记对象
2. **复制算法**：将存活对象复制到另一块内存
3. **标记-整理**：标记存活对象，整理到一端
4. **分代收集**：根据对象存活时间分别处理

```
年轻代（Young Generation）：
- Eden 区：新对象分配
- Survivor 区：存活对象复制

老年代（Old Generation）：
- 长期存活的对象
```

## 异常处理

### Q16: Checked Exception 和 Unchecked Exception 的区别？

**答：**

- **Checked Exception**：编译时异常，必须处理（IOException、SQLException）
- **Unchecked Exception**：运行时异常，可以不处理（NullPointerException、IllegalArgumentException）

```java
// Checked Exception：必须处理
public void readFile() throws IOException {
    FileReader reader = new FileReader("file.txt");
}

// Unchecked Exception：可以不处理
public void divide(int a, int b) {
    return a / b;  // 可能抛出 ArithmeticException
}
```

### Q17: finally 块一定会执行吗？

**答：** 几乎都会执行，除非：

1. System.exit() 退出 JVM
2. 守护线程中所有非守护线程结束
3. JVM 崩溃

```java
try {
    return 1;
} finally {
    System.out.println("finally 执行");  // 会执行
    // 注意：不要在 finally 中 return，会覆盖 try 中的返回值
}
```

## JVM 和类加载

### Q18: JVM 内存结构是什么？

**答：**

```
堆（Heap）：
- 年轻代：Eden + Survivor
- 老年代：Old Generation

非堆（Non-Heap）：
- 方法区/元空间：类信息、常量
- 直接内存：NIO 使用

线程私有：
- 虚拟机栈：方法调用栈
- 本地方法栈：Native 方法
- 程序计数器：当前指令地址
```

### Q19: 什么是类加载机制？

**答：** 类加载分为 5 个阶段：

1. **加载**：读取 .class 文件
2. **验证**：验证字节码
3. **准备**：分配内存，设置默认值
4. **解析**：符号引用转为直接引用
5. **初始化**：执行类构造器

```java
// 类加载时机
1. new 对象
2. 访问静态变量或方法
3. 反射调用
4. 初始化子类
5. 启动类（main方法所在类）
```

### Q20: 双亲委派模型是什么？

**答：** 类加载器在加载类时，先委托父加载器加载，父加载器无法加载时才自己加载。

```
Bootstrap ClassLoader（启动类加载器）
        ↑
Extension ClassLoader（扩展类加载器）
        ↑
Application ClassLoader（应用类加载器）
        ↑
Custom ClassLoader（自定义类加载器）
```

好处：

1. 避免类的重复加载
2. 保护核心类不被篡改

## 其他常见问题

### Q21: `int` 和 `Integer` 的区别？

**答：**

- `int`：基本类型，默认值 0
- `Integer`：包装类，默认值 null，提供了实用方法

```java
int a = 10;           // 基本类型
Integer b = 10;       // 自动装箱
int c = b;            // 自动拆箱

// Integer 缓存：-128 到 127
Integer x = 127;
Integer y = 127;
System.out.println(x == y);  // true（来自缓存）

Integer m = 128;
Integer n = 128;
System.out.println(m == n);  // false（不在缓存范围）
```

### Q22: 什么是序列化？

**答：** 序列化是将对象转换为字节流的过程，反序列化是字节流转换为对象。

```java
// 实现 Serializable 接口
public class User implements Serializable {
    private static final long serialVersionUID = 1L;
    private String name;
    private transient String password;  // transient：不序列化
}

// 序列化
ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream("user.ser"));
out.writeObject(user);

// 反序列化
ObjectInputStream in = new ObjectInputStream(new FileInputStream("user.ser"));
User user = (User) in.readObject();
```

### Q23: 什么是反射？有什么用？

**答：** 反射是在运行时动态获取类信息和操作对象的能力。

```java
// 获取 Class 对象
Class<?> clazz = Class.forName("com.example.User");

// 创建对象
Object obj = clazz.newInstance();

// 获取方法
Method method = clazz.getMethod("getName");
Object result = method.invoke(obj);

// 获取字段
Field field = clazz.getDeclaredField("name");
field.setAccessible(true);  // 访问私有字段
field.set(obj, "张三");
```

用途：

- 框架开发（Spring、Hibernate）
- 动态代理
- 注解处理

### Q24: `hashCode()` 和 `equals()` 的关系？

**答：**

1. 如果两个对象 `equals()` 返回 true，它们的 `hashCode()` 必须相同
2. 如果 `hashCode()` 相同，`equals()` 不一定返回 true
3. 重写 `equals()` 必须重写 `hashCode()`

```java
@Override
public boolean equals(Object obj) {
    if (this == obj) return true;
    if (obj == null || getClass() != obj.getClass()) return false;
    User user = (User) obj;
    return Objects.equals(id, user.id);
}

@Override
public int hashCode() {
    return Objects.hash(id);
}
```

## 总结

这些常见问题涵盖了 Java 编程的核心概念，理解这些问题对于日常开发和面试都非常重要。建议：

1. 深入理解每个概念的原理
2. 通过实际代码验证
3. 关注最佳实践和常见陷阱
4. 定期复习和更新知识

如果有其他问题，欢迎查阅相关专题文档获取更详细的说明。
