---
sidebar_position: 100
title: Java 面试题精选
---

# Java 面试题精选

> [!TIP]
> 本文精选了 50+ 道常见 Java 面试题，按主题和难度分级。建议结合相关章节深入学习。

## 🎯 基础语法（初级）

### 1. Java 的基本数据类型有哪些？占用多少字节？

**答案要点：**

- 8 种基本数据类型：
  - 整型：`byte`(1 字节)、`short`(2 字节)、`int`(4 字节)、`long`(8 字节)
  - 浮点型：`float`(4 字节)、`double`(8 字节)
  - 字符型：`char`(2 字节)
  - 布尔型：`boolean`(理论上 1 位，实际 JVM 实现可能占 1 字节)

**延伸：** 参考 [基础语法 - 数据类型](/docs/java/basic-syntax#数据类型)

---

### 2. `==` 和 `equals()` 的区别？

**答案要点：**

- `==` 比较基本类型时比较值，比较引用类型时比较内存地址
- `equals()` 是 Object 类的方法，默认比较内存地址，可以被重写来比较内容
- String、Integer 等类已重写 `equals()` 方法来比较值

**示例：**

```java
String s1 = new String("hello");
String s2 = new String("hello");
System.out.println(s1 == s2);        // false (不同对象)
System.out.println(s1.equals(s2));   // true (内容相同)
```

**延伸：** 参考 [基础语法 - 字符串创建和操作](/docs/java/basic-syntax#字符串创建和操作)

---

### 3. String、StringBuilder、StringBuffer 的区别？

**答案要点：**

- **String**: 不可变类，线程安全，适合少量字符串操作
- **StringBuilder**: 可变类，非线程安全，性能高，适合单线程大量拼接
- **StringBuffer**: 可变类，线程安全（方法加 synchronized），性能较 StringBuilder 低

**性能对比：**

```java
// 不推荐：频繁拼接会创建大量String对象
String result = "";
for (int i = 0; i < 10000; i++) {
    result += i;  // 每次循环创建新String对象
}

// 推荐：单线程用StringBuilder
StringBuilder sb = new StringBuilder();
for (int i = 0; i < 10000; i++) {
    sb.append(i);  // 在原对象上修改
}
```

**延伸：** 参考 [基础语法 - 字符串详解](/docs/java/basic-syntax#字符串详解)

---

## 🎯 面向对象（中级）

### 4. Java 面向对象的三大特性是什么？

**答案要点：**

1. **封装（Encapsulation）**: 隐藏对象内部细节，通过 public 方法访问
2. **继承（Inheritance）**: 子类继承父类的属性和方法，实现代码复用
3. **多态（Polymorphism）**: 同一接口不同实现，包括编译时多态（重载）和运行时多态（重写）

**多态示例：**

```java
Animal animal = new Dog();  // 父类引用指向子类对象
animal.makeSound();         // 运行时调用Dog的实现
```

**延伸：** 参考 [面向对象编程](/docs/java/oop)

---

### 5. 抽象类和接口的区别？

**答案要点：**

| 特性       | 抽象类               | 接口                                  |
| ---------- | -------------------- | ------------------------------------- |
| 关键字     | `abstract class`     | `interface`                           |
| 继承       | 单继承               | 多实现                                |
| 方法       | 可以有抽象和具体方法 | Java 8+ 可以有 default 和 static 方法 |
| 成员变量   | 可以有实例变量       | 只能有 public static final 常量       |
| 构造方法   | 可以有构造方法       | 不能有构造方法                        |
| 访问修饰符 | 可以有各种修饰符     | 方法默认 public abstract              |

**使用场景：**

- 抽象类：表示"是一个"（is-a）关系，有共同实现
- 接口：表示"具有某能力"（can-do）关系，定义规范

**延伸：** 参考 [面向对象 - 抽象类与接口](/docs/java/oop#抽象类)

---

### 6. 重载（Overload）和重写（Override）的区别？

**答案要点：**

**重载（Overload）- 编译时多态：**

- 同一个类中，方法名相同，参数列表不同
- 返回类型可以不同
- 发生在编译期

```java
public class Calculator {
    public int add(int a, int b) { return a + b; }
    public double add(double a, double b) { return a + b; }
    public int add(int a, int b, int c) { return a + b + c; }
}
```

**重写（Override）- 运行时多态：**

- 子类重新实现父类的方法
- 方法签名必须完全相同
- 返回类型相同或是子类型
- 访问权限不能更严格
- 发生在运行期

```java
class Animal {
    public void makeSound() { System.out.println("Some sound"); }
}

class Dog extends Animal {
    @Override
    public void makeSound() { System.out.println("Woof!"); }
}
```

**延伸：** 参考 [面向对象 - 多态](/docs/java/oop#多态)

---

## 🎯 集合框架（中级）

### 7. ArrayList 和 LinkedList 的区别？

**答案要点：**

| 特性              | ArrayList           | LinkedList               |
| ----------------- | ------------------- | ------------------------ |
| 底层结构          | 动态数组            | 双向链表                 |
| 随机访问          | O(1) - 快           | O(n) - 慢                |
| 插入/删除（中间） | O(n) - 需要移动元素 | O(1) - 只需改指针        |
| 内存占用          | 连续内存，可能浪费  | 每个节点额外存储两个指针 |
| 适用场景          | 频繁查询            | 频繁插入删除             |

**性能测试：**

```java
// ArrayList适合随机访问
List<Integer> arrayList = new ArrayList<>();
arrayList.get(1000);  // 快速

// LinkedList适合头尾操作
LinkedList<Integer> linkedList = new LinkedList<>();
linkedList.addFirst(1);  // 快速
linkedList.addLast(2);   // 快速
```

**延伸：** 参考 [集合框架 - List](/docs/java/collections#list-接口)

---

### 8. HashMap 的底层实现原理？

**答案要点：**

**JDK 1.8 之后：数组 + 链表 + 红黑树**

1. **存储结构：**

   - 数组：存储 `Node<K,V>` 节点
   - 链表：哈希冲突时使用链表
   - 红黑树：链表长度 ≥8 且数组长度 ≥64 时转为红黑树

2. **put 操作流程：**

   - 计算 key 的 hash 值：`(key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16)`
   - 确定数组索引：`(n - 1) & hash`
   - 如果位置为空，直接插入
   - 如果位置有值，比较 key，相同则覆盖，不同则链表/树插入

3. **扩容机制：**
   - 默认初始容量 16，负载因子 0.75
   - 当 `size > capacity * loadFactor` 时扩容为原来的 2 倍
   - 扩容后重新计算每个元素的位置

**代码示例：**

```java
Map<String, Integer> map = new HashMap<>();
map.put("apple", 1);   // 计算hash -> 找位置 -> 插入
map.put("banana", 2);
// 当元素数量达到 16 * 0.75 = 12 时会扩容到32
```

**延伸：** 参考 [集合框架 - HashMap 详解](/docs/java/collections#hashmap)

---

### 9. HashMap 和 ConcurrentHashMap 的区别？

**答案要点：**

| 特性      | HashMap          | ConcurrentHashMap                         |
| --------- | ---------------- | ----------------------------------------- |
| 线程安全  | 非线程安全       | 线程安全                                  |
| 并发性能  | 高（无锁）       | 较高（分段锁/CAS）                        |
| null 键值 | 允许一个 null 键 | 不允许 null 键值                          |
| 实现方式  | 简单数组+链表/树 | JDK1.7:分段锁<br/>JDK1.8:CAS+synchronized |

**ConcurrentHashMap 实现（JDK 1.8）：**

- 使用 CAS + synchronized 实现线程安全
- 只锁定当前链表或红黑树的首节点
- 多个线程可以同时操作不同的数组位置

```java
// 非线程安全
Map<String, Integer> hashMap = new HashMap<>();

// 线程安全
Map<String, Integer> concurrentMap = new ConcurrentHashMap<>();
```

**延伸：** 参考 [多线程 - 并发集合](/docs/java/multithreading#3-使用并发集合)

---

### 10. HashSet 如何保证元素不重复？

**答案要点：**

- HashSet 底层使用 HashMap 实现
- 添加元素时作为 HashMap 的 key，value 是固定的 PRESENT 对象
- 利用 HashMap 的 key 唯一性来保证元素不重复

**源码分析：**

```java
public class HashSet<E> {
    private transient HashMap<E,Object> map;
    private static final Object PRESENT = new Object();

    public boolean add(E e) {
        return map.put(e, PRESENT) == null;
    }
}
```

**重要：** 自定义对象需要重写 `hashCode()` 和 `equals()` 方法

**延伸：** 参考 [集合框架 - Set](/docs/java/collections#set-接口)

---

## 🎯 多线程（高级）

### 11. 创建线程的几种方式？

**答案要点：**

**方式一：继承 Thread 类**

```java
class MyThread extends Thread {
    @Override
    public void run() {
        System.out.println("Thread running");
    }
}
new MyThread().start();
```

**方式二：实现 Runnable 接口（推荐）**

```java
class MyRunnable implements Runnable {
    @Override
    public void run() {
        System.out.println("Runnable running");
    }
}
new Thread(new MyRunnable()).start();
```

**方式三：实现 Callable 接口（有返回值）**

```java
class MyCallable implements Callable<Integer> {
    @Override
    public Integer call() throws Exception {
        return 42;
    }
}
FutureTask<Integer> task = new FutureTask<>(new MyCallable());
new Thread(task).start();
Integer result = task.get();
```

**方式四：使用线程池**

```java
ExecutorService executor = Executors.newFixedThreadPool(5);
executor.submit(() -> System.out.println("Task running"));
```

**延伸：** 参考 [多线程 - 线程创建](/docs/java/multithreading#创建线程的方式)

---

### 12. synchronized 和 Lock 的区别？

**答案要点：**

| 特性     | synchronized         | ReentrantLock              |
| -------- | -------------------- | -------------------------- |
| 类型     | 关键字，JVM 层面     | 类，API 层面               |
| 锁释放   | 自动释放             | 手动释放（finally 中）     |
| 灵活性   | 低                   | 高（可中断、超时、公平锁） |
| 性能     | JDK1.6 优化后相当    | 略高（复杂场景）           |
| 条件变量 | 只有一个 wait/notify | 可以有多个 Condition       |

**synchronized 示例：**

```java
public synchronized void method() {
    // 自动加锁和释放
}
```

**ReentrantLock 示例：**

```java
private Lock lock = new ReentrantLock();

public void method() {
    lock.lock();
    try {
        // 业务逻辑
    } finally {
        lock.unlock();  // 必须手动释放
    }
}
```

**延伸：** 参考 [多线程 - 线程同步](/docs/java/multithreading#线程同步)

---

### 13. volatile 关键字的作用？

**答案要点：**

**两个主要作用：**

1. **保证可见性：** 一个线程修改变量，其他线程立即可见
2. **禁止指令重排序：** 保证有序性

**不能保证原子性！**

**适用场景：**

```java
// 示例：状态标志
private volatile boolean flag = false;

// 线程1
public void setFlag() {
    flag = true;  // 修改立即对其他线程可见
}

// 线程2
public void checkFlag() {
    while (!flag) {
        // 等待flag变为true
    }
}
```

**为什么不能保证原子性：**

```java
private volatile int count = 0;

// 多线程执行这个方法，最终count可能小于10000
public void increment() {
    count++;  // 分三步：读取、加1、写入，不是原子操作
}
```

**延伸：** 参考 [多线程 - volatile 详解](/docs/java/multithreading)

---

### 14. 线程池的核心参数有哪些？

**答案要点：**

**ThreadPoolExecutor 的 7 个核心参数：**

```java
public ThreadPoolExecutor(
    int corePoolSize,              // 核心线程数
    int maximumPoolSize,           // 最大线程数
    long keepAliveTime,            // 空闲线程存活时间
    TimeUnit unit,                 // 时间单位
    BlockingQueue<Runnable> workQueue,  // 任务队列
    ThreadFactory threadFactory,   // 线程工厂
    RejectedExecutionHandler handler    // 拒绝策略
)
```

**执行流程：**

1. 线程数 < corePoolSize：创建新线程执行
2. 线程数 ≥ corePoolSize：任务放入队列
3. 队列满 && 线程数 < maximumPoolSize：创建新线程
4. 队列满 && 线程数 ≥ maximumPoolSize：执行拒绝策略

**常见拒绝策略：**

- `AbortPolicy`：抛异常（默认）
- `CallerRunsPolicy`：调用者线程执行
- `DiscardPolicy`：直接丢弃
- `DiscardOldestPolicy`：丢弃最老的任务

**延伸：** 参考 [多线程 - 线程池](/docs/java/multithreading#线程池)

---

## 🎯 JVM（高级）

### 15. JVM 内存结构有哪些区域？

**答案要点：**

**运行时数据区域（JDK 8）：**

1. **程序计数器（Program Counter）**

   - 线程私有，记录当前执行的字节码指令地址
   - 不会 OOM

2. **虚拟机栈（VM Stack）**

   - 线程私有，存储局部变量、操作数栈、方法出口等
   - StackOverflowError、OutOfMemoryError

3. **本地方法栈（Native Method Stack）**

   - 为 native 方法服务

4. **堆（Heap）**

   - 线程共享，存储对象实例和数组
   - GC 主要区域
   - OutOfMemoryError: Java heap space

5. **方法区/元空间（Metaspace，JDK 8+）**
   - 线程共享，存储类信息、常量、静态变量
   - OutOfMemoryError: Metaspace

**延伸：** 参考 [JVM 基础 - 内存模型](/docs/java/jvm-basics#内存模型)

---

### 16. 垃圾回收算法有哪些？

**答案要点：**

**1. 标记-清除（Mark-Sweep）**

- 标记需要回收的对象，然后清除
- 缺点：产生内存碎片

**2. 复制算法（Copying）**

- 将内存分两块，每次只用一块
- 存活对象复制到另一块，清空当前块
- 适合新生代（对象存活率低）

**3. 标记-整理（Mark-Compact）**

- 标记后，将存活对象移到一端
- 适合老年代（对象存活率高）

**4. 分代收集**

- 新生代：复制算法
- 老年代：标记-清除或标记-整理

**延伸：** 参考 [JVM 基础 - 垃圾回收](/docs/java/jvm-basics#垃圾回收gc)

---

## 🎯 异常处理（中级）

### 17. Checked Exception 和 Unchecked Exception 的区别？

**答案要点：**

| 类型     | Checked Exception                | Unchecked Exception                             |
| -------- | -------------------------------- | ----------------------------------------------- |
| 继承关系 | Exception（除 RuntimeException） | RuntimeException                                |
| 编译检查 | 必须捕获或声明抛出               | 不强制处理                                      |
| 常见例子 | IOException、SQLException        | NullPointerException、IndexOutOfBoundsException |
| 使用场景 | 可预期的异常情况                 | 编程错误                                        |

**示例：**

```java
// Checked Exception - 必须处理
public void readFile() throws IOException {
    FileReader reader = new FileReader("file.txt");
}

// Unchecked Exception - 可不处理
public void divide(int a, int b) {
    int result = a / b;  // 可能抛出ArithmeticException
}
```

**延伸：** 参考 [异常处理](/docs/java/exception-handling)

---

### 18. try-catch-finally 的执行顺序？

**答案要点：**

**正常情况：** try → finally  
**异常情况：** try → catch → finally  
**特殊情况：** finally 一定执行（除非 System.exit()或 JVM 崩溃）

**return 优先级：**

```java
public int test() {
    try {
        return 1;  // ① 先计算返回值
    } finally {
        return 2;  // ③ finally的return会覆盖try的return
    }
    // 结果返回2
}
```

**最佳实践：** 不要在 finally 中使用 return

**延伸：** 参考 [异常处理 - try-catch-finally](/docs/java/exception-handling#try-catch-finally)

---

## 🎯 IO 流（中级）

### 19. BIO、NIO、AIO 的区别？

**答案要点：**

| 模型 | 说明                       | 特点                        |
| ---- | -------------------------- | --------------------------- |
| BIO  | Blocking IO 同步阻塞       | 一线程一连接，阻塞等待      |
| NIO  | Non-blocking IO 同步非阻塞 | 一线程多连接，Selector 轮询 |
| AIO  | Asynchronous IO 异步非阻塞 | 回调通知，真正异步          |

**BIO 示例：**

```java
ServerSocket server = new ServerSocket(8080);
while (true) {
    Socket socket = server.accept();  // 阻塞等待
    // 每个连接需要一个线程
    new Thread(() -> handleRequest(socket)).start();
}
```

**NIO 示例：**

```java
Selector selector = Selector.open();
ServerSocketChannel server = ServerSocketChannel.open();
server.configureBlocking(false);
server.register(selector, SelectionKey.OP_ACCEPT);

while (true) {
    selector.select();  // 轮询就绪的通道
    // 一个线程处理多个连接
}
```

**延伸：** 参考 [IO 流 - NIO 详解](/docs/java/io-streams#nio-new-io)

---

## 🎯 新特性（中级）

### 20. Lambda 表达式的优缺点？

**答案要点：**

**优点：**

- 简化代码，提高可读性
- 支持函数式编程
- 方便使用 Stream API

**缺点：**

- 调试困难（栈追踪不清晰）
- 不能访问非 final 的局部变量
- 可能影响性能（小对象频繁创建）

**示例对比：**

```java
// 传统方式
List<String> list = Arrays.asList("a", "b", "c");
list.forEach(new Consumer<String>() {
    @Override
    public void accept(String s) {
        System.out.println(s);
    }
});

// Lambda方式
list.forEach(s -> System.out.println(s));
```

**延伸：** 参考 [函数式编程 - Lambda 表达式](/docs/java/functional-programming#lambda-表达式)

---

## 📌 总结与学习建议

### 难度分级

- **初级（1-6 题）：** 基础语法、数据类型、String、OOP 基础
- **中级（7-19 题）：** 集合、异常、IO、部分多线程
- **高级（11-16 题）：** 多线程、JVM、性能优化

### 学习路径

1. **夯实基础** → 掌握基础语法和 OOP
2. **深入集合** → 理解常用集合的实现原理
3. **并发编程** → 掌握多线程和 JUC 工具
4. **JVM 调优** → 理解内存模型和 GC 机制
5. **实战项目** → 结合项目巩固知识

### 相关资源

- [Java 编程完整指南](/docs/java/index)
- [多线程详解](/docs/java/multithreading)
- [JVM 基础](/docs/java/jvm-basics)
- [集合框架](/docs/java/collections)
- [设计模式面试题](/docs/java-design-patterns/interview-questions)

---

**持续更新中...** 欢迎反馈和补充！
