---
sidebar_position: 8
title: 泛型编程
---

# 泛型编程

泛型（Generics）是 Java 5 引入的重要特性，提供编译时类型安全检查。本文介绍泛型类、泛型方法和通配符的使用。

## 为什么使用泛型

### 不使用泛型的问题

```java
// 不使用泛型：需要类型转换，容易出错
List list = new ArrayList();
list.add("Hello");
list.add(123);  // 可以添加任何类型

String str = (String) list.get(0);  // 需要强制转换
String num = (String) list.get(1);  // 运行时报错：ClassCastException
```

### 使用泛型的优势

```java
// 使用泛型：类型安全，无需转换
List<String> list = new ArrayList<>();
list.add("Hello");
// list.add(123);  // 编译错误

String str = list.get(0);  // 无需强制转换
```

**泛型的优势：**

- ✅ 编译时类型检查
- ✅ 消除类型转换
- ✅ 实现通用算法

## 泛型类

### 定义泛型类

```java
// 泛型类定义
public class Box<T> {
    private T content;
    
    public void set(T content) {
        this.content = content;
    }
    
    public T get() {
        return content;
    }
    
    public static void main(String[] args) {
        // 使用 String 类型
        Box<String> stringBox = new Box<>();
        stringBox.set("Hello");
        String str = stringBox.get();
        
        // 使用 Integer 类型
        Box<Integer> intBox = new Box<>();
        intBox.set(123);
        Integer num = intBox.get();
        
        // Java 7+ 菱形语法
        Box<String> box = new Box<>();  // 类型推断
    }
}
```

### 多个类型参数

```java
public class Pair<K, V> {
    private K key;
    private V value;
    
    public Pair(K key, V value) {
        this.key = key;
        this.value = value;
    }
    
    public K getKey() {
        return key;
    }
    
    public V getValue() {
        return value;
    }
    
    public void setKey(K key) {
        this.key = key;
    }
    
    public void setValue(V value) {
        this.value = value;
    }
    
    @Override
    public String toString() {
        return "Pair{" + key + "=" + value + "}";
    }
    
    public static void main(String[] args) {
        Pair<String, Integer> pair1 = new Pair<>("Age", 25);
        Pair<Integer, String> pair2 = new Pair<>(1, "First");
        
        System.out.println(pair1);  // Pair{Age=25}
        System.out.println(pair2);  // Pair{1=First}
    }
}
```

## 泛型接口

```java
// 泛型接口
public interface Comparable<T> {
    int compareTo(T other);
}

// 实现泛型接口
public class Person implements Comparable<Person> {
    private String name;
    private int age;
    
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    @Override
    public int compareTo(Person other) {
        return Integer.compare(this.age, other.age);
    }
}

// 泛型接口的另一种实现
public interface Container<T> {
    void add(T item);
    T get(int index);
    int size();
}

public class SimpleContainer<T> implements Container<T> {
    private List<T> items = new ArrayList<>();
    
    @Override
    public void add(T item) {
        items.add(item);
    }
    
    @Override
    public T get(int index) {
        return items.get(index);
    }
    
    @Override
    public int size() {
        return items.size();
    }
}
```

## 泛型方法

### 定义泛型方法

```java
public class GenericMethodExample {
    // 泛型方法
    public static <T> void printArray(T[] array) {
        for (T element : array) {
            System.out.print(element + " ");
        }
        System.out.println();
    }
    
    // 多个类型参数的泛型方法
    public static <K, V> void printPair(K key, V value) {
        System.out.println(key + " = " + value);
    }
    
    // 有返回值的泛型方法
    public static <T> T getMiddle(T... array) {
        return array[array.length / 2];
    }
    
    public static void main(String[] args) {
        Integer[] intArray = {1, 2, 3, 4, 5};
        String[] strArray = {"A", "B", "C"};
        
        printArray(intArray);   // 1 2 3 4 5
        printArray(strArray);   // A B C
        
        printPair("Name", "张三");  // Name = 张三
        printPair(1, "First");      // 1 = First
        
        String middle = getMiddle("a", "b", "c", "d", "e");
        System.out.println("中间元素: " + middle);  // c
    }
}
```

### 泛型方法和泛型类的区别

```java
// 泛型类
public class GenericClass<T> {
    private T value;
    
    // 普通方法（使用类的类型参数）
    public T getValue() {
        return value;
    }
    
    // 泛型方法（定义自己的类型参数）
    public <E> void printType(E element) {
        System.out.println("T is: " + value.getClass().getName());
        System.out.println("E is: " + element.getClass().getName());
    }
    
    // 静态泛型方法（不能使用类的类型参数）
    public static <K> void staticMethod(K element) {
        System.out.println(element);
    }
}
```

## 类型边界

### 上界通配符（extends）

```java
// 限制类型参数必须是某个类的子类
public class BoundedTypeExample {
    // 只接受 Number 及其子类
    public static <T extends Number> double sum(List<T> numbers) {
        double total = 0;
        for (T num : numbers) {
            total += num.doubleValue();
        }
        return total;
    }
    
    // 多个边界
    public static <T extends Comparable<T> & Serializable> T findMax(T[] array) {
        T max = array[0];
        for (T element : array) {
            if (element.compareTo(max) > 0) {
                max = element;
            }
        }
        return max;
    }
    
    public static void main(String[] args) {
        List<Integer> intList = Arrays.asList(1, 2, 3, 4, 5);
        List<Double> doubleList = Arrays.asList(1.5, 2.5, 3.5);
        
        System.out.println("整数和: " + sum(intList));      // 15.0
        System.out.println("浮点数和: " + sum(doubleList));  // 7.5
        
        String[] strings = {"Apple", "Banana", "Cherry"};
        System.out.println("最大值: " + findMax(strings));  // Cherry
    }
}
```

## 通配符

### 无界通配符（?）

```java
public class UnboundedWildcard {
    // 接受任何类型的 List
    public static void printList(List<?> list) {
        for (Object element : list) {
            System.out.print(element + " ");
        }
        System.out.println();
    }
    
    public static void main(String[] args) {
        List<Integer> intList = Arrays.asList(1, 2, 3);
        List<String> strList = Arrays.asList("A", "B", "C");
        
        printList(intList);  // 1 2 3
        printList(strList);  // A B C
    }
}
```

### 上界通配符（? extends T）

```java
public class UpperBoundedWildcard {
    // 只读：可以读取 Number 或其子类
    public static double sum(List<? extends Number> numbers) {
        double total = 0;
        for (Number num : numbers) {
            total += num.doubleValue();
        }
        return total;
    }
    
    public static void main(String[] args) {
        List<Integer> intList = Arrays.asList(1, 2, 3);
        List<Double> doubleList = Arrays.asList(1.5, 2.5);
        List<Float> floatList = Arrays.asList(1.0f, 2.0f);
        
        System.out.println(sum(intList));     // 6.0
        System.out.println(sum(doubleList));  // 4.0
        System.out.println(sum(floatList));   // 3.0
        
        // 不能添加元素（除了 null）
        List<? extends Number> list = new ArrayList<Integer>();
        // list.add(1);  // 编译错误
        // list.add(1.0);  // 编译错误
    }
}
```

### 下界通配符（? super T）

```java
public class LowerBoundedWildcard {
    // 只写：可以添加 Integer 或其子类
    public static void addNumbers(List<? super Integer> list) {
        for (int i = 1; i <= 5; i++) {
            list.add(i);  // 可以添加 Integer
        }
    }
    
    public static void main(String[] args) {
        List<Number> numberList = new ArrayList<>();
        List<Object> objectList = new ArrayList<>();
        
        addNumbers(numberList);
        addNumbers(objectList);
        
        System.out.println(numberList);  // [1, 2, 3, 4, 5]
        System.out.println(objectList);  // [1, 2, 3, 4, 5]
        
        // 读取时只能作为 Object
        List<? super Integer> list = new ArrayList<Number>();
        list.add(1);
        Object obj = list.get(0);  // 只能作为 Object 读取
        // Integer num = list.get(0);  // 编译错误
    }
}
```

### PECS 原则

**Producer Extends, Consumer Super**

```java
public class PECSExample {
    // Producer（生产者）：使用 extends
    // 从 src 读取数据
    public static void copy1(List<? extends Number> src, List<Number> dest) {
        for (Number num : src) {
            dest.add(num);  // 从 src 读取，向 dest 写入
        }
    }
    
    // Consumer（消费者）：使用 super
    // 向 dest 写入数据
    public static void copy2(List<Number> src, List<? super Number> dest) {
        for (Number num : src) {
            dest.add(num);  // 从 src 读取，向 dest 写入
        }
    }
    
    // 既是生产者又是消费者
    public static void copy3(List<? extends Number> src, List<? super Number> dest) {
        for (Number num : src) {
            dest.add(num);
        }
    }
    
    public static void main(String[] args) {
        List<Integer> intList = Arrays.asList(1, 2, 3);
        List<Number> numList = new ArrayList<>();
        List<Object> objList = new ArrayList<>();
        
        copy1(intList, numList);   // intList 是生产者
        copy2(numList, objList);   // objList 是消费者
        copy3(intList, objList);   // intList 生产，objList 消费
    }
}
```

## 类型擦除

Java 泛型是通过类型擦除实现的。

```java
public class TypeErasureExample {
    public static void main(String[] args) {
        List<String> stringList = new ArrayList<>();
        List<Integer> intList = new ArrayList<>();
        
        // 运行时类型相同
        System.out.println(stringList.getClass() == intList.getClass());  // true
        
        // 都是 ArrayList
        System.out.println(stringList.getClass());  // class java.util.ArrayList
        System.out.println(intList.getClass());     // class java.util.ArrayList
    }
}
```

### 类型擦除的影响

```java
public class TypeErasureLimitations {
    // 不能创建泛型数组
    // T[] array = new T[10];  // 编译错误
    
    // 不能使用基本类型作为类型参数
    // List<int> list = new ArrayList<>();  // 编译错误
    List<Integer> list = new ArrayList<>();  // 正确
    
    // 不能创建类型参数的实例
    public <T> void create() {
        // T obj = new T();  // 编译错误
    }
    
    // 不能使用 instanceof
    public <T> void check(Object obj) {
        // if (obj instanceof T) {}  // 编译错误
    }
    
    // 静态字段不能使用类型参数
    public class Generic<T> {
        // private static T value;  // 编译错误
    }
}
```

## 泛型的实际应用

### 构建通用的工具类

```java
public class CollectionUtils {
    // 查找元素
    public static <T> boolean contains(T[] array, T element) {
        for (T item : array) {
            if (item.equals(element)) {
                return true;
            }
        }
        return false;
    }
    
    // 交换元素
    public static <T> void swap(T[] array, int i, int j) {
        T temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
    
    // 反转数组
    public static <T> void reverse(T[] array) {
        int left = 0, right = array.length - 1;
        while (left < right) {
            swap(array, left++, right--);
        }
    }
    
    // 转换列表
    public static <T, R> List<R> map(List<T> list, Function<T, R> mapper) {
        List<R> result = new ArrayList<>();
        for (T element : list) {
            result.add(mapper.apply(element));
        }
        return result;
    }
}
```

### 泛型单例模式

```java
public class GenericSingleton<T> {
    private static GenericSingleton<?> instance;
    
    private GenericSingleton() {}
    
    @SuppressWarnings("unchecked")
    public static <T> GenericSingleton<T> getInstance() {
        if (instance == null) {
            synchronized (GenericSingleton.class) {
                if (instance == null) {
                    instance = new GenericSingleton<T>();
                }
            }
        }
        return (GenericSingleton<T>) instance;
    }
}
```

### 泛型建造者模式

```java
public class Builder<T> {
    private final T object;
    
    public Builder(Class<T> clazz) throws Exception {
        this.object = clazz.newInstance();
    }
    
    public Builder<T> set(String propertyName, Object value) throws Exception {
        Field field = object.getClass().getDeclaredField(propertyName);
        field.setAccessible(true);
        field.set(object, value);
        return this;
    }
    
    public T build() {
        return object;
    }
}

// 使用示例
class User {
    private String name;
    private int age;
    
    // getters and setters
}

// User user = new Builder<>(User.class)
//     .set("name", "张三")
//     .set("age", 25)
//     .build();
```

## 最佳实践

### 1. 优先使用泛型类型

```java
// 不好
List list = new ArrayList();
list.add("Hello");
String str = (String) list.get(0);

// 好
List<String> list = new ArrayList<>();
list.add("Hello");
String str = list.get(0);
```

### 2. 使用有意义的类型参数名

```java
// 单个类型参数
E - Element（集合元素）
T - Type（类型）
K - Key（键）
V - Value（值）
N - Number（数字）
S, U, V - 第2、3、4个类型参数

// 示例
public class List<E> {}
public class Map<K, V> {}
public class Function<T, R> {}
```

### 3. 限制类型参数

```java
// 好：限制类型范围
public <T extends Comparable<T>> T findMax(T[] array) {
    // 可以安全调用 compareTo
}
```

### 4. 使用通配符提高灵活性

```java
// 不好：过于严格
public void process(List<Number> list) {}

// 好：更灵活
public void process(List<? extends Number> list) {}
```

### 5. 避免原始类型

```java
// 不好：原始类型（Raw Type）
List list = new ArrayList();

// 好：使用泛型
List<String> list = new ArrayList<>();
// 或使用通配符
List<?> list = new ArrayList<>();
```

## 总结

本文介绍了 Java 泛型编程的核心内容：

- ✅ 泛型类和泛型接口
- ✅ 泛型方法
- ✅ 类型边界（extends）
- ✅ 通配符：?、? extends T、? super T
- ✅ PECS 原则
- ✅ 类型擦除
- ✅ 泛型的实际应用

掌握泛型后，继续学习 [注解](./annotations) 和 [反射](./reflection)。
