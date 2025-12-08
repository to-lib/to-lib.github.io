---
sidebar_position: 20
---

# 迭代器模式 (Iterator Pattern)

## 模式定义

**迭代器模式**是一种行为型设计模式，它提供了一种方法来依次访问一个聚合对象中的各个元素，而不需要暴露该对象的底层表示。

## 问题分析

当需要访问集合中的元素时，直接暴露集合的内部结构会导致：

- 客户端与集合实现紧耦合
- 难以切换集合实现
- 难以扩展

## 解决方案

定义一个迭代器接口，由具体集合提供迭代器实现。

```
┌─────────────┐
│  Iterator   │
│+ hasNext()  │
│+ next()     │
└──────┬──────┘
       △
       │
  ┌────┴──────────┐
  │               │
┌────────┐   ┌─────────┐
│Iterator│   │Iterator │
│   A    │   │    B    │
└────────┘   └─────────┘
       △               △
       │               │
   ┌───┴───────────────┴───┐
   │                       │
┌──────────┐         ┌──────────┐
│Collection│         │Collection│
│    A     │         │    B     │
└──────────┘         └──────────┘
```

## 代码实现

### 1. 定义迭代器接口

```java
public interface Iterator<T> {
    boolean hasNext();
    T next();
}
```

### 2. 定义聚合接口

```java
public interface Iterable<T> {
    Iterator<T> createIterator();
}
```

### 3. 具体迭代器

```java
public class ArrayIterator<T> implements Iterator<T> {
    private T[] data;
    private int index = 0;
    
    public ArrayIterator(T[] data) {
        this.data = data;
    }
    
    @Override
    public boolean hasNext() {
        return index < data.length;
    }
    
    @Override
    public T next() {
        if (!hasNext()) {
            throw new NoSuchElementException();
        }
        return data[index++];
    }
}

public class ListIterator<T> implements Iterator<T> {
    private List<T> data;
    private int index = 0;
    
    public ListIterator(List<T> data) {
        this.data = data;
    }
    
    @Override
    public boolean hasNext() {
        return index < data.size();
    }
    
    @Override
    public T next() {
        if (!hasNext()) {
            throw new NoSuchElementException();
        }
        return data.get(index++);
    }
}
```

### 4. 具体聚合类

```java
public class SimpleList<T> implements Iterable<T> {
    private List<T> items = new ArrayList<>();
    
    public void add(T item) {
        items.add(item);
    }
    
    @Override
    public Iterator<T> createIterator() {
        return new ListIterator<>(items);
    }
}

public class SimpleArray<T> implements Iterable<T> {
    private T[] items;
    
    public SimpleArray(T[] items) {
        this.items = items;
    }
    
    @Override
    public Iterator<T> createIterator() {
        return new ArrayIterator<>(items);
    }
}
```

### 5. 客户端使用

```java
public class Demo {
    public static void main(String[] args) {
        // 使用List
        SimpleList<String> list = new SimpleList<>();
        list.add("Alice");
        list.add("Bob");
        list.add("Charlie");
        
        Iterator<String> iterator = list.createIterator();
        while (iterator.hasNext()) {
            System.out.println(iterator.next());
        }
        
        // 使用Array
        String[] array = {"X", "Y", "Z"};
        SimpleArray<String> simpleArray = new SimpleArray<>(array);
        
        Iterator<String> arrayIterator = simpleArray.createIterator();
        while (arrayIterator.hasNext()) {
            System.out.println(arrayIterator.next());
        }
    }
}
```

## 实际应用示例

### 二叉树遍历

```java
public class TreeNode {
    private int value;
    private TreeNode left;
    private TreeNode right;
    
    public TreeNode(int value) {
        this.value = value;
    }
    
    public void setLeft(TreeNode left) {
        this.left = left;
    }
    
    public void setRight(TreeNode right) {
        this.right = right;
    }
    
    public int getValue() {
        return value;
    }
    
    public TreeNode getLeft() {
        return left;
    }
    
    public TreeNode getRight() {
        return right;
    }
}

// 前序遍历迭代器
public class PreOrderIterator implements Iterator<Integer> {
    private Stack<TreeNode> stack = new Stack<>();
    
    public PreOrderIterator(TreeNode root) {
        if (root != null) {
            stack.push(root);
        }
    }
    
    @Override
    public boolean hasNext() {
        return !stack.isEmpty();
    }
    
    @Override
    public Integer next() {
        TreeNode node = stack.pop();
        
        if (node.getRight() != null) {
            stack.push(node.getRight());
        }
        if (node.getLeft() != null) {
            stack.push(node.getLeft());
        }
        
        return node.getValue();
    }
}

// 中序遍历迭代器
public class InOrderIterator implements Iterator<Integer> {
    private Stack<TreeNode> stack = new Stack<>();
    private TreeNode current;
    
    public InOrderIterator(TreeNode root) {
        this.current = root;
    }
    
    @Override
    public boolean hasNext() {
        return current != null || !stack.isEmpty();
    }
    
    @Override
    public Integer next() {
        while (current != null) {
            stack.push(current);
            current = current.getLeft();
        }
        
        TreeNode node = stack.pop();
        current = node.getRight();
        
        return node.getValue();
    }
}
```

### 文件系统遍历

```java
public interface FileSystemElement {
    Iterator<FileSystemElement> createIterator();
}

public class File implements FileSystemElement {
    private String name;
    
    public File(String name) {
        this.name = name;
    }
    
    @Override
    public Iterator<FileSystemElement> createIterator() {
        return new EmptyIterator();
    }
    
    @Override
    public String toString() {
        return "File: " + name;
    }
}

public class Directory implements FileSystemElement {
    private String name;
    private List<FileSystemElement> children = new ArrayList<>();
    
    public Directory(String name) {
        this.name = name;
    }
    
    public void addChild(FileSystemElement child) {
        children.add(child);
    }
    
    @Override
    public Iterator<FileSystemElement> createIterator() {
        return new DirectoryIterator(children.iterator());
    }
    
    @Override
    public String toString() {
        return "Directory: " + name;
    }
}

public class EmptyIterator implements Iterator<FileSystemElement> {
    @Override
    public boolean hasNext() {
        return false;
    }
    
    @Override
    public FileSystemElement next() {
        throw new NoSuchElementException();
    }
}

public class DirectoryIterator implements Iterator<FileSystemElement> {
    private Iterator<FileSystemElement> iterator;
    
    public DirectoryIterator(Iterator<FileSystemElement> iterator) {
        this.iterator = iterator;
    }
    
    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }
    
    @Override
    public FileSystemElement next() {
        return iterator.next();
    }
}
```

### 翻页迭代器

```java
public class Page<T> {
    private List<T> items;
    private int pageSize;
    
    public Page(List<T> items, int pageSize) {
        this.items = items;
        this.pageSize = pageSize;
    }
    
    public int getTotalPages() {
        return (items.size() + pageSize - 1) / pageSize;
    }
    
    public List<T> getPage(int pageNumber) {
        int start = pageNumber * pageSize;
        int end = Math.min(start + pageSize, items.size());
        return items.subList(start, end);
    }
}

public class PageIterator<T> implements Iterator<List<T>> {
    private Page<T> page;
    private int currentPage = 0;
    
    public PageIterator(Page<T> page) {
        this.page = page;
    }
    
    @Override
    public boolean hasNext() {
        return currentPage < page.getTotalPages();
    }
    
    @Override
    public List<T> next() {
        if (!hasNext()) {
            throw new NoSuchElementException();
        }
        return page.getPage(currentPage++);
    }
}
```

## Java中的迭代器

```java
// List迭代器
List<String> list = new ArrayList<>();
Iterator<String> iterator = list.iterator();
while (iterator.hasNext()) {
    System.out.println(iterator.next());
}

// 增强for循环（使用Iterable接口）
for (String item : list) {
    System.out.println(item);
}

// 流API
list.stream().forEach(System.out::println);

// 各种迭代方式
ListIterator<String> listIterator = list.listIterator();  // 可双向迭代
Enumeration<String> enumeration = Collections.enumeration(list);  // 老式迭代
```

## 优缺点

### 优点
- ✅ 访问对象无需暴露内部结构
- ✅ 支持多种遍历方式
- ✅ 符合单一职责原则
- ✅ 符合开闭原则

### 缺点
- ❌ 增加类和对象数量
- ❌ 某些场景可能性能较差

## 适用场景

- ✓ 需要访问聚合对象
- ✓ 支持多种遍历方式
- ✓ 隐藏内部结构
- ✓ 集合框架

## 内部迭代器 vs 外部迭代器

| 特性 | 内部迭代器 | 外部迭代器 |
|------|---------|----------|
| 控制权 | 迭代器 | 客户端 |
| 简洁性 | 高 | 低 |
| 灵活性 | 低 | 高 |
| 例子 | forEach | Iterator |
