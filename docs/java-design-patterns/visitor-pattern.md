---
sidebar_position: 23
---

# 访问者模式 (Visitor Pattern)

## 模式定义

**访问者模式**是一种行为型设计模式，它表示一个作用于某对象结构中各个元素的操作。它让你可以在不改变各个元素的类的前提下定义作用于这些元素的新操作。

## 问题分析

当对象结构中的元素经常面临多种不同的操作时：

- 直接在元素类中添加方法会很复杂
- 新操作需要修改每个元素类
- 违反开闭原则

## 解决方案

定义一个访问者接口，不同的操作通过不同的访问者实现：

```
┌────────────────┐
│    Visitor     │
│+ visit(A)      │
│+ visit(B)      │
└────────┬───────┘
         △
    ┌────┴──────────┐
    │               │
┌────────┐    ┌──────────┐
│Visitor1│    │Visitor2  │
└────────┘    └──────────┘

┌────────────────┐
│   Element      │
│+ accept(v)     │
└────────┬───────┘
         △
    ┌────┴──────────┐
    │               │
┌────────┐    ┌──────────┐
│ElementA│    │ElementB  │
└────────┘    └──────────┘
```

## 代码实现

### 1. 定义访问者接口

```java
public interface ItemVisitor {
    void visit(Book book);
    void visit(Food food);
}
```

### 2. 定义元素接口

```java
public interface ShoppingItem {
    void accept(ItemVisitor visitor);
}
```

### 3. 具体元素

```java
public class Book implements ShoppingItem {
    private String title;
    private double price;
    
    public Book(String title, double price) {
        this.title = title;
        this.price = price;
    }
    
    public String getTitle() { return title; }
    public double getPrice() { return price; }
    
    @Override
    public void accept(ItemVisitor visitor) {
        visitor.visit(this);
    }
}

public class Food implements ShoppingItem {
    private String name;
    private double price;
    
    public Food(String name, double price) {
        this.name = name;
        this.price = price;
    }
    
    public String getName() { return name; }
    public double getPrice() { return price; }
    
    @Override
    public void accept(ItemVisitor visitor) {
        visitor.visit(this);
    }
}
```

### 4. 具体访问者

```java
public class TaxCalculator implements ItemVisitor {
    @Override
    public void visit(Book book) {
        // 书籍免税
        System.out.println("书籍 " + book.getTitle() + " 税率: 0%");
    }
    
    @Override
    public void visit(Food food) {
        // 食品10%税率
        System.out.println("食品 " + food.getName() + " 税率: 10%");
    }
}

public class PriceCalculator implements ItemVisitor {
    private double totalPrice = 0;
    
    @Override
    public void visit(Book book) {
        totalPrice += book.getPrice();
    }
    
    @Override
    public void visit(Food food) {
        totalPrice += food.getPrice();
    }
    
    public double getTotalPrice() {
        return totalPrice;
    }
}
```

### 5. 客户端使用

```java
public class Demo {
    public static void main(String[] args) {
        List<ShoppingItem> items = new ArrayList<>();
        items.add(new Book("Java编程", 89.0));
        items.add(new Food("苹果", 5.0));
        items.add(new Book("设计模式", 79.0));
        items.add(new Food("面包", 8.0));
        
        // 计算税率
        TaxCalculator taxVisitor = new TaxCalculator();
        for (ShoppingItem item : items) {
            item.accept(taxVisitor);
        }
        
        // 计算总价
        PriceCalculator priceVisitor = new PriceCalculator();
        for (ShoppingItem item : items) {
            item.accept(priceVisitor);
        }
        System.out.println("总价: " + priceVisitor.getTotalPrice());
    }
}
```

## 实际应用示例

### 文件系统操作

```java
public interface FileSystemVisitor {
    void visit(TextFile file);
    void visit(ImageFile file);
}

public abstract class FileSystemElement {
    protected String name;
    
    public FileSystemElement(String name) {
        this.name = name;
    }
    
    public abstract void accept(FileSystemVisitor visitor);
}

public class TextFile extends FileSystemElement {
    private int lines;
    
    public TextFile(String name, int lines) {
        super(name);
        this.lines = lines;
    }
    
    public int getLines() { return lines; }
    
    @Override
    public void accept(FileSystemVisitor visitor) {
        visitor.visit(this);
    }
}

public class ImageFile extends FileSystemElement {
    private int width;
    private int height;
    
    public ImageFile(String name, int width, int height) {
        super(name);
        this.width = width;
        this.height = height;
    }
    
    public int getWidth() { return width; }
    public int getHeight() { return height; }
    
    @Override
    public void accept(FileSystemVisitor visitor) {
        visitor.visit(this);
    }
}

public class FileSizeCalculator implements FileSystemVisitor {
    private long totalSize = 0;
    
    @Override
    public void visit(TextFile file) {
        totalSize += file.getLines() * 100;  // 每行100字节
        System.out.println("文本文件 " + file.name + ": " + (file.getLines() * 100) + " 字节");
    }
    
    @Override
    public void visit(ImageFile file) {
        totalSize += file.getWidth() * file.getHeight() * 4;  // RGBA
        System.out.println("图像文件 " + file.name + ": " + 
            (file.getWidth() * file.getHeight() * 4) + " 字节");
    }
    
    public long getTotalSize() {
        return totalSize;
    }
}

public class FileTypeAnalyzer implements FileSystemVisitor {
    @Override
    public void visit(TextFile file) {
        System.out.println("文本文件: " + file.name + " (" + file.getLines() + " 行)");
    }
    
    @Override
    public void visit(ImageFile file) {
        System.out.println("图像文件: " + file.name + " (" + 
            file.getWidth() + "x" + file.getHeight() + ")");
    }
}
```

### AST处理

```java
public interface ExpressionVisitor {
    Object visit(NumberExpression expr);
    Object visit(AddExpression expr);
    Object visit(SubExpression expr);
}

public abstract class Expression {
    public abstract Object accept(ExpressionVisitor visitor);
}

public class NumberExpression extends Expression {
    private int value;
    
    public NumberExpression(int value) {
        this.value = value;
    }
    
    public int getValue() { return value; }
    
    @Override
    public Object accept(ExpressionVisitor visitor) {
        return visitor.visit(this);
    }
}

public class AddExpression extends Expression {
    private Expression left;
    private Expression right;
    
    public AddExpression(Expression left, Expression right) {
        this.left = left;
        this.right = right;
    }
    
    public Expression getLeft() { return left; }
    public Expression getRight() { return right; }
    
    @Override
    public Object accept(ExpressionVisitor visitor) {
        return visitor.visit(this);
    }
}

public class EvaluationVisitor implements ExpressionVisitor {
    @Override
    public Object visit(NumberExpression expr) {
        return expr.getValue();
    }
    
    @Override
    public Object visit(AddExpression expr) {
        int left = (Integer) expr.getLeft().accept(this);
        int right = (Integer) expr.getRight().accept(this);
        return left + right;
    }
    
    @Override
    public Object visit(SubExpression expr) {
        // 实现
        return null;
    }
}

public class PrintVisitor implements ExpressionVisitor {
    @Override
    public Object visit(NumberExpression expr) {
        System.out.print(expr.getValue());
        return null;
    }
    
    @Override
    public Object visit(AddExpression expr) {
        System.out.print("(");
        expr.getLeft().accept(this);
        System.out.print(" + ");
        expr.getRight().accept(this);
        System.out.print(")");
        return null;
    }
    
    @Override
    public Object visit(SubExpression expr) {
        // 实现
        return null;
    }
}
```

## 优缺点

### 优点
- ✅ 符合开闭原则
- ✅ 易于添加新操作
- ✅ 将数据和操作分离
- ✅ 避免类型转换

### 缺点
- ❌ 难以添加新元素类型
- ❌ 代码复杂性高
- ❌ 元素类必须暴露内部结构
- ❌ 违反迪米特法则

## Double Dispatch

访问者模式使用双重分派：

```java
// 第一次分派：多态调用accept()
item.accept(visitor);  // 通过item的具体类型

// 第二次分派：多态调用visit()
visitor.visit(this);   // 通过visitor的具体类型
```

## 适用场景

- ✓ 需要对复杂对象结构进行操作
- ✓ 操作种类多且经常变化
- ✓ 元素类型相对稳定
- ✓ 编译器、IDE
- ✓ 代码生成

## 与其他模式的关系

- **迭代器模式** - 遍历集合
- **组合模式** - 处理树形结构
- **策略模式** - 不同算法
