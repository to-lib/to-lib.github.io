---
sidebar_position: 14
---

# 组合模式 (Composite Pattern)

## 模式定义

**组合模式**是一种结构型设计模式，它允许你将对象组合成树形结构来表示"部分-整体"的层次结构，使得客户端可以以相同的方式处理单个对象和对象的组合。

## 问题分析

当需要处理具有树形结构的对象时：

- 文件系统中的文件和文件夹
- 菜单和子菜单
- 组织结构
- 操作系统中的进程树

直接处理会导致代码复杂。

## 解决方案

```
┌──────────────────────┐
│    Component         │
│  + operation()       │
│  + add(child)        │
│  + remove(child)     │
└──────────┬───────────┘
           △
           │
    ┌──────┴──────────┐
    │                 │
┌────────┐        ┌──────────┐
│  Leaf  │        │ Composite│
│(叶子)  │        │ (树枝)   │
│        │        │+ children│
└────────┘        └──────────┘
```

## 代码实现

### 1. 定义组件接口

```java
public interface FileSystemComponent {
    String getName();
    void display();
    long getSize();
}
```

### 2. 叶子节点（文件）

```java
public class File implements FileSystemComponent {
    private String name;
    private long size;
    
    public File(String name, long size) {
        this.name = name;
        this.size = size;
    }
    
    @Override
    public String getName() {
        return name;
    }
    
    @Override
    public void display() {
        System.out.println("📄 " + name + " (" + size + " bytes)");
    }
    
    @Override
    public long getSize() {
        return size;
    }
}
```

### 3. 树枝节点（文件夹）

```java
public class Directory implements FileSystemComponent {
    private String name;
    private List<FileSystemComponent> children = new ArrayList<>();
    
    public Directory(String name) {
        this.name = name;
    }
    
    public void addComponent(FileSystemComponent component) {
        children.add(component);
    }
    
    public void removeComponent(FileSystemComponent component) {
        children.remove(component);
    }
    
    @Override
    public String getName() {
        return name;
    }
    
    @Override
    public void display() {
        System.out.println("📁 " + name + "/");
        for (FileSystemComponent child : children) {
            child.display();
        }
    }
    
    @Override
    public long getSize() {
        long totalSize = 0;
        for (FileSystemComponent child : children) {
            totalSize += child.getSize();
        }
        return totalSize;
    }
}
```

### 4. 客户端使用

```java
public class Demo {
    public static void main(String[] args) {
        // 创建文件系统
        Directory root = new Directory("root");
        
        Directory documents = new Directory("Documents");
        Directory pictures = new Directory("Pictures");
        
        File file1 = new File("report.pdf", 1024);
        File file2 = new File("image.jpg", 2048);
        File file3 = new File("code.java", 512);
        
        // 组合对象
        root.addComponent(documents);
        root.addComponent(pictures);
        
        documents.addComponent(file1);
        documents.addComponent(file3);
        
        pictures.addComponent(file2);
        
        // 统一处理
        root.display();
        System.out.println("\n总大小: " + root.getSize() + " bytes");
    }
}
```

## 实际应用示例

### UI组件树

```java
public interface UIComponent {
    void render();
    void add(UIComponent component);
    void remove(UIComponent component);
}

public class Button implements UIComponent {
    private String label;
    
    public Button(String label) {
        this.label = label;
    }
    
    @Override
    public void render() {
        System.out.println("按钮: " + label);
    }
    
    @Override
    public void add(UIComponent component) {
        throw new UnsupportedOperationException("按钮不能添加子组件");
    }
    
    @Override
    public void remove(UIComponent component) {
    }
}

public class Panel implements UIComponent {
    private String title;
    private List<UIComponent> children = new ArrayList<>();
    
    public Panel(String title) {
        this.title = title;
    }
    
    @Override
    public void render() {
        System.out.println("面板: " + title);
        for (UIComponent child : children) {
            child.render();
        }
    }
    
    @Override
    public void add(UIComponent component) {
        children.add(component);
    }
    
    @Override
    public void remove(UIComponent component) {
        children.remove(component);
    }
}

// 使用
Panel mainWindow = new Panel("主窗口");
Panel leftPanel = new Panel("左面板");
Panel rightPanel = new Panel("右面板");

Button saveBtn = new Button("保存");
Button deleteBtn = new Button("删除");

mainWindow.add(leftPanel);
mainWindow.add(rightPanel);
leftPanel.add(saveBtn);
rightPanel.add(deleteBtn);

mainWindow.render();
```

### 组织结构

```java
public interface Employee {
    void show();
    void addEmployee(Employee employee);
}

public class Manager implements Employee {
    private String name;
    private String position;
    private List<Employee> employees = new ArrayList<>();
    
    public Manager(String name, String position) {
        this.name = name;
        this.position = position;
    }
    
    @Override
    public void show() {
        System.out.println(position + ": " + name);
        for (Employee employee : employees) {
            employee.show();
        }
    }
    
    @Override
    public void addEmployee(Employee employee) {
        employees.add(employee);
    }
}

public class Developer implements Employee {
    private String name;
    private String position;
    
    public Developer(String name, String position) {
        this.name = name;
        this.position = position;
    }
    
    @Override
    public void show() {
        System.out.println(position + ": " + name);
    }
    
    @Override
    public void addEmployee(Employee employee) {
        throw new UnsupportedOperationException("开发人员不能有下属");
    }
}

// 使用
Manager ceo = new Manager("张三", "CEO");
Manager techDir = new Manager("李四", "技术总监");

Developer dev1 = new Developer("王五", "Java开发");
Developer dev2 = new Developer("赵六", "前端开发");

ceo.addEmployee(techDir);
techDir.addEmployee(dev1);
techDir.addEmployee(dev2);

ceo.show();
```

### 菜单系统

```java
public class MenuItem {
    private String name;
    private List<MenuItem> subItems = new ArrayList<>();
    
    public MenuItem(String name) {
        this.name = name;
    }
    
    public void add(MenuItem item) {
        subItems.add(item);
    }
    
    public void print(int depth) {
        System.out.println("  ".repeat(depth) + "- " + name);
        for (MenuItem item : subItems) {
            item.print(depth + 1);
        }
    }
}

// 使用
MenuItem root = new MenuItem("菜单");

MenuItem file = new MenuItem("文件");
MenuItem edit = new MenuItem("编辑");
MenuItem view = new MenuItem("查看");

MenuItem fileNew = new MenuItem("新建");
MenuItem fileOpen = new MenuItem("打开");
MenuItem fileSave = new MenuItem("保存");

root.add(file);
root.add(edit);
root.add(view);

file.add(fileNew);
file.add(fileOpen);
file.add(fileSave);

root.print(0);
```

## 组合模式的两种方式

### 透明组合（transparent）
组件和容器有相同的接口。

```java
public interface Component {
    void operation();
    void add(Component component);
    void remove(Component component);
}
```

### 安全组合（safe）
容器有额外的管理方法。

```java
public interface Component {
    void operation();
}

public interface Composite extends Component {
    void add(Component component);
    void remove(Component component);
}
```

## 优缺点

### 优点
- ✅ 简化客户端代码
- ✅ 易于添加新组件
- ✅ 符合开闭原则
- ✅ 支持复杂的树形结构

### 缺点
- ❌ 设计复杂
- ❌ 类的数量增多
- ❌ 可能造成性能问题

## 适用场景

- ✓ 树形结构
- ✓ 文件系统
- ✓ UI组件层次
- ✓ 组织结构
- ✓ 菜单系统

## Java中的应用

```java
// Swing中的组合模式
JPanel panel = new JPanel();
JButton button = new JButton("按钮");
panel.add(button);

// DOM树
Element element = document.getElementById("root");
Element child = document.createElement("div");
element.appendChild(child);
```
