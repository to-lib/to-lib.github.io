---
sidebar_position: 12
---

# 原型模式 (Prototype Pattern)

## 模式定义

**原型模式**是一种创建型设计模式，它通过复制一个已经存在的对象（原型）来创建新对象，而不是从头创建，无需知道对象的具体类。

## 问题分析

在某些情况下，直接创建新对象比较困难或代价很高：

- 对象初始化成本高
- 需要大量参数
- 对象创建依赖复杂的逻辑
- 需要深度拷贝

## 解决方案

使用clone()方法复制对象，Java中通过Cloneable接口实现。

```
┌─────────────────────┐
│  Prototype(接口)    │
│  + clone()          │
└─────────────────────┘
         △
         │ implements
    ┌────┴─────────┐
    │              │
┌────────┐    ┌──────────┐
│Concrete │    │Concrete  │
│Prototype│    │Prototype │
│    1    │    │    2     │
└────────┘    └──────────┘
```

## 代码实现

### 1. 定义可复制的原型

```java
public interface Prototype {
    Prototype clone();
}
```

### 2. 具体原型类

```java
public class Document implements Prototype, Cloneable {
    private String title;
    private String content;
    private List<String> tags;
    
    public Document(String title, String content) {
        this.title = title;
        this.content = content;
        this.tags = new ArrayList<>();
    }
    
    public void addTag(String tag) {
        this.tags.add(tag);
    }
    
    @Override
    public Document clone() {
        try {
            Document cloned = (Document) super.clone();
            // 深拷贝tags列表
            cloned.tags = new ArrayList<>(this.tags);
            return cloned;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException("克隆失败", e);
        }
    }
    
    @Override
    public String toString() {
        return "Document{" +
            "title='" + title + '\'' +
            ", content='" + content + '\'' +
            ", tags=" + tags +
            '}';
    }
}
```

### 3. 原型工厂

```java
public class PrototypeFactory {
    private Map<String, Document> prototypes = new HashMap<>();
    
    public void registerPrototype(String key, Document document) {
        prototypes.put(key, document);
    }
    
    public Document createDocument(String key) {
        Document prototype = prototypes.get(key);
        if (prototype != null) {
            return prototype.clone();
        }
        return null;
    }
}
```

### 4. 客户端使用

```java
public class Demo {
    public static void main(String[] args) {
        PrototypeFactory factory = new PrototypeFactory();
        
        // 创建原型
        Document original = new Document("报告", "这是一份报告");
        original.addTag("重要");
        original.addTag("2024");
        
        // 注册原型
        factory.registerPrototype("report", original);
        
        // 克隆原型
        Document copy1 = factory.createDocument("report");
        Document copy2 = factory.createDocument("report");
        
        System.out.println("原始: " + original);
        System.out.println("副本1: " + copy1);
        System.out.println("副本2: " + copy2);
        
        // 修改副本不影响原型
        copy1.addTag("已读");
        System.out.println("\n修改后：");
        System.out.println("原始: " + original);
        System.out.println("副本1: " + copy1);
    }
}
```

## 浅拷贝 vs 深拷贝

### 浅拷贝

```java
public class ShallowCopy implements Cloneable {
    private String name;
    private List<String> items;
    
    @Override
    public ShallowCopy clone() throws CloneNotSupportedException {
        return (ShallowCopy) super.clone();
        // items引用相同，是浅拷贝
    }
}
```

### 深拷贝

```java
public class DeepCopy implements Cloneable {
    private String name;
    private List<String> items;
    
    @Override
    public DeepCopy clone() throws CloneNotSupportedException {
        DeepCopy cloned = (DeepCopy) super.clone();
        // 创建新的items列表，是深拷贝
        cloned.items = new ArrayList<>(this.items);
        return cloned;
    }
}
```

## 实际应用示例

### 游戏中的敌人克隆

```java
public class Enemy implements Cloneable {
    private String name;
    private int health;
    private int power;
    private Position position;
    
    public Enemy(String name, int health, int power, Position position) {
        this.name = name;
        this.health = health;
        this.power = power;
        this.position = position;
    }
    
    @Override
    public Enemy clone() {
        try {
            Enemy cloned = (Enemy) super.clone();
            cloned.position = new Position(position.x, position.y);
            return cloned;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
}

public class Position {
    public int x, y;
    
    public Position(int x, int y) {
        this.x = x;
        this.y = y;
    }
}

// 使用
Enemy baseEnemy = new Enemy("僵尸", 100, 10, new Position(0, 0));

// 快速创建多个敌人
List<Enemy> enemies = new ArrayList<>();
for (int i = 0; i < 10; i++) {
    Enemy clone = baseEnemy.clone();
    clone.position.x = i * 50;
    enemies.add(clone);
}
```

### 配置对象克隆

```java
public class Configuration implements Cloneable {
    private String appName;
    private Map<String, String> properties;
    private List<String> modules;
    
    @Override
    public Configuration clone() {
        try {
            Configuration cloned = (Configuration) super.clone();
            cloned.properties = new HashMap<>(this.properties);
            cloned.modules = new ArrayList<>(this.modules);
            return cloned;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
}
```

## Java中的clone方法

```java
// 数组克隆
int[] array = {1, 2, 3};
int[] cloned = array.clone();

// List克隆
List<String> original = new ArrayList<>();
List<String> cloned = new ArrayList<>(original);

// Map克隆
Map<String, String> original = new HashMap<>();
Map<String, String> cloned = new HashMap<>(original);
```

## 序列化方式实现克隆

```java
public class SerializablePrototype implements Serializable {
    private String data;
    
    public SerializablePrototype deepClone() {
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos);
            oos.writeObject(this);
            oos.close();
            
            ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
            ObjectInputStream ois = new ObjectInputStream(bais);
            SerializablePrototype cloned = (SerializablePrototype) ois.readObject();
            ois.close();
            
            return cloned;
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }
}
```

## 优缺点

### 优点
- ✅ 提高创建对象的效率
- ✅ 避免重复初始化
- ✅ 适合复杂对象的创建
- ✅ 可以在运行时改变原型

### 缺点
- ❌ 需要实现Cloneable接口
- ❌ 深拷贝可能复杂
- ❌ 容易忽略深拷贝问题

## 适用场景

- ✓ 对象创建成本高
- ✓ 需要大量相似对象
- ✓ 需要深拷贝对象
- ✓ 避免重复初始化

## 最佳实践

1. **区分浅拷贝和深拷贝** - 根据需求选择
2. **使用序列化方式** - 实现完整的深拷贝
3. **提供工厂方法** - 简化原型使用
4. **避免复杂的克隆逻辑** - 保持代码简洁
5. **文档清晰** - 说明是浅拷贝还是深拷贝
