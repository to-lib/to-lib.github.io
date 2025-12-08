---
sidebar_position: 15
---

# 享元模式 (Flyweight Pattern)

## 模式定义

**享元模式**是一种结构型设计模式，它通过共享尽可能多的相关对象，来减少内存使用。享元模式适合处理大量相似对象的情况。

## 问题分析

当系统中需要创建大量相似对象时，会导致：

- 内存占用过高
- 创建对象成本高
- 垃圾回收压力大

## 解决方案

将对象的状态分为：
- **内在状态（Intrinsic State）** - 不变的、共享的
- **外在状态（Extrinsic State）** - 变化的、不共享的

```
┌─────────────────────┐
│   FlyweightFactory  │
│  - pool             │
│  + getFlyweight()   │
└─────────────────────┘
         │
    ┌────┴─────────┐
    │              │
┌───────────┐  ┌───────────┐
│Flyweight 1│  │Flyweight 2│
│(共享对象) │  │(共享对象) │
└───────────┘  └───────────┘
```

## 代码实现

### 1. 定义享元接口

```java
public interface Flyweight {
    void display(ExternalState state);
}
```

### 2. 具体享元类

```java
public class ConcreteFlyweight implements Flyweight {
    private String sharedData;
    
    public ConcreteFlyweight(String sharedData) {
        this.sharedData = sharedData;
    }
    
    @Override
    public void display(ExternalState state) {
        System.out.println("共享数据: " + sharedData + ", 位置: " + state.location);
    }
}

// 外在状态
public class ExternalState {
    public int location;
    public int fontSize;
    public String color;
    
    public ExternalState(int location, int fontSize, String color) {
        this.location = location;
        this.fontSize = fontSize;
        this.color = color;
    }
}
```

### 3. 享元工厂

```java
public class FlyweightFactory {
    private Map<String, Flyweight> pool = new HashMap<>();
    
    public Flyweight getFlyweight(String key) {
        if (!pool.containsKey(key)) {
            pool.put(key, new ConcreteFlyweight(key));
            System.out.println("创建新享元: " + key);
        } else {
            System.out.println("复用享元: " + key);
        }
        return pool.get(key);
    }
    
    public int getPoolSize() {
        return pool.size();
    }
}
```

### 4. 客户端使用

```java
public class Demo {
    public static void main(String[] args) {
        FlyweightFactory factory = new FlyweightFactory();
        
        // 创建享元对象
        Flyweight fw1 = factory.getFlyweight("data1");
        Flyweight fw2 = factory.getFlyweight("data2");
        Flyweight fw3 = factory.getFlyweight("data1");  // 复用
        
        fw1.display(new ExternalState(10, 12, "red"));
        fw2.display(new ExternalState(20, 14, "blue"));
        fw3.display(new ExternalState(30, 16, "green"));
        
        System.out.println("对象池大小: " + factory.getPoolSize());
    }
}
```

## 实际应用示例

### 文字编辑器中的字符享元

```java
public class Character implements Flyweight {
    private char character;
    private String fontFamily;
    private int fontSize;
    
    public Character(char character, String fontFamily, int fontSize) {
        this.character = character;
        this.fontFamily = fontFamily;
        this.fontSize = fontSize;
    }
    
    @Override
    public void display(int rowNumber, int columnNumber) {
        System.out.println("字符: " + character + " at (" + rowNumber + 
            "," + columnNumber + ") 字体: " + fontFamily + " 大小: " + fontSize);
    }
}

public class CharacterFactory {
    private Map<Character, Character> pool = new HashMap<>();
    
    public Character getCharacter(char c) {
        Character character = pool.get(c);
        if (character == null) {
            character = new Character(c, "Arial", 12);
            pool.put(c, character);
        }
        return character;
    }
}
```

### 游戏中的精灵享元

```java
public class Sprite implements Flyweight {
    private String imagePath;
    private int width;
    private int height;
    
    public Sprite(String imagePath, int width, int height) {
        this.imagePath = imagePath;
        this.width = width;
        this.height = height;
    }
    
    public void render(int x, int y) {
        System.out.println("渲染精灵: " + imagePath + " at (" + x + "," + y + ")");
    }
}

public class SpriteFactory {
    private Map<String, Sprite> cache = new HashMap<>();
    
    public Sprite getSprite(String imagePath) {
        if (!cache.containsKey(imagePath)) {
            cache.put(imagePath, new Sprite(imagePath, 32, 32));
            System.out.println("加载精灵: " + imagePath);
        }
        return cache.get(imagePath);
    }
}

// 使用
SpriteFactory factory = new SpriteFactory();
Sprite enemySprite = factory.getSprite("enemy.png");
Sprite playerSprite = factory.getSprite("player.png");

// 创建大量敌人
for (int i = 0; i < 1000; i++) {
    Sprite sprite = factory.getSprite("enemy.png");
    sprite.render(i * 10, i * 5);
}
```

### 连接池享元

```java
public class DatabaseConnection implements Flyweight {
    private String connectionId;
    private boolean inUse;
    
    public DatabaseConnection(String connectionId) {
        this.connectionId = connectionId;
        this.inUse = false;
    }
    
    public void setInUse(boolean inUse) {
        this.inUse = inUse;
    }
    
    public void execute(String sql) {
        System.out.println("连接 " + connectionId + " 执行: " + sql);
    }
}

public class ConnectionPool {
    private Map<String, DatabaseConnection> pool = new HashMap<>();
    private int connectionCount = 0;
    private static final int MAX_CONNECTIONS = 10;
    
    public DatabaseConnection getConnection() {
        // 先查找空闲连接
        for (DatabaseConnection conn : pool.values()) {
            if (!conn.inUse) {
                conn.setInUse(true);
                System.out.println("复用连接");
                return conn;
            }
        }
        
        // 创建新连接
        if (connectionCount < MAX_CONNECTIONS) {
            String connId = "conn_" + (++connectionCount);
            DatabaseConnection conn = new DatabaseConnection(connId);
            conn.setInUse(true);
            pool.put(connId, conn);
            System.out.println("创建新连接: " + connId);
            return conn;
        }
        
        throw new RuntimeException("连接池已满");
    }
    
    public void releaseConnection(DatabaseConnection conn) {
        conn.setInUse(false);
        System.out.println("释放连接");
    }
}

// 使用
ConnectionPool pool = new ConnectionPool();
DatabaseConnection conn1 = pool.getConnection();
conn1.execute("SELECT * FROM users");
pool.releaseConnection(conn1);

DatabaseConnection conn2 = pool.getConnection();  // 复用conn1
conn2.execute("SELECT * FROM orders");
```

### 树节点享元

```java
public class TreeNode implements Flyweight {
    private String nodeType;  // 内在状态 - 不变
    private String icon;      // 内在状态 - 不变
    
    public TreeNode(String nodeType, String icon) {
        this.nodeType = nodeType;
        this.icon = icon;
    }
    
    public void display(String name, int depth) {
        // 外在状态 - name和depth
        System.out.println("  ".repeat(depth) + icon + " " + name);
    }
}

public class TreeNodeFactory {
    private Map<String, TreeNode> cache = new HashMap<>();
    
    public TreeNode getTreeNode(String nodeType) {
        if (!cache.containsKey(nodeType)) {
            String icon = nodeType.equals("folder") ? "📁" : "📄";
            cache.put(nodeType, new TreeNode(nodeType, icon));
        }
        return cache.get(nodeType);
    }
}

// 使用
TreeNodeFactory factory = new TreeNodeFactory();
TreeNode folderNode = factory.getTreeNode("folder");
TreeNode fileNode = factory.getTreeNode("file");

folderNode.display("Documents", 0);
fileNode.display("report.pdf", 1);
fileNode.display("image.jpg", 1);
```

## 享元模式和对象池

| 特性 | 享元模式 | 对象池 |
|------|--------|-------|
| 目的 | 减少内存 | 复用对象 |
| 共享 | 共享内在状态 | 整个对象 |
| 复杂度 | 中 | 低 |
| 应用 | 大量相似对象 | 创建代价高 |

## 优缺点

### 优点
- ✅ 大幅减少内存占用
- ✅ 提高性能
- ✅ 适合处理大量对象
- ✅ 集中管理共享数据

### 缺点
- ❌ 增加代码复杂性
- ❌ 需要分离内外状态
- ❌ 线程安全问题
- ❌ 不适合小对象

## 适用场景

- ✓ 大量相似对象
- ✓ 内存占用高
- ✓ 字符、图片、文件
- ✓ 连接池、线程池
- ✓ 游戏中的精灵

## Java中的应用

```java
// String常量池
String s1 = "hello";
String s2 = "hello";
// s1 == s2 为true

// Integer缓存
Integer i1 = 128;
Integer i2 = 128;
// i1 == i2 为false (超过127)

// JDBC连接池
DataSource ds = new HikariDataSource();
```
