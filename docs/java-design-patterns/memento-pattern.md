---
sidebar_position: 22
---

# 备忘录模式 (Memento Pattern)

## 模式定义

**备忘录模式**是一种行为型设计模式，它在不破坏对象封装性的前提下，捕获并外部化一个对象的内部状态，使得以后可以恢复到这个状态。

## 问题分析

当需要保存和恢复对象状态时：

- 直接访问对象内部会破坏封装
- 需要保存对象的多个历史状态
- 需要实现undo/redo功能

## 解决方案

```
┌──────────────┐
│   Originator │
│(原发器)      │
│- state       │
│+ save()      │
│+ restore()   │
└──────────────┘
     │
     │ creates
     ▼
┌──────────────┐
│   Memento    │
│(备忘录)      │
│- state(只读) │
└──────────────┘
     △
     │ manages
┌──────────────┐
│  Caretaker   │
│(管理者)      │
│- mementos    │
└──────────────┘
```

## 代码实现

### 1. 备忘录类

```java
public class TextMemento {
    private String state;
    private LocalDateTime timestamp;
    
    public TextMemento(String state) {
        this.state = state;
        this.timestamp = LocalDateTime.now();
    }
    
    public String getState() {
        return state;
    }
    
    public LocalDateTime getTimestamp() {
        return timestamp;
    }
}
```

### 2. 原发器类

```java
public class TextEditor {
    private String content = "";
    
    public void write(String text) {
        this.content = text;
    }
    
    public String getContent() {
        return content;
    }
    
    // 保存状态
    public TextMemento save() {
        return new TextMemento(content);
    }
    
    // 恢复状态
    public void restore(TextMemento memento) {
        this.content = memento.getState();
    }
}
```

### 3. 管理者类

```java
public class EditorHistory {
    private Stack<TextMemento> history = new Stack<>();
    
    public void save(TextEditor editor) {
        history.push(editor.save());
    }
    
    public void undo(TextEditor editor) {
        if (!history.isEmpty()) {
            TextMemento memento = history.pop();
            editor.restore(memento);
        }
    }
    
    public int getHistorySize() {
        return history.size();
    }
}
```

### 4. 客户端使用

```java
public class Demo {
    public static void main(String[] args) {
        TextEditor editor = new TextEditor();
        EditorHistory history = new EditorHistory();
        
        editor.write("Hello");
        history.save(editor);
        System.out.println("内容: " + editor.getContent());
        
        editor.write("Hello World");
        history.save(editor);
        System.out.println("内容: " + editor.getContent());
        
        editor.write("Hello World Java");
        System.out.println("内容: " + editor.getContent());
        
        // 撤销
        history.undo(editor);
        System.out.println("撤销后: " + editor.getContent());
        
        history.undo(editor);
        System.out.println("再次撤销: " + editor.getContent());
    }
}
```

## 实际应用示例

### 游戏存档

```java
public class GameMemento {
    private String playerName;
    private int level;
    private int score;
    private Map<String, Integer> inventory;
    
    public GameMemento(String playerName, int level, int score, 
                      Map<String, Integer> inventory) {
        this.playerName = playerName;
        this.level = level;
        this.score = score;
        this.inventory = new HashMap<>(inventory);
    }
    
    public String getPlayerName() { return playerName; }
    public int getLevel() { return level; }
    public int getScore() { return score; }
    public Map<String, Integer> getInventory() { return inventory; }
}

public class Game {
    private String playerName;
    private int level;
    private int score;
    private Map<String, Integer> inventory = new HashMap<>();
    
    public Game(String playerName) {
        this.playerName = playerName;
        this.level = 1;
        this.score = 0;
    }
    
    public void play() {
        level++;
        score += 100;
    }
    
    public void addToInventory(String item, int count) {
        inventory.put(item, count);
    }
    
    public GameMemento save() {
        return new GameMemento(playerName, level, score, inventory);
    }
    
    public void load(GameMemento memento) {
        this.playerName = memento.getPlayerName();
        this.level = memento.getLevel();
        this.score = memento.getScore();
        this.inventory = new HashMap<>(memento.getInventory());
    }
    
    public void displayStatus() {
        System.out.println("玩家: " + playerName + ", 等级: " + level + 
                          ", 分数: " + score);
    }
}

// 使用
Game game = new Game("Hero");
game.play();
GameMemento save1 = game.save();

game.play();
game.addToInventory("金币", 100);
game.displayStatus();

game.load(save1);
game.displayStatus();
```

### 配置管理

```java
public class ConfigMemento {
    private Map<String, String> config;
    
    public ConfigMemento(Map<String, String> config) {
        this.config = new HashMap<>(config);
    }
    
    public Map<String, String> getConfig() {
        return new HashMap<>(config);
    }
}

public class Configuration {
    private Map<String, String> config = new HashMap<>();
    
    public void set(String key, String value) {
        config.put(key, value);
    }
    
    public String get(String key) {
        return config.get(key);
    }
    
    public ConfigMemento createMemento() {
        return new ConfigMemento(config);
    }
    
    public void restoreMemento(ConfigMemento memento) {
        this.config = memento.getConfig();
    }
}
```

### 数据库事务

```java
public class TransactionMemento {
    private String sqlStatement;
    private LocalDateTime timestamp;
    
    public TransactionMemento(String sqlStatement) {
        this.sqlStatement = sqlStatement;
        this.timestamp = LocalDateTime.now();
    }
    
    public String getSqlStatement() {
        return sqlStatement;
    }
    
    public LocalDateTime getTimestamp() {
        return timestamp;
    }
}

public class DatabaseTransaction {
    private List<String> statements = new ArrayList<>();
    
    public void executeSQL(String sql) {
        statements.add(sql);
    }
    
    public TransactionMemento createSavePoint() {
        return new TransactionMemento(String.join("; ", statements));
    }
    
    public void rollback(TransactionMemento memento) {
        statements.clear();
        // 恢复到存档点
    }
}
```

## 序列化方式实现

```java
public class SerializableMemento {
    private byte[] state;
    
    public SerializableMemento(Object obj) {
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos);
            oos.writeObject(obj);
            oos.close();
            this.state = baos.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    
    public Object getState() {
        try {
            ByteArrayInputStream bais = new ByteArrayInputStream(state);
            ObjectInputStream ois = new ObjectInputStream(bais);
            Object obj = ois.readObject();
            ois.close();
            return obj;
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }
}
```

## 备忘录模式 vs 快照

| 特性 | 备忘录 | 快照 |
|------|-------|------|
| 封装 | 破坏少 | 破坏多 |
| 复杂度 | 高 | 低 |
| 灵活性 | 高 | 一般 |
| 应用 | 通用 | 简单场景 |

## 优缺点

### 优点
- ✅ 保存和恢复对象状态
- ✅ 不破坏对象封装
- ✅ 支持undo/redo
- ✅ 符合单一职责原则

### 缺点
- ❌ 代码复杂性增加
- ❌ 内存占用可能很大
- ❌ 类的数量增多

## 适用场景

- ✓ 需要保存/恢复对象状态
- ✓ undo/redo功能
- ✓ 游戏存档
- ✓ 数据库事务
- ✓ 编辑器快照

## 最佳实践

1. **合理设计Memento** - 只保存必要状态
2. **管理内存** - 限制历史记录数量
3. **性能优化** - 使用增量备份
4. **安全性** - 保护敏感信息
