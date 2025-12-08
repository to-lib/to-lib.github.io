---
sidebar_position: 17
---

# 命令模式 (Command Pattern)

## 模式定义

**命令模式**是一种行为型设计模式，它将请求（命令）封装为一个对象，从而使你可以使用不同的请求对客户端进行参数化，对请求排队，记录请求日志，以及支持可撤销的操作。

## 问题分析

当需要执行各种操作，并且可能需要支持撤销、重做、排队或日志记录时：

- 操作与对象紧耦合
- 难以支持撤销功能
- 难以实现操作的延迟执行
- 难以实现事务

## 解决方案

将操作（请求）封装成一个对象：

```
┌──────────┐
│ Command  │
│+ execute()│
└─────┬────┘
      △
      │
 ┌────┴──────────────────┐
 │                       │
┌────────┐          ┌───────────┐
│Concrete│          │Concrete   │
│Command1│          │Command2   │
└────────┘          └───────────┘
```

## 代码实现

### 1. 定义命令接口

```java
public interface Command {
    void execute();
    void undo();
}
```

### 2. 接收者（执行具体操作）

```java
public class Document {
    private StringBuilder content = new StringBuilder();
    
    public void addText(String text) {
        content.append(text);
    }
    
    public void removeLastCharacters(int count) {
        if (count > 0 && count <= content.length()) {
            content.delete(content.length() - count, content.length());
        }
    }
    
    public String getContent() {
        return content.toString();
    }
}
```

### 3. 具体命令

```java
public class AddTextCommand implements Command {
    private Document document;
    private String text;
    
    public AddTextCommand(Document document, String text) {
        this.document = document;
        this.text = text;
    }
    
    @Override
    public void execute() {
        document.addText(text);
    }
    
    @Override
    public void undo() {
        document.removeLastCharacters(text.length());
    }
}

public class DeleteTextCommand implements Command {
    private Document document;
    private String deletedText;
    
    public DeleteTextCommand(Document document, String deletedText) {
        this.document = document;
        this.deletedText = deletedText;
    }
    
    @Override
    public void execute() {
        document.removeLastCharacters(deletedText.length());
    }
    
    @Override
    public void undo() {
        document.addText(deletedText);
    }
}
```

### 4. 请求者（调用者）

```java
public class TextEditor {
    private Document document = new Document();
    private Stack<Command> history = new Stack<>();
    private Stack<Command> redoStack = new Stack<>();
    
    public void execute(Command command) {
        command.execute();
        history.push(command);
        redoStack.clear();  // 执行新命令时清空重做栈
    }
    
    public void undo() {
        if (!history.isEmpty()) {
            Command command = history.pop();
            command.undo();
            redoStack.push(command);
        }
    }
    
    public void redo() {
        if (!redoStack.isEmpty()) {
            Command command = redoStack.pop();
            command.execute();
            history.push(command);
        }
    }
    
    public String getContent() {
        return document.getContent();
    }
}
```

### 5. 客户端使用

```java
public class Demo {
    public static void main(String[] args) {
        TextEditor editor = new TextEditor();
        
        // 执行命令
        editor.execute(new AddTextCommand(editor.document, "Hello"));
        System.out.println("内容: " + editor.getContent());
        
        editor.execute(new AddTextCommand(editor.document, " World"));
        System.out.println("内容: " + editor.getContent());
        
        // 撤销
        editor.undo();
        System.out.println("撤销后: " + editor.getContent());
        
        // 重做
        editor.redo();
        System.out.println("重做后: " + editor.getContent());
    }
}
```

## 实际应用示例

### 遥控器控制家电

```java
public interface Device {
    void on();
    void off();
    void setVolume(int volume);
}

public class Television implements Device {
    @Override
    public void on() {
        System.out.println("电视打开");
    }
    
    @Override
    public void off() {
        System.out.println("电视关闭");
    }
    
    @Override
    public void setVolume(int volume) {
        System.out.println("电视音量: " + volume);
    }
}

// 命令
public interface RemoteCommand {
    void execute();
    void undo();
}

public class TurnOnCommand implements RemoteCommand {
    private Device device;
    
    public TurnOnCommand(Device device) {
        this.device = device;
    }
    
    @Override
    public void execute() {
        device.on();
    }
    
    @Override
    public void undo() {
        device.off();
    }
}

// 遥控器
public class RemoteControl {
    private Stack<RemoteCommand> history = new Stack<>();
    
    public void pressButton(RemoteCommand command) {
        command.execute();
        history.push(command);
    }
    
    public void pressUndo() {
        if (!history.isEmpty()) {
            history.pop().undo();
        }
    }
}
```

### 任务队列

```java
public interface Task extends Command {
}

public class PrintTask implements Task {
    private String message;
    
    public PrintTask(String message) {
        this.message = message;
    }
    
    @Override
    public void execute() {
        System.out.println("打印: " + message);
    }
    
    @Override
    public void undo() {
        System.out.println("取消打印");
    }
}

public class TaskQueue {
    private Queue<Task> queue = new LinkedList<>();
    
    public void addTask(Task task) {
        queue.add(task);
    }
    
    public void executeTasks() {
        while (!queue.isEmpty()) {
            queue.poll().execute();
        }
    }
}
```

### 宏命令（宏）

```java
public class MacroCommand implements Command {
    private List<Command> commands = new ArrayList<>();
    
    public void addCommand(Command command) {
        commands.add(command);
    }
    
    @Override
    public void execute() {
        for (Command command : commands) {
            command.execute();
        }
    }
    
    @Override
    public void undo() {
        for (int i = commands.size() - 1; i >= 0; i--) {
            commands.get(i).undo();
        }
    }
}

// 使用
MacroCommand macro = new MacroCommand();
macro.addCommand(new AddTextCommand(doc, "Hello"));
macro.addCommand(new AddTextCommand(doc, " World"));
macro.execute();  // 执行所有命令
```

### 事务

```java
public class DatabaseTransaction {
    private List<Command> commands = new ArrayList<>();
    private List<Command> executedCommands = new ArrayList<>();
    
    public void addCommand(Command command) {
        commands.add(command);
    }
    
    public boolean commit() {
        try {
            for (Command command : commands) {
                command.execute();
                executedCommands.add(command);
            }
            return true;
        } catch (Exception e) {
            rollback();
            return false;
        }
    }
    
    public void rollback() {
        for (int i = executedCommands.size() - 1; i >= 0; i--) {
            executedCommands.get(i).undo();
        }
        executedCommands.clear();
    }
}
```

## 优缺点

### 优点
- ✅ 将请求和执行解耦
- ✅ 支持撤销和重做
- ✅ 支持事务
- ✅ 支持命令队列
- ✅ 易于添加新命令

### 缺点
- ❌ 类和对象数量增多
- ❌ 代码复杂性增加
- ❌ 内存占用增加

## 适用场景

- ✓ 需要撤销和重做
- ✓ 需要延迟执行
- ✓ 需要排队执行
- ✓ 需要事务支持
- ✓ 需要记录操作日志
- ✓ 宏命令、脚本

## Java中的应用

```java
// Swing中的Action
Action action = new AbstractAction("Click me") {
    @Override
    public void actionPerformed(ActionEvent e) {
        // 执行操作
    }
};

// Spring中的ApplicationEvent
publisher.publishEvent(new CustomEvent(this));

// Java IO
InputStream stream = new FileInputStream("file.txt");
// stream可以看作是一个命令对象
```
