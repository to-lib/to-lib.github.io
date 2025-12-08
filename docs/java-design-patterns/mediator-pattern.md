---
sidebar_position: 21
---

# 中介者模式 (Mediator Pattern)

## 模式定义

**中介者模式**是一种行为型设计模式，它定义了一个中介对象来封装一组对象间的交互，使这些对象不必直接相互交互，而是通过中介对象来通信。

## 问题分析

当多个对象之间存在复杂的通信关系时：

- 对象之间紧耦合
- 通信逻辑复杂
- 难以维护和扩展

## 解决方案

引入中介者对象来管理对象间的通信：

```
┌──────────────┐
│   Mediator   │
│(中介者)      │
│- colleagues  │
│+ send()      │
└───────┬──────┘
        │
    ┌───┴───────────────┐
    │                   │
┌────────┐         ┌────────┐
│Colleague         │Colleague│
│    1    │        │    2    │
└────────┘        └────────┘
```

## 代码实现

### 1. 定义中介者接口

```java
public interface ChatRoomMediator {
    void registerUser(User user);
    void sendMessage(String message, User sender);
}
```

### 2. 具体中介者

```java
public class ChatRoom implements ChatRoomMediator {
    private List<User> users = new ArrayList<>();
    
    @Override
    public void registerUser(User user) {
        users.add(user);
        user.setMediator(this);
        System.out.println(user.getName() + " 加入了聊天室");
    }
    
    @Override
    public void sendMessage(String message, User sender) {
        System.out.println(sender.getName() + " 说: " + message);
        for (User user : users) {
            if (user != sender) {
                user.receive(sender.getName() + ": " + message);
            }
        }
    }
}
```

### 3. 定义同事类

```java
public abstract class User {
    protected String name;
    protected ChatRoomMediator mediator;
    
    public User(String name) {
        this.name = name;
    }
    
    public void setMediator(ChatRoomMediator mediator) {
        this.mediator = mediator;
    }
    
    public String getName() {
        return name;
    }
    
    public abstract void send(String message);
    
    public abstract void receive(String message);
}
```

### 4. 具体同事类

```java
public class ConcreteUser extends User {
    public ConcreteUser(String name) {
        super(name);
    }
    
    @Override
    public void send(String message) {
        mediator.sendMessage(message, this);
    }
    
    @Override
    public void receive(String message) {
        System.out.println(name + " 收到: " + message);
    }
}
```

### 5. 客户端使用

```java
public class Demo {
    public static void main(String[] args) {
        ChatRoomMediator chatRoom = new ChatRoom();
        
        User alice = new ConcreteUser("Alice");
        User bob = new ConcreteUser("Bob");
        User charlie = new ConcreteUser("Charlie");
        
        chatRoom.registerUser(alice);
        chatRoom.registerUser(bob);
        chatRoom.registerUser(charlie);
        
        alice.send("大家好!");
        bob.send("你好 Alice!");
    }
}
```

## 实际应用示例

### 航空管制

```java
public interface AirportMediator {
    void registerAirline(Airline airline);
    void requestLanding(Airline airline);
    void requestTakeoff(Airline airline);
}

public class Airport implements AirportMediator {
    private List<Airline> airlines = new ArrayList<>();
    private Queue<Airline> landingQueue = new LinkedList<>();
    private Queue<Airline> takeoffQueue = new LinkedList<>();
    
    @Override
    public void registerAirline(Airline airline) {
        airlines.add(airline);
        airline.setMediator(this);
    }
    
    @Override
    public void requestLanding(Airline airline) {
        landingQueue.add(airline);
        processPendingRequests();
    }
    
    @Override
    public void requestTakeoff(Airline airline) {
        takeoffQueue.add(airline);
        processPendingRequests();
    }
    
    private void processPendingRequests() {
        if (!landingQueue.isEmpty()) {
            Airline airline = landingQueue.poll();
            System.out.println(airline.getName() + " 允许着陆");
        }
        
        if (!takeoffQueue.isEmpty()) {
            Airline airline = takeoffQueue.poll();
            System.out.println(airline.getName() + " 允许起飞");
        }
    }
}

public abstract class Airline {
    protected String name;
    protected AirportMediator mediator;
    
    public Airline(String name) {
        this.name = name;
    }
    
    public void setMediator(AirportMediator mediator) {
        this.mediator = mediator;
    }
    
    public String getName() {
        return name;
    }
    
    public abstract void land();
    
    public abstract void takeoff();
}

public class Aircraft extends Airline {
    public Aircraft(String name) {
        super(name);
    }
    
    @Override
    public void land() {
        mediator.requestLanding(this);
    }
    
    @Override
    public void takeoff() {
        mediator.requestTakeoff(this);
    }
}
```

### UI对话框

```java
public interface DialogMediator {
    void registerComponent(UIComponent component);
    void send(String message, UIComponent sender);
}

public class LoginDialog implements DialogMediator {
    private UIComponent usernameField;
    private UIComponent passwordField;
    private UIComponent loginButton;
    private UIComponent registerButton;
    
    public LoginDialog() {
    }
    
    @Override
    public void registerComponent(UIComponent component) {
        component.setMediator(this);
    }
    
    @Override
    public void send(String message, UIComponent sender) {
        if (sender == loginButton) {
            login();
        } else if (sender == registerButton) {
            register();
        }
    }
    
    private void login() {
        System.out.println("用户登录");
    }
    
    private void register() {
        System.out.println("用户注册");
    }
}

public abstract class UIComponent {
    protected DialogMediator mediator;
    
    public void setMediator(DialogMediator mediator) {
        this.mediator = mediator;
    }
    
    public abstract void click();
}

public class Button extends UIComponent {
    private String label;
    
    public Button(String label) {
        this.label = label;
    }
    
    @Override
    public void click() {
        mediator.send("clicked", this);
    }
}
```

### 团队成员协调

```java
public interface TeamMediator {
    void displayTeamDetails();
    void addTeamMember(TeamMember member);
    void sendMessage(String message, TeamMember sender);
}

public class Team implements TeamMediator {
    private List<TeamMember> members = new ArrayList<>();
    
    @Override
    public void displayTeamDetails() {
        System.out.println("=== 团队成员 ===");
        for (TeamMember member : members) {
            System.out.println("- " + member.getName());
        }
    }
    
    @Override
    public void addTeamMember(TeamMember member) {
        members.add(member);
        member.setMediator(this);
    }
    
    @Override
    public void sendMessage(String message, TeamMember sender) {
        System.out.println(sender.getName() + ": " + message);
    }
}

public abstract class TeamMember {
    protected String name;
    protected TeamMediator mediator;
    
    public TeamMember(String name) {
        this.name = name;
    }
    
    public void setMediator(TeamMediator mediator) {
        this.mediator = mediator;
    }
    
    public String getName() {
        return name;
    }
    
    public abstract void work();
}

public class Developer extends TeamMember {
    public Developer(String name) {
        super(name);
    }
    
    @Override
    public void work() {
        mediator.sendMessage("完成了编码任务", this);
    }
}

public class Designer extends TeamMember {
    public Designer(String name) {
        super(name);
    }
    
    @Override
    public void work() {
        mediator.sendMessage("完成了设计方案", this);
    }
}
```

## 中介者模式 vs 观察者模式

| 特性 | 中介者 | 观察者 |
|------|-------|--------|
| 关系 | 多对多 | 一对多 |
| 通信 | 通过中介者 | 直接通知 |
| 对象知晓 | 不知道彼此 | 知道发布者 |
| 耦合度 | 低 | 中 |

## 优缺点

### 优点
- ✅ 降低对象间的耦合
- ✅ 集中管理对象间的交互
- ✅ 符合单一职责原则
- ✅ 符合开闭原则

### 缺点
- ❌ 中介者可能变得复杂
- ❌ 中介者成为瓶颈
- ❌ 不利于理解通信流程

## 适用场景

- ✓ 对象间通信复杂
- ✓ 需要集中管理交互
- ✓ 对象不应该紧耦合
- ✓ 聊天系统、协调系统
- ✓ 航空管制、UI对话框

## 最佳实践

1. **定义清晰的接口** - 明确中介者职责
2. **避免中介者复杂** - 可分解为多个中介者
3. **合理使用** - 不要过度设计
4. **结合其他模式** - 如工厂模式、观察者模式
