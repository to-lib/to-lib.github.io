---
sidebar_position: 18
---

# 状态模式 (State Pattern)

## 模式定义

**状态模式**是一种行为型设计模式，它允许一个对象在其内部状态改变时改变它的行为。对象看起来好像改变了它的类。

## 问题分析

当一个对象的行为取决于其状态，并且它必须在运行时根据状态改变行为时：

- 条件判断语句过多
- 代码复杂且难以维护
- 易出错

## 解决方案

将每个状态定义为一个单独的类，将状态转换逻辑放在这些类中：

```
┌──────────────┐
│   Context    │
│- state       │
│+ request()   │
└──────┬───────┘
       │ delegates to
       △
    ┌──────────┐
    │ State    │
    │interface │
    └────┬─────┘
         △
    ┌────┴───────────────┐
    │                    │
┌────────┐          ┌─────────┐
│State A │          │State B  │
└────────┘          └─────────┘
```

## 代码实现

### 1. 定义状态接口

```java
public interface DocumentState {
    void publish(Document document);
    void review(Document document);
    void reject(Document document);
}
```

### 2. 具体状态类

```java
public class DraftState implements DocumentState {
    @Override
    public void publish(Document document) {
        document.setState(new ReviewState());
        System.out.println("文档已提交审核");
    }
    
    @Override
    public void review(Document document) {
        System.out.println("草稿文档无法审核");
    }
    
    @Override
    public void reject(Document document) {
        System.out.println("草稿文档无法拒绝");
    }
}

public class ReviewState implements DocumentState {
    @Override
    public void publish(Document document) {
        System.out.println("文档正在审核，无法发布");
    }
    
    @Override
    public void review(Document document) {
        document.setState(new PublishedState());
        System.out.println("文档审核通过，已发布");
    }
    
    @Override
    public void reject(Document document) {
        document.setState(new DraftState());
        System.out.println("文档审核未通过，返回草稿");
    }
}

public class PublishedState implements DocumentState {
    @Override
    public void publish(Document document) {
        System.out.println("文档已发布，无需再发布");
    }
    
    @Override
    public void review(Document document) {
        System.out.println("文档已发布，无法审核");
    }
    
    @Override
    public void reject(Document document) {
        System.out.println("文档已发布，无法拒绝");
    }
}
```

### 3. 上下文类

```java
public class Document {
    private DocumentState state = new DraftState();
    
    public void setState(DocumentState state) {
        this.state = state;
    }
    
    public void publish() {
        state.publish(this);
    }
    
    public void review() {
        state.review(this);
    }
    
    public void reject() {
        state.reject(this);
    }
}
```

### 4. 客户端使用

```java
public class Demo {
    public static void main(String[] args) {
        Document document = new Document();
        
        document.publish();  // 草稿 → 审核中
        document.review();   // 审核中 → 已发布
        document.publish();  // 已发布，无需再发布
        document.reject();   // 已发布，无法拒绝
    }
}
```

## 实际应用示例

### 订单状态

```java
public interface OrderState {
    void pay(Order order);
    void ship(Order order);
    void receive(Order order);
    void cancel(Order order);
}

public class PendingPaymentState implements OrderState {
    @Override
    public void pay(Order order) {
        order.setState(new ShippingState());
        System.out.println("付款成功，订单已发货");
    }
    
    @Override
    public void ship(Order order) {
        System.out.println("尚未付款，无法发货");
    }
    
    @Override
    public void receive(Order order) {
        System.out.println("订单未发货，无法收货");
    }
    
    @Override
    public void cancel(Order order) {
        order.setState(new CancelledState());
        System.out.println("订单已取消");
    }
}

public class ShippingState implements OrderState {
    @Override
    public void pay(Order order) {
        System.out.println("订单已付款");
    }
    
    @Override
    public void ship(Order order) {
        System.out.println("订单已发货");
    }
    
    @Override
    public void receive(Order order) {
        order.setState(new DeliveredState());
        System.out.println("订单已送达");
    }
    
    @Override
    public void cancel(Order order) {
        System.out.println("订单已发货，无法取消");
    }
}

public class DeliveredState implements OrderState {
    @Override
    public void pay(Order order) {
        System.out.println("订单已送达，无需付款");
    }
    
    @Override
    public void ship(Order order) {
        System.out.println("订单已送达");
    }
    
    @Override
    public void receive(Order order) {
        System.out.println("订单已收货");
    }
    
    @Override
    public void cancel(Order order) {
        System.out.println("订单已送达，无法取消");
    }
}

public class Order {
    private OrderState state = new PendingPaymentState();
    
    public void setState(OrderState state) {
        this.state = state;
    }
    
    public void pay() {
        state.pay(this);
    }
    
    public void ship() {
        state.ship(this);
    }
    
    public void receive() {
        state.receive(this);
    }
    
    public void cancel() {
        state.cancel(this);
    }
}
```

### TCP连接状态

```java
public interface TCPState {
    void open(TCPConnection connection);
    void close(TCPConnection connection);
    void send(TCPConnection connection, String data);
}

public class EstablishedState implements TCPState {
    @Override
    public void open(TCPConnection connection) {
        System.out.println("连接已建立");
    }
    
    @Override
    public void close(TCPConnection connection) {
        connection.setState(new ClosedState());
        System.out.println("连接已关闭");
    }
    
    @Override
    public void send(TCPConnection connection, String data) {
        System.out.println("发送数据: " + data);
    }
}

public class ClosedState implements TCPState {
    @Override
    public void open(TCPConnection connection) {
        connection.setState(new EstablishedState());
        System.out.println("连接已建立");
    }
    
    @Override
    public void close(TCPConnection connection) {
        System.out.println("连接已关闭");
    }
    
    @Override
    public void send(TCPConnection connection, String data) {
        System.out.println("错误：连接已关闭，无法发送");
    }
}

public class TCPConnection {
    private TCPState state = new ClosedState();
    
    public void setState(TCPState state) {
        this.state = state;
    }
    
    public void open() {
        state.open(this);
    }
    
    public void close() {
        state.close(this);
    }
    
    public void send(String data) {
        state.send(this, data);
    }
}
```

### 游戏角色状态

```java
public interface PlayerState {
    void attack(Player player);
    void defend(Player player);
    void rest(Player player);
}

public class NormalState implements PlayerState {
    @Override
    public void attack(Player player) {
        System.out.println("正常攻击，造成10点伤害");
        player.setHealth(player.getHealth() - 5);
    }
    
    @Override
    public void defend(Player player) {
        player.setState(new DefendState());
        System.out.println("进入防守状态");
    }
    
    @Override
    public void rest(Player player) {
        System.out.println("休息，恢复20点生命");
        player.setHealth(player.getHealth() + 20);
    }
}

public class DefendState implements PlayerState {
    @Override
    public void attack(Player player) {
        player.setState(new NormalState());
        System.out.println("退出防守，进行反击");
    }
    
    @Override
    public void defend(Player player) {
        System.out.println("已在防守中，伤害减少70%");
    }
    
    @Override
    public void rest(Player player) {
        player.setState(new NormalState());
        System.out.println("停止防守");
    }
}

public class Player {
    private int health = 100;
    private PlayerState state = new NormalState();
    
    public void setState(PlayerState state) {
        this.state = state;
    }
    
    public void attack() {
        state.attack(this);
    }
    
    public void defend() {
        state.defend(this);
    }
    
    public void rest() {
        state.rest(this);
    }
    
    public int getHealth() {
        return health;
    }
    
    public void setHealth(int health) {
        this.health = Math.max(0, Math.min(100, health));
        System.out.println("当前生命值: " + this.health);
    }
}
```

## 状态模式 vs 策略模式

| 特性 | 状态模式 | 策略模式 |
|------|--------|---------|
| 目的 | 改变行为 | 选择算法 |
| 状态转换 | 由对象控制 | 由客户端控制 |
| 对象关系 | 状态机 | 无关系 |
| 适用 | 状态转换 | 不同算法 |

## 优缺点

### 优点
- ✅ 消除大量条件判断
- ✅ 符合开闭原则
- ✅ 状态转换清晰
- ✅ 便于维护和扩展

### 缺点
- ❌ 类和对象数量增多
- ❌ 代码复杂性增加
- ❌ 状态转换分散

## 适用场景

- ✓ 对象有多个状态
- ✓ 状态间有转换规则
- ✓ 不同状态有不同行为
- ✓ 条件判断复杂
- ✓ 订单、流程审批、游戏状态等

## 状态机

可以使用状态机工具库简化实现：

```java
// Spring State Machine
@Configuration
@EnableStateMachine
public class StateMachineConfig extends EnumStateMachineConfigurerAdapter<States, Events> {
    
    @Override
    public void configure(StateMachineStateConfigurer<States, Events> states) throws Exception {
        states
            .withStates()
            .initial(States.DRAFT)
            .states(EnumSet.allOf(States.class));
    }
}
```
