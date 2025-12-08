---
sidebar_position: 4
---

# 观察者模式 (Observer Pattern)

## 模式定义

**观察者模式**是一种行为型设计模式，它定义了一种一对多的依赖关系，当一个对象的状态发生改变时，所有依赖于它的对象都会得到通知并自动更新。

## 问题分析

在许多应用中，需要在某个对象的状态改变时，通知多个其他对象：

- 事件驱动编程
- MVC架构中Model的变化通知View
- 消息订阅系统
- 实时通知系统
- 事件总线

直接实现会导致对象间的高耦合。

## 解决方案

```
┌──────────────────────────────────────────┐
│         Subject（主题）                   │
│  ┌────────────────────────────────────┐ │
│  │ - observers: List<Observer>        │ │
│  │ - state                            │ │
│  │ + attach(observer)                 │ │
│  │ + detach(observer)                 │ │
│  │ + notify()                         │ │
│  │ + setState(state)                  │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
            △ notifies
            │
            ├─────────────────────────┐
            │                         │
       ┌────────────┐           ┌────────────┐
       │ Observer A │           │ Observer B │
       └────────────┘           └────────────┘
```

## 代码实现

### 1. 定义观察者接口

```java
public interface Observer {
    void update(Subject subject);
}
```

### 2. 定义主题类

```java
public class Subject {
    private List<Observer> observers = new ArrayList<>();
    private int state;
    
    // 附加观察者
    public void attach(Observer observer) {
        if (!observers.contains(observer)) {
            observers.add(observer);
        }
    }
    
    // 移除观察者
    public void detach(Observer observer) {
        observers.remove(observer);
    }
    
    // 通知所有观察者
    private void notifyObservers() {
        for (Observer observer : observers) {
            observer.update(this);
        }
    }
    
    // 设置状态
    public void setState(int state) {
        if (this.state != state) {
            this.state = state;
            notifyObservers();
        }
    }
    
    public int getState() {
        return state;
    }
}
```

### 3. 具体观察者实现

```java
public class ConcreteObserverA implements Observer {
    @Override
    public void update(Subject subject) {
        if (subject.getState() < 10) {
            System.out.println("观察者A: 状态小于10，需要处理");
        }
    }
}

public class ConcreteObserverB implements Observer {
    @Override
    public void update(Subject subject) {
        if (subject.getState() >= 10) {
            System.out.println("观察者B: 状态大于等于10，需要处理");
        }
    }
}
```

### 4. 客户端使用

```java
public class Demo {
    public static void main(String[] args) {
        Subject subject = new Subject();
        
        Observer observerA = new ConcreteObserverA();
        Observer observerB = new ConcreteObserverB();
        
        subject.attach(observerA);
        subject.attach(observerB);
        
        subject.setState(5);   // 触发observerA
        subject.setState(15);  // 触发observerB
    }
}
```

## 实际应用示例

### 事件发布订阅系统

```java
// 事件接口
public interface EventListener {
    void update(String eventType, String data);
}

// 事件发布者
public class EventSource {
    private List<EventListener> listeners = new ArrayList<>();
    
    public void subscribe(EventListener listener) {
        listeners.add(listener);
    }
    
    public void unsubscribe(EventListener listener) {
        listeners.remove(listener);
    }
    
    public void publishEvent(String eventType, String data) {
        for (EventListener listener : listeners) {
            listener.update(eventType, data);
        }
    }
}

// 具体监听器
public class EmailNotifier implements EventListener {
    @Override
    public void update(String eventType, String data) {
        System.out.println("邮件通知: " + eventType + " - " + data);
    }
}

public class SMSNotifier implements EventListener {
    @Override
    public void update(String eventType, String data) {
        System.out.println("短信通知: " + eventType + " - " + data);
    }
}

// 使用
public class Main {
    public static void main(String[] args) {
        EventSource eventSource = new EventSource();
        
        eventSource.subscribe(new EmailNotifier());
        eventSource.subscribe(new SMSNotifier());
        
        eventSource.publishEvent("ORDER_CREATED", "订单号：12345");
    }
}
```

### 股票价格监控系统

```java
public interface StockObserver {
    void update(String stockCode, double price);
}

public class StockMarket {
    private Map<String, Double> prices = new HashMap<>();
    private Map<String, List<StockObserver>> observers = new HashMap<>();
    
    public void subscribe(String stockCode, StockObserver observer) {
        observers.computeIfAbsent(stockCode, k -> new ArrayList<>()).add(observer);
    }
    
    public void priceChanged(String stockCode, double newPrice) {
        prices.put(stockCode, newPrice);
        
        List<StockObserver> stockObservers = observers.get(stockCode);
        if (stockObservers != null) {
            for (StockObserver observer : stockObservers) {
                observer.update(stockCode, newPrice);
            }
        }
    }
}

public class Investor implements StockObserver {
    private String name;
    
    public Investor(String name) {
        this.name = name;
    }
    
    @Override
    public void update(String stockCode, double price) {
        System.out.println(name + " 收到通知: " + stockCode + " 当前价格: " + price);
    }
}
```

## Java内置观察者支持

Java提供了 `Observer` 和 `Observable` 类（已过时）：

```java
// 不推荐使用，已过时
import java.util.Observer;
import java.util.Observable;

public class WeatherData extends Observable {
    private int temperature;
    
    public void setTemperature(int temp) {
        this.temperature = temp;
        setChanged();
        notifyObservers(temperature);
    }
}
```

**现代做法**：使用事件监听器或反应式编程框架。

## 观察者模式 vs 发布-订阅模式

| 特性 | 观察者模式 | 发布-订阅模式 |
|------|---------|----------|
| 耦合度 | 紧耦合 | 松耦合 |
| 中介 | 直接通知 | 通过消息中介 |
| 适用 | 简单场景 | 复杂系统 |
| 性能 | 高 | 低 |

## 优缺点

### 优点
- ✅ 实现了对象间的低耦合
- ✅ 支持一对多的通信
- ✅ 符合开闭原则
- ✅ 易于扩展和维护

### 缺点
- ❌ 观察者过多时性能下降
- ❌ 无法保证通知顺序
- ❌ 可能造成内存泄漏
- ❌ 调试困难

## 适用场景

- ✓ 事件驱动系统
- ✓ MVC架构
- ✓ 模型与视图的分离
- ✓ 消息发布订阅系统
- ✓ 实时通知系统
- ✓ 数据绑定

## 最佳实践

1. **防止内存泄漏** - 及时unsubscribe
2. **异步通知** - 耗时操作使用线程池
3. **异常处理** - 单个观察者异常不影响其他观察者
4. **顺序问题** - 明确定义通知顺序
5. **性能优化** - 避免频繁的通知
