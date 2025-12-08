---
sidebar_position: 19
---

# 责任链模式 (Chain of Responsibility Pattern)

## 模式定义

**责任链模式**是一种行为型设计模式，它让你可以将请求沿着处理者链进行传递，直到其中某个处理者对其进行处理。

## 问题分析

当需要多个对象可能处理一个请求时：

- 发送者不知道谁会处理请求
- 处理者之间需要形成一条链
- 需要灵活地添加和删除处理者

## 解决方案

将处理者组织成一条链，每个处理者决定是否处理请求或将其传给下一个处理者：

```
┌─────────────┐
│  Handler    │
│- next       │
│+ handle()   │
└──────┬──────┘
       △
       │
┌──────┴──────────────┐
│                     │
┌────────────┐  ┌──────────────┐
│ConcreteHandler│  │ConcreteHandler│
│      1     │  │       2      │
└────────────┘  └──────────────┘
```

## 代码实现

### 1. 定义处理者接口

```java
public abstract class ApprovalHandler {
    protected ApprovalHandler nextHandler;
    
    public void setNextHandler(ApprovalHandler nextHandler) {
        this.nextHandler = nextHandler;
    }
    
    public abstract void handle(Request request);
}
```

### 2. 具体处理者

```java
public class Manager extends ApprovalHandler {
    @Override
    public void handle(Request request) {
        if (request.getAmount() <= 10000) {
            System.out.println("经理批准了 " + request.getAmount() + " 元的请求");
        } else if (nextHandler != null) {
            nextHandler.handle(request);
        }
    }
}

public class Director extends ApprovalHandler {
    @Override
    public void handle(Request request) {
        if (request.getAmount() <= 50000) {
            System.out.println("主任批准了 " + request.getAmount() + " 元的请求");
        } else if (nextHandler != null) {
            nextHandler.handle(request);
        }
    }
}

public class CEO extends ApprovalHandler {
    @Override
    public void handle(Request request) {
        if (request.getAmount() <= 100000) {
            System.out.println("CEO批准了 " + request.getAmount() + " 元的请求");
        } else if (nextHandler != null) {
            nextHandler.handle(request);
        } else {
            System.out.println("请求金额过大，无法批准");
        }
    }
}
```

### 3. 请求对象

```java
public class Request {
    private double amount;
    private String description;
    
    public Request(double amount, String description) {
        this.amount = amount;
        this.description = description;
    }
    
    public double getAmount() {
        return amount;
    }
    
    public String getDescription() {
        return description;
    }
}
```

### 4. 客户端使用

```java
public class Demo {
    public static void main(String[] args) {
        // 构建责任链
        ApprovalHandler manager = new Manager();
        ApprovalHandler director = new Director();
        ApprovalHandler ceo = new CEO();
        
        manager.setNextHandler(director);
        director.setNextHandler(ceo);
        
        // 处理请求
        manager.handle(new Request(5000, "办公用品"));
        manager.handle(new Request(30000, "设备购置"));
        manager.handle(new Request(80000, "系统升级"));
        manager.handle(new Request(200000, "基础设施"));
    }
}
```

## 实际应用示例

### 日志系统

```java
public abstract class Logger {
    protected Logger next;
    
    public void setNext(Logger next) {
        this.next = next;
    }
    
    public void log(String message, Level level) {
        if (canHandle(level)) {
            write(message);
        }
        
        if (next != null) {
            next.log(message, level);
        }
    }
    
    protected abstract boolean canHandle(Level level);
    
    protected abstract void write(String message);
}

public enum Level {
    INFO, WARNING, ERROR
}

public class InfoLogger extends Logger {
    @Override
    protected boolean canHandle(Level level) {
        return level == Level.INFO;
    }
    
    @Override
    protected void write(String message) {
        System.out.println("INFO: " + message);
    }
}

public class WarningLogger extends Logger {
    @Override
    protected boolean canHandle(Level level) {
        return level == Level.WARNING;
    }
    
    @Override
    protected void write(String message) {
        System.out.println("WARNING: " + message);
    }
}

public class ErrorLogger extends Logger {
    @Override
    protected boolean canHandle(Level level) {
        return level == Level.ERROR;
    }
    
    @Override
    protected void write(String message) {
        System.out.println("ERROR: " + message);
    }
}

// 使用
Logger logger = new InfoLogger();
logger.setNext(new WarningLogger());
logger.getNext().setNext(new ErrorLogger());

logger.log("系统启动", Level.INFO);
logger.log("内存不足", Level.WARNING);
logger.log("数据库连接失败", Level.ERROR);
```

### 请求处理链

```java
public interface RequestHandler {
    boolean process(HttpRequest request);
    void setNext(RequestHandler next);
}

public class AuthenticationHandler implements RequestHandler {
    private RequestHandler next;
    
    @Override
    public boolean process(HttpRequest request) {
        System.out.println("检查身份验证");
        if (!request.hasToken()) {
            System.out.println("身份验证失败");
            return false;
        }
        
        if (next != null) {
            return next.process(request);
        }
        return true;
    }
    
    @Override
    public void setNext(RequestHandler next) {
        this.next = next;
    }
}

public class AuthorizationHandler implements RequestHandler {
    private RequestHandler next;
    
    @Override
    public boolean process(HttpRequest request) {
        System.out.println("检查授权");
        if (!request.hasPermission()) {
            System.out.println("授权失败");
            return false;
        }
        
        if (next != null) {
            return next.process(request);
        }
        return true;
    }
    
    @Override
    public void setNext(RequestHandler next) {
        this.next = next;
    }
}

public class ValidationHandler implements RequestHandler {
    private RequestHandler next;
    
    @Override
    public boolean process(HttpRequest request) {
        System.out.println("验证请求数据");
        if (!request.isValid()) {
            System.out.println("数据验证失败");
            return false;
        }
        
        if (next != null) {
            return next.process(request);
        }
        return true;
    }
    
    @Override
    public void setNext(RequestHandler next) {
        this.next = next;
    }
}
```

### 事件处理

```java
public abstract class EventHandler {
    protected EventHandler successor;
    
    public void handle(Event event) {
        if (canHandle(event)) {
            process(event);
        } else if (successor != null) {
            successor.handle(event);
        }
    }
    
    protected abstract boolean canHandle(Event event);
    
    protected abstract void process(Event event);
}

public class MouseEventHandler extends EventHandler {
    @Override
    protected boolean canHandle(Event event) {
        return event.getType() == EventType.MOUSE;
    }
    
    @Override
    protected void process(Event event) {
        System.out.println("处理鼠标事件");
    }
}

public class KeyboardEventHandler extends EventHandler {
    @Override
    protected boolean canHandle(Event event) {
        return event.getType() == EventType.KEYBOARD;
    }
    
    @Override
    protected void process(Event event) {
        System.out.println("处理键盘事件");
    }
}
```

## 实现方式对比

### 方式1：链式调用

```java
public abstract class Handler {
    protected Handler next;
    
    public void handle(Request request) {
        if (doHandle(request)) {
            return;
        }
        if (next != null) {
            next.handle(request);
        }
    }
    
    protected abstract boolean doHandle(Request request);
}
```

### 方式2：条件判断

```java
public abstract class Handler {
    protected Handler next;
    
    public void handle(Request request) {
        if (canHandle(request)) {
            process(request);
        } else if (next != null) {
            next.handle(request);
        }
    }
    
    protected abstract boolean canHandle(Request request);
    
    protected abstract void process(Request request);
}
```

## 优缺点

### 优点
- ✅ 符合单一职责原则
- ✅ 符合开闭原则
- ✅ 灵活组合处理链
- ✅ 易于添加或删除处理者

### 缺点
- ❌ 不能保证请求被处理
- ❌ 类的数量增多
- ❌ 性能可能下降（链过长）

## 适用场景

- ✓ 多个对象可能处理请求
- ✓ 发送者不知道接收者
- ✓ 处理者动态确定
- ✓ 日志系统
- ✓ 事件处理
- ✓ 请求审批流程

## Java中的应用

```java
// Servlet Filter
public class AuthFilter implements Filter {
    @Override
    public void doFilter(ServletRequest request, ServletResponse response, 
                        FilterChain chain) throws IOException, ServletException {
        // 处理请求
        chain.doFilter(request, response);
        // 处理响应
    }
}

// Java AWT事件处理
event.dispatch(handler);

// Android事件分发
public boolean dispatchTouchEvent(MotionEvent event) {
    return onTouchEvent(event);
}
```
