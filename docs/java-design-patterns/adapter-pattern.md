---
sidebar_position: 7
---

# 适配器模式 (Adapter Pattern)

## 模式定义

**适配器模式**是一种结构型设计模式，它允许将一个类的接口转换成客户端所期望的另一个接口，使得原本不兼容的类可以一起工作。

## 问题分析

在实际开发中，经常需要整合来自不同来源的类：

- 新系统需要集成第三方库
- 使用现有类但其接口不符合要求
- 需要兼容多个版本的API

直接使用会导致接口不兼容的问题。

## 解决方案

有两种实现方式：

### 方案1：类适配器（继承）

```
┌──────────────────┐
│   Target接口     │
│  + request()     │
└────────┬─────────┘
         △
         │ implements
    ┌────────────────────┐
    │    Adapter         │
    │ + request()        │
    └────────────────────┘
         │ extends
    ┌────────────────────┐
    │  Adaptee(现有类)   │
    │ + specificRequest()│
    └────────────────────┘
```

### 方案2：对象适配器（组合）

```
┌──────────────────┐
│   Target接口     │
│  + request()     │
└────────┬─────────┘
         △
         │ implements
    ┌────────────────────┐
    │    Adapter         │
    │ - adaptee          │
    │ + request()        │
    └────────────────────┘
         │ uses
    ┌────────────────────┐
    │  Adaptee(现有类)   │
    │ + specificRequest()│
    └────────────────────┘
```

## 代码实现

### 场景：支付系统集成

假设系统原有的支付接口和第三方支付库的接口不兼容。

#### 1. 定义系统的目标接口

```java
public interface PaymentGateway {
    boolean processPayment(double amount);
    String getTransactionId();
}
```

#### 2. 第三方支付库（不能修改）

```java
public class ThirdPartyPaymentService {
    public void pay(double amount, String currency) {
        System.out.println("第三方支付: " + amount + " " + currency);
    }
    
    public String getTransactionCode() {
        return "TXN123456";
    }
}
```

#### 3. 对象适配器实现

```java
public class PaymentAdapter implements PaymentGateway {
    private ThirdPartyPaymentService thirdPartyService;
    
    public PaymentAdapter(ThirdPartyPaymentService service) {
        this.thirdPartyService = service;
    }
    
    @Override
    public boolean processPayment(double amount) {
        try {
            thirdPartyService.pay(amount, "CNY");
            return true;
        } catch (Exception e) {
            return false;
        }
    }
    
    @Override
    public String getTransactionId() {
        return thirdPartyService.getTransactionCode();
    }
}
```

#### 4. 客户端使用

```java
public class PaymentProcessor {
    private PaymentGateway gateway;
    
    public PaymentProcessor(PaymentGateway gateway) {
        this.gateway = gateway;
    }
    
    public void checkout(double amount) {
        if (gateway.processPayment(amount)) {
            System.out.println("交易ID: " + gateway.getTransactionId());
        } else {
            System.out.println("支付失败");
        }
    }
    
    public static void main(String[] args) {
        // 使用第三方支付服务
        ThirdPartyPaymentService service = new ThirdPartyPaymentService();
        PaymentGateway adapter = new PaymentAdapter(service);
        
        PaymentProcessor processor = new PaymentProcessor(adapter);
        processor.checkout(100.0);
    }
}
```

## 实际应用示例

### 类适配器（使用继承）

```java
public class PrinterAdapter extends Printer implements USB {
    @Override
    public void transferData() {
        print();
    }
    
    private void print() {
        System.out.println("打印文档");
    }
}
```

### 数据格式适配器

```java
// 目标接口
public interface DataConverter {
    String convertToJSON();
}

// 现有类
public class XMLData {
    public String getXML() {
        return "<data><name>张三</name></data>";
    }
}

// 适配器
public class XMLToJSONAdapter implements DataConverter {
    private XMLData xmlData;
    
    public XMLToJSONAdapter(XMLData xmlData) {
        this.xmlData = xmlData;
    }
    
    @Override
    public String convertToJSON() {
        String xml = xmlData.getXML();
        // 简单转换示例
        return "{\"name\":\"张三\"}";
    }
}
```

### 日志适配器

```java
// 系统接口
public interface Logger {
    void log(String message);
}

// 第三方日志库
public class Log4j {
    public void info(String msg) {
        System.out.println("[INFO] " + msg);
    }
}

// 适配器
public class Log4jAdapter implements Logger {
    private Log4j log4j = new Log4j();
    
    @Override
    public void log(String message) {
        log4j.info(message);
    }
}
```

## 类适配器 vs 对象适配器

| 特性 | 类适配器 | 对象适配器 |
|------|--------|---------|
| 实现方式 | 继承 | 组合 |
| 灵活性 | 低 | 高 |
| 所需代码 | 少 | 多 |
| 可复用 | 差 | 好 |
| 推荐 | × | ✓ |

## Java中的适配器

```java
// 集合框架的适配器
List<String> list = Arrays.asList("a", "b", "c");

// IO流的适配器
Reader reader = new InputStreamReader(inputStream, "UTF-8");

// 事件监听器的适配器
public abstract class WindowAdapter extends WindowListener {
    // 默认实现空方法
}
```

## 优缺点

### 优点
- ✅ 增加了类的复用率
- ✅ 提高了类的透明性
- ✅ 符合开闭原则
- ✅ 灵活地集成不兼容的类

### 缺点
- ❌ 过多使用会增加代码复杂性
- ❌ 不利于理解代码流程
- ❌ 增加系统维护难度

## 适用场景

- ✓ 系统需要使用现有的类
- ✓ 与第三方库集成
- ✓ 需要兼容多个版本
- ✓ 接口转换
- ✓ 数据格式转换

## 区分相似模式

- **适配器** - 转换接口以实现兼容
- **装饰器** - 添加新功能而保持接口
- **外观** - 简化复杂子系统
- **代理** - 控制对象访问
