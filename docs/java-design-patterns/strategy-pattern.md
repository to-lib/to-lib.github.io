---
sidebar_position: 5
---

# 策略模式 (Strategy Pattern)

## 模式定义

**策略模式**是一种行为型设计模式，它定义了一族算法，将每个算法封装起来，使它们可以相互替换。策略模式使得算法的变化独立于使用算法的客户。

## 问题分析

如果有多种算法实现同一功能，直接在代码中使用条件判断会导致：

```java
// 不好的做法
public class PaymentProcessor {
    public void pay(String method, double amount) {
        if ("CREDIT_CARD".equals(method)) {
            // 信用卡支付逻辑
        } else if ("PAYPAL".equals(method)) {
            // PayPal支付逻辑
        } else if ("WECHAT".equals(method)) {
            // 微信支付逻辑
        }
    }
}
```

**问题**：
- 条件判断复杂
- 添加新支付方式需要修改代码
- 违反开闭原则

## 解决方案

```
┌─────────────────────────┐
│  Strategy（策略接口）    │
│  + execute()            │
└──────────┬──────────────┘
           △
           │ implements
    ┌──────┼──────┐
    │      │      │
┌────────┐ │ ┌────────┐
│Algorithm│ │ │Algorithm│
│   A     │ │ │   B     │
└────────┘ │ └────────┘
           │
        ┌──────┐
        │Algorithm C│
        └──────┘

┌─────────────────────────────┐
│  Context（环境）            │
│  - strategy: Strategy       │
│  + executeStrategy()        │
└─────────────────────────────┘
```

## 代码实现

### 1. 定义策略接口

```java
public interface PaymentStrategy {
    void pay(double amount);
}
```

### 2. 具体策略实现

```java
public class CreditCardPayment implements PaymentStrategy {
    private String cardNumber;
    
    public CreditCardPayment(String cardNumber) {
        this.cardNumber = cardNumber;
    }
    
    @Override
    public void pay(double amount) {
        System.out.println("使用信用卡 " + cardNumber + " 支付: ¥" + amount);
    }
}

public class PayPalPayment implements PaymentStrategy {
    private String email;
    
    public PayPalPayment(String email) {
        this.email = email;
    }
    
    @Override
    public void pay(double amount) {
        System.out.println("使用PayPal账户 " + email + " 支付: $" + amount);
    }
}

public class WeChatPayment implements PaymentStrategy {
    private String userId;
    
    public WeChatPayment(String userId) {
        this.userId = userId;
    }
    
    @Override
    public void pay(double amount) {
        System.out.println("使用微信账户 " + userId + " 支付: ¥" + amount);
    }
}
```

### 3. 环境类

```java
public class ShoppingCart {
    private PaymentStrategy paymentStrategy;
    private double totalPrice;
    
    public void setPaymentStrategy(PaymentStrategy strategy) {
        this.paymentStrategy = strategy;
    }
    
    public void addItem(double price) {
        this.totalPrice += price;
    }
    
    public void checkout() {
        if (paymentStrategy == null) {
            throw new IllegalStateException("请先选择支付方式");
        }
        paymentStrategy.pay(totalPrice);
    }
}
```

### 4. 客户端使用

```java
public class Demo {
    public static void main(String[] args) {
        ShoppingCart cart = new ShoppingCart();
        cart.addItem(100);
        cart.addItem(50);
        
        // 使用信用卡支付
        cart.setPaymentStrategy(new CreditCardPayment("1234 5678 9012 3456"));
        cart.checkout();
        
        // 使用微信支付
        cart.setPaymentStrategy(new WeChatPayment("user123"));
        cart.checkout();
        
        // 使用PayPal支付
        cart.setPaymentStrategy(new PayPalPayment("user@example.com"));
        cart.checkout();
    }
}
```

## 实际应用示例

### 数据排序策略

```java
public interface SortingStrategy {
    void sort(int[] array);
}

public class BubbleSort implements SortingStrategy {
    @Override
    public void sort(int[] array) {
        System.out.println("使用冒泡排序");
        // 冒泡排序实现
    }
}

public class QuickSort implements SortingStrategy {
    @Override
    public void sort(int[] array) {
        System.out.println("使用快速排序");
        // 快速排序实现
    }
}

public class Sorter {
    private SortingStrategy strategy;
    
    public Sorter(SortingStrategy strategy) {
        this.strategy = strategy;
    }
    
    public void sort(int[] array) {
        strategy.sort(array);
    }
}

// 使用
Sorter sorter = new Sorter(new QuickSort());
sorter.sort(array);
```

### 文件压缩策略

```java
public interface CompressionStrategy {
    void compress(String filePath);
}

public class ZipCompression implements CompressionStrategy {
    @Override
    public void compress(String filePath) {
        System.out.println("使用ZIP格式压缩: " + filePath);
    }
}

public class RarCompression implements CompressionStrategy {
    @Override
    public void compress(String filePath) {
        System.out.println("使用RAR格式压缩: " + filePath);
    }
}

public class FileCompressor {
    private CompressionStrategy strategy;
    
    public void setStrategy(CompressionStrategy strategy) {
        this.strategy = strategy;
    }
    
    public void compressFile(String filePath) {
        strategy.compress(filePath);
    }
}
```

### 折扣计算策略

```java
public interface DiscountStrategy {
    double applyDiscount(double originalPrice);
}

public class NoDiscount implements DiscountStrategy {
    @Override
    public double applyDiscount(double originalPrice) {
        return originalPrice;
    }
}

public class StudentDiscount implements DiscountStrategy {
    @Override
    public double applyDiscount(double originalPrice) {
        return originalPrice * 0.8; // 8折
    }
}

public class VIPDiscount implements DiscountStrategy {
    @Override
    public double applyDiscount(double originalPrice) {
        return originalPrice * 0.5; // 5折
    }
}

public class PriceCalculator {
    private DiscountStrategy discountStrategy = new NoDiscount();
    
    public void setDiscountStrategy(DiscountStrategy strategy) {
        this.discountStrategy = strategy;
    }
    
    public double calculatePrice(double originalPrice) {
        return discountStrategy.applyDiscount(originalPrice);
    }
}
```

## 策略模式 vs 状态模式

| 特性 | 策略模式 | 状态模式 |
|------|---------|---------|
| 目的 | 选择算法 | 改变行为 |
| 改变时机 | 客户端指定 | 对象状态改变 |
| 耦合度 | 低 | 中 |
| 使用场景 | 多种算法 | 状态转换 |

## 优缺点

### 优点
- ✅ 避免多重条件判断
- ✅ 符合开闭原则，易于扩展
- ✅ 算法复用性好
- ✅ 符合单一职责原则
- ✅ 可运行时切换算法

### 缺点
- ❌ 增加类和对象数量
- ❌ 客户端需了解所有策略
- ❌ 对于少数算法可能过度设计

## 适用场景

- ✓ 多种算法实现同一功能
- ✓ 避免使用条件判断选择算法
- ✓ 需要动态选择算法
- ✓ 算法经常需要更换或扩展
- ✓ 支付方式、排序算法、压缩方式等

## 最佳实践

1. **优先使用策略模式** - 替代多重if-else
2. **配合工厂模式** - 创建策略对象
3. **不可变策略** - 策略类通常无状态
4. **参数化** - 在构造函数中注入依赖
5. **文档清晰** - 说明每种策略的适用场景
