---
sidebar_position: 16
---

# 模板方法模式 (Template Method Pattern)

## 模式定义

**模板方法模式**是一种行为型设计模式，它定义了一个算法的骨架，但将某些步骤的实现延迟到子类。这样可以使子类在不改变算法结构的前提下，重新定义算法的某些步骤。

## 问题分析

当多个类有相似的算法流程，但实现细节不同时：

- 代码重复
- 维护困难
- 易出错

## 解决方案

在父类中定义算法的骨架，将变化的部分留给子类实现。

```
┌─────────────────────────┐
│   AbstractClass         │
│  - templateMethod()     │
│  + primitiveOperation1()│
│  + primitiveOperation2()│
└────────────┬────────────┘
             △
             │ extends
    ┌────────┴──────────┐
    │                   │
┌───────────┐      ┌──────────┐
│ConcreteClass1    │ConcreteClass2│
└───────────┘      └──────────┘
```

## 代码实现

### 1. 定义抽象类

```java
public abstract class CoffeeMaker {
    // 模板方法 - 定义算法骨架
    public final void makeCoffee() {
        boilWater();
        brewCoffee();
        pourInCup();
        addCondiments();
    }
    
    // 具体步骤
    private void boilWater() {
        System.out.println("烧水");
    }
    
    private void pourInCup() {
        System.out.println("倒入杯子");
    }
    
    // 抽象方法 - 由子类实现
    protected abstract void brewCoffee();
    
    protected abstract void addCondiments();
}
```

### 2. 具体实现类

```java
public class AmericanCoffeeMaker extends CoffeeMaker {
    @Override
    protected void brewCoffee() {
        System.out.println("用热水冲咖啡粉");
    }
    
    @Override
    protected void addCondiments() {
        System.out.println("添加糖和奶油");
    }
}

public class EspressoCoffeeMaker extends CoffeeMaker {
    @Override
    protected void brewCoffee() {
        System.out.println("用意式咖啡机压制咖啡");
    }
    
    @Override
    protected void addCondiments() {
        System.out.println("添加牛奶");
    }
}
```

### 3. 客户端使用

```java
public class Demo {
    public static void main(String[] args) {
        CoffeeMaker americanMaker = new AmericanCoffeeMaker();
        americanMaker.makeCoffee();
        
        System.out.println("\n---\n");
        
        CoffeeMaker espressoMaker = new EspressoCoffeeMaker();
        espressoMaker.makeCoffee();
    }
}
```

## 实际应用示例

### 文件导出

```java
public abstract class DataExporter {
    public final void export(String filename) {
        validateData();
        transformData();
        writeToFile(filename);
        compress(filename);
    }
    
    protected abstract void validateData();
    
    protected abstract void transformData();
    
    protected abstract void writeToFile(String filename);
    
    protected abstract void compress(String filename);
}

public class CSVExporter extends DataExporter {
    @Override
    protected void validateData() {
        System.out.println("验证CSV数据");
    }
    
    @Override
    protected void transformData() {
        System.out.println("转换为CSV格式");
    }
    
    @Override
    protected void writeToFile(String filename) {
        System.out.println("写入CSV文件: " + filename);
    }
    
    @Override
    protected void compress(String filename) {
        System.out.println("压缩CSV文件");
    }
}

public class XMLExporter extends DataExporter {
    @Override
    protected void validateData() {
        System.out.println("验证XML数据");
    }
    
    @Override
    protected void transformData() {
        System.out.println("转换为XML格式");
    }
    
    @Override
    protected void writeToFile(String filename) {
        System.out.println("写入XML文件: " + filename);
    }
    
    @Override
    protected void compress(String filename) {
        System.out.println("压缩XML文件");
    }
}
```

### 游戏角色行动

```java
public abstract class GameCharacter {
    public final void act() {
        moveToTarget();
        attack();
        defend();
        specialAbility();
    }
    
    protected abstract void moveToTarget();
    
    protected abstract void attack();
    
    protected abstract void defend();
    
    protected abstract void specialAbility();
}

public class Warrior extends GameCharacter {
    @Override
    protected void moveToTarget() {
        System.out.println("战士冲向敌人");
    }
    
    @Override
    protected void attack() {
        System.out.println("战士挥动大剑");
    }
    
    @Override
    protected void defend() {
        System.out.println("战士举起盾牌");
    }
    
    @Override
    protected void specialAbility() {
        System.out.println("战士发动旋风斩");
    }
}

public class Mage extends GameCharacter {
    @Override
    protected void moveToTarget() {
        System.out.println("法师移动到安全距离");
    }
    
    @Override
    protected void attack() {
        System.out.println("法师释放火球");
    }
    
    @Override
    protected void defend() {
        System.out.println("法师启动魔法护盾");
    }
    
    @Override
    protected void specialAbility() {
        System.out.println("法师释放冰雨术");
    }
}
```

### 数据处理流程

```java
public abstract class DataProcessor {
    public final void process(String inputFile) {
        String data = readData(inputFile);
        data = cleanData(data);
        data = transformData(data);
        analyzeData(data);
        saveResult(data);
    }
    
    protected abstract String readData(String inputFile);
    
    protected abstract String cleanData(String data);
    
    protected abstract String transformData(String data);
    
    protected abstract void analyzeData(String data);
    
    protected abstract void saveResult(String data);
}

public class JSONDataProcessor extends DataProcessor {
    @Override
    protected String readData(String inputFile) {
        System.out.println("从JSON文件读取数据");
        return "json_data";
    }
    
    @Override
    protected String cleanData(String data) {
        System.out.println("清理JSON数据");
        return data;
    }
    
    @Override
    protected String transformData(String data) {
        System.out.println("转换JSON数据");
        return data;
    }
    
    @Override
    protected void analyzeData(String data) {
        System.out.println("分析JSON数据");
    }
    
    @Override
    protected void saveResult(String data) {
        System.out.println("保存结果");
    }
}
```

### HTTP请求处理

```java
public abstract class HttpHandler {
    public final void handleRequest(HttpRequest request) {
        if (!authenticate(request)) {
            return;
        }
        
        if (!validate(request)) {
            return;
        }
        
        process(request);
        log(request);
    }
    
    protected abstract boolean authenticate(HttpRequest request);
    
    protected abstract boolean validate(HttpRequest request);
    
    protected abstract void process(HttpRequest request);
    
    protected abstract void log(HttpRequest request);
}

public class UserHandler extends HttpHandler {
    @Override
    protected boolean authenticate(HttpRequest request) {
        System.out.println("验证用户身份");
        return true;
    }
    
    @Override
    protected boolean validate(HttpRequest request) {
        System.out.println("验证用户数据");
        return true;
    }
    
    @Override
    protected void process(HttpRequest request) {
        System.out.println("处理用户请求");
    }
    
    @Override
    protected void log(HttpRequest request) {
        System.out.println("记录用户操作日志");
    }
}
```

## 模板方法模式 vs 策略模式

| 特性 | 模板方法 | 策略模式 |
|------|--------|---------|
| 继承 | 使用 | 不使用 |
| 算法选择 | 子类 | 客户端 |
| 灵活性 | 中 | 高 |
| 代码复用 | 好 | 一般 |

## 钩子方法

可选的钩子方法让子类更灵活：

```java
public abstract class CoffeeWithHook {
    public final void makeCoffee() {
        boilWater();
        brewCoffee();
        pourInCup();
        
        if (wantCondiments()) {
            addCondiments();
        }
    }
    
    protected abstract void brewCoffee();
    
    protected abstract void addCondiments();
    
    // 钩子方法 - 可选
    protected boolean wantCondiments() {
        return true;
    }
    
    private void boilWater() {
        System.out.println("烧水");
    }
    
    private void pourInCup() {
        System.out.println("倒入杯子");
    }
}

public class BlackCoffee extends CoffeeWithHook {
    @Override
    protected void brewCoffee() {
        System.out.println("冲咖啡");
    }
    
    @Override
    protected void addCondiments() {
        System.out.println("添加调料");
    }
    
    @Override
    protected boolean wantCondiments() {
        return false;  // 不要调料
    }
}
```

## 优缺点

### 优点
- ✅ 提高代码复用性
- ✅ 反向控制
- ✅ 降低代码重复
- ✅ 易于维护

### 缺点
- ❌ 增加了抽象类数量
- ❌ 违反了里氏替换原则的某些方面
- ❌ 算法骨架改变需要修改所有子类

## 适用场景

- ✓ 多个类有相似的算法
- ✓ 算法骨架相同，细节不同
- ✓ 需要代码复用
- ✓ 涉及通用处理流程

## Java中的应用

```java
// Collections.sort使用模板方法
Collections.sort(list, comparator);

// AbstractList
public abstract class AbstractList<E> {
    public void add(int index, E element) {
        // 模板方法
    }
}

// Servlet
public abstract class HttpServlet {
    protected void service(HttpRequest req, HttpResponse res) {
        // 模板方法
        if ("GET".equals(req.getMethod())) {
            doGet(req, res);
        }
    }
    
    protected abstract void doGet(HttpRequest req, HttpResponse res);
}
```
