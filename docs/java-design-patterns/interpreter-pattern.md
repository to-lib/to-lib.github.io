---
sidebar_position: 24
---

# 解释器模式 (Interpreter Pattern)

## 模式定义

**解释器模式**是一种行为型设计模式，它给定一个语言，定义它的文法表示，并定义一个解释器来处理这个语言中的句子。

## 问题分析

当需要处理特定语言或表达式时：

- 需要定义和解析语言
- 需要处理语言中的各种构造
- 频繁添加新的语言结构

## 解决方案

```
┌─────────────────┐
│   Expression    │
│+ interpret()    │
└────────┬────────┘
         △
    ┌────┴─────────────┐
    │                  │
┌────────────┐   ┌──────────────┐
│Terminal    │   │Non-Terminal  │
│Expression  │   │Expression    │
└────────────┘   └──────────────┘
```

## 代码实现

### 1. 定义表达式抽象类

```java
public interface Expression {
    boolean interpret(String context);
}
```

### 2. 终结符表达式

```java
// 终结符表达式 - 叶子节点
public class TerminalExpression implements Expression {
    private String data;
    
    public TerminalExpression(String data) {
        this.data = data;
    }
    
    @Override
    public boolean interpret(String context) {
        return context.contains(data);
    }
}
```

### 3. 非终结符表达式

```java
// 非终结符表达式 - 组合节点
public class AndExpression implements Expression {
    private Expression expr1;
    private Expression expr2;
    
    public AndExpression(Expression expr1, Expression expr2) {
        this.expr1 = expr1;
        this.expr2 = expr2;
    }
    
    @Override
    public boolean interpret(String context) {
        return expr1.interpret(context) && expr2.interpret(context);
    }
}

public class OrExpression implements Expression {
    private Expression expr1;
    private Expression expr2;
    
    public OrExpression(Expression expr1, Expression expr2) {
        this.expr1 = expr1;
        this.expr2 = expr2;
    }
    
    @Override
    public boolean interpret(String context) {
        return expr1.interpret(context) || expr2.interpret(context);
    }
}
```

### 4. 客户端使用

```java
public class Demo {
    public static void main(String[] args) {
        Expression isMale = new TerminalExpression("male");
        Expression isAdult = new TerminalExpression("adult");
        
        // 男性且成人
        Expression maleAndAdult = new AndExpression(isMale, isAdult);
        
        System.out.println("male adult: " + maleAndAdult.interpret("male adult"));
        System.out.println("male: " + maleAndAdult.interpret("male"));
        System.out.println("female adult: " + maleAndAdult.interpret("female adult"));
    }
}
```

## 实际应用示例

### 数学表达式求值

```java
public interface MathExpression {
    int evaluate();
}

public class Number implements MathExpression {
    private int value;
    
    public Number(int value) {
        this.value = value;
    }
    
    @Override
    public int evaluate() {
        return value;
    }
}

public class Add implements MathExpression {
    private MathExpression left;
    private MathExpression right;
    
    public Add(MathExpression left, MathExpression right) {
        this.left = left;
        this.right = right;
    }
    
    @Override
    public int evaluate() {
        return left.evaluate() + right.evaluate();
    }
}

public class Subtract implements MathExpression {
    private MathExpression left;
    private MathExpression right;
    
    public Subtract(MathExpression left, MathExpression right) {
        this.left = left;
        this.right = right;
    }
    
    @Override
    public int evaluate() {
        return left.evaluate() - right.evaluate();
    }
}

public class Multiply implements MathExpression {
    private MathExpression left;
    private MathExpression right;
    
    public Multiply(MathExpression left, MathExpression right) {
        this.left = left;
        this.right = right;
    }
    
    @Override
    public int evaluate() {
        return left.evaluate() * right.evaluate();
    }
}

// 使用: (5 + 3) * 2 = 16
MathExpression expr = new Multiply(
    new Add(new Number(5), new Number(3)),
    new Number(2)
);
System.out.println("结果: " + expr.evaluate());
```

### SQL查询表达式

```java
public interface SQLExpression {
    String toSQL();
}

public class SelectExpression implements SQLExpression {
    private String columns;
    private SQLExpression from;
    
    public SelectExpression(String columns, SQLExpression from) {
        this.columns = columns;
        this.from = from;
    }
    
    @Override
    public String toSQL() {
        return "SELECT " + columns + " " + from.toSQL();
    }
}

public class FromExpression implements SQLExpression {
    private String table;
    private SQLExpression where;
    
    public FromExpression(String table, SQLExpression where) {
        this.table = table;
        this.where = where;
    }
    
    @Override
    public String toSQL() {
        if (where != null) {
            return "FROM " + table + " " + where.toSQL();
        }
        return "FROM " + table;
    }
}

public class WhereExpression implements SQLExpression {
    private String condition;
    
    public WhereExpression(String condition) {
        this.condition = condition;
    }
    
    @Override
    public String toSQL() {
        return "WHERE " + condition;
    }
}

// 使用
SQLExpression expr = new SelectExpression("*",
    new FromExpression("users",
        new WhereExpression("age > 18")
    )
);
System.out.println(expr.toSQL());
// SELECT * FROM users WHERE age > 18
```

### 正则表达式模式匹配

```java
public interface PatternExpression {
    boolean matches(String text);
}

public class LiteralPattern implements PatternExpression {
    private String literal;
    
    public LiteralPattern(String literal) {
        this.literal = literal;
    }
    
    @Override
    public boolean matches(String text) {
        return text.contains(literal);
    }
}

public class StarPattern implements PatternExpression {
    private PatternExpression inner;
    
    public StarPattern(PatternExpression inner) {
        this.inner = inner;
    }
    
    @Override
    public boolean matches(String text) {
        // 匹配0次或多次
        return true;
    }
}

public class SequencePattern implements PatternExpression {
    private PatternExpression first;
    private PatternExpression second;
    
    public SequencePattern(PatternExpression first, PatternExpression second) {
        this.first = first;
        this.second = second;
    }
    
    @Override
    public boolean matches(String text) {
        return first.matches(text) && second.matches(text);
    }
}
```

## 与其他模式的关系

- **访问者模式** - 在已有对象结构上增加操作
- **组合模式** - 处理树形结构
- **工厂模式** - 创建表达式对象

## 优缺点

### 优点
- ✅ 易于改变和扩展语法
- ✅ 实现语法简单
- ✅ 符合开闭原则

### 缺点
- ❌ 语法复杂时性能下降
- ❌ 维护困难
- ❌ 类数量多

## 适用场景

- ✓ 需要解析表达式
- ✓ SQL查询
- ✓ 配置文件解析
- ✓ 脚本语言
- ✓ 模式匹配

## Java中的应用

```java
// 正则表达式 - Pattern是解释器模式
Pattern pattern = Pattern.compile("[a-z]+");
Matcher matcher = pattern.matcher("hello");

// Spring EL表达式
ExpressionParser parser = new SpelExpressionParser();
Expression expr = parser.parseExpression("1 + 2");
int result = (Integer) expr.getValue();

// SQL处理
Statement stmt = connection.createStatement();
ResultSet rs = stmt.executeQuery("SELECT * FROM users");
```

## 何时使用

解释器模式适合：
- 语法相对简单的语言
- 性能不是首要问题
- 需要灵活扩展语法

对于复杂语言，更好的选择是：
- 使用专业的解析器生成工具（如ANTLR）
- 使用编译器框架
