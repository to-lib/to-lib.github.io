---
sidebar_position: 15
title: 设计模式
---

# JavaScript 设计模式

> [!TIP]
> 设计模式是解决常见编程问题的最佳实践，掌握它们能写出更优雅、可维护的代码。

## 🎯 单例模式

确保一个类只有一个实例。

```javascript
class Singleton {
  static instance = null;

  static getInstance() {
    if (!Singleton.instance) {
      Singleton.instance = new Singleton();
    }
    return Singleton.instance;
  }

  constructor() {
    if (Singleton.instance) {
      return Singleton.instance;
    }
  }
}

const a = Singleton.getInstance();
const b = Singleton.getInstance();
console.log(a === b); // true
```

### 应用场景

```javascript
// 全局状态管理
class Store {
  static instance = null;
  state = {};

  static getInstance() {
    if (!Store.instance) {
      Store.instance = new Store();
    }
    return Store.instance;
  }

  getState() {
    return this.state;
  }

  setState(newState) {
    this.state = { ...this.state, ...newState };
  }
}
```

## 📢 观察者模式

对象间一对多的依赖关系，当一个对象改变时，所有依赖者都会收到通知。

```javascript
class Subject {
  observers = [];

  addObserver(observer) {
    this.observers.push(observer);
  }

  removeObserver(observer) {
    this.observers = this.observers.filter((o) => o !== observer);
  }

  notify(data) {
    this.observers.forEach((observer) => observer.update(data));
  }
}

class Observer {
  constructor(name) {
    this.name = name;
  }

  update(data) {
    console.log(`${this.name} 收到通知:`, data);
  }
}

// 使用
const subject = new Subject();
const observer1 = new Observer("观察者1");
const observer2 = new Observer("观察者2");

subject.addObserver(observer1);
subject.addObserver(observer2);
subject.notify("数据更新了"); // 两个观察者都收到通知
```

## 📡 发布订阅模式

比观察者模式更解耦，通过事件中心通信。

```javascript
class EventEmitter {
  events = {};

  on(event, callback) {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    this.events[event].push(callback);
    return () => this.off(event, callback);
  }

  off(event, callback) {
    if (!this.events[event]) return;
    this.events[event] = this.events[event].filter((cb) => cb !== callback);
  }

  emit(event, ...args) {
    if (!this.events[event]) return;
    this.events[event].forEach((callback) => callback(...args));
  }

  once(event, callback) {
    const wrapper = (...args) => {
      callback(...args);
      this.off(event, wrapper);
    };
    this.on(event, wrapper);
  }
}

// 使用
const bus = new EventEmitter();

bus.on("login", (user) => console.log(`${user} 登录了`));
bus.on("login", (user) => console.log(`欢迎 ${user}`));

bus.emit("login", "Alice");
// Alice 登录了
// 欢迎 Alice
```

### 观察者 vs 发布订阅

| 特点     | 观察者模式         | 发布订阅模式 |
| -------- | ------------------ | ------------ |
| 耦合度   | 观察者知道被观察者 | 完全解耦     |
| 中间层   | 无                 | 事件中心     |
| 通信方式 | 直接调用           | 通过事件名   |

## 🏭 工厂模式

封装对象创建逻辑。

```javascript
// 简单工厂
class UserFactory {
  static create(type) {
    switch (type) {
      case "admin":
        return new Admin();
      case "user":
        return new User();
      case "guest":
        return new Guest();
      default:
        throw new Error("未知用户类型");
    }
  }
}

class Admin {
  role = "admin";
  permissions = ["read", "write", "delete"];
}

class User {
  role = "user";
  permissions = ["read", "write"];
}

class Guest {
  role = "guest";
  permissions = ["read"];
}

// 使用
const admin = UserFactory.create("admin");
const user = UserFactory.create("user");
```

## 🎭 策略模式

定义一系列算法，使它们可以互换。

```javascript
// 策略对象
const strategies = {
  add: (a, b) => a + b,
  subtract: (a, b) => a - b,
  multiply: (a, b) => a * b,
  divide: (a, b) => a / b,
};

// 上下文
class Calculator {
  execute(strategy, a, b) {
    return strategies[strategy](a, b);
  }
}

const calc = new Calculator();
calc.execute("add", 5, 3); // 8
calc.execute("multiply", 5, 3); // 15
```

### 表单验证示例

```javascript
const validators = {
  required: (value) => value.trim() !== "" || "此字段必填",
  email: (value) =>
    /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value) || "邮箱格式不正确",
  minLength: (min) => (value) =>
    value.length >= min || `最少需要 ${min} 个字符`,
};

function validate(value, rules) {
  for (const rule of rules) {
    const result = rule(value);
    if (result !== true) return result;
  }
  return true;
}

// 使用
validate("", [validators.required]); // "此字段必填"
validate("test@", [validators.email]); // "邮箱格式不正确"
validate("test@example.com", [validators.email]); // true
```

## 🔌 代理模式

为对象提供一个代理，控制对原对象的访问。

```javascript
// 使用 Proxy 实现
const user = {
  name: "Alice",
  age: 25,
  _password: "secret",
};

const userProxy = new Proxy(user, {
  get(target, prop) {
    // 禁止访问私有属性
    if (prop.startsWith("_")) {
      throw new Error("不能访问私有属性");
    }
    return target[prop];
  },

  set(target, prop, value) {
    // 验证年龄
    if (prop === "age" && (value < 0 || value > 150)) {
      throw new Error("年龄无效");
    }
    target[prop] = value;
    return true;
  },
});

userProxy.name; // 'Alice'
// userProxy._password;  // Error: 不能访问私有属性
userProxy.age = 200; // Error: 年龄无效
```

### 缓存代理

```javascript
function createCachedFetch(fetcher) {
  const cache = new Map();

  return new Proxy(fetcher, {
    apply(target, thisArg, args) {
      const key = JSON.stringify(args);

      if (cache.has(key)) {
        console.log("从缓存返回");
        return cache.get(key);
      }

      const result = target.apply(thisArg, args);
      cache.set(key, result);
      return result;
    },
  });
}
```

## 🎨 装饰器模式

动态地给对象添加额外的职责。

```javascript
// 函数装饰器
function withLogging(fn) {
  return function (...args) {
    console.log(`调用 ${fn.name}，参数:`, args);
    const result = fn.apply(this, args);
    console.log(`结果:`, result);
    return result;
  };
}

function add(a, b) {
  return a + b;
}

const loggedAdd = withLogging(add);
loggedAdd(2, 3);
// 调用 add，参数: [2, 3]
// 结果: 5
```

### 类装饰器（ES 提案）

```javascript
// TypeScript / Babel 装饰器
function log(target, name, descriptor) {
  const original = descriptor.value;

  descriptor.value = function (...args) {
    console.log(`调用 ${name}`);
    return original.apply(this, args);
  };

  return descriptor;
}

class Calculator {
  @log
  add(a, b) {
    return a + b;
  }
}
```

## 💡 模式选择指南

| 场景          | 推荐模式   |
| ------------- | ---------- |
| 全局唯一实例  | 单例模式   |
| 状态变化通知  | 观察者模式 |
| 组件间通信    | 发布订阅   |
| 复杂对象创建  | 工厂模式   |
| 多种算法切换  | 策略模式   |
| 访问控制/缓存 | 代理模式   |
| 扩展对象功能  | 装饰器模式 |

## 🔗 相关资源

- [闭包与作用域](/docs/frontend/javascript/closure)
- [原型链](/docs/frontend/javascript/prototype)
- [函数式编程](/docs/frontend/javascript/functional)

---

**下一步**：学习 [手写实现](/docs/frontend/javascript/implementations) 掌握常见面试题。
