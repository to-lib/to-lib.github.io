---
sidebar_position: 12
title: this 关键字
---

# JavaScript this 关键字

> [!TIP] > `this` 是 JavaScript 中最容易混淆的概念之一。理解 this 的绑定规则是掌握 JS 的关键。

## 🎯 this 是什么？

`this` 是函数执行时的上下文对象，它的值取决于函数**如何被调用**，而不是在哪里定义。

## 📦 四种绑定规则

### 1. 默认绑定

独立函数调用时，`this` 指向全局对象（严格模式下为 `undefined`）。

```javascript
function sayHi() {
  console.log(this); // window（浏览器）
}
sayHi();

// 严格模式
("use strict");
function sayHi() {
  console.log(this); // undefined
}
```

### 2. 隐式绑定

通过对象调用时，`this` 指向调用该方法的对象。

```javascript
const user = {
  name: "Alice",
  greet() {
    console.log(`Hi, I'm ${this.name}`);
  },
};

user.greet(); // Hi, I'm Alice
```

#### ⚠️ 隐式丢失

```javascript
const user = {
  name: "Alice",
  greet() {
    console.log(this.name);
  },
};

const greet = user.greet;
greet(); // undefined（丢失了 this）

// 回调函数也会丢失
setTimeout(user.greet, 100); // undefined
```

### 3. 显式绑定

使用 `call`、`apply`、`bind` 明确指定 this。

```javascript
function greet(greeting) {
  console.log(`${greeting}, ${this.name}`);
}

const user = { name: "Alice" };

// call - 立即调用，逐个传参
greet.call(user, "Hello"); // Hello, Alice

// apply - 立即调用，数组传参
greet.apply(user, ["Hi"]); // Hi, Alice

// bind - 返回新函数，永久绑定
const boundGreet = greet.bind(user);
boundGreet("Hey"); // Hey, Alice
```

### 4. new 绑定

使用 `new` 调用构造函数时，`this` 指向新创建的对象。

```javascript
function Person(name) {
  this.name = name;
  // 隐式返回 this
}

const alice = new Person("Alice");
console.log(alice.name); // Alice
```

## ⚡ 箭头函数

箭头函数**没有自己的 this**，它继承外层作用域的 this。

```javascript
const user = {
  name: "Alice",
  greet() {
    // 普通函数 - 有自己的 this
    setTimeout(function () {
      console.log(this.name); // undefined
    }, 100);

    // 箭头函数 - 继承外层 this
    setTimeout(() => {
      console.log(this.name); // Alice
    }, 100);
  },
};

user.greet();
```

### 箭头函数的特点

```javascript
const obj = {
  // ❌ 箭头函数作为方法 - this 指向外层
  badMethod: () => {
    console.log(this); // window
  },

  // ✅ 普通函数作为方法
  goodMethod() {
    console.log(this); // obj
  },
};
```

## 🔢 优先级

从高到低：

1. **new 绑定** - `new Foo()`
2. **显式绑定** - `call/apply/bind`
3. **隐式绑定** - `obj.method()`
4. **默认绑定** - `func()`

```javascript
function foo() {
  console.log(this.a);
}

const obj1 = { a: 2 };
const obj2 = { a: 3 };

const bar = foo.bind(obj1);
bar.call(obj2); // 2（bind 优先于 call）

new bar(); // undefined（new 优先于 bind）
```

## 🎮 实际应用

### 事件处理

```javascript
class Button {
  constructor(text) {
    this.text = text;
  }

  // ❌ 直接使用会丢失 this
  handleClick() {
    console.log(this.text);
  }

  // ✅ 方案1：箭头函数
  handleClick = () => {
    console.log(this.text);
  };

  // ✅ 方案2：bind
  constructor(text) {
    this.text = text;
    this.handleClick = this.handleClick.bind(this);
  }
}

const btn = new Button("Click me");
document.addEventListener("click", btn.handleClick);
```

### React 组件

```jsx
class Counter extends React.Component {
  state = { count: 0 };

  // ✅ 箭头函数自动绑定
  increment = () => {
    this.setState({ count: this.state.count + 1 });
  };

  render() {
    return <button onClick={this.increment}>{this.state.count}</button>;
  }
}
```

## 💡 判断 this 的技巧

1. 箭头函数？→ 继承外层 this
2. `new` 调用？→ 新创建的对象
3. `call/apply/bind`？→ 指定的对象
4. 对象方法调用？→ 调用的对象
5. 都不是？→ 全局对象或 undefined

## 🔗 相关资源

- [闭包与作用域](/docs/frontend/javascript/closure)
- [原型链](/docs/frontend/javascript/prototype)
- [ES6+](/docs/frontend/javascript/es6)

---

**下一步**：学习 [深浅拷贝](/docs/frontend/javascript/copy) 掌握对象复制技巧。
