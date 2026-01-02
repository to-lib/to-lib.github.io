---
sidebar_position: 16
title: 手写实现
---

# JavaScript 手写实现

> [!TIP]
> 手写实现常见功能是前端面试的高频考点，也能帮助你深入理解 JavaScript 原理。

## 🎯 防抖与节流

### 防抖 (Debounce)

停止触发后延迟执行。

```javascript
function debounce(fn, delay, immediate = false) {
  let timer = null;

  return function (...args) {
    const callNow = immediate && !timer;

    clearTimeout(timer);

    timer = setTimeout(() => {
      timer = null;
      if (!immediate) {
        fn.apply(this, args);
      }
    }, delay);

    if (callNow) {
      fn.apply(this, args);
    }
  };
}

// 使用
const search = debounce((query) => {
  console.log("搜索:", query);
}, 300);

input.addEventListener("input", (e) => search(e.target.value));
```

### 节流 (Throttle)

固定频率执行。

```javascript
function throttle(fn, limit) {
  let lastTime = 0;
  let timer = null;

  return function (...args) {
    const now = Date.now();

    if (now - lastTime >= limit) {
      lastTime = now;
      fn.apply(this, args);
    } else if (!timer) {
      timer = setTimeout(() => {
        lastTime = Date.now();
        timer = null;
        fn.apply(this, args);
      }, limit - (now - lastTime));
    }
  };
}

// 使用
window.addEventListener("scroll", throttle(handleScroll, 100));
```

## 📦 深拷贝

```javascript
function deepClone(obj, hash = new WeakMap()) {
  if (obj === null || typeof obj !== "object") {
    return obj;
  }

  // 处理循环引用
  if (hash.has(obj)) {
    return hash.get(obj);
  }

  // 处理特殊对象
  if (obj instanceof Date) return new Date(obj);
  if (obj instanceof RegExp) return new RegExp(obj);
  if (obj instanceof Map) {
    const map = new Map();
    hash.set(obj, map);
    obj.forEach((v, k) => map.set(deepClone(k, hash), deepClone(v, hash)));
    return map;
  }
  if (obj instanceof Set) {
    const set = new Set();
    hash.set(obj, set);
    obj.forEach((v) => set.add(deepClone(v, hash)));
    return set;
  }

  // 处理数组和普通对象
  const clone = Array.isArray(obj) ? [] : {};
  hash.set(obj, clone);

  for (const key of Reflect.ownKeys(obj)) {
    clone[key] = deepClone(obj[key], hash);
  }

  return clone;
}
```

## 🔧 call / apply / bind

### 手写 call

```javascript
Function.prototype.myCall = function (context, ...args) {
  context = context ?? globalThis;
  context = Object(context);

  const key = Symbol();
  context[key] = this;

  const result = context[key](...args);
  delete context[key];

  return result;
};

// 测试
function greet(greeting) {
  return `${greeting}, ${this.name}`;
}
greet.myCall({ name: "Alice" }, "Hello"); // "Hello, Alice"
```

### 手写 apply

```javascript
Function.prototype.myApply = function (context, args = []) {
  context = context ?? globalThis;
  context = Object(context);

  const key = Symbol();
  context[key] = this;

  const result = context[key](...args);
  delete context[key];

  return result;
};
```

### 手写 bind

```javascript
Function.prototype.myBind = function (context, ...args) {
  const fn = this;

  return function bound(...newArgs) {
    // 处理 new 调用
    if (new.target) {
      return new fn(...args, ...newArgs);
    }
    return fn.apply(context, [...args, ...newArgs]);
  };
};

// 测试
const bound = greet.myBind({ name: "Alice" }, "Hi");
bound(); // "Hi, Alice"
```

## 🤝 Promise

### 基础实现

```javascript
class MyPromise {
  static PENDING = "pending";
  static FULFILLED = "fulfilled";
  static REJECTED = "rejected";

  constructor(executor) {
    this.status = MyPromise.PENDING;
    this.value = undefined;
    this.reason = undefined;
    this.onFulfilledCallbacks = [];
    this.onRejectedCallbacks = [];

    const resolve = (value) => {
      if (this.status === MyPromise.PENDING) {
        this.status = MyPromise.FULFILLED;
        this.value = value;
        this.onFulfilledCallbacks.forEach((fn) => fn());
      }
    };

    const reject = (reason) => {
      if (this.status === MyPromise.PENDING) {
        this.status = MyPromise.REJECTED;
        this.reason = reason;
        this.onRejectedCallbacks.forEach((fn) => fn());
      }
    };

    try {
      executor(resolve, reject);
    } catch (error) {
      reject(error);
    }
  }

  then(onFulfilled, onRejected) {
    onFulfilled = typeof onFulfilled === "function" ? onFulfilled : (v) => v;
    onRejected =
      typeof onRejected === "function"
        ? onRejected
        : (e) => {
            throw e;
          };

    return new MyPromise((resolve, reject) => {
      const handle = (callback, value) => {
        queueMicrotask(() => {
          try {
            const result = callback(value);
            if (result instanceof MyPromise) {
              result.then(resolve, reject);
            } else {
              resolve(result);
            }
          } catch (error) {
            reject(error);
          }
        });
      };

      if (this.status === MyPromise.FULFILLED) {
        handle(onFulfilled, this.value);
      } else if (this.status === MyPromise.REJECTED) {
        handle(onRejected, this.reason);
      } else {
        this.onFulfilledCallbacks.push(() => handle(onFulfilled, this.value));
        this.onRejectedCallbacks.push(() => handle(onRejected, this.reason));
      }
    });
  }

  catch(onRejected) {
    return this.then(null, onRejected);
  }

  finally(onFinally) {
    return this.then(
      (value) => MyPromise.resolve(onFinally()).then(() => value),
      (reason) =>
        MyPromise.resolve(onFinally()).then(() => {
          throw reason;
        })
    );
  }

  static resolve(value) {
    if (value instanceof MyPromise) return value;
    return new MyPromise((resolve) => resolve(value));
  }

  static reject(reason) {
    return new MyPromise((_, reject) => reject(reason));
  }
}
```

### Promise.all

```javascript
MyPromise.all = function (promises) {
  return new MyPromise((resolve, reject) => {
    const results = [];
    let count = 0;

    if (promises.length === 0) {
      return resolve(results);
    }

    promises.forEach((promise, index) => {
      MyPromise.resolve(promise).then(
        (value) => {
          results[index] = value;
          count++;
          if (count === promises.length) {
            resolve(results);
          }
        },
        (reason) => reject(reason)
      );
    });
  });
};
```

### Promise.race

```javascript
MyPromise.race = function (promises) {
  return new MyPromise((resolve, reject) => {
    promises.forEach((promise) => {
      MyPromise.resolve(promise).then(resolve, reject);
    });
  });
};
```

## 🆕 new 操作符

```javascript
function myNew(Constructor, ...args) {
  // 创建新对象，继承构造函数原型
  const obj = Object.create(Constructor.prototype);

  // 执行构造函数
  const result = Constructor.apply(obj, args);

  // 返回对象（如果构造函数返回对象则使用它）
  return result instanceof Object ? result : obj;
}

// 测试
function Person(name) {
  this.name = name;
}
const p = myNew(Person, "Alice");
console.log(p.name); // "Alice"
console.log(p instanceof Person); // true
```

## 🔗 instanceof

```javascript
function myInstanceof(obj, Constructor) {
  if (obj === null || typeof obj !== "object") {
    return false;
  }

  let proto = Object.getPrototypeOf(obj);

  while (proto !== null) {
    if (proto === Constructor.prototype) {
      return true;
    }
    proto = Object.getPrototypeOf(proto);
  }

  return false;
}

// 测试
myInstanceof([], Array); // true
myInstanceof({}, Array); // false
```

## 📋 数组方法

### Array.prototype.map

```javascript
Array.prototype.myMap = function (callback, thisArg) {
  const result = [];

  for (let i = 0; i < this.length; i++) {
    if (i in this) {
      result[i] = callback.call(thisArg, this[i], i, this);
    }
  }

  return result;
};
```

### Array.prototype.reduce

```javascript
Array.prototype.myReduce = function (callback, initialValue) {
  let accumulator = initialValue;
  let startIndex = 0;

  if (accumulator === undefined) {
    accumulator = this[0];
    startIndex = 1;
  }

  for (let i = startIndex; i < this.length; i++) {
    if (i in this) {
      accumulator = callback(accumulator, this[i], i, this);
    }
  }

  return accumulator;
};
```

### Array.prototype.flat

```javascript
Array.prototype.myFlat = function (depth = 1) {
  const result = [];

  const flatten = (arr, d) => {
    for (const item of arr) {
      if (Array.isArray(item) && d > 0) {
        flatten(item, d - 1);
      } else {
        result.push(item);
      }
    }
  };

  flatten(this, depth);
  return result;
};

// 测试
[1, [2, [3, [4]]]].myFlat(2); // [1, 2, 3, [4]]
```

## 🔗 相关资源

- [原型链](/docs/frontend/javascript/prototype)
- [异步编程](/docs/frontend/javascript/async)
- [设计模式](/docs/frontend/javascript/design-patterns)

---

**下一步**：学习 [数据结构](/docs/frontend/javascript/data-structures) 掌握基础算法。
