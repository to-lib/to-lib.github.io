---
sidebar_position: 13
title: 深浅拷贝
---

# JavaScript 深浅拷贝

> [!TIP]
> 理解引用类型的拷贝机制，是避免数据意外修改的关键。

## 🎯 为什么需要拷贝？

```javascript
// 基本类型 - 值传递
let a = 1;
let b = a;
b = 2;
console.log(a); // 1（不受影响）

// 引用类型 - 引用传递
let obj1 = { name: "Alice" };
let obj2 = obj1;
obj2.name = "Bob";
console.log(obj1.name); // 'Bob'（被修改了！）
```

## 📦 浅拷贝

只复制第一层，嵌套对象仍共享引用。

### 对象浅拷贝

```javascript
const original = {
  name: "Alice",
  info: { age: 25 },
};

// 方法1：展开运算符（推荐）
const copy1 = { ...original };

// 方法2：Object.assign
const copy2 = Object.assign({}, original);

// 验证
copy1.name = "Bob"; // ✅ 不影响原对象
copy1.info.age = 30; // ❌ 影响原对象
console.log(original.info.age); // 30
```

### 数组浅拷贝

```javascript
const arr = [1, 2, { value: 3 }];

// 方法1：展开运算符
const copy1 = [...arr];

// 方法2：slice
const copy2 = arr.slice();

// 方法3：concat
const copy3 = [].concat(arr);

// 方法4：Array.from
const copy4 = Array.from(arr);

// 验证
copy1[0] = 100; // ✅ 不影响原数组
copy1[2].value = 999; // ❌ 影响原数组
```

## 🔄 深拷贝

复制所有层级，完全独立。

### JSON 方法（简单但有限制）

```javascript
const original = {
  name: "Alice",
  info: { age: 25 },
};

const copy = JSON.parse(JSON.stringify(original));

copy.info.age = 30;
console.log(original.info.age); // 25 ✅
```

#### ⚠️ JSON 方法的限制

```javascript
const obj = {
  func: () => {}, // ❌ 函数丢失
  date: new Date(), // ❌ 变成字符串
  regex: /abc/, // ❌ 变成空对象
  undef: undefined, // ❌ 属性丢失
  symbol: Symbol(), // ❌ 属性丢失
  circular: null, // ❌ 循环引用报错
};

obj.circular = obj;
JSON.stringify(obj); // Error!
```

### structuredClone（现代方案，推荐）

```javascript
const original = {
  name: "Alice",
  info: { age: 25 },
  date: new Date(),
  arr: [1, 2, 3],
  map: new Map([["key", "value"]]),
  set: new Set([1, 2, 3]),
};

const copy = structuredClone(original);

copy.info.age = 30;
console.log(original.info.age); // 25 ✅
```

#### structuredClone 的优势

- ✅ 支持 Date、Map、Set、ArrayBuffer 等
- ✅ 支持循环引用
- ❌ 不支持函数、Symbol、DOM 节点

### 递归实现

```javascript
function deepClone(obj, hash = new WeakMap()) {
  // 基本类型直接返回
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

  // 创建新对象/数组
  const clone = Array.isArray(obj) ? [] : {};
  hash.set(obj, clone);

  // 递归复制属性
  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      clone[key] = deepClone(obj[key], hash);
    }
  }

  return clone;
}

// 使用
const copy = deepClone(original);
```

### 使用第三方库

```javascript
// lodash
import _ from "lodash";
const copy = _.cloneDeep(original);
```

## 📊 方法对比

| 方法                   | 深度 | 函数 | Date | 循环引用 |
| ---------------------- | ---- | ---- | ---- | -------- |
| `{...obj}`             | 浅   | ✅   | ✅   | ✅       |
| `Object.assign`        | 浅   | ✅   | ✅   | ✅       |
| `JSON.parse/stringify` | 深   | ❌   | ❌   | ❌       |
| `structuredClone`      | 深   | ❌   | ✅   | ✅       |
| `lodash.cloneDeep`     | 深   | ✅   | ✅   | ✅       |

## 🎮 实际应用

### 状态管理

```javascript
// Redux reducer - 必须返回新对象
function reducer(state, action) {
  switch (action.type) {
    case "UPDATE_USER":
      return {
        ...state,
        user: {
          ...state.user,
          ...action.payload,
        },
      };
    default:
      return state;
  }
}
```

### 缓存原始数据

```javascript
// 保存原始数据用于重置
const originalData = structuredClone(data);

function reset() {
  data = structuredClone(originalData);
}
```

### 避免副作用

```javascript
// ❌ 直接修改参数
function process(obj) {
  obj.processed = true; // 影响原对象
  return obj;
}

// ✅ 使用副本
function process(obj) {
  const copy = { ...obj };
  copy.processed = true;
  return copy;
}
```

## 💡 最佳实践

1. **简单对象** → 使用展开运算符 `{...obj}`
2. **需要深拷贝** → 优先使用 `structuredClone`
3. **需要拷贝函数** → 使用 `lodash.cloneDeep`
4. **性能敏感** → 考虑 Immutable.js

## 🔗 相关资源

- [JavaScript 基础语法](/docs/frontend/javascript/fundamentals)
- [ES6+](/docs/frontend/javascript/es6)

---

**下一步**：学习 [函数式编程](/docs/frontend/javascript/functional) 了解不可变性的更多应用。
