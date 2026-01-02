---
sidebar_position: 4
title: 异步编程
---

# JavaScript 异步编程

> [!TIP]
> 异步编程让 JavaScript 能够处理耗时操作（如网络请求）而不阻塞页面。

## 🎯 异步概念

### 同步 vs 异步

```javascript
// 同步：按顺序执行，会阻塞
console.log("1");
console.log("2");
console.log("3");
// 输出: 1, 2, 3

// 异步：不阻塞后续代码
console.log("1");
setTimeout(() => console.log("2"), 0);
console.log("3");
// 输出: 1, 3, 2
```

## ⏱️ 定时器

```javascript
// 延迟执行
const timeoutId = setTimeout(() => {
  console.log("3秒后执行");
}, 3000);

// 取消
clearTimeout(timeoutId);

// 重复执行
const intervalId = setInterval(() => {
  console.log("每秒执行");
}, 1000);

// 取消
clearInterval(intervalId);
```

## 🤝 Promise

### 基础用法

```javascript
// 创建 Promise
const promise = new Promise((resolve, reject) => {
  setTimeout(() => {
    const success = true;
    if (success) {
      resolve("成功");
    } else {
      reject("失败");
    }
  }, 1000);
});

// 使用 Promise
promise
  .then((result) => console.log(result))
  .catch((error) => console.error(error))
  .finally(() => console.log("完成"));
```

### Promise 状态

```
pending → fulfilled (resolve)
       → rejected  (reject)
```

### 链式调用

```javascript
fetch("/api/user")
  .then((response) => response.json())
  .then((user) => fetch(`/api/posts/${user.id}`))
  .then((response) => response.json())
  .then((posts) => console.log(posts))
  .catch((error) => console.error(error));
```

### Promise 方法

```javascript
// 全部成功
Promise.all([p1, p2, p3]).then((results) => console.log(results)); // [r1, r2, r3]

// 任一成功
Promise.race([p1, p2, p3]).then((result) => console.log(result)); // 最快的结果

// 全部完成（不管成功失败）
Promise.allSettled([p1, p2, p3]).then((results) => console.log(results));

// 任一成功（忽略失败）
Promise.any([p1, p2, p3]).then((result) => console.log(result));
```

## ⚡ async/await

### 基础用法

```javascript
async function fetchUser() {
  try {
    const response = await fetch("/api/user");
    const user = await response.json();
    return user;
  } catch (error) {
    console.error("获取失败:", error);
  }
}

// 调用
const user = await fetchUser();
```

### 错误处理

```javascript
// try-catch
async function getData() {
  try {
    const data = await fetch("/api/data");
    return await data.json();
  } catch (error) {
    console.error(error);
    return null;
  }
}

// 或使用 .catch()
const data = await fetch("/api/data").catch((e) => null);
```

### 并行执行

```javascript
// 顺序执行（较慢）
const user = await fetchUser();
const posts = await fetchPosts();

// 并行执行（更快）
const [user, posts] = await Promise.all([fetchUser(), fetchPosts()]);
```

## 🌐 Fetch API

### GET 请求

```javascript
// 基础请求
const response = await fetch("/api/users");
const users = await response.json();

// 带参数
const response = await fetch("/api/users?page=1&limit=10");

// 检查状态
if (!response.ok) {
  throw new Error(`HTTP error! status: ${response.status}`);
}
```

### POST 请求

```javascript
const response = await fetch("/api/users", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    name: "Alice",
    email: "alice@example.com",
  }),
});

const result = await response.json();
```

### 完整示例

```javascript
async function createUser(userData) {
  try {
    const response = await fetch("/api/users", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(userData),
    });

    if (!response.ok) {
      throw new Error(`Error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("创建用户失败:", error);
    throw error;
  }
}
```

### 其他请求方法

```javascript
// PUT
await fetch("/api/users/1", {
  method: "PUT",
  body: JSON.stringify(data),
});

// PATCH
await fetch("/api/users/1", {
  method: "PATCH",
  body: JSON.stringify({ name: "New Name" }),
});

// DELETE
await fetch("/api/users/1", {
  method: "DELETE",
});
```

## 🎮 实用示例

### 加载数据并渲染

```javascript
async function loadUsers() {
  const list = document.querySelector("#user-list");

  try {
    list.innerHTML = "<li>加载中...</li>";

    const response = await fetch("/api/users");
    const users = await response.json();

    list.innerHTML = users.map((u) => `<li>${u.name}</li>`).join("");
  } catch (error) {
    list.innerHTML = "<li>加载失败</li>";
  }
}

loadUsers();
```

### 防抖搜索

```javascript
function debounce(fn, delay) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

const search = debounce(async (query) => {
  const response = await fetch(`/api/search?q=${query}`);
  const results = await response.json();
  renderResults(results);
}, 300);

input.addEventListener("input", (e) => {
  search(e.target.value);
});
```

## 🔗 相关资源

- [JavaScript 入门](/docs/frontend/javascript/)
- [DOM 操作](/docs/frontend/javascript/dom)
- [ES6+](/docs/frontend/javascript/es6)

---

**下一步**：学习 [ES6+](/docs/frontend/javascript/es6) 了解现代 JavaScript 特性。
