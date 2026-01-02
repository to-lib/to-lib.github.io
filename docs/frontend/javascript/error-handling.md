---
sidebar_position: 8
title: 错误处理
---

# 错误处理

> [!TIP]
> 良好的错误处理让代码更健壮，帮助快速定位和解决问题。

## 🎯 try...catch...finally

### 基础语法

```javascript
try {
  // 可能出错的代码
  const data = JSON.parse(invalidJson);
} catch (error) {
  // 错误处理
  console.error("解析失败:", error.message);
} finally {
  // 无论是否出错都会执行
  console.log("清理工作");
}
```

### Error 对象属性

```javascript
try {
  throw new Error("Something went wrong");
} catch (error) {
  console.log(error.name); // 'Error'
  console.log(error.message); // 'Something went wrong'
  console.log(error.stack); // 调用栈信息
}
```

## 📦 错误类型

### 内置错误类型

| 类型             | 说明           |
| ---------------- | -------------- |
| `Error`          | 通用错误       |
| `SyntaxError`    | 语法错误       |
| `TypeError`      | 类型错误       |
| `ReferenceError` | 引用未定义变量 |
| `RangeError`     | 数值超出范围   |

```javascript
// TypeError
null.foo;

// ReferenceError
console.log(undefinedVar);

// SyntaxError
eval("var a = ");

// RangeError
new Array(-1);
```

### 自定义错误

```javascript
class ValidationError extends Error {
  constructor(message, field) {
    super(message);
    this.name = "ValidationError";
    this.field = field;
  }
}

function validateEmail(email) {
  if (!email.includes("@")) {
    throw new ValidationError("Invalid email format", "email");
  }
}

try {
  validateEmail("invalid");
} catch (error) {
  if (error instanceof ValidationError) {
    console.log(`字段 ${error.field}: ${error.message}`);
  }
}
```

## 🔄 异步错误处理

### Promise 错误

```javascript
// .catch() 方法
fetch("/api/data")
  .then((response) => response.json())
  .catch((error) => {
    console.error("请求失败:", error);
  });

// Promise 链错误传递
fetch("/api/data")
  .then((response) => {
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
  })
  .then((data) => console.log(data))
  .catch((error) => console.error(error));
```

### async/await 错误

```javascript
async function fetchData() {
  try {
    const response = await fetch("/api/data");

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("请求失败:", error);
    throw error; // 重新抛出或返回默认值
  }
}
```

### 并行请求错误

```javascript
async function fetchAll() {
  try {
    const results = await Promise.all([
      fetch("/api/users"),
      fetch("/api/posts"),
    ]);
    return results;
  } catch (error) {
    // 任一请求失败都会进入这里
    console.error("请求失败:", error);
  }
}

// 使用 allSettled 获取所有结果
async function fetchAllSafe() {
  const results = await Promise.allSettled([
    fetch("/api/users"),
    fetch("/api/posts"),
  ]);

  results.forEach((result, index) => {
    if (result.status === "fulfilled") {
      console.log(`请求 ${index} 成功`);
    } else {
      console.log(`请求 ${index} 失败:`, result.reason);
    }
  });
}
```

## 🛡️ 防御性编程

### 参数验证

```javascript
function divide(a, b) {
  if (typeof a !== "number" || typeof b !== "number") {
    throw new TypeError("参数必须是数字");
  }
  if (b === 0) {
    throw new RangeError("除数不能为零");
  }
  return a / b;
}
```

### 可选链和空值合并

```javascript
// 可选链 - 安全访问嵌套属性
const city = user?.address?.city;

// 空值合并 - 提供默认值
const name = user.name ?? "Anonymous";

// 组合使用
const street = user?.address?.street ?? "Unknown";
```

### 类型守卫

```javascript
function processValue(value) {
  if (value === null || value === undefined) {
    return "No value";
  }

  if (Array.isArray(value)) {
    return value.join(", ");
  }

  if (typeof value === "object") {
    return JSON.stringify(value);
  }

  return String(value);
}
```

## 🌐 全局错误处理

### 浏览器环境

```javascript
// 捕获未处理的错误
window.onerror = function (message, source, line, col, error) {
  console.error("Global error:", { message, source, line, col });
  // 返回 true 阻止默认处理
  return true;
};

// 捕获未处理的 Promise rejection
window.onunhandledrejection = function (event) {
  console.error("Unhandled rejection:", event.reason);
};
```

### 错误日志上报

```javascript
function reportError(error) {
  fetch("/api/errors", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: error.message,
      stack: error.stack,
      url: window.location.href,
      timestamp: Date.now(),
    }),
  }).catch(() => {
    // 上报失败时静默处理
  });
}
```

## 💡 最佳实践

### 1. 提供有意义的错误信息

```javascript
// ❌ 不好
throw new Error("Error");

// ✅ 好
throw new Error(`User ${userId} not found in database`);
```

### 2. 只捕获能处理的错误

```javascript
// ❌ 吞掉所有错误
try {
  doSomething();
} catch (e) {
  // 什么都不做
}

// ✅ 处理或重新抛出
try {
  doSomething();
} catch (error) {
  if (error instanceof NetworkError) {
    showRetryButton();
  } else {
    throw error; // 无法处理的错误继续抛出
  }
}
```

### 3. 使用 finally 清理资源

```javascript
async function processFile(path) {
  const file = await openFile(path);
  try {
    return await file.read();
  } finally {
    await file.close(); // 确保文件被关闭
  }
}
```

## 🔗 相关资源

- [异步编程](/docs/frontend/javascript/async)
- [ES6+](/docs/frontend/javascript/es6)

---

**下一步**：学习 [模块化](/docs/frontend/javascript/modules) 组织代码结构。
