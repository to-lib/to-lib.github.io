---
sidebar_position: 3
title: DOM 操作
---

# JavaScript DOM 操作

> [!TIP]
> DOM（Document Object Model）是网页的编程接口，允许 JavaScript 操作页面元素。

## 🎯 获取元素

```javascript
// 单个元素
document.getElementById("header");
document.querySelector(".card"); // 第一个匹配
document.querySelector("#header");

// 多个元素
document.getElementsByClassName("item");
document.getElementsByTagName("div");
document.querySelectorAll(".card"); // 所有匹配

// 特殊元素
document.body;
document.head;
document.documentElement; // <html>
```

## ✏️ 修改元素

### 内容

```javascript
const el = document.querySelector("#title");

// 文本内容
el.textContent = "新标题";

// HTML 内容
el.innerHTML = "<strong>加粗</strong>标题";

// 表单值
document.querySelector("#input").value = "输入值";
```

### 属性

```javascript
const link = document.querySelector("a");

// 读取/设置属性
link.getAttribute("href");
link.setAttribute("href", "https://example.com");
link.removeAttribute("target");

// 直接访问
link.href = "https://example.com";
link.id = "my-link";

// data 属性
el.dataset.userId = "123"; // data-user-id="123"
el.dataset.userId; // '123'
```

### 样式

```javascript
const box = document.querySelector(".box");

// 单个样式
box.style.color = "red";
box.style.backgroundColor = "#f0f0f0";
box.style.fontSize = "16px";

// 多个样式
box.style.cssText = "color: red; background: #f0f0f0;";

// 类操作
box.classList.add("active");
box.classList.remove("hidden");
box.classList.toggle("selected");
box.classList.contains("active"); // true/false
box.className = "card active"; // 替换所有类
```

## 🏗️ 创建和删除

### 创建元素

```javascript
// 创建
const div = document.createElement("div");
div.textContent = "新元素";
div.className = "card";

// 插入
parent.appendChild(div); // 末尾
parent.insertBefore(div, reference); // 在某元素前
parent.append(div1, div2); // 多个元素
parent.prepend(div); // 开头

// insertAdjacentHTML
el.insertAdjacentHTML("beforebegin", "<p>之前</p>");
el.insertAdjacentHTML("afterbegin", "<p>内部开头</p>");
el.insertAdjacentHTML("beforeend", "<p>内部末尾</p>");
el.insertAdjacentHTML("afterend", "<p>之后</p>");
```

### 删除元素

```javascript
el.remove(); // 删除自己
parent.removeChild(child); // 删除子元素
el.innerHTML = ""; // 清空内容
```

### 克隆元素

```javascript
const clone = el.cloneNode(true); // true 包含子元素
```

## 🖱️ 事件处理

### 添加事件

```javascript
// addEventListener（推荐）
button.addEventListener("click", function (event) {
  console.log("点击了", event.target);
});

// 箭头函数
button.addEventListener("click", (e) => {
  console.log("点击了");
});

// 移除事件
const handler = () => console.log("click");
button.addEventListener("click", handler);
button.removeEventListener("click", handler);
```

### 常用事件

```javascript
// 鼠标事件
el.addEventListener("click", fn);
el.addEventListener("dblclick", fn);
el.addEventListener("mouseenter", fn);
el.addEventListener("mouseleave", fn);

// 键盘事件
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
  }
});
input.addEventListener("keyup", fn);

// 表单事件
input.addEventListener("input", fn); // 输入时
input.addEventListener("change", fn); // 值改变后
input.addEventListener("focus", fn);
input.addEventListener("blur", fn);
form.addEventListener("submit", (e) => {
  e.preventDefault(); // 阻止提交
});

// 页面事件
window.addEventListener("load", fn);
document.addEventListener("DOMContentLoaded", fn);
window.addEventListener("scroll", fn);
window.addEventListener("resize", fn);
```

### 事件对象

```javascript
el.addEventListener("click", (event) => {
  event.target; // 触发事件的元素
  event.currentTarget; // 绑定事件的元素
  event.type; // 事件类型
  event.preventDefault(); // 阻止默认行为
  event.stopPropagation(); // 阻止冒泡
});
```

### 事件委托

```javascript
// 不在每个 li 上绑定，而是在父元素上绑定
ul.addEventListener("click", (e) => {
  if (e.target.tagName === "LI") {
    console.log("点击了", e.target.textContent);
  }
});
```

## 📐 元素尺寸和位置

```javascript
// 元素尺寸
el.offsetWidth; // 包含边框
el.offsetHeight;
el.clientWidth; // 不包含边框
el.clientHeight;

// 元素位置
el.offsetTop; // 相对于定位父元素
el.offsetLeft;
el.getBoundingClientRect(); // 相对于视口

// 滚动
el.scrollTop; // 滚动距离
el.scrollLeft;
el.scrollIntoView({ behavior: "smooth" });

// 视口尺寸
window.innerWidth;
window.innerHeight;
```

## 💡 实用示例

### 待办列表

```javascript
const form = document.querySelector("#todo-form");
const input = document.querySelector("#todo-input");
const list = document.querySelector("#todo-list");

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  const li = document.createElement("li");
  li.textContent = text;
  list.appendChild(li);
  input.value = "";
});

// 点击删除
list.addEventListener("click", (e) => {
  if (e.target.tagName === "LI") {
    e.target.remove();
  }
});
```

### 模态框

```javascript
const modal = document.querySelector("#modal");
const openBtn = document.querySelector("#open-modal");
const closeBtn = document.querySelector("#close-modal");

openBtn.addEventListener("click", () => {
  modal.classList.add("active");
});

closeBtn.addEventListener("click", () => {
  modal.classList.remove("active");
});

// 点击遮罩关闭
modal.addEventListener("click", (e) => {
  if (e.target === modal) {
    modal.classList.remove("active");
  }
});
```

## 🔗 相关资源

- [JavaScript 入门](/docs/frontend/javascript/)
- [基础语法](/docs/frontend/javascript/fundamentals)
- [异步编程](/docs/frontend/javascript/async)

---

**下一步**：学习 [异步编程](/docs/frontend/javascript/async) 处理网络请求。
