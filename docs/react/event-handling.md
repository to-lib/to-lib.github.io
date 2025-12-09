---
sidebar_position: 7
title: 事件处理
---

# React 事件处理

> [!TIP]
> React 事件处理与 DOM 事件类似，但使用 camelCase 命名，并且传递函数而不是字符串。

## 📚 基础语法

### DOM 事件 vs React 事件

```jsx
// HTML/DOM 事件
<button onclick="handleClick()">Click</button>

// React 事件
<button onClick={handleClick}>Click</button>
```

**主要区别：**

1. 使用 camelCase（`onClick` 而不是 `onclick`）
2. 传递函数引用（`{handleClick}` 而不是 `"handleClick()"`）
3. 不能通过返回 `false` 阻止默认行为，必须使用 `preventDefault`

## 🎯 常用事件

### 点击事件

```jsx
function Button() {
  const handleClick = () => {
    console.log("Button clicked!");
  };

  return <button onClick={handleClick}>Click Me</button>;
}

// 内联箭头函数
function Button() {
  return <button onClick={() => console.log("Clicked")}>Click Me</button>;
}
```

### 表单事件

```jsx
function Form() {
  const [value, setValue] = useState("");

  const handleChange = (e) => {
    setValue(e.target.value);
  };

  const handleSubmit = (e) => {
    e.preventDefault(); // 阻止表单默认提交
    console.log("Submitted:", value);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        value={value}
        onChange={handleChange}
        onFocus={() => console.log("Input focused")}
        onBlur={() => console.log("Input blurred")}
      />
      <button type="submit">Submit</button>
    </form>
  );
}
```

### 键盘事件

```jsx
function SearchBox() {
  const [query, setQuery] = useState("");

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      console.log("Search:", query);
    }
    if (e.key === "Escape") {
      setQuery("");
    }
  };

  return (
    <input
      value={query}
      onChange={(e) => setQuery(e.target.value)}
      onKeyDown={handleKeyDown}
      onKeyPress={(e) => console.log("Key pressed:", e.key)}
      onKeyUp={(e) => console.log("Key released:", e.key)}
      placeholder="Press Enter to search, Esc to clear"
    />
  );
}
```

### 鼠标事件

```jsx
function Hover() {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onMouseMove={(e) => console.log("Mouse at:", e.clientX, e.clientY)}
      onClick={(e) => console.log("Clicked at:", e.clientX, e.clientY)}
      onDoubleClick={() => console.log("Double clicked!")}
      style={{
        background: isHovered ? "lightblue" : "white",
        padding: "20px",
      }}
    >
      {isHovered ? "Hovering!" : "Hover over me"}
    </div>
  );
}
```

## 📋 事件对象

### 合成事件（SyntheticEvent）

React 封装了原生事件，提供跨浏览器一致的接口。

```jsx
function Input() {
  const handleEvent = (e) => {
    console.log("React 合成事件:", e);
    console.log("原生事件:", e.nativeEvent);

    // 常用属性
    console.log("目标元素:", e.target);
    console.log("当前元素:", e.currentTarget);
    console.log("事件类型:", e.type);
    console.log("按键:", e.key);
    console.log("鼠标位置:", e.clientX, e.clientY);
  };

  return <input onChange={handleEvent} />;
}
```

### 阻止默认行为

```jsx
function Link() {
  const handleClick = (e) => {
    e.preventDefault(); // 阻止链接跳转
    console.log("Link clicked, but not navigating");
  };

  return (
    <a href="https://example.com" onClick={handleClick}>
      Click
    </a>
  );
}
```

### 阻止事件冒泡

```jsx
function Nested() {
  const handleParentClick = () => {
    console.log("Parent clicked");
  };

  const handleChildClick = (e) => {
    e.stopPropagation(); // 阻止冒泡
    console.log("Child clicked");
  };

  return (
    <div
      onClick={handleParentClick}
      style={{ padding: "20px", background: "lightgray" }}
    >
      Parent Div
      <button onClick={handleChildClick}>
        Child Button (Click won't bubble)
      </button>
    </div>
  );
}
```

## 🔧 传递参数

### 方法一：箭头函数

```jsx
function List({ items }) {
  const handleDelete = (id) => {
    console.log("Delete item:", id);
  };

  return (
    <ul>
      {items.map((item) => (
        <li key={item.id}>
          {item.name}
          <button onClick={() => handleDelete(item.id)}>Delete</button>
        </li>
      ))}
    </ul>
  );
}
```

### 方法二：bind

```jsx
function List({ items }) {
  const handleDelete = (id) => {
    console.log("Delete item:", id);
  };

  return (
    <ul>
      {items.map((item) => (
        <li key={item.id}>
          {item.name}
          <button onClick={handleDelete.bind(null, item.id)}>Delete</button>
        </li>
      ))}
    </ul>
  );
}
```

### 方法三：data 属性

```jsx
function List({ items }) {
  const handleDelete = (e) => {
    const id = e.currentTarget.dataset.id;
    console.log("Delete item:", id);
  };

  return (
    <ul>
      {items.map((item) => (
        <li key={item.id}>
          {item.name}
          <button data-id={item.id} onClick={handleDelete}>
            Delete
          </button>
        </li>
      ))}
    </ul>
  );
}
```

## 📱 常见事件类型

### 鼠标和指针事件

```jsx
function EventDemo() {
  return (
    <div
      onClick={(e) => console.log("Click")}
      onDoubleClick={(e) => console.log("Double Click")}
      onContextMenu={(e) => console.log("Right Click")}
      onMouseDown={(e) => console.log("Mouse Down")}
      onMouseUp={(e) => console.log("Mouse Up")}
      onMouseEnter={(e) => console.log("Mouse Enter")}
      onMouseLeave={(e) => console.log("Mouse Leave")}
      onMouseMove={(e) => console.log("Mouse Move")}
    >
      Interact with me
    </div>
  );
}
```

### 键盘事件

```jsx
function KeyboardDemo() {
  const handleKey = (e) => {
    console.log("Key:", e.key);
    console.log("Code:", e.code);
    console.log("Ctrl:", e.ctrlKey);
    console.log("Shift:", e.shiftKey);
    console.log("Alt:", e.altKey);
  };

  return (
    <input onKeyDown={handleKey} onKeyPress={handleKey} onKeyUp={handleKey} />
  );
}
```

### 焦点事件

```jsx
function FocusDemo() {
  return (
    <input
      onFocus={() => console.log("Focused")}
      onBlur={() => console.log("Blurred")}
      placeholder="Focus on me"
    />
  );
}
```

### 表单事件

```jsx
function FormDemo() {
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        console.log("Form submitted");
      }}
      onChange={() => console.log("Form changed")}
    >
      <input onChange={(e) => console.log("Input changed:", e.target.value)} />
      <select onChange={(e) => console.log("Select changed:", e.target.value)}>
        <option>Option 1</option>
        <option>Option 2</option>
      </select>
      <button type="submit">Submit</button>
    </form>
  );
}
```

### 剪贴板事件

```jsx
function ClipboardDemo() {
  return (
    <input
      onCopy={() => console.log("Copied")}
      onCut={() => console.log("Cut")}
      onPaste={(e) => console.log("Pasted:", e.clipboardData.getData("text"))}
    />
  );
}
```

## 💡 最佳实践

### 1. 避免内联箭头函数

```jsx
// ✗ 不好 - 每次渲染都创建新函数
function List({ items }) {
  return (
    <ul>
      {items.map((item) => (
        <li key={item.id}>
          <button onClick={() => console.log(item.id)}>Click</button>
        </li>
      ))}
    </ul>
  );
}

// ✓ 好 - 使用 useCallback
function List({ items }) {
  const handleClick = useCallback((id) => {
    console.log(id);
  }, []);

  return (
    <ul>
      {items.map((item) => (
        <li key={item.id}>
          <button onClick={() => handleClick(item.id)}>Click</button>
        </li>
      ))}
    </ul>
  );
}
```

### 2. 事件处理函数命名

```jsx
// ✓ 统一使用 handle 前缀
function Component() {
  const handleClick = () => {};
  const handleChange = () => {};
  const handleSubmit = () => {};

  return (
    <form onSubmit={handleSubmit}>
      <input onChange={handleChange} />
      <button onClick={handleClick}>Submit</button>
    </form>
  );
}
```

### 3. 提取复杂逻辑

```jsx
// ✗ 不好 - 逻辑太复杂
<button
  onClick={(e) => {
    e.preventDefault();
    if (isValid) {
      saveData();
      updateUI();
      showNotification();
    }
  }}
>
  Save
</button>;

// ✓ 好 - 提取函数
const handleSave = (e) => {
  e.preventDefault();
  if (isValid) {
    saveData();
    updateUI();
    showNotification();
  }
};

<button onClick={handleSave}>Save</button>;
```

### 4. 防抖和节流

```jsx
import { debounce } from "lodash";
import { useCallback } from "react";

function SearchBox() {
  // 防抖：延迟执行
  const handleSearch = useCallback(
    debounce((value) => {
      console.log("Search:", value);
    }, 300),
    []
  );

  return <input onChange={(e) => handleSearch(e.target.value)} />;
}
```

## 📖 实用示例

### 可拖拽元素

```jsx
function DraggableBox() {
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const handleMouseDown = (e) => {
    setIsDragging(true);
    setDragStart({
      x: e.clientX - position.x,
      y: e.clientY - position.y,
    });
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  return (
    <div
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      style={{
        position: "absolute",
        left: position.x,
        top: position.y,
        width: "100px",
        height: "100px",
        background: "lightblue",
        cursor: isDragging ? "grabbing" : "grab",
      }}
    >
      Drag me!
    </div>
  );
}
```

### 文件上传

```jsx
function FileUpload() {
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    setFile(droppedFile);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      style={{
        border: "2px dashed gray",
        padding: "20px",
        textAlign: "center",
      }}
    >
      <input type="file" onChange={handleFileChange} />
      <p>Or drag and drop a file here</p>
      {file && <p>Selected: {file.name}</p>}
    </div>
  );
}
```

---

**下一步**: 学习 [条件渲染](./conditional-rendering) 了解如何根据条件显示内容，或查看 [列表和 Keys](./lists-and-keys) 学习列表渲染。
