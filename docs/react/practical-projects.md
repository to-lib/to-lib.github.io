---
sidebar_position: 93
title: 实战案例
---

# React 实战案例

> [!TIP]
> 通过实战项目巩固 React 知识。本文提供完整的小型项目示例。

## 📝 Todo 应用

完整的待办事项应用，包含增删改查、筛选、本地存储。

```jsx
import { useState, useEffect } from "react";

function TodoApp() {
  const [todos, setTodos] = useState(() => {
    const saved = localStorage.getItem("todos");
    return saved ? JSON.parse(saved) : [];
  });
  const [input, setInput] = useState("");
  const [filter, setFilter] = useState("all"); // all, active, completed

  // 保存到本地存储
  useEffect(() => {
    localStorage.setItem("todos", JSON.stringify(todos));
  }, [todos]);

  const addTodo = () => {
    if (input.trim()) {
      setTodos([
        ...todos,
        {
          id: Date.now(),
          text: input,
          completed: false,
        },
      ]);
      setInput("");
    }
  };

  const toggleTodo = (id) => {
    setTodos(
      todos.map((todo) =>
        todo.id === id ? { ...todo, completed: !todo.completed } : todo
      )
    );
  };

  const deleteTodo = (id) => {
    setTodos(todos.filter((todo) => todo.id !== id));
  };

  const filteredTodos = todos.filter((todo) => {
    if (filter === "active") return !todo.completed;
    if (filter === "completed") return todo.completed;
    return true;
  });

  return (
    <div className="todo-app">
      <h1>Todo List</h1>

      <div className="input-section">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && addTodo()}
          placeholder="添加待办事项..."
        />
        <button onClick={addTodo}>添加</button>
      </div>

      <div className="filters">
        <button onClick={() => setFilter("all")}>全部</button>
        <button onClick={() => setFilter("active")}>进行中</button>
        <button onClick={() => setFilter("completed")}>已完成</button>
      </div>

      <ul>
        {filteredTodos.map((todo) => (
          <li key={todo.id}>
            <input
              type="checkbox"
              checked={todo.completed}
              onChange={() => toggleTodo(todo.id)}
            />
            <span
              style={{
                textDecoration: todo.completed ? "line-through" : "none",
              }}
            >
              {todo.text}
            </span>
            <button onClick={() => deleteTodo(todo.id)}>删除</button>
          </li>
        ))}
      </ul>

      <div className="stats">
        <span>总计: {todos.length}</span>
        <span>进行中: {todos.filter((t) => !t.completed).length}</span>
        <span>已完成: {todos.filter((t) => t.completed).length}</span>
      </div>
    </div>
  );
}
```

## 🛒 购物车

简单的购物车系统，使用 Context 管理状态。

```jsx
import { createContext, useContext, useReducer } from "react";

const CartContext = createContext();

function cartReducer(state, action) {
  switch (action.type) {
    case "ADD_ITEM":
      const existingIndex = state.items.findIndex(
        (i) => i.id === action.payload.id
      );
      if (existingIndex > -1) {
        const newItems = [...state.items];
        newItems[existingIndex].quantity++;
        return { ...state, items: newItems };
      }
      return {
        ...state,
        items: [...state.items, { ...action.payload, quantity: 1 }],
      };

    case "REMOVE_ITEM":
      return {
        ...state,
        items: state.items.filter((i) => i.id !== action.payload),
      };

    case "UPDATE_QUANTITY":
      return {
        ...state,
        items: state.items.map((i) =>
          i.id === action.payload.id
            ? { ...i, quantity: action.payload.quantity }
            : i
        ),
      };

    default:
      return state;
  }
}

function CartProvider({ children }) {
  const [state, dispatch] = useReducer(cartReducer, { items: [] });

  const addItem = (item) => dispatch({ type: "ADD_ITEM", payload: item });
  const removeItem = (id) => dispatch({ type: "REMOVE_ITEM", payload: id });
  const updateQuantity = (id, quantity) =>
    dispatch({ type: "UPDATE_QUANTITY", payload: { id, quantity } });

  const total = state.items.reduce(
    (sum, item) => sum + item.price * item.quantity,
    0
  );

  return (
    <CartContext.Provider
      value={{
        items: state.items,
        addItem,
        removeItem,
        updateQuantity,
        total,
      }}
    >
      {children}
    </CartContext.Provider>
  );
}

function useCart() {
  return useContext(CartContext);
}

// 使用
function ProductList() {
  const { addItem } = useCart();
  const products = [
    { id: 1, name: "产品 A", price: 100 },
    { id: 2, name: "产品 B", price: 200 },
  ];

  return (
    <div>
      {products.map((product) => (
        <div key={product.id}>
          <h3>{product.name}</h3>
          <p>¥{product.price}</p>
          <button onClick={() => addItem(product)}>加入购物车</button>
        </div>
      ))}
    </div>
  );
}

function Cart() {
  const { items, removeItem, updateQuantity, total } = useCart();

  return (
    <div>
      <h2>购物车</h2>
      {items.map((item) => (
        <div key={item.id}>
          <span>{item.name}</span>
          <input
            type="number"
            value={item.quantity}
            onChange={(e) => updateQuantity(item.id, parseInt(e.target.value))}
          />
          <span>¥{item.price * item.quantity}</span>
          <button onClick={() => removeItem(item.id)}>删除</button>
        </div>
      ))}
      <h3>总计: ¥{total}</h3>
    </div>
  );
}
```

## 🔍 搜索过滤器

带防抖的实时搜索功能。

```jsx
import { useState, useEffect, useMemo } from "react";

function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}

function SearchFilter() {
  const [searchTerm, setSearchTerm] = useState("");
  const [items, setItems] = useState([]);
  const debouncedSearchTerm = useDebounce(searchTerm, 500);

  useEffect(() => {
    if (debouncedSearchTerm) {
      // 模拟 API 调用
      fetch(`/api/search?q=${debouncedSearchTerm}`)
        .then((res) => res.json())
        .then(setItems);
    }
  }, [debouncedSearchTerm]);

  return (
    <div>
      <input
        type="search"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        placeholder="搜索..."
      />
      <ul>
        {items.map((item) => (
          <li key={item.id}>{item.name}</li>
        ))}
      </ul>
    </div>
  );
}
```

---

**更多学习**：查看 [面试题](./interview-questions) 或返回 [概览](./index)
