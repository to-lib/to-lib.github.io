---
sidebar_position: 16
title: Refs 和 DOM
---

# Refs 和 DOM 操作

> [!TIP]
> Refs 提供了访问 DOM 节点或 React 元素的方式。理解 Refs 对于集成第三方库、聚焦输入框、触发动画等场景至关重要。

## 📚 什么是 Refs？

Refs 是 React 提供的一种访问 DOM 节点或组件实例的方式，绕过了声明式的数据流。

### 何时使用 Refs

✅ **适合使用的场景：**

- 管理焦点、文本选择或媒体播放
- 触发强制动画
- 集成第三方 DOM 库
- 测量 DOM 元素尺寸

❌ **避免使用的场景：**

- 任何可以声明式完成的事情
- 不要过度使用 Refs

## 🎯 useRef Hook

### 基础用法

```jsx
import { useRef } from "react";

function TextInput() {
  const inputRef = useRef(null);

  const focusInput = () => {
    // 访问 DOM 节点
    inputRef.current.focus();
  };

  return (
    <div>
      <input ref={inputRef} type="text" />
      <button onClick={focusInput}>聚焦输入框</button>
    </div>
  );
}
```

### Ref 对象结构

```jsx
const myRef = useRef(initialValue);

console.log(myRef);
// { current: initialValue }

myRef.current = newValue; // 可以修改
```

## 🔄 多种 Ref 类型

### 1. DOM Refs

```jsx
function MediaPlayer() {
  const videoRef = useRef(null);

  const play = () => videoRef.current.play();
  const pause = () => videoRef.current.pause();
  const mute = () => {
    videoRef.current.muted = !videoRef.current.muted;
  };

  return (
    <div>
      <video ref={videoRef} src="video.mp4" />
      <button onClick={play}>播放</button>
      <button onClick={pause}>暂停</button>
      <button onClick={mute}>静音/取消静音</button>
    </div>
  );
}
```

### 2. 保存可变值

```jsx
function Timer() {
  const [seconds, setSeconds] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const intervalRef = useRef(null);

  const start = () => {
    if (intervalRef.current) return; // 已在运行

    setIsRunning(true);
    intervalRef.current = setInterval(() => {
      setSeconds((s) => s + 1);
    }, 1000);
  };

  const stop = () => {
    setIsRunning(false);
    clearInterval(intervalRef.current);
    intervalRef.current = null;
  };

  const reset = () => {
    stop();
    setSeconds(0);
  };

  // 组件卸载时清理
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  return (
    <div>
      <h2>时间: {seconds}秒</h2>
      <button onClick={start} disabled={isRunning}>
        开始
      </button>
      <button onClick={stop} disabled={!isRunning}>
        停止
      </button>
      <button onClick={reset}>重置</button>
    </div>
  );
}
```

### 3. 保存前一个值

```jsx
function usePrevious(value) {
  const ref = useRef();

  useEffect(() => {
    ref.current = value;
  }, [value]);

  return ref.current;
}

// 使用
function Counter() {
  const [count, setCount] = useState(0);
  const prevCount = usePrevious(count);

  return (
    <div>
      <p>当前: {count}</p>
      <p>之前: {prevCount}</p>
      <button onClick={() => setCount(count + 1)}>+1</button>
    </div>
  );
}
```

## 🔗 forwardRef - 转发 Refs

父组件可能需要访问子组件的 DOM 节点：

### 基础用法

```jsx
import { forwardRef, useRef } from "react";

// 子组件使用 forwardRef 包裹
const CustomInput = forwardRef((props, ref) => {
  return <input ref={ref} {...props} />;
});

// 父组件使用
function App() {
  const inputRef = useRef(null);

  const focusInput = () => {
    inputRef.current.focus();
  };

  return (
    <div>
      <CustomInput ref={inputRef} placeholder="输入文本" />
      <button onClick={focusInput}>聚焦</button>
    </div>
  );
}
```

### 复杂示例：自定义组件库

```jsx
const FancyButton = forwardRef((props, ref) => {
  const { children, variant = "primary", ...rest } = props;

  return (
    <button ref={ref} className={`btn btn-${variant}`} {...rest}>
      {children}
    </button>
  );
});

// 使用
function App() {
  const buttonRef = useRef(null);

  useEffect(() => {
    // 自动聚焦按钮
    buttonRef.current.focus();
  }, []);

  return <FancyButton ref={buttonRef}>点击我</FancyButton>;
}
```

## 🎨 useImperativeHandle

自定义暴露给父组件的实例值：

### 基础用法

```jsx
import { forwardRef, useRef, useImperativeHandle } from "react";

const FancyInput = forwardRef((props, ref) => {
  const inputRef = useRef(null);

  // 自定义暴露的方法
  useImperativeHandle(ref, () => ({
    focus: () => {
      inputRef.current.focus();
    },
    clear: () => {
      inputRef.current.value = "";
    },
    setValue: (value) => {
      inputRef.current.value = value;
    },
  }));

  return <input ref={inputRef} {...props} />;
});

// 父组件使用
function App() {
  const inputRef = useRef(null);

  const handleClear = () => {
    inputRef.current.clear();
  };

  const handleSetValue = () => {
    inputRef.current.setValue("Hello World");
  };

  return (
    <div>
      <FancyInput ref={inputRef} />
      <button onClick={() => inputRef.current.focus()}>聚焦</button>
      <button onClick={handleClear}>清空</button>
      <button onClick={handleSetValue}>设置值</button>
    </div>
  );
}
```

### 视频播放器示例

```jsx
const VideoPlayer = forwardRef((props, ref) => {
  const videoRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);

  useImperativeHandle(ref, () => ({
    play() {
      videoRef.current.play();
      setIsPlaying(true);
    },
    pause() {
      videoRef.current.pause();
      setIsPlaying(false);
    },
    getCurrentTime() {
      return videoRef.current.currentTime;
    },
    setCurrentTime(time) {
      videoRef.current.currentTime = time;
    },
    getIsPlaying() {
      return isPlaying;
    },
  }));

  return (
    <video
      ref={videoRef}
      src={props.src}
      onPlay={() => setIsPlaying(true)}
      onPause={() => setIsPlaying(false)}
    />
  );
});

// 使用
function App() {
  const playerRef = useRef(null);

  return (
    <div>
      <VideoPlayer ref={playerRef} src="video.mp4" />
      <button onClick={() => playerRef.current.play()}>播放</button>
      <button onClick={() => playerRef.current.pause()}>暂停</button>
      <button
        onClick={() => {
          const time = playerRef.current.getCurrentTime();
          alert(`当前时间: ${time}秒`);
        }}
      >
        获取时间
      </button>
    </div>
  );
}
```

## 📏 测量 DOM 元素

### 获取元素尺寸

```jsx
function MeasureElement() {
  const divRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (divRef.current) {
      const { width, height } = divRef.current.getBoundingClientRect();
      setDimensions({ width, height });
    }
  }, []);

  return (
    <div>
      <div
        ref={divRef}
        style={{ width: 200, height: 100, background: "lightblue" }}
      >
        测量我
      </div>
      <p>宽度: {dimensions.width}px</p>
      <p>高度: {dimensions.height}px</p>
    </div>
  );
}
```

### ResizeObserver 监听尺寸变化

```jsx
function useElementSize(ref) {
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (!ref.current) return;

    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setSize({ width, height });
    });

    observer.observe(ref.current);

    return () => observer.disconnect();
  }, [ref]);

  return size;
}

// 使用
function ResizableBox() {
  const boxRef = useRef(null);
  const size = useElementSize(boxRef);

  return (
    <div>
      <div
        ref={boxRef}
        style={{
          resize: "both",
          overflow: "auto",
          width: 200,
          height: 100,
          border: "1px solid black",
        }}
      >
        拖动调整大小
      </div>
      <p>宽度: {Math.round(size.width)}px</p>
      <p>高度: {Math.round(size.height)}px</p>
    </div>
  );
}
```

## 🎯 实际应用场景

### 1. 自动聚焦输入框

```jsx
function SearchBar() {
  const inputRef = useRef(null);

  useEffect(() => {
    // 页面加载时自动聚焦
    inputRef.current.focus();
  }, []);

  return <input ref={inputRef} type="search" placeholder="搜索..." />;
}
```

### 2. 滚动到视图

```jsx
function TodoList({ todos }) {
  const bottomRef = useRef(null);

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    // 新增 todo 时滚动到底部
    scrollToBottom();
  }, [todos.length]);

  return (
    <div>
      <ul>
        {todos.map((todo) => (
          <li key={todo.id}>{todo.text}</li>
        ))}
        <li ref={bottomRef} />
      </ul>
    </div>
  );
}
```

### 3. 文本选择

```jsx
function TextSelector() {
  const textRef = useRef(null);

  const selectAll = () => {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(textRef.current);
    selection.removeAllRanges();
    selection.addRange(range);
  };

  return (
    <div>
      <p ref={textRef}>这是一段可以被选中的文本。点击按钮选中全部。</p>
      <button onClick={selectAll}>全选</button>
    </div>
  );
}
```

### 4. 集成第三方库（Chart.js）

```jsx
import { useEffect, useRef } from "react";
import Chart from "chart.js/auto";

function ChartComponent({ data }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    const ctx = canvasRef.current.getContext("2d");

    // 销毁旧图表
    if (chartRef.current) {
      chartRef.current.destroy();
    }

    // 创建新图表
    chartRef.current = new Chart(ctx, {
      type: "bar",
      data: data,
      options: {
        responsive: true,
      },
    });

    // 清理
    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
      }
    };
  }, [data]);

  return <canvas ref={canvasRef} />;
}
```

### 5. 表单验证

```jsx
function LoginForm() {
  const usernameRef = useRef(null);
  const passwordRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();

    const username = usernameRef.current.value;
    const password = passwordRef.current.value;

    if (!username) {
      usernameRef.current.focus();
      alert("请输入用户名");
      return;
    }

    if (password.length < 6) {
      passwordRef.current.focus();
      alert("密码至少 6 位");
      return;
    }

    console.log("登录:", { username, password });
  };

  return (
    <form onSubmit={handleSubmit}>
      <input ref={usernameRef} type="text" placeholder="用户名" />
      <input ref={passwordRef} type="password" placeholder="密码" />
      <button type="submit">登录</button>
    </form>
  );
}
```

## 💡 最佳实践

### 1. 避免过度使用 Refs

```jsx
// ✗ 不好：用 Ref 管理可变状态
function Counter() {
  const countRef = useRef(0);

  const increment = () => {
    countRef.current++;
    // 组件不会重新渲染！
  };

  return <div>{countRef.current}</div>;
}

// ✓ 好：用 State 管理 UI 状态
function Counter() {
  const [count, setCount] = useState(0);

  return (
    <div>
      {count}
      <button onClick={() => setCount(count + 1)}>+</button>
    </div>
  );
}
```

### 2. Ref 不触发重新渲染

```jsx
function Example() {
  const countRef = useRef(0);
  const [, forceUpdate] = useState();

  const increment = () => {
    countRef.current++;
    forceUpdate({}); // 强制重新渲染（不推荐）
  };

  return <div>{countRef.current}</div>;
}
```

### 3. Callback Refs

当需要在 ref 设置时执行代码：

```jsx
function MeasureExample() {
  const [height, setHeight] = useState(0);

  // Callback ref
  const measureRef = (node) => {
    if (node !== null) {
      setHeight(node.getBoundingClientRect().height);
    }
  };

  return (
    <div>
      <div ref={measureRef}>我会被测量</div>
      <p>高度：{height}px</p>
    </div>
  );
}
```

### 4. 多个 Refs

使用数组或 Map 存储多个 refs：

```jsx
function ItemList({ items }) {
  const itemRefs = useRef(new Map());

  const scrollToItem = (id) => {
    const node = itemRefs.current.get(id);
    node?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };

  return (
    <ul>
      {items.map((item) => (
        <li
          key={item.id}
          ref={(node) => {
            if (node) {
              itemRefs.current.set(item.id, node);
            } else {
              itemRefs.current.delete(item.id);
            }
          }}
        >
          {item.name}
          <button onClick={() => scrollToItem(item.id)}>滚动到此</button>
        </li>
      ))}
    </ul>
  );
}
```

## 🚨 常见错误

### 1. 在渲染期间访问 Ref

```jsx
// ✗ 错误：渲染期间读取 ref
function Example() {
  const ref = useRef(0);

  return <div>{ref.current}</div>; // 不会更新
}

// ✓ 正确：在事件处理器中读取
function Example() {
  const ref = useRef(0);
  const [display, setDisplay] = useState(0);

  const handleClick = () => {
    setDisplay(ref.current);
  };

  return (
    <div>
      {display}
      <button onClick={handleClick}>显示</button>
    </div>
  );
}
```

### 2. Ref 为 null

```jsx
// ✗ 错误：可能为 null
const focusInput = () => {
  inputRef.current.focus(); // 可能报错
};

// ✓ 正确：判空
const focusInput = () => {
  inputRef.current?.focus();
  // 或
  if (inputRef.current) {
    inputRef.current.focus();
  }
};
```

---

**下一步**：学习 [错误边界](./error-boundaries) 处理组件错误，或查看 [组件组合模式](./composition-patterns) 了解高级组件模式。
