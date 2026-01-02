---
sidebar_position: 6
title: 调试技巧
---

# 前端调试技巧

> [!TIP]
> 掌握调试技巧能大幅提升开发效率，快速定位和解决问题。

## 🔧 Chrome DevTools

### 打开方式

- `F12` 或 `Ctrl+Shift+I`（Mac: `Cmd+Option+I`）
- 右键 → 检查

### 常用面板

| 面板        | 用途                 |
| ----------- | -------------------- |
| Elements    | 查看/修改 DOM 和 CSS |
| Console     | 执行 JS、查看日志    |
| Sources     | 调试 JS、查看源码    |
| Network     | 网络请求分析         |
| Performance | 性能分析             |
| Application | 存储、缓存、PWA      |

## 📝 Console 技巧

### 日志方法

```javascript
console.log("普通日志");
console.info("信息");
console.warn("警告");
console.error("错误");

// 分组
console.group("分组标题");
console.log("内容1");
console.log("内容2");
console.groupEnd();

// 表格
console.table([
  { name: "Alice", age: 25 },
  { name: "Bob", age: 30 },
]);

// 计时
console.time("操作");
// ... 操作
console.timeEnd("操作"); // 操作: 123.45ms

// 计数
console.count("点击"); // 点击: 1
console.count("点击"); // 点击: 2

// 断言
console.assert(1 === 2, "1 不等于 2"); // 输出错误

// 清空
console.clear();
```

### 格式化输出

```javascript
// 样式
console.log("%c红色文字", "color: red; font-size: 20px;");

// 对象
console.dir(document.body); // DOM 对象详情

// 堆栈追踪
console.trace("调用堆栈");
```

## 🔍 断点调试

### 设置断点

1. **行断点**：点击行号
2. **条件断点**：右键行号 → Add conditional breakpoint
3. **日志断点**：右键行号 → Add logpoint（不暂停）

### 代码内断点

```javascript
function process(data) {
  debugger; // 执行到此处会暂停
  return data.map((x) => x * 2);
}
```

### 调试控制

| 按钮/快捷键 | 功能                   |
| ----------- | ---------------------- |
| F8          | 继续执行               |
| F10         | 单步跳过（不进入函数） |
| F11         | 单步进入               |
| Shift+F11   | 单步跳出               |

### 观察表达式

在 Sources 面板的 Watch 区域添加变量或表达式，实时查看值。

## 🌐 网络调试

### 查看请求

1. 打开 Network 面板
2. 刷新页面或触发请求
3. 点击请求查看详情

### 请求详情

- **Headers**：请求头、响应头
- **Payload**：请求参数
- **Preview**：响应预览
- **Response**：响应原文
- **Timing**：时间分析

### 过滤请求

```
# 按类型
XHR  Fetch  JS  CSS  Img

# 按状态
status-code:404

# 按域名
domain:api.example.com

# 正则
/api\/user/
```

### 模拟网络

- **Throttling**：模拟网络速度（3G、离线）
- **Disable cache**：禁用缓存

## 📱 移动端调试

### 模拟设备

1. 点击设备图标或 `Ctrl+Shift+M`
2. 选择设备或自定义尺寸
3. 可以模拟触摸、位置、传感器

### Android 真机

```bash
# 1. 手机开启 USB 调试
# 2. 连接电脑
# 3. Chrome 访问
chrome://inspect/#devices
```

### iOS 真机

```bash
# 需要 Mac
# 1. Safari → 偏好设置 → 高级 → 显示开发菜单
# 2. iPhone 开启 Web 检查器
# 3. Safari → 开发 → 设备名
```

## ⚡ 性能调试

### Performance 面板

1. 点击录制按钮
2. 执行要分析的操作
3. 停止录制
4. 分析火焰图

### 关键指标

- **FP**：首次绘制
- **FCP**：首次内容绘制
- **LCP**：最大内容绘制
- **TBT**：总阻塞时间

### 内存分析

```javascript
// 获取内存快照
// Memory 面板 → Take heap snapshot

// 检测内存泄漏
// 1. 拍快照
// 2. 执行操作
// 3. 再拍快照
// 4. 对比差异
```

## 🔧 实用技巧

### 元素选择

```javascript
// 控制台直接使用
$0; // 当前选中的元素
$1; // 上一个选中的元素
$("selector"); // 等于 document.querySelector
$$("selector"); // 等于 document.querySelectorAll
```

### 复制数据

```javascript
copy(JSON.stringify(data, null, 2)); // 复制到剪贴板
```

### 监听事件

```javascript
// 监听元素事件
monitorEvents($0, "click");

// 停止监听
unmonitorEvents($0);

// 获取事件监听器
getEventListeners($0);
```

### 覆盖响应

1. Sources 面板 → Overrides
2. 选择本地文件夹
3. 右键网络请求 → Override content

### 代码片段

1. Sources → Snippets
2. 新建片段保存常用脚本
3. 右键 → Run 执行

## 💡 调试清单

- [ ] 确认问题是否可复现
- [ ] 检查控制台错误
- [ ] 检查网络请求
- [ ] 使用断点逐步调试
- [ ] 检查数据流和状态
- [ ] 对比预期和实际结果

## 🔗 相关资源

- [浏览器原理](/docs/frontend/browser/)
- [性能优化](/docs/frontend/advanced/performance)

---

**下一步**：学习 [WebSocket](/docs/frontend/browser/websocket) 实现实时通信。
