---
sidebar_position: 5
title: Canvas 与 SVG
---

# Canvas 与 SVG

> [!TIP]
> Canvas 和 SVG 是网页中实现图形和可视化的两种主要技术。

## 🎨 Canvas 基础

Canvas 是一个位图画布，通过 JavaScript 绘制图形。

### 创建画布

```html
<canvas id="canvas" width="400" height="300"></canvas>
```

```javascript
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
```

### 基本绘制

```javascript
// 矩形
ctx.fillStyle = "#3b82f6";
ctx.fillRect(10, 10, 100, 80); // 填充矩形

ctx.strokeStyle = "#ef4444";
ctx.strokeRect(130, 10, 100, 80); // 描边矩形

ctx.clearRect(30, 30, 40, 40); // 清除区域

// 路径
ctx.beginPath();
ctx.moveTo(10, 120);
ctx.lineTo(100, 200);
ctx.lineTo(10, 200);
ctx.closePath();
ctx.fill();
```

### 绘制圆形

```javascript
ctx.beginPath();
ctx.arc(200, 150, 50, 0, Math.PI * 2);
ctx.fillStyle = "#10b981";
ctx.fill();

// 弧形
ctx.beginPath();
ctx.arc(320, 150, 50, 0, Math.PI);
ctx.stroke();
```

### 文字

```javascript
ctx.font = "24px Arial";
ctx.fillStyle = "#000";
ctx.fillText("Hello Canvas", 50, 50);
ctx.strokeText("Outlined", 50, 100);

// 文字对齐
ctx.textAlign = "center"; // left, right, center
ctx.textBaseline = "middle"; // top, middle, bottom
```

### 图像

```javascript
const img = new Image();
img.src = "image.jpg";
img.onload = () => {
  ctx.drawImage(img, 0, 0); // 原始大小
  ctx.drawImage(img, 0, 0, 100, 100); // 指定大小
  ctx.drawImage(img, 0, 0, 50, 50, 100, 100, 100, 100); // 裁剪
};
```

### 变换

```javascript
ctx.save(); // 保存状态

ctx.translate(100, 100); // 平移
ctx.rotate(Math.PI / 4); // 旋转（弧度）
ctx.scale(2, 2); // 缩放

ctx.fillRect(-25, -25, 50, 50);

ctx.restore(); // 恢复状态
```

### 动画

```javascript
function animate() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 更新和绘制
  x += dx;
  ctx.beginPath();
  ctx.arc(x, 150, 20, 0, Math.PI * 2);
  ctx.fill();

  requestAnimationFrame(animate);
}

animate();
```

## 🔷 SVG 基础

SVG 是矢量图形，使用 XML 格式描述。

### 基本形状

```html
<svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
  <!-- 矩形 -->
  <rect
    x="10"
    y="10"
    width="100"
    height="80"
    fill="#3b82f6"
    stroke="#1d4ed8"
    stroke-width="2"
  />

  <!-- 圆形 -->
  <circle cx="200" cy="50" r="40" fill="#10b981" />

  <!-- 椭圆 -->
  <ellipse cx="320" cy="50" rx="50" ry="30" fill="#f59e0b" />

  <!-- 线条 -->
  <line x1="10" y1="150" x2="100" y2="200" stroke="#ef4444" stroke-width="3" />

  <!-- 多边形 -->
  <polygon points="150,150 200,200 100,200" fill="#8b5cf6" />

  <!-- 路径 -->
  <path d="M 250,150 L 300,200 L 250,200 Z" fill="#ec4899" />
</svg>
```

### 路径命令

| 命令  | 说明           |
| ----- | -------------- |
| M x,y | 移动到         |
| L x,y | 直线到         |
| H x   | 水平线到       |
| V y   | 垂直线到       |
| C     | 三次贝塞尔曲线 |
| Q     | 二次贝塞尔曲线 |
| A     | 弧线           |
| Z     | 闭合路径       |

```html
<path d="M 10,80 Q 95,10 180,80" stroke="#000" fill="none" />
```

### 文字

```html
<text x="50" y="50" font-size="24" fill="#333"> Hello SVG </text>

<text x="100" y="100" text-anchor="middle"> Centered Text </text>
```

### 渐变

```html
<defs>
  <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" stop-color="#3b82f6" />
    <stop offset="100%" stop-color="#8b5cf6" />
  </linearGradient>
</defs>

<rect x="10" y="10" width="200" height="100" fill="url(#gradient1)" />
```

### 分组和复用

```html
<defs>
  <g id="icon">
    <circle cx="10" cy="10" r="8" />
    <line x1="16" y1="16" x2="24" y2="24" stroke-width="2" />
  </g>
</defs>

<use href="#icon" x="50" y="50" fill="#3b82f6" stroke="#3b82f6" />
<use href="#icon" x="100" y="50" fill="#ef4444" stroke="#ef4444" />
```

### CSS 动画

```css
@keyframes pulse {
  0%,
  100% {
    r: 40;
  }
  50% {
    r: 50;
  }
}

circle {
  animation: pulse 1s ease-in-out infinite;
}
```

### JavaScript 操作

```javascript
const svg = document.querySelector("svg");
const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
circle.setAttribute("cx", "100");
circle.setAttribute("cy", "100");
circle.setAttribute("r", "50");
circle.setAttribute("fill", "#3b82f6");
svg.appendChild(circle);

// 修改属性
circle.setAttribute("r", "60");
```

## ⚖️ Canvas vs SVG

| 特性     | Canvas         | SVG              |
| -------- | -------------- | ---------------- |
| 类型     | 位图（像素）   | 矢量             |
| 缩放     | 会模糊         | 无损             |
| DOM      | 单一元素       | 每个图形都是 DOM |
| 事件     | 需手动检测     | 原生支持         |
| 性能     | 大量对象更好   | 少量复杂图形更好 |
| 适用场景 | 游戏、图像处理 | 图标、图表、地图 |

## 💡 选择建议

```
游戏开发     → Canvas
图片处理     → Canvas
数据可视化   → SVG
图标系统     → SVG
交互式地图   → SVG
粒子动画     → Canvas
```

## 🔗 相关资源

- [HTML 入门](/docs/frontend/html/)
- [CSS 动画](/docs/frontend/css/animation)

---

**下一步**：学习 [无障碍开发](/docs/frontend/html/accessibility) 构建包容性网页。
