---
sidebar_position: 1
title: 前端学习指南
---

# 前端基础学习指南

> [!TIP]
> 本指南涵盖 HTML、CSS、JavaScript 三大核心技术，以及浏览器原理和进阶主题，帮助你建立扎实的前端开发基础。

## 🎯 学习路线

```mermaid
graph LR
    A[HTML] --> B[CSS]
    B --> C[JavaScript]
    C --> D[浏览器原理]
    D --> E[进阶主题]
    E --> F[React/Vue]

    style A fill:#e34c26
    style B fill:#264de4
    style C fill:#f7df1e
    style D fill:#8b5cf6
    style E fill:#10b981
    style F fill:#61dafb
```

## 📚 内容概览

### HTML - 网页结构

| 主题                                            | 内容               |
| ----------------------------------------------- | ------------------ |
| [HTML 入门](/docs/frontend/html/)               | 文档结构、基础语法 |
| [常用元素](/docs/frontend/html/elements)        | 文本、图片、链接   |
| [表单](/docs/frontend/html/forms)               | 输入控件、验证     |
| [语义化](/docs/frontend/html/semantic)          | 语义标签、SEO      |
| [Canvas 与 SVG](/docs/frontend/html/canvas-svg) | 图形绘制、矢量图   |
| [无障碍开发](/docs/frontend/html/accessibility) | ARIA、键盘导航     |

### CSS - 网页样式

| 主题                                        | 内容                       |
| ------------------------------------------- | -------------------------- |
| [CSS 入门](/docs/frontend/css/)             | 语法、引入方式             |
| [选择器](/docs/frontend/css/selectors)      | 选择器类型、优先级         |
| [布局](/docs/frontend/css/layout)           | Flexbox、Grid              |
| [响应式](/docs/frontend/css/responsive)     | 媒体查询、移动优先         |
| [动画与过渡](/docs/frontend/css/animation)  | transition、animation      |
| [CSS 新特性](/docs/frontend/css/modern-css) | 容器查询、:has()、层叠层   |
| [移动端适配](/docs/frontend/css/mobile)     | viewport、rem/vw、1px 问题 |

### JavaScript - 网页交互

| 主题                                                  | 内容                     |
| ----------------------------------------------------- | ------------------------ |
| [JS 入门](/docs/frontend/javascript/)                 | 语言特点、运行环境       |
| [基础语法](/docs/frontend/javascript/fundamentals)    | 变量、函数、对象         |
| [DOM 操作](/docs/frontend/javascript/dom)             | 元素操作、事件           |
| [异步编程](/docs/frontend/javascript/async)           | Promise、async/await     |
| [ES6+](/docs/frontend/javascript/es6)                 | 现代 JavaScript          |
| [闭包与作用域](/docs/frontend/javascript/closure)     | 词法作用域、闭包应用     |
| [原型链](/docs/frontend/javascript/prototype)         | 原型、继承、Class        |
| [this 关键字](/docs/frontend/javascript/this)         | 绑定规则、箭头函数       |
| [深浅拷贝](/docs/frontend/javascript/copy)            | 引用类型、克隆方法       |
| [函数式编程](/docs/frontend/javascript/functional)    | 纯函数、组合、柯里化     |
| [设计模式](/docs/frontend/javascript/design-patterns) | 单例、发布订阅、策略模式 |
| [手写实现](/docs/frontend/javascript/implementations) | call/bind、Promise、防抖 |
| [数据结构](/docs/frontend/javascript/data-structures) | 栈、队列、链表、树       |
| [TypeScript](/docs/frontend/javascript/typescript)    | 类型系统、接口、泛型     |
| [错误处理](/docs/frontend/javascript/error-handling)  | try/catch、异步错误      |
| [模块化](/docs/frontend/javascript/modules)           | ESM、CommonJS            |
| [正则表达式](/docs/frontend/javascript/regex)         | 模式匹配、常用模式       |

### 浏览器 - 运行环境

| 主题                                          | 内容                       |
| --------------------------------------------- | -------------------------- |
| [浏览器原理](/docs/frontend/browser/)         | 渲染流程、Event Loop       |
| [存储机制](/docs/frontend/browser/storage)    | Cookie、Storage、IndexedDB |
| [HTTP 网络](/docs/frontend/browser/network)   | Fetch、CORS、请求优化      |
| [Web Workers](/docs/frontend/browser/workers) | 多线程、Service Worker     |
| [跨域详解](/docs/frontend/browser/cors)       | CORS、代理、JSONP          |
| [调试技巧](/docs/frontend/browser/debugging)  | DevTools、断点、性能分析   |
| [WebSocket](/docs/frontend/browser/websocket) | 实时通信、心跳机制         |

### 进阶主题

| 主题                                            | 内容                      |
| ----------------------------------------------- | ------------------------- |
| [性能优化](/docs/frontend/advanced/performance) | Core Web Vitals、加载优化 |
| [前端安全](/docs/frontend/advanced/security)    | XSS、CSRF、CSP            |
| [工程化](/docs/frontend/advanced/engineering)   | 包管理、构建工具、规范    |
| [前端监控](/docs/frontend/advanced/monitoring)  | 错误监控、性能监控、埋点  |

## 🔗 进阶学习

完成基础后，推荐继续学习：

- [React 开发指南](/docs/react) - 现代前端框架
- [TypeScript](/docs/react/typescript) - 类型安全
