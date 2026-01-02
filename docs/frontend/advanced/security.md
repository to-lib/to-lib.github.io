---
sidebar_position: 2
title: 前端安全
---

# 前端安全

> [!CAUTION]
> 安全是前端开发的重要组成部分。了解常见漏洞和防护措施可以保护用户数据安全。

## 🛡️ XSS (跨站脚本攻击)

攻击者注入恶意脚本到网页中。

### 类型

1. **存储型 XSS**：恶意脚本存储在服务器
2. **反射型 XSS**：恶意脚本通过 URL 参数注入
3. **DOM 型 XSS**：在客户端 JavaScript 中触发

### 攻击示例

```javascript
// 用户输入
const userInput = '<script>alert("XSS")</script>';

// ❌ 危险：直接插入 HTML
element.innerHTML = userInput;
```

### 防护措施

```javascript
// ✅ 使用 textContent
element.textContent = userInput;

// ✅ 转义 HTML
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ✅ 使用安全的模板库（React, Vue 自动转义）
```

### Content Security Policy (CSP)

```html
<!-- 通过 meta 标签 -->
<meta
  http-equiv="Content-Security-Policy"
  content="default-src 'self'; script-src 'self' https://trusted.com"
/>

<!-- 或通过 HTTP 头 -->
Content-Security-Policy: default-src 'self'; script-src 'self'
```

常用指令：

| 指令          | 说明                |
| ------------- | ------------------- |
| `default-src` | 默认策略            |
| `script-src`  | JavaScript 来源     |
| `style-src`   | CSS 来源            |
| `img-src`     | 图片来源            |
| `connect-src` | AJAX/WebSocket 来源 |

## 🔐 CSRF (跨站请求伪造)

攻击者诱导用户在已登录网站执行非预期操作。

### 攻击示例

```html
<!-- 恶意网站 -->
<img src="https://bank.com/transfer?to=attacker&amount=1000" />
```

### 防护措施

```javascript
// 1. CSRF Token
// 服务端生成 token，前端每次请求携带

fetch("/api/transfer", {
  method: "POST",
  headers: {
    "X-CSRF-Token": csrfToken, // 从页面或 Cookie 获取
  },
  body: JSON.stringify(data),
});
```

```javascript
// 2. SameSite Cookie
// 服务端设置
Set-Cookie: sessionId=abc; SameSite=Strict
```

```javascript
// 3. 验证 Origin/Referer 头
// 服务端验证请求来源
```

## 🔒 其他安全措施

### 点击劫持防护

```javascript
// 防止页面被嵌入 iframe
if (window.top !== window.self) {
  window.top.location = window.self.location;
}

// 更好的方式：使用 HTTP 头
X-Frame-Options: DENY
Content-Security-Policy: frame-ancestors 'none'
```

### 安全的 Cookie

```javascript
// 服务端设置安全 Cookie
Set-Cookie: token=abc;
  HttpOnly;     // 禁止 JS 访问
  Secure;       // 仅 HTTPS
  SameSite=Strict;  // 防止 CSRF
```

### 输入验证

```javascript
// ✅ 验证并清理用户输入
function validateEmail(email) {
  const pattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
  return pattern.test(email);
}

function sanitizeInput(input) {
  return input.trim().replace(/[<>]/g, ""); // 移除 HTML 标签
}
```

### URL 验证

```javascript
// ❌ 危险：直接使用用户提供的 URL
window.location = userUrl;

// ✅ 验证 URL
function isSafeUrl(url) {
  try {
    const parsed = new URL(url);
    return ["http:", "https:"].includes(parsed.protocol);
  } catch {
    return false;
  }
}

if (isSafeUrl(userUrl)) {
  window.location = userUrl;
}
```

### 敏感数据处理

```javascript
// ❌ 不要在前端存储敏感信息
localStorage.setItem("creditCard", "1234-5678-9012-3456");

// ❌ 不要在 URL 中传递敏感信息
window.location = `/page?token=${secretToken}`;

// ✅ 敏感数据应该
// 1. 通过 HTTPS 传输
// 2. 存储在 httpOnly Cookie 中
// 3. 必要时使用 sessionStorage（页面关闭即清除）
```

## 🔍 安全检查清单

- [ ] 所有用户输入都经过验证和转义
- [ ] 使用 HTTPS
- [ ] 设置适当的 CSP 策略
- [ ] Cookie 设置 HttpOnly, Secure, SameSite
- [ ] 实现 CSRF 防护
- [ ] 防止点击劫持
- [ ] 不在前端存储敏感数据
- [ ] 第三方库保持更新
- [ ] 错误信息不泄露敏感信息

## 💡 安全开发原则

1. **最小权限原则**：只请求必要的权限
2. **纵深防御**：多层防护
3. **默认安全**：默认配置应该是安全的
4. **不信任任何输入**：验证所有用户输入

## 🔗 相关资源

- [HTTP 网络](/docs/frontend/browser/network)
- [浏览器存储](/docs/frontend/browser/storage)

---

**下一步**：学习 [前端工程化](/docs/frontend/advanced/engineering) 构建现代项目。
