---
sidebar_position: 3
title: 表单
---

# HTML 表单

> [!TIP]
> 表单是收集用户输入的核心方式。本文介绍表单元素、输入类型和表单验证。

## 🎯 表单结构

```html
<form action="/submit" method="POST">
  <label for="name">姓名：</label>
  <input type="text" id="name" name="name" required />

  <button type="submit">提交</button>
</form>
```

### form 属性

| 属性         | 说明                                             |
| ------------ | ------------------------------------------------ |
| `action`     | 提交地址                                         |
| `method`     | GET 或 POST                                      |
| `enctype`    | 编码类型（上传文件需设为 `multipart/form-data`） |
| `novalidate` | 禁用浏览器验证                                   |

## 📝 输入类型

### 文本输入

```html
<!-- 文本 -->
<input type="text" placeholder="请输入..." />

<!-- 密码 -->
<input type="password" placeholder="密码" />

<!-- 多行文本 -->
<textarea rows="4" cols="50">默认文本</textarea>
```

### 选择输入

```html
<!-- 单选 -->
<input type="radio" name="gender" value="male" id="male" />
<label for="male">男</label>
<input type="radio" name="gender" value="female" id="female" />
<label for="female">女</label>

<!-- 多选 -->
<input type="checkbox" id="agree" name="agree" />
<label for="agree">我同意条款</label>

<!-- 下拉选择 -->
<select name="city">
  <option value="">请选择城市</option>
  <option value="beijing">北京</option>
  <option value="shanghai">上海</option>
</select>
```

### 特殊类型

```html
<!-- 邮箱（自动验证格式） -->
<input type="email" placeholder="email@example.com" />

<!-- 电话 -->
<input type="tel" placeholder="手机号" />

<!-- URL -->
<input type="url" placeholder="https://..." />

<!-- 数字 -->
<input type="number" min="0" max="100" step="1" />

<!-- 范围滑块 -->
<input type="range" min="0" max="100" value="50" />

<!-- 日期 -->
<input type="date" />
<input type="time" />
<input type="datetime-local" />

<!-- 颜色选择器 -->
<input type="color" value="#ff0000" />

<!-- 文件上传 -->
<input type="file" accept="image/*" multiple />

<!-- 隐藏字段 -->
<input type="hidden" name="token" value="abc123" />
```

## 🏷️ 标签与分组

### label 标签

```html
<!-- 方式1：for 关联 -->
<label for="email">邮箱：</label>
<input type="email" id="email" name="email" />

<!-- 方式2：包裹 -->
<label>
  邮箱：
  <input type="email" name="email" />
</label>
```

### fieldset 分组

```html
<fieldset>
  <legend>个人信息</legend>

  <label for="name">姓名：</label>
  <input type="text" id="name" name="name" />

  <label for="age">年龄：</label>
  <input type="number" id="age" name="age" />
</fieldset>
```

## ✅ 表单验证

### HTML5 验证属性

```html
<!-- 必填 -->
<input type="text" required />

<!-- 最小/最大长度 -->
<input type="text" minlength="3" maxlength="20" />

<!-- 数值范围 -->
<input type="number" min="1" max="100" />

<!-- 正则验证 -->
<input type="text" pattern="[0-9]{6}" title="请输入6位数字" />
```

### 完整表单示例

```html
<form action="/register" method="POST">
  <fieldset>
    <legend>注册信息</legend>

    <div>
      <label for="username">用户名：</label>
      <input
        type="text"
        id="username"
        name="username"
        required
        minlength="3"
        maxlength="16"
        pattern="[a-zA-Z0-9_]+"
        title="只能包含字母、数字和下划线"
      />
    </div>

    <div>
      <label for="email">邮箱：</label>
      <input type="email" id="email" name="email" required />
    </div>

    <div>
      <label for="password">密码：</label>
      <input
        type="password"
        id="password"
        name="password"
        required
        minlength="8"
      />
    </div>

    <div>
      <label for="age">年龄：</label>
      <input type="number" id="age" name="age" min="1" max="150" />
    </div>

    <div>
      <input type="checkbox" id="terms" name="terms" required />
      <label for="terms">我同意服务条款</label>
    </div>
  </fieldset>

  <button type="submit">注册</button>
  <button type="reset">重置</button>
</form>
```

## 🎨 输入提示

```html
<!-- 占位符 -->
<input type="text" placeholder="请输入姓名" />

<!-- 自动完成建议 -->
<input type="text" list="cities" />
<datalist id="cities">
  <option value="北京" />
  <option value="上海" />
  <option value="广州" />
</datalist>

<!-- 自动填充 -->
<input type="text" name="name" autocomplete="name" />
<input type="email" name="email" autocomplete="email" />
```

## 🔘 按钮

```html
<!-- 提交按钮 -->
<button type="submit">提交</button>
<input type="submit" value="提交" />

<!-- 重置按钮 -->
<button type="reset">重置</button>

<!-- 普通按钮 -->
<button type="button" onclick="handleClick()">点击</button>
```

## 🔗 相关资源

- [HTML 入门](/docs/frontend/html/)
- [常用元素](/docs/frontend/html/elements)
- [JavaScript DOM](/docs/frontend/javascript/dom)

---

**下一步**：学习 [语义化](/docs/frontend/html/semantic) 提升网页可访问性和 SEO。
