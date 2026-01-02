---
sidebar_position: 10
title: 正则表达式
---

# 正则表达式

> [!TIP]
> 正则表达式是处理字符串的强大工具，用于模式匹配、搜索和替换。

## 🎯 基础语法

### 创建正则

```javascript
// 字面量（推荐）
const regex1 = /pattern/flags;

// 构造函数（动态模式时使用）
const regex2 = new RegExp('pattern', 'flags');

// 示例
const emailRegex = /^\w+@\w+\.\w+$/;
```

### 常用标志

| 标志 | 说明                   |
| ---- | ---------------------- |
| `g`  | 全局匹配，查找所有匹配 |
| `i`  | 忽略大小写             |
| `m`  | 多行模式               |
| `s`  | 允许 `.` 匹配换行符    |
| `u`  | Unicode 模式           |

```javascript
/hello/i.test("Hello"); // true
"aaa".match(/a/g); // ['a', 'a', 'a']
```

## 📦 字符匹配

### 基础字符

```javascript
/abc/       // 匹配 'abc'
/./         // 匹配任意字符（除换行）
/\./        // 匹配点号（转义）
```

### 字符类

```javascript
/[abc]/ / // 匹配 a、b 或 c
  [a - z] / // 匹配小写字母
  /[A-Z]/ / // 匹配大写字母
  [0 - 9] / // 匹配数字
  /[^abc]/; // 匹配除 a、b、c 外的字符
```

### 预定义字符类

| 符号 | 等价于           | 说明       |
| ---- | ---------------- | ---------- |
| `\d` | `[0-9]`          | 数字       |
| `\D` | `[^0-9]`         | 非数字     |
| `\w` | `[a-zA-Z0-9_]`   | 单词字符   |
| `\W` | `[^a-zA-Z0-9_]`  | 非单词字符 |
| `\s` | `[\t\n\r\f\v ]`  | 空白字符   |
| `\S` | `[^\t\n\r\f\v ]` | 非空白字符 |

```javascript
/\d{3}/.test("123"); // true
/\w+/.test("hello_123"); // true
```

## 🔢 量词

```javascript
/a?/        // 0 或 1 个
/a*/        // 0 或多个
/a+/        // 1 或多个
/a{3}/      // 恰好 3 个
/a{2,4}/    // 2 到 4 个
/a{2,}/     // 至少 2 个
```

### 贪婪 vs 非贪婪

```javascript
// 贪婪（默认）- 尽可能多匹配
"aaaaab".match(/a+/); // ['aaaaa']

// 非贪婪 - 尽可能少匹配
"aaaaab".match(/a+?/); // ['a']
```

## 📍 位置匹配

```javascript
/^hello/    // 以 hello 开头
/world$/    // 以 world 结尾
/\bword\b/  // 单词边界

// 示例
/^hello$/.test('hello');      // true（完全匹配）
/\bcat\b/.test('a cat here'); // true（独立单词）
/\bcat\b/.test('category');   // false
```

## 📦 分组与引用

### 捕获组

```javascript
const regex = /(\d{4})-(\d{2})-(\d{2})/;
const match = "2024-01-15".match(regex);

console.log(match[0]); // '2024-01-15' (完整匹配)
console.log(match[1]); // '2024' (第一组)
console.log(match[2]); // '01' (第二组)
console.log(match[3]); // '15' (第三组)
```

### 命名捕获组

```javascript
const regex = /(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2})/;
const match = "2024-01-15".match(regex);

console.log(match.groups.year); // '2024'
console.log(match.groups.month); // '01'
console.log(match.groups.day); // '15'
```

### 非捕获组

```javascript
// (?:...) 只分组不捕获
/(?:Mr|Mrs)\. (\w+)/.exec("Mr. Smith");
// ['Mr. Smith', 'Smith'] - 只有一个捕获组
```

### 反向引用

```javascript
// \1 引用第一个捕获组
/(\w)\1/.test("aa"); // true（重复字符）
/(["'])(.*?)\1/.exec('"hello"'); // 匹配引号对
```

## 🔄 常用方法

### test() - 测试匹配

```javascript
/\d+/.test("abc123"); // true
/^[a-z]+$/.test("abc"); // true
```

### match() - 获取匹配

```javascript
"hello world".match(/\w+/); // ['hello']
"hello world".match(/\w+/g); // ['hello', 'world']
```

### matchAll() - 获取所有匹配

```javascript
const str = "2024-01-15 and 2024-02-20";
const regex = /(\d{4})-(\d{2})-(\d{2})/g;

for (const match of str.matchAll(regex)) {
  console.log(match[0], match[1], match[2]);
}
```

### replace() - 替换

```javascript
"hello world".replace(/world/, "JS");
// 'hello JS'

"foo bar foo".replace(/foo/g, "baz");
// 'baz bar baz'

// 使用捕获组
"John Smith".replace(/(\w+) (\w+)/, "$2, $1");
// 'Smith, John'

// 使用函数
"hello".replace(/./g, (char) => char.toUpperCase());
// 'HELLO'
```

### split() - 分割

```javascript
"a, b,  c".split(/,\s*/); // ['a', 'b', 'c']
"one1two2three".split(/\d/); // ['one', 'two', 'three']
```

## 💡 常用模式

### 邮箱验证

```javascript
const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
emailRegex.test("user@example.com"); // true
```

### 手机号（中国）

```javascript
const phoneRegex = /^1[3-9]\d{9}$/;
phoneRegex.test("13812345678"); // true
```

### URL 匹配

```javascript
const urlRegex = /^https?:\/\/[\w.-]+(?:\/[\w./?%&=-]*)?$/;
urlRegex.test("https://example.com/path?query=1"); // true
```

### 提取数字

```javascript
const numbers = "Price: $12.99, Qty: 3".match(/\d+\.?\d*/g);
// ['12.99', '3']
```

### HTML 标签

```javascript
const text = "<div>Hello</div>";
text.replace(/<[^>]+>/g, ""); // 'Hello'
```

## ⚠️ 注意事项

### 1. 转义特殊字符

```javascript
// 特殊字符需要转义: . * + ? ^ $ { } [ ] ( ) | \
const regex = /\$\d+\.\d{2}/; // 匹配 $12.99
```

### 2. 避免灾难性回溯

```javascript
// ❌ 危险 - 可能导致性能问题
/(a+)+b/.test("aaaaaaaaaaaaaaaaaaaaac");

// ✅ 改进
/a+b/.test("aaaaaaaaaaaaaaaaaaaaac");
```

## 🔗 相关资源

- [基础语法](/docs/frontend/javascript/fundamentals)
- [ES6+](/docs/frontend/javascript/es6)

---

**下一步**：学习 [浏览器原理](/docs/frontend/browser/) 了解运行环境。
