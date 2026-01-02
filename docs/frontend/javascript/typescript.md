---
sidebar_position: 11
title: TypeScript 入门
---

# TypeScript 入门

> [!TIP]
> TypeScript 是 JavaScript 的超集，添加了静态类型检查，让代码更可靠、更易维护。

## 🎯 为什么用 TypeScript？

- **类型安全** - 编译时发现错误
- **更好的 IDE 支持** - 智能提示、自动补全
- **代码可读性** - 类型即文档
- **重构友好** - 改动时自动检查

## 📦 基础类型

```typescript
// 基本类型
let name: string = "Alice";
let age: number = 25;
let isActive: boolean = true;

// 数组
let numbers: number[] = [1, 2, 3];
let names: Array<string> = ["Alice", "Bob"];

// 元组
let point: [number, number] = [10, 20];

// any（避免使用）
let anything: any = "hello";

// unknown（更安全的 any）
let userInput: unknown = getData();

// void
function log(msg: string): void {
  console.log(msg);
}

// null 和 undefined
let nothing: null = null;
let undef: undefined = undefined;
```

## 📝 接口

### 定义对象结构

```typescript
interface User {
  name: string;
  age: number;
  email?: string; // 可选属性
  readonly id: number; // 只读属性
}

const user: User = {
  id: 1,
  name: "Alice",
  age: 25,
};

// user.id = 2;  // ❌ 只读属性不能修改
```

### 函数类型

```typescript
interface SearchFunc {
  (query: string, page: number): Promise<string[]>;
}

const search: SearchFunc = async (query, page) => {
  return [];
};
```

### 接口继承

```typescript
interface Person {
  name: string;
}

interface Student extends Person {
  grade: number;
}

const student: Student = {
  name: "Alice",
  grade: 3,
};
```

## 🎨 类型别名

```typescript
// 基础类型别名
type ID = string | number;
type Point = { x: number; y: number };

// 联合类型
type Status = "pending" | "success" | "error";

// 交叉类型
type Admin = User & { permissions: string[] };
```

### interface vs type

```typescript
// interface - 适合定义对象结构，可扩展
interface User {
  name: string;
}
interface User {
  age: number;
} // 自动合并

// type - 适合联合类型、元组、复杂类型
type Result = Success | Error;
type Pair = [string, number];
```

## 🔧 泛型

### 基础泛型

```typescript
// 泛型函数
function identity<T>(value: T): T {
  return value;
}

identity<string>("hello");
identity(42); // 类型推断为 number

// 泛型接口
interface Box<T> {
  value: T;
}

const box: Box<number> = { value: 42 };
```

### 泛型约束

```typescript
interface HasLength {
  length: number;
}

function logLength<T extends HasLength>(item: T): number {
  console.log(item.length);
  return item.length;
}

logLength("hello"); // ✅
logLength([1, 2, 3]); // ✅
// logLength(123);    // ❌ number 没有 length
```

### 常用泛型工具

```typescript
interface User {
  name: string;
  age: number;
  email: string;
}

// Partial - 所有属性变可选
type PartialUser = Partial<User>;

// Required - 所有属性变必需
type RequiredUser = Required<User>;

// Pick - 选择部分属性
type UserName = Pick<User, "name">;

// Omit - 排除部分属性
type UserWithoutEmail = Omit<User, "email">;

// Record - 键值对类型
type UserMap = Record<string, User>;
```

## ⚡ 类型断言

```typescript
// as 语法（推荐）
const input = document.getElementById("input") as HTMLInputElement;
input.value = "hello";

// 尖括号语法（JSX 中不可用）
const input2 = <HTMLInputElement>document.getElementById("input");

// 非空断言
function process(value: string | null) {
  console.log(value!.length); // 确定不为 null
}
```

## 🔀 类型守卫

```typescript
// typeof
function process(value: string | number) {
  if (typeof value === "string") {
    return value.toUpperCase();
  }
  return value * 2;
}

// instanceof
function handle(error: Error | TypeError) {
  if (error instanceof TypeError) {
    console.log("类型错误");
  }
}

// in
interface Cat {
  meow(): void;
}
interface Dog {
  bark(): void;
}

function speak(animal: Cat | Dog) {
  if ("meow" in animal) {
    animal.meow();
  } else {
    animal.bark();
  }
}
```

## 📦 在项目中使用

### 安装

```bash
pnpm add -D typescript
npx tsc --init  # 生成 tsconfig.json
```

### 基础配置

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "outDir": "./dist"
  },
  "include": ["src/**/*"]
}
```

## 🔗 相关资源

- [ES6+](/docs/frontend/javascript/es6)
- [React TypeScript](/docs/react/typescript)

---

**下一步**：学习 [this 关键字](/docs/frontend/javascript/this) 掌握 JavaScript 中的 this。
