---
sidebar_position: 12
title: 表单处理
---

# React 表单处理

> [!TIP]
> 表单是 Web 应用的重要组成部分，React 提供了受控组件和非受控组件两种方式处理表单。React 19 引入了强大的 **Actions** 特性，进一步简化了表单处理。

## 🆕 React 19 Actions（推荐）

React 19 引入了表单 Actions，可以直接将异步函数传递给 `<form>` 的 `action` 属性，自动处理 pending 状态和错误。

### 使用 useActionState

```jsx
import { useActionState } from "react";

function ContactForm() {
  async function submitForm(prevState, formData) {
    const email = formData.get("email");
    const message = formData.get("message");

    if (!email.includes("@")) {
      return { error: "请输入有效的邮箱地址" };
    }

    await sendMessage({ email, message });
    return { success: "消息已发送！" };
  }

  const [state, formAction, isPending] = useActionState(submitForm, {});

  return (
    <form action={formAction}>
      <input name="email" type="email" placeholder="邮箱" required />
      <textarea name="message" placeholder="留言内容" required />
      <button type="submit" disabled={isPending}>
        {isPending ? "发送中..." : "发送"}
      </button>
      {state.error && <p className="error">{state.error}</p>}
      {state.success && <p className="success">{state.success}</p>}
    </form>
  );
}
```

### 使用 useFormStatus

`useFormStatus` 可以在子组件中获取父表单的状态，无需传递 props：

```jsx
"use client";

import { useFormStatus } from "react-dom";

function SubmitButton() {
  const { pending, data } = useFormStatus();

  return (
    <button type="submit" disabled={pending}>
      {pending ? "提交中..." : "提交"}
    </button>
  );
}

function MyForm({ action }) {
  return (
    <form action={action}>
      <input name="name" placeholder="姓名" />
      <SubmitButton />
    </form>
  );
}
```

### 使用 useOptimistic

实现乐观 UI 更新，让用户立即看到操作结果：

```jsx
import { useOptimistic, useState } from "react";

function Messages({ messages, sendMessage }) {
  const [optimisticMessages, addOptimisticMessage] = useOptimistic(
    messages,
    (state, newMessage) => [...state, { ...newMessage, sending: true }]
  );

  async function handleSubmit(formData) {
    const text = formData.get("message");
    const newMessage = { id: Date.now(), text };

    addOptimisticMessage(newMessage); // 立即显示
    await sendMessage(newMessage); // 后台发送
  }

  return (
    <div>
      <ul>
        {optimisticMessages.map((msg) => (
          <li key={msg.id} style={{ opacity: msg.sending ? 0.5 : 1 }}>
            {msg.text}
            {msg.sending && " (发送中...)"}
          </li>
        ))}
      </ul>
      <form action={handleSubmit}>
        <input name="message" placeholder="输入消息" />
        <button type="submit">发送</button>
      </form>
    </div>
  );
}
```

## 🎯 受控组件

受控组件的表单数据由 React 组件的 state 管理。

### 基础输入

```jsx
function Form() {
  const [name, setName] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Submitted:", name);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Enter name"
      />
      <button type="submit">Submit</button>
    </form>
  );
}
```

### 多个输入

```jsx
function ContactForm() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    message: "",
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Form data:", formData);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        name="name"
        value={formData.name}
        onChange={handleChange}
        placeholder="Name"
      />
      <input
        name="email"
        type="email"
        value={formData.email}
        onChange={handleChange}
        placeholder="Email"
      />
      <textarea
        name="message"
        value={formData.message}
        onChange={handleChange}
        placeholder="Message"
      />
      <button type="submit">Send</button>
    </form>
  );
}
```

### 选择框和复选框

```jsx
function PreferencesForm() {
  const [preferences, setPreferences] = useState({
    theme: "light",
    notifications: true,
    newsletter: false,
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setPreferences((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  return (
    <form>
      {/* 选择框 */}
      <select name="theme" value={preferences.theme} onChange={handleChange}>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
        <option value="auto">Auto</option>
      </select>

      {/* 复选框 */}
      <label>
        <input
          type="checkbox"
          name="notifications"
          checked={preferences.notifications}
          onChange={handleChange}
        />
        Enable Notifications
      </label>

      <label>
        <input
          type="checkbox"
          name="newsletter"
          checked={preferences.newsletter}
          onChange={handleChange}
        />
        Subscribe to Newsletter
      </label>
    </form>
  );
}
```

## 🔓 非受控组件

非受控组件使用 ref 直接访问 DOM 元素的值。

### 基础用法

```jsx
function UncontrolledForm() {
  const nameRef = useRef();
  const emailRef = useRef();

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Name:", nameRef.current.value);
    console.log("Email:", emailRef.current.value);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input ref={nameRef} type="text" defaultValue="Alice" />
      <input ref={emailRef} type="email" />
      <button type="submit">Submit</button>
    </form>
  );
}
```

## ✅ 表单验证

### 手动验证

```jsx
function LoginForm() {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });
  const [errors, setErrors] = useState({});

  const validate = () => {
    const newErrors = {};

    if (!formData.email) {
      newErrors.email = "Email is required";
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = "Email is invalid";
    }

    if (!formData.password) {
      newErrors.password = "Password is required";
    } else if (formData.password.length < 6) {
      newErrors.password = "Password must be at least 6 characters";
    }

    return newErrors;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const validationErrors = validate();

    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }

    console.log("Form is valid!", formData);
    setErrors({});
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    // 清除该字段的错误
    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: "" }));
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <input
          name="email"
          type="email"
          value={formData.email}
          onChange={handleChange}
          placeholder="Email"
        />
        {errors.email && <span className="error">{errors.email}</span>}
      </div>

      <div>
        <input
          name="password"
          type="password"
          value={formData.password}
          onChange={handleChange}
          placeholder="Password"
        />
        {errors.password && <span className="error">{errors.password}</span>}
      </div>

      <button type="submit">Login</button>
    </form>
  );
}
```

## 🛠️ React Hook Form

React Hook Form 是一个高性能、灵活的表单库。

### 基础用法

```jsx
import { useForm } from "react-hook-form";

function RegistrationForm() {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm();

  const onSubmit = (data) => {
    console.log(data);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input
        {...register("name", { required: "Name is required" })}
        placeholder="Name"
      />
      {errors.name && <span>{errors.name.message}</span>}

      <input
        {...register("email", {
          required: "Email is required",
          pattern: {
            value: /\S+@\S+\.\S+/,
            message: "Invalid email",
          },
        })}
        placeholder="Email"
      />
      {errors.email && <span>{errors.email.message}</span>}

      <input
        {...register("password", {
          required: "Password is required",
          minLength: {
            value: 6,
            message: "Password must be at least 6 characters",
          },
        })}
        type="password"
        placeholder="Password"
      />
      {errors.password && <span>{errors.password.message}</span>}

      <button type="submit">Register</button>
    </form>
  );
}
```

### 复杂验证

```jsx
function AdvancedForm() {
  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
  } = useForm();

  const password = watch("password");

  const onSubmit = (data) => {
    console.log(data);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input
        {...register("username", {
          required: "Username is required",
          minLength: {
            value: 3,
            message: "Username must be at least 3 characters",
          },
          validate: async (value) => {
            // 异步验证
            const exists = await checkUsernameExists(value);
            return !exists || "Username already taken";
          },
        })}
        placeholder="Username"
      />
      {errors.username && <span>{errors.username.message}</span>}

      <input
        {...register("password", {
          required: "Password is required",
          minLength: {
            value: 8,
            message: "Password must be at least 8 characters",
          },
        })}
        type="password"
        placeholder="Password"
      />
      {errors.password && <span>{errors.password.message}</span>}

      <input
        {...register("confirmPassword", {
          required: "Please confirm password",
          validate: (value) => value === password || "Passwords do not match",
        })}
        type="password"
        placeholder="Confirm Password"
      />
      {errors.confirmPassword && <span>{errors.confirmPassword.message}</span>}

      <button type="submit">Submit</button>
    </form>
  );
}
```

## 📖 实用示例

### 动态表单字段

```jsx
function DynamicForm() {
  const [fields, setFields] = useState([{ id: 1, value: "" }]);

  const addField = () => {
    setFields([...fields, { id: Date.now(), value: "" }]);
  };

  const removeField = (id) => {
    setFields(fields.filter((field) => field.id !== id));
  };

  const handleChange = (id, value) => {
    setFields(
      fields.map((field) => (field.id === id ? { ...field, value } : field))
    );
  };

  return (
    <form>
      {fields.map((field) => (
        <div key={field.id}>
          <input
            value={field.value}
            onChange={(e) => handleChange(field.id, e.target.value)}
          />
          <button type="button" onClick={() => removeField(field.id)}>
            Remove
          </button>
        </div>
      ))}
      <button type="button" onClick={addField}>
        Add Field
      </button>
    </form>
  );
}
```

### 文件上传

```jsx
function FileUploadForm() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);

    // 预览图片
    if (selectedFile && selectedFile.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });

    console.log("Uploaded!", response);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input type="file" accept="image/*" onChange={handleFileChange} />

      {preview && (
        <img src={preview} alt="Preview" style={{ maxWidth: "200px" }} />
      )}

      <button type="submit" disabled={!file}>
        Upload
      </button>
    </form>
  );
}
```

### 多步表单

```jsx
function MultiStepForm() {
  const [step, setStep] = useState(1);
  const [formData, setFormData] = useState({
    personal: {},
    address: {},
    payment: {},
  });

  const handleNext = () => setStep(step + 1);
  const handlePrev = () => setStep(step - 1);

  const handleChange = (section, data) => {
    setFormData((prev) => ({
      ...prev,
      [section]: { ...prev[section], ...data },
    }));
  };

  return (
    <div>
      {step === 1 && (
        <PersonalInfo
          data={formData.personal}
          onChange={(data) => handleChange("personal", data)}
          onNext={handleNext}
        />
      )}

      {step === 2 && (
        <AddressInfo
          data={formData.address}
          onChange={(data) => handleChange("address", data)}
          onNext={handleNext}
          onPrev={handlePrev}
        />
      )}

      {step === 3 && (
        <PaymentInfo
          data={formData.payment}
          onChange={(data) => handleChange("payment", data)}
          onPrev={handlePrev}
          onSubmit={() => console.log("Final:", formData)}
        />
      )}
    </div>
  );
}
```

---

**下一步**: 查看 [样式方案](/docs/react/styling-solutions) 了解如何为表单添加样式，或学习 [TypeScript](/docs/react/typescript) 为表单添加类型安全。
