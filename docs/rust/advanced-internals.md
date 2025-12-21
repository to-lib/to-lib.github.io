---
sidebar_position: 15
title: 底层原理
---

# 底层原理

本文深入探讨 Rust 的底层原理，适合想要深入理解语言内部机制的开发者。

## 内存布局

### 基本类型布局

```rust
use std::mem::{size_of, align_of};

fn main() {
    // 基本类型
    println!("bool: size={}, align={}", size_of::<bool>(), align_of::<bool>());
    println!("i32: size={}, align={}", size_of::<i32>(), align_of::<i32>());
    println!("i64: size={}, align={}", size_of::<i64>(), align_of::<i64>());

    // 指针
    println!("&i32: size={}", size_of::<&i32>());  // 8 on 64-bit
    println!("Box<i32>: size={}", size_of::<Box<i32>>());  // 8
}
```

### 结构体布局

```rust
// 默认布局（可能有填充）
struct A { a: u8, b: u32, c: u8 }  // size=12 (有填充)

// 紧凑布局
#[repr(C)]
struct B { a: u8, b: u32, c: u8 }  // C 兼容布局

#[repr(packed)]
struct C { a: u8, b: u32, c: u8 }  // size=6 (无填充)

// 透明布局（单字段）
#[repr(transparent)]
struct Wrapper(i32);
```

### 枚举布局

```rust
use std::mem::size_of;

enum Option<T> { None, Some(T) }  // 利用空位优化
enum Result<T, E> { Ok(T), Err(E) }

fn main() {
    // 空位优化
    println!("Option<&i32>: {}", size_of::<Option<&i32>>());  // 8，非 16
    println!("Option<Box<i32>>: {}", size_of::<Option<Box<i32>>>());  // 8
}
```

## 自定义智能指针

### 实现 Deref

```rust
use std::ops::{Deref, DerefMut};

struct MyBox<T>(T);

impl<T> MyBox<T> {
    fn new(x: T) -> MyBox<T> { MyBox(x) }
}

impl<T> Deref for MyBox<T> {
    type Target = T;
    fn deref(&self) -> &T { &self.0 }
}

impl<T> DerefMut for MyBox<T> {
    fn deref_mut(&mut self) -> &mut T { &mut self.0 }
}

fn main() {
    let x = MyBox::new(5);
    assert_eq!(*x, 5);
}
```

### 实现 Drop

```rust
struct CustomSmartPointer { data: String }

impl Drop for CustomSmartPointer {
    fn drop(&mut self) {
        println!("Dropping with data: {}", self.data);
    }
}

fn main() {
    let c = CustomSmartPointer { data: String::from("hello") };
    drop(c);  // 显式调用
    println!("CustomSmartPointer dropped");
}
```

### 引用计数指针

```rust
use std::cell::Cell;
use std::ptr::NonNull;

struct MyRc<T> {
    ptr: NonNull<RcBox<T>>,
}

struct RcBox<T> {
    value: T,
    ref_count: Cell<usize>,
}

impl<T> MyRc<T> {
    fn new(value: T) -> Self {
        let boxed = Box::new(RcBox {
            value,
            ref_count: Cell::new(1),
        });
        MyRc { ptr: NonNull::new(Box::into_raw(boxed)).unwrap() }
    }

    fn strong_count(&self) -> usize {
        unsafe { self.ptr.as_ref().ref_count.get() }
    }
}

impl<T> Clone for MyRc<T> {
    fn clone(&self) -> Self {
        unsafe {
            let count = self.ptr.as_ref().ref_count.get();
            self.ptr.as_ref().ref_count.set(count + 1);
        }
        MyRc { ptr: self.ptr }
    }
}

impl<T> Drop for MyRc<T> {
    fn drop(&mut self) {
        unsafe {
            let count = self.ptr.as_ref().ref_count.get();
            self.ptr.as_ref().ref_count.set(count - 1);
            if count == 1 {
                drop(Box::from_raw(self.ptr.as_ptr()));
            }
        }
    }
}
```

## 自己实现 Future

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

// 简单的延迟 Future
struct Delay {
    when: std::time::Instant,
}

impl Future for Delay {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if std::time::Instant::now() >= self.when {
            Poll::Ready(())
        } else {
            // 注册唤醒器
            let waker = cx.waker().clone();
            let when = self.when;
            std::thread::spawn(move || {
                std::thread::sleep(when - std::time::Instant::now());
                waker.wake();
            });
            Poll::Pending
        }
    }
}

async fn example() {
    let delay = Delay { when: std::time::Instant::now() + std::time::Duration::from_secs(1) };
    delay.await;
    println!("Delay complete!");
}
```

### 组合 Future

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

// Join 两个 Future
struct Join<A, B> {
    a: Option<A>,
    b: Option<B>,
    a_result: Option<A::Output>,
    b_result: Option<B::Output>,
}

impl<A, B> Future for Join<A, B>
where
    A: Future + Unpin,
    B: Future + Unpin,
{
    type Output = (A::Output, B::Output);

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Some(a) = self.a.as_mut() {
            if let Poll::Ready(result) = Pin::new(a).poll(cx) {
                self.a_result = Some(result);
                self.a = None;
            }
        }
        if let Some(b) = self.b.as_mut() {
            if let Poll::Ready(result) = Pin::new(b).poll(cx) {
                self.b_result = Some(result);
                self.b = None;
            }
        }
        if self.a.is_none() && self.b.is_none() {
            Poll::Ready((self.a_result.take().unwrap(), self.b_result.take().unwrap()))
        } else {
            Poll::Pending
        }
    }
}
```

## Mini Runtime

```rust
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

type Task = Pin<Box<dyn Future<Output = ()> + Send>>;

struct MiniRuntime {
    tasks: Mutex<VecDeque<Task>>,
}

impl MiniRuntime {
    fn new() -> Arc<Self> {
        Arc::new(MiniRuntime { tasks: Mutex::new(VecDeque::new()) })
    }

    fn spawn(&self, future: impl Future<Output = ()> + Send + 'static) {
        self.tasks.lock().unwrap().push_back(Box::pin(future));
    }

    fn run(&self) {
        loop {
            let task = { self.tasks.lock().unwrap().pop_front() };
            match task {
                Some(mut task) => {
                    let waker = dummy_waker();
                    let mut cx = Context::from_waker(&waker);
                    match task.as_mut().poll(&mut cx) {
                        Poll::Ready(()) => {}
                        Poll::Pending => {
                            self.tasks.lock().unwrap().push_back(task);
                        }
                    }
                }
                None => break,
            }
        }
    }
}

fn dummy_waker() -> Waker {
    fn clone(_: *const ()) -> RawWaker { RawWaker::new(std::ptr::null(), &VTABLE) }
    fn wake(_: *const ()) {}
    fn wake_by_ref(_: *const ()) {}
    fn drop(_: *const ()) {}

    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop);
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}
```

## FFI

### 调用 C 函数

```rust
// 链接 C 库
#[link(name = "c")]
extern "C" {
    fn abs(input: i32) -> i32;
    fn strlen(s: *const i8) -> usize;
}

fn main() {
    unsafe {
        println!("abs(-5) = {}", abs(-5));
    }
}
```

### 暴露 Rust 函数给 C

```rust
#[no_mangle]
pub extern "C" fn rust_add(a: i32, b: i32) -> i32 {
    a + b
}

#[no_mangle]
pub extern "C" fn rust_string_len(s: *const std::ffi::c_char) -> usize {
    if s.is_null() { return 0; }
    unsafe { std::ffi::CStr::from_ptr(s).to_bytes().len() }
}
```

### 处理 C 字符串

```rust
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

fn rust_string_to_c(s: &str) -> *mut c_char {
    CString::new(s).unwrap().into_raw()
}

fn c_string_to_rust(s: *const c_char) -> String {
    unsafe { CStr::from_ptr(s).to_string_lossy().into_owned() }
}

#[no_mangle]
pub extern "C" fn free_rust_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)); }
    }
}
```

### cbindgen 生成 C 头文件

```toml
# Cargo.toml
[lib]
crate-type = ["cdylib"]

[build-dependencies]
cbindgen = "0.26"
```

```rust
// build.rs
fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    cbindgen::generate(&crate_dir)
        .expect("Unable to generate bindings")
        .write_to_file("target/include/mylib.h");
}
```

## Java FFI (JNI)

Rust 可以通过 JNI (Java Native Interface) 与 Java 代码交互，这对于需要高性能计算的 Java 应用特别有用。

### 准备工作

```toml
# Cargo.toml
[lib]
crate-type = ["cdylib"]

[dependencies]
jni = "0.21"
```

### Rust 调用示例

首先定义 Java 类和 native 方法：

```java
// Java: com/example/RustLib.java
package com.example;

public class RustLib {
    static {
        System.loadLibrary("rust_jni");
    }

    // 声明 native 方法
    public static native String hello(String name);
    public static native int add(int a, int b);
    public static native byte[] processData(byte[] data);

    public static void main(String[] args) {
        System.out.println(hello("World"));
        System.out.println("2 + 3 = " + add(2, 3));
    }
}
```

然后在 Rust 中实现这些 native 方法：

```rust
use jni::objects::{JByteArray, JClass, JString};
use jni::sys::{jbyteArray, jint, jstring};
use jni::JNIEnv;

// 方法命名规则：Java_包名_类名_方法名
// 包名中的 . 替换为 _

#[no_mangle]
pub extern "system" fn Java_com_example_RustLib_hello(
    mut env: JNIEnv,
    _class: JClass,
    name: JString,
) -> jstring {
    // 从 Java String 获取 Rust String
    let name: String = env.get_string(&name)
        .expect("Couldn't get java string!")
        .into();

    // 创建返回的 Java String
    let output = format!("Hello, {}! From Rust", name);
    env.new_string(output)
        .expect("Couldn't create java string!")
        .into_raw()
}

#[no_mangle]
pub extern "system" fn Java_com_example_RustLib_add(
    _env: JNIEnv,
    _class: JClass,
    a: jint,
    b: jint,
) -> jint {
    a + b
}

#[no_mangle]
pub extern "system" fn Java_com_example_RustLib_processData(
    env: JNIEnv,
    _class: JClass,
    data: JByteArray,
) -> jbyteArray {
    // 获取 Java 字节数组
    let input = env.convert_byte_array(&data)
        .expect("Couldn't convert byte array");

    // 处理数据（示例：每个字节 +1）
    let output: Vec<u8> = input.iter().map(|&b| b.wrapping_add(1)).collect();

    // 返回新的 Java 字节数组
    env.byte_array_from_slice(&output)
        .expect("Couldn't create byte array")
        .into_raw()
}
```

### 处理复杂对象

```rust
use jni::objects::{JClass, JObject, JValue};
use jni::JNIEnv;

#[no_mangle]
pub extern "system" fn Java_com_example_RustLib_createUser(
    mut env: JNIEnv,
    _class: JClass,
    name: JString,
    age: jint,
) -> jobject {
    // 获取 Java 类
    let user_class = env.find_class("com/example/User")
        .expect("Couldn't find User class");

    // 获取构造函数
    let constructor = env.get_method_id(&user_class, "<init>", "(Ljava/lang/String;I)V")
        .expect("Couldn't find constructor");

    // 创建对象
    let name = env.get_string(&name).expect("Couldn't get string");
    let user = env.new_object_unchecked(
        &user_class,
        constructor,
        &[JValue::Object(&JObject::from(name)), JValue::Int(age)],
    ).expect("Couldn't create User object");

    user.into_raw()
}

#[no_mangle]
pub extern "system" fn Java_com_example_RustLib_processUser(
    mut env: JNIEnv,
    _class: JClass,
    user: JObject,
) {
    // 获取字段值
    let name_field = env.get_field(&user, "name", "Ljava/lang/String;")
        .expect("Couldn't get name field");

    // 调用方法
    let result = env.call_method(&user, "getName", "()Ljava/lang/String;", &[])
        .expect("Couldn't call getName");

    if let JValue::Object(name_obj) = result {
        let name: String = env.get_string(&JString::from(name_obj))
            .expect("Couldn't get string")
            .into();
        println!("User name from Rust: {}", name);
    }
}
```

### 异常处理

```rust
use jni::objects::{JClass, JThrowable};
use jni::JNIEnv;

#[no_mangle]
pub extern "system" fn Java_com_example_RustLib_riskyOperation(
    mut env: JNIEnv,
    _class: JClass,
    value: jint,
) -> jint {
    if value < 0 {
        // 抛出 Java 异常
        env.throw_new("java/lang/IllegalArgumentException", "Value must be non-negative")
            .expect("Couldn't throw exception");
        return -1;
    }

    // 检查是否有异常
    if env.exception_check().unwrap() {
        env.exception_describe().unwrap();
        env.exception_clear().unwrap();
        return -1;
    }

    value * 2
}
```

### 编译和使用

```bash
# 编译 Rust 库
cargo build --release

# macOS 上的库名
# target/release/librust_jni.dylib

# Linux 上的库名
# target/release/librust_jni.so

# Windows 上的库名
# target/release/rust_jni.dll

# 编译 Java
javac -d target com/example/RustLib.java

# 运行（需要指定库路径）
java -Djava.library.path=target/release -cp target com.example.RustLib
```

### 内存管理注意事项

| 问题         | 解决方案                                                 |
| ------------ | -------------------------------------------------------- |
| 局部引用限制 | 使用 `env.push_local_frame()` 和 `env.pop_local_frame()` |
| 长期持有对象 | 使用 `env.new_global_ref()` 创建全局引用                 |
| 字符串内存   | JString 转换后及时释放                                   |
| 异常安全     | 在调用 Java 方法后检查异常                               |

```rust
// 处理大量局部引用
#[no_mangle]
pub extern "system" fn Java_com_example_RustLib_processMany(
    mut env: JNIEnv,
    _class: JClass,
    count: jint,
) {
    for i in 0..count {
        // 创建局部引用帧，避免引用溢出
        env.push_local_frame(10).expect("Couldn't push frame");

        let s = env.new_string(format!("Item {}", i))
            .expect("Couldn't create string");
        // 使用 s...

        // 弹出帧，自动释放所有局部引用
        env.pop_local_frame(&JObject::null()).expect("Couldn't pop frame");
    }
}
```

## 总结

| 主题     | 关键点                                  |
| -------- | --------------------------------------- |
| 内存布局 | `#repr(C)`, `#repr(packed)`, 空位优化   |
| 智能指针 | 实现 `Deref`, `Drop`                    |
| Future   | 实现 `poll`, 返回 `Poll::Ready/Pending` |
| Runtime  | 任务队列 + Waker 机制                   |
| FFI      | `extern "C"`, `#[no_mangle]`, `CString` |

深入理解这些底层原理，能让你更好地驾驭 Rust！
