---
sidebar_position: 14
title: 不安全 Rust
---

# 不安全 Rust

Rust 的核心是内存安全,但有时需要执行不安全操作。本文介绍如何正确使用 `unsafe` 关键字。

## 为什么需要 Unsafe

### 安全 vs 不安全

```rust
// 安全代码
fn safe_function() {
    let mut v = vec![1, 2, 3];
    v.push(4);
}

// 不安全代码
unsafe fn unsafe_function() {
    // 可以执行不安全操作
}
```

### Unsafe 的使用场景

1. 解引用裸指针
2. 调用不安全函数或方法
3. 访问或修改可变静态变量
4. 实现不安全 trait
5. 访问 union 的字段

## 裸指针

### 创建裸指针

```rust
fn main() {
    let mut num = 5;
    
    // 创建不可变裸指针
    let r1 = &num as *const i32;
    
    // 创建可变裸指针
    let r2 = &mut num as *mut i32;
    
    // 裸指针可以在安全代码中创建
    println!("r1: {:p}", r1);
    println!("r2: {:p}", r2);
}
```

### 解引用裸指针

```rust
fn main() {
    let mut num = 5;
    
    let r1 = &num as *const i32;
    let r2 = &mut num as *mut i32;
    
    // 解引用裸指针需要 unsafe 块
    unsafe {
        println!("r1: {}", *r1);
        println!("r2: {}", *r2);
    }
}
```

### 裸指针 vs 引用

```rust
fn main() {
    let mut num = 5;
    
    // 可以同时创建多个裸指针
    let r1 = &num as *const i32;
    let r2 = &num as *const i32;
    let r3 = &mut num as *mut i32;
    
    // 裸指针不保证指向有效内存
    let address = 0x012345usize;
    let r = address as *const i32;
    
    unsafe {
        // 可能导致未定义行为
        // println!("r: {}", *r);
    }
}
```

## 不安全函数和方法

### 定义不安全函数

```rust
unsafe fn dangerous() {
    println!("执行危险操作");
}

fn main() {
    // 调用不安全函数需要 unsafe 块
    unsafe {
        dangerous();
    }
}
```

### 安全抽象封装不安全代码

```rust
use std::slice;

fn split_at_mut(slice: &mut [i32], mid: usize) -> (&mut [i32], &mut [i32]) {
    let len = slice.len();
    let ptr = slice.as_mut_ptr();
    
    assert!(mid <= len);
    
    unsafe {
        (
            slice::from_raw_parts_mut(ptr, mid),
            slice::from_raw_parts_mut(ptr.add(mid), len - mid),
        )
    }
}

fn main() {
    let mut v = vec![1, 2, 3, 4, 5, 6];
    let (left, right) = split_at_mut(&mut v, 3);
    
    println!("左: {:?}", left);   // [1, 2, 3]
    println!("右: {:?}", right);  // [4, 5, 6]
}
```

## 外部函数接口 (FFI)

### 调用 C 函数

```rust
extern "C" {
    fn abs(input: i32) -> i32;
}

fn main() {
    unsafe {
        println!("C 语言 abs(-3): {}", abs(-3));
    }
}
```

### 从其他语言调用 Rust

```rust
#[no_mangle]
pub extern "C" fn call_from_c() {
    println!("从 C 调用 Rust!");
}

#[no_mangle]
pub extern "C" fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}
```

### 使用 libc

```rust
use std::ffi::CString;
use std::os::raw::c_char;

extern "C" {
    fn puts(s: *const c_char) -> i32;
}

fn main() {
    let c_string = CString::new("Hello from Rust!").unwrap();
    
    unsafe {
        puts(c_string.as_ptr());
    }
}
```

## 访问或修改可变静态变量

### 静态变量

```rust
static HELLO_WORLD: &str = "Hello, world!";

fn main() {
    println!("{}", HELLO_WORLD);
}
```

### 可变静态变量

```rust
static mut COUNTER: u32 = 0;

fn add_to_counter(inc: u32) {
    unsafe {
        COUNTER += inc;
    }
}

fn main() {
    add_to_counter(3);
    
    unsafe {
        println!("COUNTER: {}", COUNTER);
    }
}
```

## 实现不安全 Trait

### 定义和实现不安全 Trait

```rust
unsafe trait Foo {
    fn foo(&self);
}

struct MyType;

unsafe impl Foo for MyType {
    fn foo(&self) {
        println!("实现不安全 trait");
    }
}

fn main() {
    let my = MyType;
    my.foo();
}
```

### Send 和 Sync Trait

```rust
// Send: 可以在线程间转移所有权
// Sync: 可以在线程间共享引用

use std::rc::Rc;

// Rc<T> 不是 Send 或 Sync
fn example() {
    let rc = Rc::new(5);
    
    // 以下代码会编译错误
    // std::thread::spawn(move || {
    //     println!("{}", rc);
    // });
}
```

## 联合体 (Union)

### 定义和使用 Union

```rust
#[repr(C)]
union MyUnion {
    f1: u32,
    f2: f32,
}

fn main() {
    let u = MyUnion { f1: 1 };
    
    unsafe {
        println!("u.f1: {}", u.f1);
        
        // 访问 f2 是未定义行为
        // println!("u.f2: {}", u.f2);
    }
}
```

## 内存操作

### 手动分配和释放内存

```rust
use std::alloc::{alloc, dealloc, Layout};

fn main() {
    unsafe {
        let layout = Layout::from_size_align(4, 4).unwrap();
        let ptr = alloc(layout);
        
        if ptr.is_null() {
            panic!("分配失败");
        }
        
        // 使用内存
        *ptr = 42;
        println!("值: {}", *ptr);
        
        // 释放内存
        dealloc(ptr, layout);
    }
}
```

### 原始指针运算

```rust
fn main() {
    let arr = [1, 2, 3, 4, 5];
    let ptr = arr.as_ptr();
    
    unsafe {
        println!("第0个: {}", *ptr);
        println!("第1个: {}", *ptr.add(1));
        println!("第2个: {}", *ptr.add(2));
        
        // 使用 offset
        println!("offset(3): {}", *ptr.offset(3));
    }
}
```

## Unsafe 最佳实践

### 1. 最小化 Unsafe 块

```rust
// 不好:整个函数都是 unsafe
unsafe fn bad_example(ptr: *const i32) {
    let value = *ptr;
    println!("{}", value);
    let x = 5 + 3;  // 安全操作也在 unsafe 中
}

// 好:只将必要部分放在 unsafe 中
fn good_example(ptr: *const i32) {
    let value = unsafe { *ptr };
    println!("{}", value);
    let x = 5 + 3;  // 安全操作在外面
}
```

### 2. 提供安全抽象

```rust
pub struct SafeWrapper {
    data: *mut i32,
    len: usize,
}

impl SafeWrapper {
    pub fn new(data: Vec<i32>) -> Self {
        let mut data = data;
        let ptr = data.as_mut_ptr();
        let len = data.len();
        std::mem::forget(data);
        
        SafeWrapper { data: ptr, len }
    }
    
    pub fn get(&self, index: usize) -> Option<i32> {
        if index < self.len {
            unsafe { Some(*self.data.add(index)) }
        } else {
            None
        }
    }
}

impl Drop for SafeWrapper {
    fn drop(&mut self) {
        unsafe {
            Vec::from_raw_parts(self.data, self.len, self.len);
        }
    }
}
```

### 3. 文档化不安全性

```rust
/// # Safety
///
/// `ptr` 必须指向有效的 i32 值
/// `ptr` 必须正确对齐
/// `ptr` 指向的值必须初始化
unsafe fn read_value(ptr: *const i32) -> i32 {
    *ptr
}
```

### 4. 使用类型系统保证安全

```rust
use std::marker::PhantomData;

struct RawPtr<T> {
    ptr: *const T,
    _marker: PhantomData<T>,
}

impl<T> RawPtr<T> {
    unsafe fn new(ptr: *const T) -> Self {
        RawPtr {
            ptr,
            _marker: PhantomData,
        }
    }
    
    unsafe fn as_ref(&self) -> &T {
        &*self.ptr
    }
}
```

## 常见陷阱

### 悬垂指针

```rust
fn dangle() -> *const i32 {
    let x = 5;
    &x as *const i32  // 错误:返回指向栈上变量的指针
}

// 正确方式
fn not_dangle() -> Box<i32> {
    Box::new(5)
}
```

### 数据竞争

```rust
use std::thread;

static mut SHARED: i32 = 0;

fn main() {
    // 错误:多个线程同时访问可变静态变量
    let handle1 = thread::spawn(|| unsafe {
        SHARED += 1;
    });
    
    let handle2 = thread::spawn(|| unsafe {
        SHARED += 1;
    });
    
    handle1.join().unwrap();
    handle2.join().unwrap();
    
    // 结果不确定!
}
```

### 未初始化内存

```rust
fn main() {
    let mut x: i32;
    
    // 错误:使用未初始化的变量
    // println!("{}", x);
    
    unsafe {
        let mut uninit: std::mem::MaybeUninit<i32> = std::mem::MaybeUninit::uninit();
        // 使用前必须初始化
        uninit.write(42);
        let init = uninit.assume_init();
        println!("{}", init);
    }
}
```

## 工具和验证

### Miri

```bash
# 安装 Miri
rustup +nightly component add miri

# 运行 Miri
cargo +nightly miri test
```

### Address Sanitizer

```bash
# 使用 AddressSanitizer
RUSTFLAGS="-Z sanitizer=address" cargo +nightly run
```

### 使用 valgrind

```bash
cargo build
valgrind --leak-check=full ./target/debug/your_program
```

## 总结

本文介绍了 Rust 的不安全特性:

- ✅ unsafe 关键字和使用场景
- ✅ 裸指针:创建、解引用、运算
- ✅ 不安全函数和方法
- ✅ FFI:调用 C 函数,从 C 调用 Rust
- ✅ 可变静态变量
- ✅ 不安全 trait 实现
- ✅ Union 联合体
- ✅ 内存操作:手动分配/释放
- ✅ 最佳实践和常见陷阱
- ✅ 验证工具:Miri、ASan、Valgrind

**关键原则:**

1. 只在必要时使用 unsafe
2. 最小化 unsafe 块范围
3. 提供安全的抽象接口
4. 详细文档化安全要求
5. 使用工具验证正确性

恭喜!你已完成所有 Rust 文档的学习,掌握了从基础到高级的完整知识体系!
