---
sidebar_position: 12
title: 宏编程
---

# 宏编程

宏是 Rust 的元编程工具,允许你编写生成代码的代码。Rust 有两种宏:声明宏和过程宏。

## 宏概述

### 宏 vs 函数

```rust
// 函数:运行时调用
fn add(a: i32, b: i32) -> i32 {
    a + b
}

// 宏:编译时展开
macro_rules! add_macro {
    ($a:expr, $b:expr) => {
        $a + $b
    };
}

fn main() {
    let result1 = add(1, 2);
    let result2 = add_macro!(1, 2);
}
```

### 宏的优势

1. **可变参数** - 接受任意数量的参数
2. **编译时展开** - 在编译期生成代码
3. **元编程** - 减少样板代码

## 声明宏 macro_rules

### 基础语法

```rust
macro_rules! say_hello {
    () => {
        println!("Hello!");
    };
}

fn main() {
    say_hello!();
}
```

### 带参数的宏

```rust
macro_rules! create_function {
    ($func_name:ident) => {
        fn $func_name() {
            println!("函数 {:?} 被调用", stringify!($func_name));
        }
    };
}

create_function!(foo);
create_function!(bar);

fn main() {
    foo();
    bar();
}
```

### 模式匹配

```rust
macro_rules! calculate {
    (eval $e:expr) => {
        {
            let val = $e;
            println!("{} = {}", stringify!($e), val);
        }
    };
}

fn main() {
    calculate!(eval 1 + 2);
    calculate!(eval (1 + 2) * 3);
}
```

### 指示符

```rust
macro_rules! types {
    // expr: 表达式
    ($e:expr) => { println!("expression: {}", $e); };
}

macro_rules! items {
    // ident: 标识符
    ($i:ident) => { let $i = 42; };
}

macro_rules! patterns {
    // ty: 类型
    ($t:ty) => {
        fn type_name() -> &'static str {
            stringify!($t)
        }
    };
}
```

常用指示符:

- `expr` - 表达式
- `ident` - 标识符
- `ty` - 类型
- `pat` - 模式
- `stmt` - 语句
- `block` - 代码块
- `item` - 项(函数、结构体等)
- `literal` - 字面量
- `tt` - token tree

### 重复

```rust
macro_rules! vec_of_strings {
    ($($x:expr),*) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x.to_string());
            )*
            temp_vec
        }
    };
}

fn main() {
    let v = vec_of_strings!("a", "b", "c");
    println!("{:?}", v);
}
```

### 多个分支

```rust
macro_rules! test {
    // 无参数
    () => {
        println!("无参数");
    };
    
    // 单个参数
    ($x:expr) => {
        println!("单个参数: {}", $x);
    };
    
    // 多个参数
    ($x:expr, $y:expr) => {
        println!("两个参数: {}, {}", $x, $y);
    };
}

fn main() {
    test!();
    test!(1);
    test!(1, 2);
}
```

## 实用声明宏示例

### HashMap 初始化宏

```rust
macro_rules! hashmap {
    ($($key:expr => $value:expr),* $(,)?) => {
        {
            let mut map = ::std::collections::HashMap::new();
            $(
                map.insert($key, $value);
            )*
            map
        }
    };
}

fn main() {
    let map = hashmap! {
        "key1" => "value1",
        "key2" => "value2",
    };
    println!("{:?}", map);
}
```

### 计算最小值宏

```rust
macro_rules! min {
    ($x:expr) => ($x);
    ($x:expr, $($y:expr),+) => {
        std::cmp::min($x, min!($($y),+))
    };
}

fn main() {
    println!("{}", min!(1));
    println!("{}", min!(1, 2));
    println!("{}", min!(5, 2 * 3, 4));
}
```

### 测试宏

```rust
macro_rules! test_case {
    ($name:ident, $input:expr, $expected:expr) => {
        #[test]
        fn $name() {
            assert_eq!(my_function($input), $expected);
        }
    };
}

fn my_function(x: i32) -> i32 {
    x * 2
}

#[cfg(test)]
mod tests {
    use super::*;
    
    test_case!(test_zero, 0, 0);
    test_case!(test_one, 1, 2);
    test_case!(test_five, 5, 10);
}
```

## 过程宏

过程宏有三种类型:

1. **派生宏(Derive)** - `#[derive(MyMacro)]`
2. **属性宏** - `#[my_attribute]`
3. **函数式过程宏** - `my_macro!()`

### 创建过程宏项目

```bash
cargo new my_macro --lib
```

```toml
# Cargo.toml
[lib]
proc-macro = true

[dependencies]
syn = "2.0"
quote = "1.0"
proc-macro2 = "1.0"
```

## 派生宏

### 实现 derive 宏

```rust
// my_macro/src/lib.rs
use proc_macro::TokenStream;
use quote::quote;
use syn;

#[proc_macro_derive(HelloMacro)]
pub fn hello_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    impl_hello_macro(&ast)
}

fn impl_hello_macro(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let gen = quote! {
        impl HelloMacro for #name {
            fn hello_macro() {
                println!("Hello, Macro! My name is {}!", stringify!(#name));
            }
        }
    };
    gen.into()
}
```

### 使用派生宏

```rust
// 使用宏的项目
use my_macro::HelloMacro;

trait HelloMacro {
    fn hello_macro();
}

#[derive(HelloMacro)]
struct Pancakes;

fn main() {
    Pancakes::hello_macro();
}
```

### Builder 模式派生宏

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(Builder)]
pub fn derive_builder(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let builder_name = format!("{}Builder", name);
    let builder_ident = syn::Ident::new(&builder_name, name.span());
    
    let fields = if let syn::Data::Struct(syn::DataStruct {
        fields: syn::Fields::Named(syn::FieldsNamed { ref named, .. }),
        ..
    }) = input.data
    {
        named
    } else {
        unimplemented!();
    };
    
    let builder_fields = fields.iter().map(|f| {
        let name = &f.ident;
        let ty = &f.ty;
        quote! { #name: Option<#ty> }
    });
    
    let methods = fields.iter().map(|f| {
        let name = &f.ident;
        let ty = &f.ty;
        quote! {
            pub fn #name(&mut self, #name: #ty) -> &mut Self {
                self.#name = Some(#name);
                self
            }
        }
    });
    
    let build_fields = fields.iter().map(|f| {
        let name = &f.ident;
        quote! {
            #name: self.#name.take().unwrap()
        }
    });
    
    let expanded = quote! {
        pub struct #builder_ident {
            #(#builder_fields,)*
        }
        
        impl #builder_ident {
            #(#methods)*
            
            pub fn build(&mut self) -> #name {
                #name {
                    #(#build_fields,)*
                }
            }
        }
        
        impl #name {
            pub fn builder() -> #builder_ident {
                #builder_ident {
                    #(#name: None,)*
                }
            }
        }
    };
    
    TokenStream::from(expanded)
}
```

## 属性宏

### 定义属性宏

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn route(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let path = attr.to_string();
    
    let expanded = quote! {
        #input_fn
        
        inventory::submit! {
            Route {
                path: #path,
                handler: #fn_name,
            }
        }
    };
    
    TokenStream::from(expanded)
}
```

### 使用属性宏

```rust
#[route("/users")]
fn get_users() {
    println!("Getting users");
}

#[route("/posts")]
fn get_posts() {
    println!("Getting posts");
}
```

## 函数式过程宏

### 定义函数式宏

```rust
use proc_macro::TokenStream;

#[proc_macro]
pub fn sql(input: TokenStream) -> TokenStream {
    let sql_query = input.to_string();
    
    // 处理 SQL 查询
    let output = format!("execute_query(\"{}\")", sql_query);
    
    output.parse().unwrap()
}
```

### 使用函数式宏

```rust
use my_macro::sql;

fn main() {
    sql!(SELECT * FROM users WHERE id = 1);
}
```

## 宏调试

### 使用 cargo expand

```bash
# 安装
cargo install cargo-expand

# 展开宏
cargo expand

# 展开特定模块
cargo expand my_module
```

### 使用 trace_macros

```rust
#![feature(trace_macros)]

fn main() {
    trace_macros!(true);
    let v = vec![1, 2, 3];
    trace_macros!(false);
}
```

### 使用 log_syntax

```rust
#![feature(log_syntax)]

macro_rules! example {
    () => {
        log_syntax!(Hello, world!);
    };
}

fn main() {
    example!();
}
```

## 常用宏示例

### 枚举遍历宏

```rust
macro_rules! enum_iterator {
    ($name:ident { $($variant:ident),* $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum $name {
            $($variant),*
        }
        
        impl $name {
            fn variants() -> &'static [$name] {
                &[$($name::$variant),*]
            }
        }
    };
}

enum_iterator!(Color {
    Red,
    Green,
    Blue,
});

fn main() {
    for color in Color::variants() {
        println!("{:?}", color);
    }
}
```

### 错误处理宏

```rust
macro_rules! try_option {
    ($expr:expr) => {
        match $expr {
            Some(val) => val,
            None => return None,
        }
    };
}

fn divide(a: i32, b: i32) -> Option<i32> {
    if b == 0 {
        return None;
    }
    Some(a / b)
}

fn calculate() -> Option<i32> {
    let x = try_option!(divide(10, 2));
    let y = try_option!(divide(20, 4));
    Some(x + y)
}
```

## 最佳实践

### 1. 保持宏简单

```rust
// 好:简单明了
macro_rules! double {
    ($x:expr) => {
        $x * 2
    };
}

// 避免:过于复杂
// macro_rules! complex {
//     (很多复杂的模式匹配...)
// }
```

### 2. 提供清晰的错误消息

```rust
macro_rules! check_range {
    ($val:expr, $min:expr, $max:expr) => {
        {
            let v = $val;
            if v < $min || v > $max {
                panic!(
                    "值 {} 超出范围 [{}, {}]",
                    v, $min, $max
                );
            }
            v
        }
    };
}
```

### 3. 文档化宏

```rust
/// 创建一个 HashMap 并初始化键值对
///
/// # Examples
///
/// ```
/// let map = hashmap! {
///     "key1" => 1,
///     "key2" => 2,
/// };
/// ```
#[macro_export]
macro_rules! hashmap {
    // 实现...
}
```

### 4. 使用 `#[macro_export]`

```rust
#[macro_export]
macro_rules! public_macro {
    () => {
        println!("This macro is public");
    };
}
```

## 总结

本文详细介绍了 Rust 的宏编程:

- ✅ 声明宏 macro_rules!
- ✅ 过程宏(派生、属性、函数式)
- ✅ 宏指示符和重复
- ✅ 实用宏示例
- ✅ 宏调试技巧
- ✅ 最佳实践

**关键要点:**

1. 宏在编译时展开,减少运行时开销
2. 声明宏用于简单的代码生成
3. 过程宏提供更强大的元编程能力
4. 使用 cargo expand 调试宏
5. 保持宏简单,提供清晰的文档

掌握宏编程后,继续学习 [不安全 Rust](./unsafe-rust) 或 [最佳实践](./best-practices)。
