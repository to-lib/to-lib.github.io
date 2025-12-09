---
sidebar_position: 8
title: 智能指针
---

# 智能指针

智能指针是拥有数据所有权并提供额外功能的数据结构。

## `Box<T>`

### 基本使用

```rust
fn main() {
    let b = Box::new(5);
    println!("b = {}", b);
}  // b 离开作用域，堆上的数据被释放
```

### 递归类型

```rust
enum List {
    Cons(i32, Box<List>),
    Nil,
}

use List::{Cons, Nil};

fn main() {
    let list = Cons(1, Box::new(Cons(2, Box::new(Cons(3, Box::new(Nil))))));
}
```

## `Rc<T>`

引用计数智能指针，允许多个所有者。

```rust
use std::rc::Rc;

enum List {
    Cons(i32, Rc<List>),
    Nil,
}

use List::{Cons, Nil};

fn main() {
    let a = Rc::new(Cons(5, Rc::new(Cons(10, Rc::new(Nil)))));
    println!("引用计数: {}", Rc::strong_count(&a));  // 1
    
    let b = Cons(3, Rc::clone(&a));
    println!("引用计数: {}", Rc::strong_count(&a));  // 2
    
    {
        let c = Cons(4, Rc::clone(&a));
        println!("引用计数: {}", Rc::strong_count(&a));  // 3
    }
    
    println!("引用计数: {}", Rc::strong_count(&a));  // 2
}
```

## `RefCell<T>`

内部可变性模式，运行时检查借用规则。

```rust
use std::cell::RefCell;

fn main() {
    let x = RefCell::new(5);
    
    *x.borrow_mut() += 1;
    
    println!("x = {:?}", x.borrow());
}
```

### `Rc<RefCell<T>>`

```rust
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug)]
enum List {
    Cons(Rc<RefCell<i32>>, Rc<List>),
    Nil,
}

use List::{Cons, Nil};

fn main() {
    let value = Rc::new(RefCell::new(5));
    
    let a = Rc::new(Cons(Rc::clone(&value), Rc::new(Nil)));
    let b = Cons(Rc::new(RefCell::new(6)), Rc::clone(&a));
    let c = Cons(Rc::new(RefCell::new(10)), Rc::clone(&a));
    
    *value.borrow_mut() += 10;
    
    println!("a = {:?}", a);
    println!("b = {:?}", b);
    println!("c = {:?}", c);
}
```

## `Weak<T>`

弱引用，不增加引用计数。

```rust
use std::cell::RefCell;
use std::rc::{Rc, Weak};

#[derive(Debug)]
struct Node {
    value: i32,
    parent: RefCell<Weak<Node>>,
    children: RefCell<Vec<Rc<Node>>>,
}

fn main() {
    let leaf = Rc::new(Node {
        value: 3,
        parent: RefCell::new(Weak::new()),
        children: RefCell::new(vec![]),
    });
    
    let branch = Rc::new(Node {
        value: 5,
        parent: RefCell::new(Weak::new()),
        children: RefCell::new(vec![Rc::clone(&leaf)]),
    });
    
    *leaf.parent.borrow_mut() = Rc::downgrade(&branch);
}
```

## 总结

- ✅ `Box<T>`：堆分配
- ✅ `Rc<T>`：引用计数
- ✅ `RefCell<T>`：内部可变性
- ✅ `Weak<T>`：弱引用

这些是 Rust 编程的核心主题。继续探索 Rust 生态系统吧！
