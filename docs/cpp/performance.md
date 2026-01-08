---
sidebar_position: 29
title: 性能优化
---

# C++ 性能优化

C++ 性能优化技巧和最佳实践。

## 🚀 编译优化

```bash
# 优化级别
g++ -O0 main.cpp  # 无优化（调试用）
g++ -O2 main.cpp  # 推荐优化级别
g++ -O3 main.cpp  # 激进优化
g++ -Os main.cpp  # 优化体积

# 启用 LTO
g++ -flto -O2 main.cpp

# 生成性能分析信息
g++ -pg main.cpp  # gprof
```

## 📦 内存优化

```cpp
// 预分配容器容量
std::vector<int> v;
v.reserve(1000);  // 避免多次扩容

// 使用 emplace 原地构造
v.emplace_back(1, 2, 3);  // 避免临时对象

// 移动而非拷贝
std::string s1 = "Hello";
std::string s2 = std::move(s1);

// 返回值优化 (RVO)
std::vector<int> createVector() {
    return std::vector<int>{1, 2, 3};  // 编译器优化
}
```

## ⚡ 代码优化

```cpp
// 使用 const 引用避免拷贝
void process(const std::string& s);

// 缓存友好的数据结构
// 使用 vector 而非 list

// 循环优化
for (size_t i = 0, n = vec.size(); i < n; ++i) { }

// 使用 constexpr 编译期计算
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

// 避免虚函数调用开销（热路径）
// 使用 CRTP 或 final
```

## 🔍 性能分析工具

```bash
# perf (Linux)
perf record ./main
perf report

# gprof
g++ -pg main.cpp -o main
./main
gprof main gmon.out

# Valgrind callgrind
valgrind --tool=callgrind ./main
```

## 📊 缓存优化

```cpp
// 结构体成员对齐
struct alignas(64) CacheLine {
    int data[16];
};

// 数据局部性
// 按行访问二维数组
for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
        matrix[i][j] = 0;  // 好

// 避免 false sharing
struct alignas(64) Counter {
    std::atomic<int> value;
};
```

## ⚡ 优化建议

1. **先测量再优化** - 使用 profiler
2. **优化热点代码** - 80/20 法则
3. **避免过早优化** - 先保证正确性
4. **使用合适的数据结构**
5. **减少内存分配**
