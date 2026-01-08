---
sidebar_position: 7.5
title: 文件 I/O
---

# C++ 文件输入输出

C++ 使用流（stream）进行文件操作，提供类型安全的 I/O。

## 🎯 文件流类

```cpp
#include <fstream>

// ifstream - 输入文件流（读取）
// ofstream - 输出文件流（写入）
// fstream  - 双向文件流
```

## 📄 文本文件操作

### 写入文件

```cpp
#include <fstream>
#include <iostream>

int main() {
    std::ofstream file("output.txt");

    if (file.is_open()) {
        file << "Hello, World!" << std::endl;
        file << "Line 2" << std::endl;
        file << 42 << " " << 3.14 << std::endl;
        file.close();
    } else {
        std::cerr << "无法打开文件" << std::endl;
    }

    return 0;
}
```

### 读取文件

```cpp
#include <fstream>
#include <string>

int main() {
    std::ifstream file("input.txt");

    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::cout << line << std::endl;
        }
        file.close();
    }

    // 读取特定类型
    std::ifstream data("data.txt");
    int num;
    double value;
    data >> num >> value;

    return 0;
}
```

### 追加模式

```cpp
std::ofstream file("log.txt", std::ios::app);
file << "New log entry" << std::endl;
```

## 📦 二进制文件

```cpp
#include <fstream>

struct Record {
    int id;
    char name[50];
    double score;
};

// 写入二进制
void writeBinary() {
    std::ofstream file("data.bin", std::ios::binary);

    Record r = {1, "张三", 95.5};
    file.write(reinterpret_cast<char*>(&r), sizeof(r));

    file.close();
}

// 读取二进制
void readBinary() {
    std::ifstream file("data.bin", std::ios::binary);

    Record r;
    file.read(reinterpret_cast<char*>(&r), sizeof(r));

    std::cout << r.name << ": " << r.score << std::endl;
    file.close();
}
```

## 🔍 文件位置操作

```cpp
std::fstream file("data.txt", std::ios::in | std::ios::out);

// 获取当前位置
std::streampos pos = file.tellg();  // 读
std::streampos pos2 = file.tellp(); // 写

// 移动位置
file.seekg(0, std::ios::beg);    // 移到开头
file.seekg(0, std::ios::end);    // 移到结尾
file.seekg(10, std::ios::cur);   // 从当前位置移动
```

## 🔄 字符串流

```cpp
#include <sstream>

// 字符串输出流
std::ostringstream oss;
oss << "Value: " << 42 << ", Pi: " << 3.14;
std::string result = oss.str();

// 字符串输入流
std::string data = "10 20 30";
std::istringstream iss(data);
int a, b, c;
iss >> a >> b >> c;

// 双向字符串流
std::stringstream ss;
ss << 100;
int num;
ss >> num;
```

## 📋 C++17 文件系统

```cpp
#include <filesystem>
namespace fs = std::filesystem;

// 检查文件存在
if (fs::exists("file.txt")) { }

// 文件大小
auto size = fs::file_size("file.txt");

// 遍历目录
for (const auto& entry : fs::directory_iterator(".")) {
    std::cout << entry.path() << std::endl;
}

// 创建/删除目录
fs::create_directory("newdir");
fs::remove("file.txt");

// 复制/移动文件
fs::copy("src.txt", "dst.txt");
fs::rename("old.txt", "new.txt");
```

## ⚡ 最佳实践

1. **检查文件是否打开** - `is_open()`
2. **使用 RAII** - 作用域结束自动关闭
3. **处理错误** - 检查 `fail()`, `eof()`, `bad()`
4. **使用 filesystem** - C++17 跨平台文件操作
