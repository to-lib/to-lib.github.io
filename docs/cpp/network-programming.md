---
sidebar_position: 28
title: 网络编程
---

# C++ 网络编程

C++ 网络编程基础，使用 POSIX Socket 和现代库。

## 🎯 TCP 客户端

```cpp
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

int main() {
    // 创建 socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);

    // 服务器地址
    sockaddr_in server{};
    server.sin_family = AF_INET;
    server.sin_port = htons(8080);
    inet_pton(AF_INET, "127.0.0.1", &server.sin_addr);

    // 连接
    connect(sock, (sockaddr*)&server, sizeof(server));

    // 发送数据
    const char* msg = "Hello Server";
    send(sock, msg, strlen(msg), 0);

    // 接收数据
    char buffer[1024] = {0};
    recv(sock, buffer, sizeof(buffer), 0);
    std::cout << "Response: " << buffer << std::endl;

    close(sock);
    return 0;
}
```

## 🖥️ TCP 服务器

```cpp
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

int main() {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(8080);

    bind(server_fd, (sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 3);

    std::cout << "Listening on port 8080..." << std::endl;

    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    int client = accept(server_fd, (sockaddr*)&client_addr, &client_len);

    char buffer[1024] = {0};
    recv(client, buffer, sizeof(buffer), 0);
    std::cout << "Received: " << buffer << std::endl;

    send(client, "Hello Client", 12, 0);

    close(client);
    close(server_fd);
    return 0;
}
```

## 📡 UDP 通信

```cpp
// UDP 发送
int sock = socket(AF_INET, SOCK_DGRAM, 0);
sockaddr_in dest{};
dest.sin_family = AF_INET;
dest.sin_port = htons(8080);
inet_pton(AF_INET, "127.0.0.1", &dest.sin_addr);

sendto(sock, "Hello", 5, 0, (sockaddr*)&dest, sizeof(dest));

// UDP 接收
char buffer[1024];
sockaddr_in from{};
socklen_t fromLen = sizeof(from);
recvfrom(sock, buffer, sizeof(buffer), 0, (sockaddr*)&from, &fromLen);
```

## 🔧 现代 C++ 网络库

推荐使用第三方库简化开发：

- **Boost.Asio** - 异步 I/O 库
- **libcurl** - HTTP 客户端
- **Poco** - 网络框架
- **cpp-httplib** - 轻量 HTTP 库

```cpp
// cpp-httplib 示例
#include "httplib.h"

int main() {
    httplib::Server svr;

    svr.Get("/", [](const auto& req, auto& res) {
        res.set_content("Hello World!", "text/plain");
    });

    svr.listen("0.0.0.0", 8080);
}
```

## ⚡ 最佳实践

1. **使用 RAII** - 封装 socket 资源
2. **处理错误** - 检查返回值
3. **非阻塞/异步** - 高并发场景
4. **使用现代库** - 简化开发
