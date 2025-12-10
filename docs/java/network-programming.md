---
sidebar_position: 11
title: Java 网络编程
---

# Java 网络编程

Java 提供了强大的网络编程 API，支持 TCP、UDP 协议以及 HTTP 通信。本文介绍 Socket 编程、URL 连接和 HTTP Client 的使用。

## TCP Socket 编程

TCP 是面向连接的可靠传输协议，使用 Socket 和 ServerSocket 实现。

### 服务器端

```java
import java.io.*;
import java.net.*;

public class TCPServer {
    public static void main(String[] args) {
        int port = 8888;

        try (ServerSocket serverSocket = new ServerSocket(port)) {
            System.out.println("服务器启动，监听端口: " + port);

            while (true) {
                // 等待客户端连接
                Socket clientSocket = serverSocket.accept();
                System.out.println("客户端已连接: " + clientSocket.getInetAddress());

                // 处理客户端请求（建议使用多线程）
                handleClient(clientSocket);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void handleClient(Socket socket) {
        try (
            BufferedReader in = new BufferedReader(
                new InputStreamReader(socket.getInputStream())
            );
            PrintWriter out = new PrintWriter(
                socket.getOutputStream(), true
            )
        ) {
            String message;
            while ((message = in.readLine()) != null) {
                System.out.println("收到消息: " + message);

                // 回复客户端
                out.println("服务器收到: " + message);

                if ("bye".equalsIgnoreCase(message)) {
                    break;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                socket.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
```

### 客户端

```java
import java.io.*;
import java.net.*;

public class TCPClient {
    public static void main(String[] args) {
        String host = "localhost";
        int port = 8888;

        try (
            Socket socket = new Socket(host, port);
            PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
            BufferedReader in = new BufferedReader(
                new InputStreamReader(socket.getInputStream())
            );
            BufferedReader console = new BufferedReader(
                new InputStreamReader(System.in)
            )
        ) {
            System.out.println("已连接到服务器: " + host + ":" + port);

            String userInput;
            while ((userInput = console.readLine()) != null) {
                // 发送消息
                out.println(userInput);

                // 接收响应
                String response = in.readLine();
                System.out.println("服务器响应: " + response);

                if ("bye".equalsIgnoreCase(userInput)) {
                    break;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

### 多线程服务器

```java
import java.io.*;
import java.net.*;
import java.util.concurrent.*;

public class MultiThreadedServer {
    private static final int PORT = 8888;
    private static final ExecutorService threadPool =
        Executors.newFixedThreadPool(10);

    public static void main(String[] args) {
        try (ServerSocket serverSocket = new ServerSocket(PORT)) {
            System.out.println("多线程服务器启动，端口: " + PORT);

            while (true) {
                Socket clientSocket = serverSocket.accept();
                System.out.println("新客户端连接: " +
                    clientSocket.getInetAddress());

                // 提交到线程池处理
                threadPool.submit(new ClientHandler(clientSocket));
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            threadPool.shutdown();
        }
    }

    static class ClientHandler implements Runnable {
        private final Socket socket;

        public ClientHandler(Socket socket) {
            this.socket = socket;
        }

        @Override
        public void run() {
            try (
                BufferedReader in = new BufferedReader(
                    new InputStreamReader(socket.getInputStream())
                );
                PrintWriter out = new PrintWriter(
                    socket.getOutputStream(), true
                )
            ) {
                String message;
                while ((message = in.readLine()) != null) {
                    System.out.println(Thread.currentThread().getName() +
                        " 收到: " + message);
                    out.println("Echo: " + message);

                    if ("bye".equalsIgnoreCase(message)) {
                        break;
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                try {
                    socket.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
```

## UDP Socket 编程

UDP 是无连接的不可靠传输协议，使用 DatagramSocket 和 DatagramPacket。

### UDP 服务器

```java
import java.net.*;

public class UDPServer {
    public static void main(String[] args) {
        int port = 9999;

        try (DatagramSocket socket = new DatagramSocket(port)) {
            System.out.println("UDP 服务器启动，端口: " + port);

            byte[] buffer = new byte[1024];

            while (true) {
                // 接收数据
                DatagramPacket receivePacket =
                    new DatagramPacket(buffer, buffer.length);
                socket.receive(receivePacket);

                String message = new String(
                    receivePacket.getData(),
                    0,
                    receivePacket.getLength()
                );
                System.out.println("收到消息: " + message);
                System.out.println("来自: " +
                    receivePacket.getAddress() + ":" +
                    receivePacket.getPort());

                // 发送响应
                String response = "服务器收到: " + message;
                byte[] responseData = response.getBytes();
                DatagramPacket sendPacket = new DatagramPacket(
                    responseData,
                    responseData.length,
                    receivePacket.getAddress(),
                    receivePacket.getPort()
                );
                socket.send(sendPacket);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

### UDP 客户端

```java
import java.net.*;
import java.util.Scanner;

public class UDPClient {
    public static void main(String[] args) {
        String host = "localhost";
        int port = 9999;

        try (
            DatagramSocket socket = new DatagramSocket();
            Scanner scanner = new Scanner(System.in)
        ) {
            InetAddress address = InetAddress.getByName(host);

            while (true) {
                System.out.print("请输入消息: ");
                String message = scanner.nextLine();

                // 发送数据
                byte[] sendData = message.getBytes();
                DatagramPacket sendPacket = new DatagramPacket(
                    sendData,
                    sendData.length,
                    address,
                    port
                );
                socket.send(sendPacket);

                // 接收响应
                byte[] receiveData = new byte[1024];
                DatagramPacket receivePacket =
                    new DatagramPacket(receiveData, receiveData.length);
                socket.receive(receivePacket);

                String response = new String(
                    receivePacket.getData(),
                    0,
                    receivePacket.getLength()
                );
                System.out.println("服务器响应: " + response);

                if ("bye".equalsIgnoreCase(message)) {
                    break;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

## URL 和 URLConnection

### URL 基本操作

```java
import java.net.*;
import java.io.*;

public class URLExample {
    public static void main(String[] args) {
        try {
            URL url = new URL("https://www.example.com:80/path?query=value#fragment");

            // 获取 URL 各部分
            System.out.println("协议: " + url.getProtocol());     // https
            System.out.println("主机: " + url.getHost());         // www.example.com
            System.out.println("端口: " + url.getPort());         // 80
            System.out.println("路径: " + url.getPath());         // /path
            System.out.println("查询: " + url.getQuery());        // query=value
            System.out.println("锚点: " + url.getRef());          // fragment
            System.out.println("文件: " + url.getFile());         // /path?query=value

            // 读取 URL 内容
            readURLContent(url);
        } catch (MalformedURLException e) {
            System.err.println("URL 格式错误");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void readURLContent(URL url) throws IOException {
        try (
            BufferedReader reader = new BufferedReader(
                new InputStreamReader(url.openStream())
            )
        ) {
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }
        }
    }
}
```

### URLConnection

```java
import java.net.*;
import java.io.*;

public class URLConnectionExample {
    public static void main(String[] args) {
        try {
            URL url = new URL("https://api.example.com/data");
            URLConnection connection = url.openConnection();

            // 设置请求属性
            connection.setRequestProperty("User-Agent", "Java Client");
            connection.setRequestProperty("Accept", "application/json");
            connection.setConnectTimeout(5000);  // 连接超时 5 秒
            connection.setReadTimeout(5000);     // 读取超时 5 秒

            // 获取响应头
            System.out.println("Content-Type: " +
                connection.getContentType());
            System.out.println("Content-Length: " +
                connection.getContentLength());
            System.out.println("Last-Modified: " +
                connection.getLastModified());

            // 读取内容
            try (
                BufferedReader reader = new BufferedReader(
                    new InputStreamReader(connection.getInputStream())
                )
            ) {
                String line;
                while ((line = reader.readLine()) != null) {
                    System.out.println(line);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

## HTTP Client（JDK 11+）

JDK 11 引入了新的 HTTP Client API，提供了更现代和强大的 HTTP 通信功能。

### GET 请求

```java
import java.net.URI;
import java.net.http.*;
import java.time.Duration;

public class HttpClientGetExample {
    public static void main(String[] args) {
        try {
            // 创建 HttpClient
            HttpClient client = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_2)
                .connectTimeout(Duration.ofSeconds(10))
                .build();

            // 创建请求
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://api.github.com/users/octocat"))
                .header("Accept", "application/json")
                .GET()
                .build();

            // 发送请求（同步）
            HttpResponse<String> response = client.send(
                request,
                HttpResponse.BodyHandlers.ofString()
            );

            // 处理响应
            System.out.println("状态码: " + response.statusCode());
            System.out.println("响应头: " + response.headers());
            System.out.println("响应体: " + response.body());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

### POST 请求

```java
import java.net.URI;
import java.net.http.*;

public class HttpClientPostExample {
    public static void main(String[] args) {
        try {
            HttpClient client = HttpClient.newHttpClient();

            // JSON 数据
            String json = """
                {
                    "name": "张三",
                    "email": "zhangsan@example.com"
                }
                """;

            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://api.example.com/users"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json))
                .build();

            HttpResponse<String> response = client.send(
                request,
                HttpResponse.BodyHandlers.ofString()
            );

            System.out.println("状态码: " + response.statusCode());
            System.out.println("响应: " + response.body());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

### 异步请求

```java
import java.net.URI;
import java.net.http.*;
import java.util.concurrent.CompletableFuture;

public class HttpClientAsyncExample {
    public static void main(String[] args) {
        HttpClient client = HttpClient.newHttpClient();

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create("https://api.github.com/users/octocat"))
            .build();

        // 异步发送请求
        CompletableFuture<HttpResponse<String>> futureResponse =
            client.sendAsync(request, HttpResponse.BodyHandlers.ofString());

        // 处理响应
        futureResponse
            .thenApply(HttpResponse::body)
            .thenAccept(System.out::println)
            .join();  // 等待完成

        System.out.println("请求已发送，等待响应...");
    }
}
```

### 下载文件

```java
import java.net.URI;
import java.net.http.*;
import java.nio.file.*;

public class FileDownloadExample {
    public static void main(String[] args) {
        try {
            HttpClient client = HttpClient.newHttpClient();

            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://example.com/file.zip"))
                .build();

            // 下载到文件
            Path downloadPath = Paths.get("downloaded-file.zip");
            HttpResponse<Path> response = client.send(
                request,
                HttpResponse.BodyHandlers.ofFile(downloadPath)
            );

            System.out.println("文件已下载到: " + response.body());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

## InetAddress

InetAddress 类用于表示 IP 地址。

```java
import java.net.*;

public class InetAddressExample {
    public static void main(String[] args) {
        try {
            // 获取本机地址
            InetAddress localhost = InetAddress.getLocalHost();
            System.out.println("本机名称: " + localhost.getHostName());
            System.out.println("本机IP: " + localhost.getHostAddress());

            // 根据主机名获取地址
            InetAddress google = InetAddress.getByName("www.google.com");
            System.out.println("Google IP: " + google.getHostAddress());

            // 获取所有IP地址
            InetAddress[] allAddresses =
                InetAddress.getAllByName("www.google.com");
            for (InetAddress addr : allAddresses) {
                System.out.println("IP: " + addr.getHostAddress());
            }

            // 测试可达性
            boolean reachable = google.isReachable(5000);
            System.out.println("是否可达: " + reachable);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

## 实战示例：聊天室

### 服务器端

```java
import java.io.*;
import java.net.*;
import java.util.*;
import java.util.concurrent.*;

public class ChatServer {
    private static final int PORT = 8888;
    private static Set<PrintWriter> clientWriters =
        ConcurrentHashMap.newKeySet();

    public static void main(String[] args) {
        System.out.println("聊天服务器启动...");

        ExecutorService pool = Executors.newCachedThreadPool();

        try (ServerSocket serverSocket = new ServerSocket(PORT)) {
            while (true) {
                Socket socket = serverSocket.accept();
                pool.execute(new ClientHandler(socket));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static class ClientHandler implements Runnable {
        private Socket socket;
        private PrintWriter out;
        private String userName;

        public ClientHandler(Socket socket) {
            this.socket = socket;
        }

        @Override
        public void run() {
            try (
                BufferedReader in = new BufferedReader(
                    new InputStreamReader(socket.getInputStream())
                )
            ) {
                out = new PrintWriter(socket.getOutputStream(), true);

                // 获取用户名
                out.println("请输入你的用户名:");
                userName = in.readLine();

                synchronized (clientWriters) {
                    clientWriters.add(out);
                }

                broadcast(userName + " 加入了聊天室");

                // 处理消息
                String message;
                while ((message = in.readLine()) != null) {
                    broadcast(userName + ": " + message);
                }
            } catch (IOException e) {
                System.out.println("连接错误: " + e.getMessage());
            } finally {
                if (userName != null) {
                    broadcast(userName + " 离开了聊天室");
                }
                if (out != null) {
                    clientWriters.remove(out);
                }
                try {
                    socket.close();
                } catch (IOException e) {
                }
            }
        }

        private void broadcast(String message) {
            System.out.println(message);
            synchronized (clientWriters) {
                for (PrintWriter writer : clientWriters) {
                    writer.println(message);
                }
            }
        }
    }
}
```

### 客户端

```java
import java.io.*;
import java.net.*;

public class ChatClient {
    private static final String HOST = "localhost";
    private static final int PORT = 8888;

    public static void main(String[] args) {
        try (
            Socket socket = new Socket(HOST, PORT);
            BufferedReader in = new BufferedReader(
                new InputStreamReader(socket.getInputStream())
            );
            PrintWriter out = new PrintWriter(
                socket.getOutputStream(), true
            );
            BufferedReader console = new BufferedReader(
                new InputStreamReader(System.in)
            )
        ) {
            // 启动接收线程
            Thread receiveThread = new Thread(() -> {
                try {
                    String message;
                    while ((message = in.readLine()) != null) {
                        System.out.println(message);
                    }
                } catch (IOException e) {
                }
            });
            receiveThread.start();

            // 发送消息
            String userInput;
            while ((userInput = console.readLine()) != null) {
                out.println(userInput);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

## 最佳实践

### 1. 资源管理

```java
// ✅ 使用 try-with-resources 自动关闭资源
try (
    Socket socket = new Socket(host, port);
    InputStream in = socket.getInputStream();
    OutputStream out = socket.getOutputStream()
) {
    // 使用 socket
} catch (IOException e) {
    e.printStackTrace();
}
```

### 2. 超时设置

```java
// ✅ 设置合适的超时时间
Socket socket = new Socket();
socket.connect(new InetSocketAddress(host, port), 5000);  // 连接超时
socket.setSoTimeout(10000);  // 读取超时
```

### 3. 线程池处理并发

```java
// ✅ 使用线程池而不是为每个连接创建新线程
ExecutorService threadPool = Executors.newFixedThreadPool(10);
threadPool.submit(new ClientHandler(socket));
```

### 4. 异常处理

```java
// ✅ 处理网络异常
try {
    // 网络操作
} catch (SocketTimeoutException e) {
    System.err.println("连接超时");
} catch (UnknownHostException e) {
    System.err.println("未知主机");
} catch (IOException e) {
    System.err.println("IO 错误: " + e.getMessage());
}
```

## 总结

- **TCP Socket**：面向连接的可靠传输，适用于需要可靠性的场景
- **UDP Socket**：无连接的快速传输，适用于实时性要求高的场景
- **URL/URLConnection**：访问 Web 资源的传统方式
- **HTTP Client**：JDK 11+ 提供的现代 HTTP API，支持同步/异步请求
- **多线程服务器**：使用线程池处理并发连接
- **资源管理**：使用 try-with-resources 确保资源正确释放

掌握 Java 网络编程能够开发各种网络应用，如聊天室、文件传输、HTTP 服务等。
