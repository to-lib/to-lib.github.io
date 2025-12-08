---
sidebar_position: 9
---

# 快速参考

## 核心 API 速查表

### 服务器启动模板

```java
EventLoopGroup bossGroup = new NioEventLoopGroup(1);
EventLoopGroup workerGroup = new NioEventLoopGroup();

try {
    ServerBootstrap bootstrap = new ServerBootstrap();
    bootstrap.group(bossGroup, workerGroup)
        .channel(NioServerSocketChannel.class)
        .option(ChannelOption.SO_BACKLOG, 128)
        .childOption(ChannelOption.SO_KEEPALIVE, true)
        .childHandler(new ChannelInitializer<SocketChannel>() {
            @Override
            protected void initChannel(SocketChannel ch) {
                ch.pipeline()
                    .addLast(new StringDecoder(CharsetUtil.UTF_8))
                    .addLast(new StringEncoder(CharsetUtil.UTF_8))
                    .addLast(new MyHandler());
            }
        });

    ChannelFuture future = bootstrap.bind(port).sync();
    future.channel().closeFuture().sync();
} finally {
    bossGroup.shutdownGracefully();
    workerGroup.shutdownGracefully();
}
```

### 客户端连接模板

```java
EventLoopGroup group = new NioEventLoopGroup();

try {
    Bootstrap bootstrap = new Bootstrap();
    bootstrap.group(group)
        .channel(NioSocketChannel.class)
        .option(ChannelOption.TCP_NODELAY, true)
        .handler(new ChannelInitializer<SocketChannel>() {
            @Override
            protected void initChannel(SocketChannel ch) {
                ch.pipeline()
                    .addLast(new StringDecoder(CharsetUtil.UTF_8))
                    .addLast(new StringEncoder(CharsetUtil.UTF_8))
                    .addLast(new MyHandler());
            }
        });

    ChannelFuture future = bootstrap.connect(host, port).sync();
    future.channel().closeFuture().sync();
} finally {
    group.shutdownGracefully();
}
```

## Handler 常用方法

### Inbound 事件（读取数据）

| 方法 | 触发时机 |
|------|---------|
| `channelRegistered` | Channel 注册到 EventLoop |
| `channelActive` | Channel 连接建立 |
| `channelRead` | 收到数据 |
| `channelReadComplete` | 一次读完成 |
| `userEventTriggered` | 用户事件 |
| `channelInactive` | Channel 连接关闭 |
| `exceptionCaught` | 异常发生 |

### Outbound 事件（写入数据）

| 方法 | 作用 |
|------|------|
| `bind` | 绑定端口 |
| `connect` | 连接服务器 |
| `write` | 写入数据 |
| `flush` | 刷新缓冲 |
| `close` | 关闭连接 |

## ByteBuf 常用操作

### 创建 ByteBuf

```java
// 堆内存
ByteBuf buf = Unpooled.buffer(1024);

// 直接内存
ByteBuf directBuf = Unpooled.directBuffer(1024);

// 从数据创建
ByteBuf copy = Unpooled.copiedBuffer("Hello", CharsetUtil.UTF_8);
```

### 读写 ByteBuf

```java
// 写入
buf.writeInt(123);
buf.writeByte(45);
buf.writeBytes(new byte[]{1, 2, 3});

// 读取
int value = buf.readInt();
byte b = buf.readByte();

// 获取（不移动指针）
int val = buf.getInt(0);
```

### 释放 ByteBuf

```java
// 方式1：直接释放
buf.release();

// 方式2：安全释放
ReferenceCountUtil.release(msg);

// 方式3：SimpleChannelInboundHandler 自动释放
public class MyHandler extends SimpleChannelInboundHandler<ByteBuf> {
    protected void channelRead0(ChannelHandlerContext ctx, ByteBuf msg) {
        // msg 自动释放
    }
}
```

## 常用编解码器

```java
// 字符串
pipeline.addLast(new StringDecoder(CharsetUtil.UTF_8));
pipeline.addLast(new StringEncoder(CharsetUtil.UTF_8));

// 按行分割
pipeline.addLast(new LineBasedFrameDecoder(1024));

// 长度字段
pipeline.addLast(
    new LengthFieldBasedFrameDecoder(65535, 0, 4, 0, 4)
);

// 空闲检测
pipeline.addLast(new IdleStateHandler(60, 30, 0, TimeUnit.SECONDS));

// 日志
pipeline.addLast(new LoggingHandler(LogLevel.DEBUG));
```

## Channel 常用操作

```java
// 写入数据
ChannelFuture future = ctx.writeAndFlush(msg);

// 获取连接信息
InetSocketAddress remote = (InetSocketAddress) channel.remoteAddress();
InetSocketAddress local = (InetSocketAddress) channel.localAddress();

// 检查状态
channel.isActive();
channel.isOpen();
channel.isWritable();

// 关闭连接
channel.close();

// 属性存储
channel.attr(AttributeKey.valueOf("key")).set(value);
```

## Pipeline 操作

```java
// 添加 Handler
pipeline.addLast("name", new MyHandler());
pipeline.addFirst(new MyHandler());
pipeline.addBefore("name", new MyHandler());
pipeline.addAfter("name", new MyHandler());

// 获取 Handler
MyHandler handler = pipeline.get(MyHandler.class);
ChannelHandler handler2 = pipeline.get("name");

// 移除 Handler
pipeline.remove("name");
pipeline.removeFirst();
pipeline.removeLast();

// 替换 Handler
pipeline.replace("old", "new", new NewHandler());
```

## ChannelOption 常用配置

| 选项 | 说明 | 推荐值 |
|------|------|--------|
| `SO_BACKLOG` | TCP backlog 大小 | 128-2048 |
| `SO_KEEPALIVE` | 保活探针 | true |
| `TCP_NODELAY` | 关闭 Nagle 算法 | true |
| `SO_SNDBUF` | 发送缓冲大小 | 256KB |
| `SO_RCVBUF` | 接收缓冲大小 | 256KB |
| `ALLOCATOR` | ByteBuf 分配器 | PooledByteBufAllocator |
| `WRITE_BUFFER_WATER_MARK` | 写缓冲水位线 | 32KB-1MB |

## 异步操作

```java
// 写入并监听结果
ChannelFuture future = ctx.writeAndFlush(msg);
future.addListener(f -> {
    if (f.isSuccess()) {
        System.out.println("写入成功");
    } else {
        System.out.println("写入失败: " + f.cause());
    }
});

// 异步等待（阻塞）
try {
    future.sync();
} catch (InterruptedException e) {
    e.printStackTrace();
}

// 检查完成状态
if (future.isDone()) {
    if (future.isSuccess()) {
        // 成功
    }
}
```

## 任务调度

```java
// 一次性延迟任务
ctx.executor().schedule(() -> {
    System.out.println("延迟执行");
}, 1, TimeUnit.SECONDS);

// 周期性任务
ctx.executor().scheduleAtFixedRate(() -> {
    System.out.println("定时执行");
}, 1, 1, TimeUnit.SECONDS);

// 异步执行
ctx.executor().execute(() -> {
    System.out.println("异步执行");
});
```

## 异常处理

```java
@Override
public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
    // 记录异常
    // logger.error("异常: ", cause);
    cause.printStackTrace();
    
    // 关闭连接
    ctx.close();
}
```

## 常用工具类

```java
// ByteBuf 工具
ByteBufUtil.hexDump(buf);           // 16 进制输出
ByteBufUtil.appendPrettyHexDump(sb, buf); // 格式化输出

// 引用计数工具
ReferenceCountUtil.release(msg);    // 安全释放
ReferenceCountUtil.retain(msg);     // 增加引用计数

// 字符集
CharsetUtil.UTF_8;
CharsetUtil.US_ASCII;
CharsetUtil.ISO_8859_1;

// 缓冲分配器
PooledByteBufAllocator.DEFAULT;
UnpooledByteBufAllocator.DEFAULT;

// Promise
ChannelPromise promise = ctx.newPromise();
promise.setSuccess();
promise.setFailure(cause);
```

## 性能相关参数

### JVM 参数

```bash
# 直接内存大小
-XX:MaxDirectMemorySize=2G

# 堆大小
-Xms2G -Xmx2G

# GC 优化
-XX:+UseG1GC -XX:MaxGCPauseMillis=50

# 禁用偏向锁（高并发下）
-XX:-UseBiasedLocking
```

### 系统参数（Linux）

```bash
# 增加系统最大文件描述符
ulimit -n 100000

# TCP 参数调优
sysctl -w net.core.somaxconn=65535
sysctl -w net.ipv4.tcp_max_syn_backlog=65535
sysctl -w net.ipv4.tcp_tw_reuse=1
sysctl -w net.ipv4.tcp_fin_timeout=30
```

## 快速诊断

### 检查清单

- [ ] 是否正确释放了 ByteBuf？
- [ ] 是否在 EventLoop 中执行了阻塞操作？
- [ ] 是否正确传播了事件？
- [ ] 是否处理了异常？
- [ ] 是否配置了合理的线程池大小？
- [ ] 是否设置了连接超时？
- [ ] 是否监控了内存使用？
- [ ] 是否有内存泄漏检测？

### 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `LeakDetectedException` | 内存泄漏 | 检查 ByteBuf 释放 |
| `RejectedExecutionException` | 线程池满 | 增加线程池大小 |
| `OutOfMemoryError` | 内存溢出 | 释放资源、增加堆大小 |
| `ClosedChannelException` | Channel 已关闭 | 检查连接状态 |
| `ConnectException` | 连接失败 | 检查服务器地址、防火墙 |

## 学习资源

### 官方资源
- [Netty 官网](https://netty.io/)
- [API 文档](https://netty.io/4.1/api/)
- [GitHub](https://github.com/netty/netty)

### 推荐书籍
- 《Netty in Action》
- 《Netty 权威指南》

### 常用依赖版本

```xml
<!-- 最新稳定版 -->
<dependency>
    <groupId>io.netty</groupId>
    <artifactId>netty-all</artifactId>
    <version>4.1.100.Final</version>
</dependency>

<!-- 长期支持版 -->
<!-- 4.1.x 系列，支持 JDK 8+ -->

<!-- 最新开发版 -->
<!-- 5.0.x 系列（预发布），需要 JDK 11+ -->
```

## Maven POM 配置示例

```xml
<dependency>
    <groupId>io.netty</groupId>
    <artifactId>netty-all</artifactId>
    <version>4.1.100.Final</version>
</dependency>

<!-- 日志 -->
<dependency>
    <groupId>org.slf4j</groupId>
    <artifactId>slf4j-api</artifactId>
    <version>1.7.36</version>
</dependency>

<!-- 日志实现 -->
<dependency>
    <groupId>ch.qos.logback</groupId>
    <artifactId>logback-classic</artifactId>
    <version>1.4.1</version>
</dependency>
```

## 一行代码参考

```java
// 启用内存泄漏检测
System.setProperty("io.netty.leakDetection.level", "PARANOID");

// 关闭 Nagle 算法
bootstrap.childOption(ChannelOption.TCP_NODELAY, true);

// 使用池化分配器
bootstrap.childOption(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT);

// 设置心跳
pipeline.addLast(new IdleStateHandler(60, 30, 0, TimeUnit.SECONDS));

// 限制消息大小
pipeline.addLast(new LengthFieldBasedFrameDecoder(1024*1024, 0, 4, 0, 4));

// 记录日志
pipeline.addLast(new LoggingHandler(LogLevel.DEBUG));

// 安全释放
ReferenceCountUtil.release(msg);
```