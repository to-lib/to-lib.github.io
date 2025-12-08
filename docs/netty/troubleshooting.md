---
sidebar_position: 8
---

# 常见问题与排查

## 常见问题

### Q1: 内存泄漏问题

**问题表现：** 内存持续增长，GC 后不释放

**常见原因：**
1. [ByteBuf](./bytebuf.md) 没有正确释放
2. Handler 中接收的消息没有传播给下一个 Handler

**解决方案：**

```java
// ✗ 错误：消息没有被处理，导致泄漏
@Override
public void channelRead(ChannelHandlerContext ctx, Object msg) {
    System.out.println("接收到消息");
    // 没有释放或继续传播
}

// ✓ 正确：方案1 - 释放消息
@Override
public void channelRead(ChannelHandlerContext ctx, Object msg) {
    try {
        ByteBuf buf = (ByteBuf) msg;
        processData(buf);
    } finally {
        ReferenceCountUtil.release(msg);
    }
}

// ✓ 正确：方案2 - 继续传播给下一个 Handler
@Override
public void channelRead(ChannelHandlerContext ctx, Object msg) {
    // 处理消息
    ctx.fireChannelRead(msg); // 传播给下一个 Handler，由它负责释放
}

// ✓ 正确：方案3 - SimpleChannelInboundHandler 自动释放
public class MyHandler extends SimpleChannelInboundHandler<ByteBuf> {
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, ByteBuf msg) {
        // msg 会在方法结束后自动释放
        processData(msg);
    }
}
```

### Q2: 延迟过高

**问题表现：** 处理消息的延迟很高，不符合预期

**常见原因：**
1. 在 [EventLoop](./core-components.md#eventloop) 线程中执行阻塞操作
2. 事件处理链过长
3. 消息处理效率低

**解决方案：**

```java
// ✗ 错误：阻塞 EventLoop
@Override
public void channelRead(ChannelHandlerContext ctx, Object msg) {
    try {
        Thread.sleep(1000); // 阻塞！
    } catch (InterruptedException e) {
        e.printStackTrace();
    }
    ctx.write(msg);
}

// ✓ 正确：使用独立线程执行阻塞操作
private ExecutorService executor = Executors.newFixedThreadPool(10);

@Override
public void channelRead(ChannelHandlerContext ctx, Object msg) {
    executor.execute(() -> {
        try {
            Thread.sleep(1000); // 在独立线程执行
            ctx.writeAndFlush(msg);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    });
}

// ✓ 更好：直接使用 EventLoop 的 execute
@Override
public void channelRead(ChannelHandlerContext ctx, Object msg) {
    ctx.executor().execute(() -> {
        // 这个操作会在 EventLoop 的任务队列中异步执行
        try {
            Thread.sleep(1000);
            ctx.writeAndFlush(msg);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    });
}
```

### Q3: 连接堆积

**问题表现：** 服务器无法处理新连接，连接数持续增长

**常见原因：**
1. Worker EventLoop 处理速度过慢
2. 消息处理耗时过长
3. 并发能力不足

**解决方案：**

```java
// 增加 EventLoop 数量
int cpuCores = Runtime.getRuntime().availableProcessors();
EventLoopGroup workerGroup = new NioEventLoopGroup(cpuCores * 2);

// 配置合理的缓冲区和水位线
bootstrap.childOption(ChannelOption.SO_BACKLOG, 2048);
bootstrap.childOption(
    ChannelOption.WRITE_BUFFER_WATER_MARK,
    new WriteBufferWaterMark(32 * 1024, 1024 * 1024)
);

// 优化消息处理效率
// 避免在 Handler 中执行复杂操作
// 使用线程池异步处理业务逻辑
```

### Q4: OutOfMemoryError

**问题表现：** 抛出 OutOfMemoryError 异常

**常见原因：**
1. 直接内存溢出
2. 堆内存溢出
3. 大消息处理不当

**解决方案：**

```java
// 监控内存使用
MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
MemoryUsage directUsage = memoryBean.getNonHeapMemoryUsage();

System.out.println("堆内存使用: " + heapUsage.getUsed());
System.out.println("直接内存使用: " + directUsage.getUsed());

// 限制消息大小
pipeline.addLast(
    new LengthFieldBasedFrameDecoder(1024 * 1024, 0, 4, 0, 4) // 最大 1MB
);

// 及时释放大对象
@Override
public void channelRead(ChannelHandlerContext ctx, Object msg) {
    try {
        processBigMessage(msg);
    } finally {
        ReferenceCountUtil.release(msg);
    }
}

// 增加直接内存大小
// java -XX:MaxDirectMemorySize=1G ...
```

### Q5: 客户端连接拒绝

**问题表现：** 客户端连接超时或被拒绝

**常见原因：**
1. 服务器连接数达到上限
2. TCP backlog 队列满
3. 资源不足

**解决方案：**

```java
// 增加 backlog 大小
ServerBootstrap bootstrap = new ServerBootstrap();
bootstrap.option(ChannelOption.SO_BACKLOG, 4096);

// 增加系统调优
// Linux: sysctl -w net.core.somaxconn=65535
//        sysctl -w net.ipv4.tcp_max_syn_backlog=65535

// 在应用层实现连接池和限流
public class ConnectionManager {
    private static final int MAX_CONNECTIONS = 10000;
    private static final AtomicInteger activeConnections = new AtomicInteger();
    
    public static boolean acceptConnection() {
        if (activeConnections.get() >= MAX_CONNECTIONS) {
            return false;
        }
        activeConnections.incrementAndGet();
        return true;
    }
}
```

## 性能问题排查

### 使用 Netty 内置日志

```java
import io.netty.handler.logging.LoggingHandler;
import io.netty.handler.logging.LogLevel;

// 添加到 Pipeline
pipeline.addFirst("logging", new LoggingHandler(LogLevel.DEBUG));

// 查看每个操作的详细信息
// CONNECT, CLOSE, WRITE, READ 等所有操作都会被记录
```

### 使用 JFR（Java Flight Recorder）

```bash
# 启用 JFR 录制
java -XX:+UnlockCommercialFeatures \
     -XX:+FlightRecorder \
     -XX:StartFlightRecording=delay=20s,duration=60s,filename=myapp.jfr \
     -jar myapp.jar

# 分析录制数据
jmc -open myapp.jfr
```

### 使用 JProfiler

```java
// 在代码中标记关键点
@Profiled(name = "messageProcessing")
@Override
public void channelRead(ChannelHandlerContext ctx, Object msg) {
    // 处理消息
}
```

### 使用 async-profiler

```bash
# 记录 CPU 使用情况
./profiler.sh -d 30 -f cpu.html jps

# 生成火焰图
./profiler.sh -d 30 -f flame.html -cg jps
```

## 调试技巧

### 1. 启用详细日志

```java
// 添加 SLF4J 配置
// logback.xml 中添加：
<logger name="io.netty" level="DEBUG"/>

// 或在代码中设置
System.setProperty("io.netty.leakDetection.level", "PARANOID");
```

### 2. 检查内存泄漏

```java
// 启用 Netty 的内存泄漏检测
System.setProperty("io.netty.leakDetection.level", "PARANOID");

// 可选值：DISABLED, SIMPLE, ADVANCED, PARANOID
// PARANOID 模式检测最严格，会记录所有 ByteBuf 的分配堆栈
```

### 3. Handler 执行顺序追踪

```java
public class TraceHandler extends ChannelDuplexHandler {
    private String name;

    public TraceHandler(String name) {
        this.name = name;
    }

    @Override
    public void channelRead(ChannelHandlerContext ctx, Object msg) {
        System.out.println("[Inbound] " + name);
        ctx.fireChannelRead(msg);
    }

    @Override
    public void write(ChannelHandlerContext ctx, Object msg, 
                      ChannelPromise promise) {
        System.out.println("[Outbound] " + name);
        ctx.write(msg, promise);
    }
}

// 在 Pipeline 中添加追踪 Handler
pipeline.addLast(new TraceHandler("Handler1"));
pipeline.addLast(new TraceHandler("Handler2"));
pipeline.addLast(new TraceHandler("Handler3"));
```

### 4. 异步操作追踪

```java
public class AsyncTraceHandler extends ChannelInboundHandlerAdapter {
    @Override
    public void channelRead(ChannelHandlerContext ctx, Object msg) {
        long startTime = System.currentTimeMillis();
        String threadName = Thread.currentThread().getName();
        
        ctx.executor().execute(() -> {
            long elapsed = System.currentTimeMillis() - startTime;
            String asyncThreadName = Thread.currentThread().getName();
            
            System.out.println(String.format(
                "异步执行延迟: %dms, 原线程: %s, 当前线程: %s",
                elapsed, threadName, asyncThreadName
            ));
        });

        ctx.fireChannelRead(msg);
    }
}
```

## 常见错误代码

### ✗ 错误1：内存泄漏

```java
@Override
public void channelRead(ChannelHandlerContext ctx, Object msg) {
    ByteBuf buf = (ByteBuf) msg;
    String data = buf.toString(CharsetUtil.UTF_8);
    System.out.println(data);
    // 没有释放！
}
```

**修复：**
```java
@Override
public void channelRead(ChannelHandlerContext ctx, Object msg) {
    ByteBuf buf = (ByteBuf) msg;
    try {
        String data = buf.toString(CharsetUtil.UTF_8);
        System.out.println(data);
    } finally {
        ReferenceCountUtil.release(msg);
    }
}
```

### ✗ 错误2：阻塞 EventLoop

```java
@Override
public void channelRead(ChannelHandlerContext ctx, Object msg) {
    try {
        Thread.sleep(5000); // 大错特错！
    } catch (InterruptedException e) {
        e.printStackTrace();
    }
    ctx.writeAndFlush(msg);
}
```

**修复：**
```java
private ExecutorService executor = Executors.newFixedThreadPool(10);

@Override
public void channelRead(ChannelHandlerContext ctx, Object msg) {
    executor.execute(() -> {
        try {
            Thread.sleep(5000);
            ctx.writeAndFlush(msg);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    });
}
```

### ✗ 错误3：未传播事件

```java
@Override
public void channelRead(ChannelHandlerContext ctx, Object msg) {
    // 处理完就完了，没有传播给下一个 Handler
    ByteBuf buf = (ByteBuf) msg;
    System.out.println(buf.toString(CharsetUtil.UTF_8));
}
```

**修复：**
```java
@Override
public void channelRead(ChannelHandlerContext ctx, Object msg) {
    ByteBuf buf = (ByteBuf) msg;
    System.out.println(buf.toString(CharsetUtil.UTF_8));
    ctx.fireChannelRead(msg); // 继续传播
}

// 或者使用 SimpleChannelInboundHandler 自动处理
public class MyHandler extends SimpleChannelInboundHandler<ByteBuf> {
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, ByteBuf msg) {
        System.out.println(msg.toString(CharsetUtil.UTF_8));
        // 自动释放，无需手动释放
    }
}
```

## 监控和告警

### 建议的监控指标

```
1. 连接相关
   - 活跃连接数
   - 总连接数
   - 连接建立速率
   - 连接断开速率

2. 流量相关
   - 入站字节数/秒
   - 出站字节数/秒
   - 消息处理速率

3. 内存相关
   - 堆内存使用率
   - 直接内存使用量
   - GC 频率和耗时

4. 延迟相关
   - 消息处理P95/P99延迟
   - EventLoop 任务队列长度

5. 异常相关
   - 异常发生次数
   - 异常类型分布
```

### 简单的监控实现

```java
public class MonitoringMetrics {
    private AtomicLong totalConnections = new AtomicLong();
    private AtomicLong activeConnections = new AtomicLong();
    private AtomicLong totalBytesRead = new AtomicLong();
    private AtomicLong totalBytesWritten = new AtomicLong();
    private AtomicLong totalMessagesProcessed = new AtomicLong();

    public void recordConnection() {
        totalConnections.incrementAndGet();
        activeConnections.incrementAndGet();
    }

    public void recordDisconnection() {
        activeConnections.decrementAndGet();
    }

    public void recordBytesRead(long bytes) {
        totalBytesRead.addAndGet(bytes);
    }

    public void recordMessageProcessed() {
        totalMessagesProcessed.incrementAndGet();
    }

    public void printMetrics() {
        System.out.println("=== Metrics ===");
        System.out.println("Active Connections: " + activeConnections.get());
        System.out.println("Total Connections: " + totalConnections.get());
        System.out.println("Total Bytes Read: " + totalBytesRead.get());
        System.out.println("Total Messages: " + totalMessagesProcessed.get());
    }
}
```

---
[下一章：快速参考](./quick-reference.md)