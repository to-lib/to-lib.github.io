---
sidebar_position: 7
---

# 性能优化与进阶

> [!IMPORTANT]
> 性能优化是一个系统工程,需要从内存、连接、吞吐量、线程等多个维度综合考虑。在优化前务必先做性能测试,建立基准,然后针对性优化。

## 内存优化

### ByteBuf 内存池

```java
// 使用池化分配器（默认）
ChannelConfig config = channel.config();
ByteBufAllocator allocator = config.getAllocator();

// 确认使用了 PooledByteBufAllocator
if (allocator instanceof PooledByteBufAllocator) {
    System.out.println("使用池化分配器");
}

// 手动配置
ServerBootstrap bootstrap = new ServerBootstrap();
bootstrap.childOption(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT);

// 池化分配器参数调优
System.setProperty("io.netty.allocator.numHeapArenas", "16");
System.setProperty("io.netty.allocator.numDirectArenas", "16");
System.setProperty("io.netty.allocator.pageSize", "8192");
System.setProperty("io.netty.allocator.maxOrder", "11");
```

### 使用 ObjectPool 重用对象

```java
import io.netty.util.Recycler;

public class MyMessage {
    private static final Recycler<MyMessage> RECYCLER = 
        new Recycler<MyMessage>() {
            @Override
            protected MyMessage newObject(Handle<MyMessage> handle) {
                return new MyMessage(handle);
            }
        };

    private final Recycler.Handle<MyMessage> handle;
    private byte type;
    private String data;

    private MyMessage(Recycler.Handle<MyMessage> handle) {
        this.handle = handle;
    }

    public static MyMessage newInstance() {
        return RECYCLER.get();
    }

    public void set(byte type, String data) {
        this.type = type;
        this.data = data;
    }

    public void recycle() {
        this.type = 0;
        this.data = null;
        handle.recycle(this);
    }

    // getters...
}

// 使用
MyMessage msg = MyMessage.newInstance();
msg.set((byte) 1, "hello");
try {
    // 使用 msg
} finally {
    msg.recycle(); // 回收对象
}
```

## 连接管理

### 心跳检测

```java
public class HeartbeatHandler extends ChannelInboundHandlerAdapter {
    private static final long HEARTBEAT_INTERVAL = 30; // 秒

    @Override
    public void userEventTriggered(ChannelHandlerContext ctx, Object evt) 
            throws Exception {
        if (evt instanceof IdleStateEvent) {
            IdleStateEvent event = (IdleStateEvent) evt;
            
            if (event.state() == IdleState.WRITER_IDLE) {
                // 30 秒没有写，发送心跳
                ctx.writeAndFlush(new HeartbeatMessage());
            } else if (event.state() == IdleState.READER_IDLE) {
                // 30 秒没有读，断开连接
                System.out.println("连接超时，断开连接");
                ctx.close();
            }
        }
        super.userEventTriggered(ctx, evt);
    }
}

// 在 Pipeline 中添加
pipeline.addLast(
    new IdleStateHandler(30, 30, 0, TimeUnit.SECONDS)
);
pipeline.addLast(new HeartbeatHandler());
```

### 连接限制

```java
public class ConnectionLimitHandler extends ChannelInboundHandlerAdapter {
    private static final int MAX_CONNECTIONS = 10000;
    private static final AtomicInteger activeConnections = new AtomicInteger(0);

    @Override
    public void channelActive(ChannelHandlerContext ctx) throws Exception {
        int count = activeConnections.incrementAndGet();
        
        if (count > MAX_CONNECTIONS) {
            System.out.println("连接数达到上限，拒绝新连接");
            ctx.close();
            activeConnections.decrementAndGet();
        } else {
            System.out.println("当前连接数: " + count);
            ctx.fireChannelActive();
        }
    }

    @Override
    public void channelInactive(ChannelHandlerContext ctx) throws Exception {
        activeConnections.decrementAndGet();
        ctx.fireChannelInactive();
    }
}
```

### 连接空闲检测

```java
// 在 Pipeline 中添加
ChannelInitializer<SocketChannel> initializer = 
    new ChannelInitializer<SocketChannel>() {
    @Override
    protected void initChannel(SocketChannel ch) {
        ChannelPipeline pipeline = ch.pipeline();
        
        // IdleStateHandler 会触发 IdleStateEvent
        // 参数：读空闲超时、写空闲超时、读写都空闲超时（秒）
        pipeline.addLast(new IdleStateHandler(60, 30, 0));
        pipeline.addLast(new ConnectionIdleHandler());
    }
};

public class ConnectionIdleHandler extends ChannelInboundHandlerAdapter {
    @Override
    public void userEventTriggered(ChannelHandlerContext ctx, Object evt) 
            throws Exception {
        if (evt instanceof IdleStateEvent) {
            IdleStateEvent event = (IdleStateEvent) evt;
            
            switch (event.state()) {
                case READER_IDLE:
                    System.out.println("读空闲");
                    break;
                case WRITER_IDLE:
                    System.out.println("写空闲");
                    break;
                case ALL_IDLE:
                    System.out.println("读写空闲");
                    break;
            }
        }
        super.userEventTriggered(ctx, evt);
    }
}
```

## 吞吐量优化

### 写缓冲区优化

```java
// 配置写缓冲区大小
ServerBootstrap bootstrap = new ServerBootstrap();
bootstrap.childOption(
    ChannelOption.SO_SNDBUF, 
    256 * 1024  // 256KB 发送缓冲
);

bootstrap.childOption(
    ChannelOption.SO_RCVBUF,
    256 * 1024  // 256KB 接收缓冲
);

// 配置写水位线
bootstrap.childOption(
    ChannelOption.WRITE_BUFFER_WATER_MARK,
    new WriteBufferWaterMark(32 * 1024, 1024 * 1024) // 低位 32KB，高位 1MB
);
```

### 批量写入

```java
public class BatchWriteHandler extends ChannelInboundHandlerAdapter {
    private static final int BATCH_SIZE = 100;
    private Queue<Object> queue = new LinkedList<>();

    @Override
    public void channelRead(ChannelHandlerContext ctx, Object msg) {
        queue.offer(msg);
        
        if (queue.size() >= BATCH_SIZE) {
            flushBatch(ctx);
        }
    }

    private void flushBatch(ChannelHandlerContext ctx) {
        while (!queue.isEmpty()) {
            ctx.write(queue.poll());
        }
        ctx.flush();
    }

    @Override
    public void channelReadComplete(ChannelHandlerContext ctx) {
        flushBatch(ctx);
    }
}
```

### 关闭 Nagle 算法

```java
// Nagle 算法会延迟小包发送，关闭它可以降低延迟
ServerBootstrap bootstrap = new ServerBootstrap();
bootstrap.childOption(ChannelOption.TCP_NODELAY, true);
```

## 线程优化

### 线程池大小配置

```java
// 根据 CPU 核心数配置 EventLoop 数量
int cpuCores = Runtime.getRuntime().availableProcessors();

// Boss EventLoop：通常只需 1 个
EventLoopGroup bossGroup = new NioEventLoopGroup(1);

// Worker EventLoop：通常是 CPU 核心数的 1-2 倍
EventLoopGroup workerGroup = new NioEventLoopGroup(cpuCores * 2);

// 对于存在大量数据库操作的应用，可以增加更多线程
// EventLoopGroup workerGroup = new NioEventLoopGroup(cpuCores * 4);
```

### 任务执行器优化

```java
// 对于耗时操作，使用独立的业务线程池
ExecutorService businessPool = new ThreadPoolExecutor(
    cpuCores,
    cpuCores * 2,
    60,
    TimeUnit.SECONDS,
    new LinkedBlockingQueue<>(1000),
    new ThreadFactory() {
        private AtomicInteger counter = new AtomicInteger();
        
        @Override
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r);
            t.setName("business-" + counter.incrementAndGet());
            return t;
        }
    },
    new ThreadPoolExecutor.CallerRunsPolicy()
);

// 在 Handler 中使用
public class BusinessHandler extends SimpleChannelInboundHandler<MyMessage> {
    private ExecutorService businessPool;

    @Override
    protected void channelRead0(ChannelHandlerContext ctx, MyMessage msg) {
        // 提交到业务线程池
        businessPool.execute(() -> {
            try {
                Object result = handleMessage(msg);
                ctx.writeAndFlush(result);
            } catch (Exception e) {
                ctx.fireExceptionCaught(e);
            }
        });
    }

    private Object handleMessage(MyMessage msg) {
        // 耗时操作
        return null;
    }
}
```

## 监控与调试

### Netty 监控

```java
// 启用调试日志
import io.netty.handler.logging.LogLevel;
import io.netty.handler.logging.LoggingHandler;

ServerBootstrap bootstrap = new ServerBootstrap();
bootstrap.handler(new LoggingHandler(LogLevel.DEBUG));
bootstrap.childHandler(new ChannelInitializer<SocketChannel>() {
    @Override
    protected void initChannel(SocketChannel ch) {
        ch.pipeline().addFirst("logging", 
            new LoggingHandler(LogLevel.DEBUG));
    }
});
```

### 连接统计

```java
public class ConnectionStatsHandler extends ChannelDuplexHandler {
    private static final AtomicLong totalConnections = new AtomicLong();
    private static final AtomicLong activeConnections = new AtomicLong();
    private static final AtomicLong totalBytesRead = new AtomicLong();
    private static final AtomicLong totalBytesWritten = new AtomicLong();

    @Override
    public void channelActive(ChannelHandlerContext ctx) throws Exception {
        totalConnections.incrementAndGet();
        activeConnections.incrementAndGet();
        ctx.fireChannelActive();
    }

    @Override
    public void channelInactive(ChannelHandlerContext ctx) throws Exception {
        activeConnections.decrementAndGet();
        ctx.fireChannelInactive();
    }

    @Override
    public void channelRead(ChannelHandlerContext ctx, Object msg) throws Exception {
        if (msg instanceof ByteBuf) {
            ByteBuf buf = (ByteBuf) msg;
            totalBytesRead.addAndGet(buf.readableBytes());
        }
        ctx.fireChannelRead(msg);
    }

    @Override
    public void write(ChannelHandlerContext ctx, Object msg, 
                      ChannelPromise promise) throws Exception {
        if (msg instanceof ByteBuf) {
            ByteBuf buf = (ByteBuf) msg;
            totalBytesWritten.addAndGet(buf.readableBytes());
        }
        ctx.write(msg, promise);
    }

    public static void printStats() {
        System.out.println("=== Netty 连接统计 ===");
        System.out.println("总连接数: " + totalConnections.get());
        System.out.println("活跃连接数: " + activeConnections.get());
        System.out.println("总接收字节数: " + totalBytesRead.get());
        System.out.println("总发送字节数: " + totalBytesWritten.get());
    }
}
```

### 定期输出统计信息

```java
// 在 EventLoop 中定期输出统计
public class StatsServerHandler extends ChannelInboundHandlerAdapter {
    @Override
    public void channelActive(ChannelHandlerContext ctx) throws Exception {
        // 每 30 秒输出一次统计信息
        ctx.executor().scheduleAtFixedRate(
            ConnectionStatsHandler::printStats,
            30,
            30,
            TimeUnit.SECONDS
        );
        ctx.fireChannelActive();
    }
}
```

## 最佳实践检查清单

### 内存

- [ ] 使用 [PooledByteBufAllocator](./bytebuf.md#pooledbytebufallocator推荐)
- [ ] 及时释放 ByteBuf
- [ ] 重用对象避免频繁创建
- [ ] 监控内存使用情况

### 连接

- [ ] 配置心跳检测
- [ ] 设置连接超时
- [ ] 限制最大连接数
- [ ] 优雅关闭连接

### 性能

- [ ] 关闭 Nagle 算法（TCP_NODELAY）
- [ ] 配置合理的缓冲区大小
- [ ] 避免阻塞 EventLoop
- [ ] 使用独立线程池处理业务逻辑

### 监控

- [ ] 记录关键业务指标
- [ ] 监控连接数和流量
- [ ] 定期检查内存使用
- [ ] 设置告警阈值

### 安全

- [ ] 验证请求数据
- [ ] 限制消息大小
- [ ] 实现 DDoS 防护
- [ ] 使用 SSL/TLS 加密

## 性能测试

```java
public class PerformanceTest {
    public static void main(String[] args) throws InterruptedException {
        int threads = 100;
        int messagesPerThread = 1000;
        int totalMessages = threads * messagesPerThread;

        CountDownLatch latch = new CountDownLatch(threads);
        long startTime = System.currentTimeMillis();

        for (int i = 0; i < threads; i++) {
            new Thread(() -> {
                try {
                    for (int j = 0; j < messagesPerThread; j++) {
                        // 发送消息
                    }
                } finally {
                    latch.countDown();
                }
            }).start();
        }

        latch.await();
        long endTime = System.currentTimeMillis();
        
        long duration = endTime - startTime;
        double throughput = (double) totalMessages / (duration / 1000.0);

        System.out.println("总消息数: " + totalMessages);
        System.out.println("总耗时: " + duration + " ms");
        System.out.println("吞吐量: " + throughput + " msg/s");
    }
}
```

---
[下一章：常见问题与排查](./troubleshooting.md)
