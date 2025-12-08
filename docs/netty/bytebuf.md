---
sidebar_position: 4
---

# ByteBuf 详解

## ByteBuf 概述

ByteBuf 是 Netty 的核心数据结构，用于替代 Java NIO 的 ByteBuffer，提供更高效和易用的字节缓冲区。

### ByteBuf vs ByteBuffer

| 特性 | ByteBuffer | ByteBuf |
|------|-----------|---------|
| **API 复杂度** | 复杂 | 简单易用 |
| **读写指针** | 共享一个指针 | 独立的读写指针 |
| **动态大小** | 固定大小 | 可动态扩展 |
| **内存管理** | 需手动管理 | 自动引用计数 |
| **零拷贝** | 不支持 | 支持 |
| **线程安全** | 非线程安全 | 非线程安全 |

## ByteBuf 结构

```
┌─────────────────────────────────────────────────┐
│  读取区域（Readable）│ 写入区域（Writable） │预留 │
├─────────────────────────────────────────────────┤
│                                                 │
0          readerIndex                 writerIndex  capacity
          ◄──────── 可读字节数 ───────►
                                  ◄── 可写字节数 ──►
```

### 创建 ByteBuf

```java
// 方式1：非池化 ByteBuf（不推荐用于生产）
ByteBuf buf = Unpooled.buffer(10);
ByteBuf buf2 = Unpooled.directBuffer(10);

// 方式2：池化 ByteBuf（推荐，高性能）
ByteBufAllocator allocator = PooledByteBufAllocator.DEFAULT;
ByteBuf buf3 = allocator.buffer(1024);
ByteBuf buf4 = allocator.directBuffer(1024);

// 方式3：包装现有字节数组
byte[] array = {1, 2, 3, 4, 5};
ByteBuf buf5 = Unpooled.wrappedBuffer(array);

// 方式4：拷贝字节数组
ByteBuf buf6 = Unpooled.copiedBuffer(array);

// 方式5：从字符串创建
ByteBuf buf7 = Unpooled.copiedBuffer("Hello", CharsetUtil.UTF_8);

// 方式6：在 Handler 中获取
@Override
public void channelRead(ChannelHandlerContext ctx, Object msg) {
    ByteBuf in = (ByteBuf) msg;
    // 直接使用
}
```

## 读写操作

### 基本读写

```java
ByteBuf buf = Unpooled.buffer(10);

// ========== 写入 ==========
buf.writeInt(100);           // 写入 4 字节整数
buf.writeByte(10);           // 写入 1 字节
buf.writeBytes(new byte[]{1, 2, 3});  // 写入字节数组
buf.writeCharSequence("Hello", CharsetUtil.UTF_8);  // 写入字符串

System.out.println("Writable: " + buf.writableBytes()); // 可写字节数

// ========== 读取 ==========
int value = buf.readInt();   // 读取 4 字节整数
byte b = buf.readByte();     // 读取 1 字节
byte[] data = new byte[3];
buf.readBytes(data);         // 读取字节数组
String str = buf.readCharSequence(5, CharsetUtil.UTF_8);

System.out.println("Readable: " + buf.readableBytes()); // 可读字节数
```

### 读取方式对比

```java
ByteBuf buf = Unpooled.copiedBuffer("Hello", CharsetUtil.UTF_8);

// 方式1：直接读取（会移动 readerIndex）
byte b = buf.readByte();
System.out.println("读取后 readerIndex: " + buf.readerIndex());

// 方式2：获取指定位置数据（不移动 readerIndex）
byte b2 = buf.getByte(0);
System.out.println("getByte 后 readerIndex: " + buf.readerIndex());

// 方式3：获取数据而不移动指针
String str = buf.getCharSequence(0, 5, CharsetUtil.UTF_8);
```

## 内存分配方式

### Heap Buffer（堆内存）

```java
// 存储在 JVM 堆内存中
ByteBuf heapBuf = Unpooled.buffer(1024);

// 优点：
// - GC 自动管理
// - 易于使用
// - 快速分配和销毁

// 缺点：
// - 性能不如直接内存
// - 需要额外的 copy 操作进行 I/O
```

### Direct Buffer（直接内存）

```java
// 存储在堆外内存（native memory）
ByteBuf directBuf = Unpooled.directBuffer(1024);

// 优点：
// - 性能高，减少 GC 压力
// - 直接 I/O，无需 copy

// 缺点：
// - 分配和释放更慢
// - 需要手动管理内存
// - 可能引起 OutOfMemory
```

## 引用计数和资源释放

Netty 使用引用计数来管理 ByteBuf 的生命周期。

```java
// 新创建的 ByteBuf，引用计数为 1
ByteBuf buf = allocator.buffer(1024);
System.out.println("创建后，引用计数: " + buf.refCnt()); // 1

// 增加引用计数
buf.retain();
System.out.println("retain 后: " + buf.refCnt()); // 2

// 减少引用计数
buf.release();
System.out.println("release 后: " + buf.refCnt()); // 1

// 最后一次 release，ByteBuf 被回收
buf.release();
System.out.println("release 后: " + buf.refCnt()); // 0

// 尝试再次使用会抛异常
// buf.writeInt(100); // IllegalReferenceCountException
```

### Handler 中的资源管理

```java
public class ByteBufHandler extends ChannelInboundHandlerAdapter {
    @Override
    public void channelRead(ChannelHandlerContext ctx, Object msg) {
        ByteBuf in = (ByteBuf) msg;
        try {
            // 处理 ByteBuf
            byte[] data = new byte[in.readableBytes()];
            in.readBytes(data);
            System.out.println("数据: " + Arrays.toString(data));
        } finally {
            // Netty 会自动释放输入的 ByteBuf
            ReferenceCountUtil.release(msg);
        }
    }

    @Override
    public void write(ChannelHandlerContext ctx, Object msg, ChannelPromise promise) {
        // 如果写入 ByteBuf，需要传递给下一个 Handler
        ByteBuf response = ctx.alloc().buffer(128);
        response.writeCharSequence("OK", CharsetUtil.UTF_8);
        ctx.writeAndFlush(response); // Netty 会自动释放
    }
}
```

## 零拷贝操作

### CompositeByteBuf

多个 ByteBuf 组合成一个，避免复制：

```java
ByteBuf buf1 = Unpooled.wrappedBuffer(new byte[]{1, 2, 3});
ByteBuf buf2 = Unpooled.wrappedBuffer(new byte[]{4, 5, 6});

// 组合多个 ByteBuf，不进行复制
CompositeByteBuf composite = Unpooled.compositeBuffer();
composite.addComponents(true, buf1, buf2); // true 表示自动增长

// 读取组合的数据
byte[] data = new byte[composite.readableBytes()];
composite.readBytes(data);
System.out.println("组合数据: " + Arrays.toString(data)); // [1,2,3,4,5,6]
```

### Slice（切片）

```java
ByteBuf buf = Unpooled.wrappedBuffer(new byte[]{1, 2, 3, 4, 5});

// 创建 slice，共享底层数据，不复制
ByteBuf slice = buf.slice(1, 3); // 从位置 1 开始，长度 3

// 修改 slice 会影响原 ByteBuf
slice.setByte(0, 10);
System.out.println("原 buf: " + buf.getByte(1)); // 10

// 但 slice 有独立的读写指针
slice.readByte();
System.out.println("slice readerIndex: " + slice.readerIndex()); // 1
System.out.println("原 buf readerIndex: " + buf.readerIndex()); // 0
```

### Duplicate（复制引用）

```java
ByteBuf buf = Unpooled.wrappedBuffer(new byte[]{1, 2, 3, 4, 5});

// 创建 duplicate，独立的读写指针，但共享数据
ByteBuf dup = buf.duplicate();

buf.readByte();
System.out.println("原 buf readerIndex: " + buf.readerIndex()); // 1
System.out.println("dup readerIndex: " + dup.readerIndex()); // 0

// 修改 dup 会影响原 ByteBuf
dup.setByte(0, 10);
System.out.println("原 buf 第 0 位: " + buf.getByte(0)); // 10
```

## ByteBuf 分配器

### PooledByteBufAllocator（推荐）

```java
// 默认使用 PooledByteBufAllocator
ChannelConfig config = channel.config();
config.setOption(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT);

// 优势：
// - 重用已分配的内存，减少 GC
// - 性能高
// - 适合长连接场景
```

### UnpooledByteBufAllocator

```java
// 不使用内存池，每次创建新的 ByteBuf
ByteBufAllocator allocator = UnpooledByteBufAllocator.DEFAULT;
ByteBuf buf = allocator.buffer(1024);

// 适合：
// - 一次性操作
// - 内存需求不频繁
```

## 最佳实践

```java
public class ByteBufBestPractices extends ChannelInboundHandlerAdapter {
    
    // ✓ 正确：处理接收的消息
    @Override
    public void channelRead(ChannelHandlerContext ctx, Object msg) {
        ByteBuf in = (ByteBuf) msg;
        try {
            // 处理消息
            processMessage(in);
        } finally {
            // 释放引用
            ReferenceCountUtil.release(msg);
        }
    }

    // ✓ 正确：写入响应
    private void sendResponse(ChannelHandlerContext ctx, String response) {
        ByteBuf buf = ctx.alloc().buffer();
        buf.writeCharSequence(response, CharsetUtil.UTF_8);
        ctx.writeAndFlush(buf); // Netty 自动释放
    }

    // ✗ 错误：内存泄漏
    @Override
    public void channelRead2(ChannelHandlerContext ctx, Object msg) {
        ByteBuf in = (ByteBuf) msg;
        // 没有释放，导致内存泄漏！
        processMessage(in);
    }

    // ✗ 错误：未正确释放
    private void badWrite(ChannelHandlerContext ctx, String msg) {
        ByteBuf buf = ctx.alloc().buffer();
        buf.writeCharSequence(msg, CharsetUtil.UTF_8);
        ctx.channel().writeAndFlush(buf.copy()); // 这样会导致泄漏
    }

    private void processMessage(ByteBuf in) {
        byte[] data = new byte[in.readableBytes()];
        in.readBytes(data);
        System.out.println("处理数据: " + Arrays.toString(data));
    }
}
```

## 常用工具类

```java
// ReferenceCountUtil - 安全释放对象
Object msg = ...;
ReferenceCountUtil.release(msg);

// ByteBufUtil - ByteBuf 工具方法
String hex = ByteBufUtil.hexDump(buf);
ByteBufUtil.appendPrettyHexDump(sb, buf);

// PooledByteBufAllocator.DEFAULT - 获取默认分配器
ByteBuf buf = PooledByteBufAllocator.DEFAULT.buffer(256);
```

---
[下一章：编码与解码](./codec.md)