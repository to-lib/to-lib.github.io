---
sidebar_position: 5
---

# 编码与解码

> [!TIP]
> 编解码器是 Netty 网络编程的核心，负责将字节数据与业务对象之间进行转换。本章讲解 Netty 内置编解码器的使用方法，以及如何自定义编解码器实现自定义协议。

## 内置编解码器

### 字符串编解码

```java
// 在 Pipeline 中添加
pipeline.addLast(new StringDecoder(CharsetUtil.UTF_8));
pipeline.addLast(new StringEncoder(CharsetUtil.UTF_8));

// Handler 中接收已解码的字符串
public class StringHandler extends SimpleChannelInboundHandler<String> {
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, String msg) {
        System.out.println("接收到字符串: " + msg);
        ctx.writeAndFlush("服务器收到: " + msg);
    }
}
```

### 对象编码解码

```java
// Java 对象序列化（不推荐用于生产环境）
pipeline.addLast(new ObjectDecoder(1024, ClassResolvers.cacheDisabled(null)));
pipeline.addLast(new ObjectEncoder());

// Handler
public class ObjectHandler extends SimpleChannelInboundHandler<Object> {
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, Object msg) {
        MyMessage message = (MyMessage) msg;
        System.out.println("接收到对象: " + message);
    }
}

// 序列化对象需要实现 Serializable 接口
public class MyMessage implements Serializable {
    private static final long serialVersionUID = 1L;
    public String content;
}
```

## 自定义编码器

### ByteToMessageDecoder（解码器基类）

解码器用于将 ByteBuf 转换为 Java 对象。

```java
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.ByteToMessageDecoder;
import java.util.List;

// 例子：解析固定长度报文
public class FixedLengthDecoder extends ByteToMessageDecoder {
    private final int frameLength;

    public FixedLengthDecoder(int frameLength) {
        this.frameLength = frameLength;
    }

    @Override
    protected void decode(ChannelHandlerContext ctx, ByteBuf in, List<Object> out) {
        // 检查是否有足够的数据
        if (in.readableBytes() < frameLength) {
            return; // 等待更多数据
        }

        // 读取一个完整的帧
        ByteBuf frame = in.readBytes(frameLength);
        out.add(frame); // 传递给下一个 Handler
    }
}

// 使用
ChannelInitializer<SocketChannel> initializer =
    new ChannelInitializer<SocketChannel>() {
    @Override
    protected void initChannel(SocketChannel ch) {
        ch.pipeline().addLast(new FixedLengthDecoder(10));
        ch.pipeline().addLast(new FrameHandler());
    }
};
```

### MessageToByteEncoder（编码器基类）

编码器用于将 Java 对象转换为 ByteBuf。

```java
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.MessageToByteEncoder;

public class CustomEncoder extends MessageToByteEncoder<String> {
    @Override
    protected void encode(ChannelHandlerContext ctx, String msg, ByteBuf out) {
        // 将字符串编码为字节
        byte[] bytes = msg.getBytes(CharsetUtil.UTF_8);

        // 写入长度 + 内容
        out.writeInt(bytes.length);
        out.writeBytes(bytes);
    }
}
```

## 分界符解码器

### DelimitedFrameDecoder（分界符解码）

```java
// 按行分割
pipeline.addLast(new LineBasedFrameDecoder(1024));
pipeline.addLast(new StringDecoder(CharsetUtil.UTF_8));

// 按自定义分界符分割
ByteBuf delimiter = Unpooled.copiedBuffer("##", CharsetUtil.UTF_8);
pipeline.addLast(new DelimitedFrameDecoder(1024, delimiter));
pipeline.addLast(new StringDecoder(CharsetUtil.UTF_8));
```

## 长度字段解码器

### LengthFieldBasedFrameDecoder（最常用）

用于解析包含长度字段的协议。

```
常见的协议格式：
┌───────────┬──────────┬──────────────┐
│  长度字段  │  业务数据 │  长度字段    │
│ (4字节)   │(变长)    │  指向数据长度 │
└───────────┴──────────┴──────────────┘
```

### 常见用法

```java
// 场景1：长度字段在数据头部，包含长度字段本身
//  格式: [数据长度(4字节)] [数据...]
pipeline.addLast(
    new LengthFieldBasedFrameDecoder(
        1024,     // maxFrameLength：最大帧长度
        0,        // lengthFieldOffset：长度字段的起始位置
        4,        // lengthFieldLength：长度字段的长度（4字节 int）
        0,        // lengthAdjustment：长度字段之后到真实数据之间的字节数
        4         // initialBytesToStrip：从头部移除的字节数
    )
);

// 场景2：长度字段不包含自身长度
//  格式: [长度(4字节)] [数据...]
//  长度字段的值只表示数据部分的长度
pipeline.addLast(
    new LengthFieldBasedFrameDecoder(
        1024,     // maxFrameLength
        0,        // lengthFieldOffset
        4,        // lengthFieldLength
        0,        // lengthAdjustment：长度字段值需要加 4（字段本身）
        4         // initialBytesToStrip
    )
);

// 场景3：长度字段在中间位置
//  格式: [头部(2字节)] [长度(4字节)] [数据...]
pipeline.addLast(
    new LengthFieldBasedFrameDecoder(
        1024,     // maxFrameLength
        2,        // lengthFieldOffset：长度字段在位置 2
        4,        // lengthFieldLength
        0,        // lengthAdjustment
        0         // initialBytesToStrip：不移除任何字节，保留完整帧
    )
);

// 完整例子
ChannelInitializer<SocketChannel> initializer =
    new ChannelInitializer<SocketChannel>() {
    @Override
    protected void initChannel(SocketChannel ch) {
        ch.pipeline()
            .addLast(new LengthFieldBasedFrameDecoder(65535, 0, 4, 0, 4))
            .addLast(new StringDecoder(CharsetUtil.UTF_8))
            .addLast(new MyMessageHandler());
    }
};
```

## 自定义协议编解码

### 定义协议格式

```java
// 协议格式: [长度(4字节)] [类型(1字节)] [数据...]
public class MyMessage {
    private byte type;
    private String data;

    public MyMessage(byte type, String data) {
        this.type = type;
        this.data = data;
    }

    public byte getType() {
        return type;
    }

    public String getData() {
        return data;
    }
}
```

### 编码器

```java
public class MyMessageEncoder extends MessageToByteEncoder<MyMessage> {
    @Override
    protected void encode(ChannelHandlerContext ctx, MyMessage msg, ByteBuf out) {
        // 获取消息数据
        String data = msg.getData();
        byte[] bytes = data.getBytes(CharsetUtil.UTF_8);

        // 写入长度（包含类型字段 1 字节）
        out.writeInt(bytes.length + 1);

        // 写入类型
        out.writeByte(msg.getType());

        // 写入数据
        out.writeBytes(bytes);
    }
}
```

### 解码器

```java
public class MyMessageDecoder extends ByteToMessageDecoder {
    private final int HEADER_SIZE = 5; // 长度(4) + 类型(1)

    @Override
    protected void decode(ChannelHandlerContext ctx, ByteBuf in, List<Object> out) {
        // 检查头部数据是否完整
        if (in.readableBytes() < HEADER_SIZE) {
            return;
        }

        // 标记读位置
        in.markReaderIndex();

        // 读取消息长度
        int length = in.readInt();

        // 检查数据是否完整
        if (in.readableBytes() < length) {
            in.resetReaderIndex(); // 重置读位置
            return;
        }

        // 读取类型
        byte type = in.readByte();

        // 读取数据
        byte[] data = new byte[length - 1];
        in.readBytes(data);
        String message = new String(data, CharsetUtil.UTF_8);

        // 创建消息对象
        MyMessage msg = new MyMessage(type, message);
        out.add(msg);
    }
}
```

### 使用自定义编解码器

```java
ServerBootstrap bootstrap = new ServerBootstrap();
bootstrap.childHandler(new ChannelInitializer<SocketChannel>() {
    @Override
    protected void initChannel(SocketChannel ch) {
        ChannelPipeline pipeline = ch.pipeline();

        // 添加解码器
        pipeline.addLast(new MyMessageDecoder());

        // 添加编码器
        pipeline.addLast(new MyMessageEncoder());

        // 添加业务处理 Handler
        pipeline.addLast(new MyMessageHandler());
    }
});

// 业务处理
public class MyMessageHandler extends SimpleChannelInboundHandler<MyMessage> {
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, MyMessage msg) {
        System.out.println("类型: " + msg.getType() + ", 数据: " + msg.getData());

        // 发送响应
        MyMessage response = new MyMessage((byte) 1, "服务器响应");
        ctx.writeAndFlush(response);
    }
}
```

## 常用编解码器总结

| 编解码器                         | 用途          | 场景                   |
| -------------------------------- | ------------- | ---------------------- |
| **StringDecoder/Encoder**        | 字符串编解码  | 文本协议               |
| **ObjectDecoder/Encoder**        | 对象序列化    | 简单场景（不推荐生产） |
| **LineBasedFrameDecoder**        | 按行分割      | HTTP、Redis 等         |
| **DelimitedFrameDecoder**        | 按分界符分割  | 自定义分界符           |
| **FixedLengthFrameDecoder**      | 固定长度帧    | 固定大小消息           |
| **LengthFieldBasedFrameDecoder** | 长度字段帧    | 大多数二进制协议       |
| **Base64Decoder/Encoder**        | Base64 编解码 | Base64 数据            |
| **ZlibCodecFactory**             | 压缩编解码    | 压缩传输               |

## 最佳实践

```java
// ✓ 正确：解码器中合理检查数据完整性
@Override
protected void decode(ChannelHandlerContext ctx, ByteBuf in, List<Object> out) {
    if (in.readableBytes() < MIN_LENGTH) {
        return; // 等待更多数据
    }
    // 处理数据
}

// ✓ 正确：编码器中检查入参
@Override
protected void encode(ChannelHandlerContext ctx, MyMessage msg, ByteBuf out) {
    if (msg == null) {
        return;
    }
    // 编码
}

// ✗ 错误：编码器中进行 I/O 操作
@Override
protected void encode(ChannelHandlerContext ctx, MyMessage msg, ByteBuf out) {
    // 不要做：从数据库读取数据（阻塞）
    // Object obj = database.query(msg.getId());
    out.writeBytes(...)
}
```

---

[下一章：实战案例](/docs/netty/practical-examples)
