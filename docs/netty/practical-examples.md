---
sidebar_position: 6
---

# 实战案例

> [!TIP]
> 本章节提供完整可运行的示例代码,涵盖 Echo 服务器、聊天系统和 HTTP 服务器等常见场景。建议跟随示例动手实践,加深理解。

## 案例 1：简单的 Echo 服务器和客户端

### Echo 服务器

```java
import io.netty.bootstrap.ServerBootstrap;
import io.netty.buffer.Unpooled;
import io.netty.channel.*;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.handler.codec.string.StringDecoder;
import io.netty.handler.codec.string.StringEncoder;
import io.netty.util.CharsetUtil;

public class EchoServer {
    private int port;

    public EchoServer(int port) {
        this.port = port;
    }

    public void start() throws InterruptedException {
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
                        ChannelPipeline pipeline = ch.pipeline();
                        pipeline.addLast(new StringDecoder(CharsetUtil.UTF_8));
                        pipeline.addLast(new StringEncoder(CharsetUtil.UTF_8));
                        pipeline.addLast(new EchoServerHandler());
                    }
                });

            ChannelFuture future = bootstrap.bind(port).sync();
            System.out.println("Echo Server started on port " + port);
            future.channel().closeFuture().sync();
        } finally {
            bossGroup.shutdownGracefully();
            workerGroup.shutdownGracefully();
        }
    }

    public static void main(String[] args) throws InterruptedException {
        new EchoServer(8080).start();
    }
}

public class EchoServerHandler extends SimpleChannelInboundHandler<String> {
    @Override
    public void channelActive(ChannelHandlerContext ctx) {
        InetSocketAddress remote = (InetSocketAddress) ctx.channel().remoteAddress();
        System.out.println("客户端连接: " + remote);
        ctx.writeAndFlush("欢迎连接 Echo 服务器\n");
    }

    @Override
    protected void channelRead0(ChannelHandlerContext ctx, String msg) {
        System.out.println("服务器接收: " + msg);
        ctx.writeAndFlush("服务器回复: " + msg);
    }

    @Override
    public void channelInactive(ChannelHandlerContext ctx) {
        System.out.println("客户端断开连接");
    }

    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
        cause.printStackTrace();
        ctx.close();
    }
}
```

> [!NOTE] > **代码要点:**
>
> - `bossGroup` 和 `workerGroup` 分离提高性能
> - `SO_KEEPALIVE` 开启 TCP 保活机制
> - `SimpleChannelInboundHandler` 自动释放 ByteBuf
> - `channelActive` 在连接建立时触发

### Echo 客户端

```java
import io.netty.bootstrap.Bootstrap;
import io.netty.channel.*;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.handler.codec.string.StringDecoder;
import io.netty.handler.codec.string.StringEncoder;
import io.netty.util.CharsetUtil;
import java.util.Scanner;

public class EchoClient {
    private String host;
    private int port;

    public EchoClient(String host, int port) {
        this.host = host;
        this.port = port;
    }

    public void start() throws InterruptedException {
        EventLoopGroup group = new NioEventLoopGroup();
        try {
            Bootstrap bootstrap = new Bootstrap();
            bootstrap.group(group)
                .channel(NioSocketChannel.class)
                .option(ChannelOption.TCP_NODELAY, true)
                .handler(new ChannelInitializer<SocketChannel>() {
                    @Override
                    protected void initChannel(SocketChannel ch) {
                        ChannelPipeline pipeline = ch.pipeline();
                        pipeline.addLast(new StringDecoder(CharsetUtil.UTF_8));
                        pipeline.addLast(new StringEncoder(CharsetUtil.UTF_8));
                        pipeline.addLast(new EchoClientHandler());
                    }
                });

            ChannelFuture future = bootstrap.connect(host, port).sync();
            System.out.println("连接服务器成功");

            // 从控制台读取输入
            Scanner scanner = new Scanner(System.in);
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                if ("quit".equalsIgnoreCase(line)) {
                    break;
                }
                future.channel().writeAndFlush(line + "\n");
            }

            future.channel().closeFuture().sync();
        } finally {
            group.shutdownGracefully();
        }
    }

    public static void main(String[] args) throws InterruptedException {
        new EchoClient("localhost", 8080).start();
    }
}

public class EchoClientHandler extends SimpleChannelInboundHandler<String> {
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, String msg) {
        System.out.println("客户端接收: " + msg);
    }

    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
        cause.printStackTrace();
        ctx.close();
    }
}
```

## 案例 2：聊天服务器

多客户端支持的聊天系统。

### 聊天消息类

```java
public class ChatMessage {
    public static final byte JOIN = 1;
    public static final byte LEAVE = 2;
    public static final byte CHAT = 3;

    private byte type;
    private String username;
    private String content;

    public ChatMessage(byte type, String username, String content) {
        this.type = type;
        this.username = username;
        this.content = content;
    }

    // getters...
}
```

### 聊天服务器

```java
public class ChatServer {
    private Map<String, Channel> clients = new ConcurrentHashMap<>();
    private int port;

    public ChatServer(int port) {
        this.port = port;
    }

    public void start() throws InterruptedException {
        EventLoopGroup bossGroup = new NioEventLoopGroup(1);
        EventLoopGroup workerGroup = new NioEventLoopGroup();

        try {
            ServerBootstrap bootstrap = new ServerBootstrap();
            bootstrap.group(bossGroup, workerGroup)
                .channel(NioServerSocketChannel.class)
                .childHandler(new ChannelInitializer<SocketChannel>() {
                    @Override
                    protected void initChannel(SocketChannel ch) {
                        ChannelPipeline pipeline = ch.pipeline();

                        // 长度字段解码器
                        pipeline.addLast(
                            new LengthFieldBasedFrameDecoder(65535, 0, 4, 0, 4)
                        );

                        // 自定义解码器
                        pipeline.addLast(new ChatMessageDecoder());
                        pipeline.addLast(new ChatMessageEncoder());
                        pipeline.addLast(new ChatServerHandler(clients));
                    }
                });

            ChannelFuture future = bootstrap.bind(port).sync();
            System.out.println("Chat Server started on port " + port);
            future.channel().closeFuture().sync();
        } finally {
            bossGroup.shutdownGracefully();
            workerGroup.shutdownGracefully();
        }
    }

    public static void main(String[] args) throws InterruptedException {
        new ChatServer(8080).start();
    }
}

public class ChatServerHandler extends SimpleChannelInboundHandler<ChatMessage> {
    private Map<String, Channel> clients;

    public ChatServerHandler(Map<String, Channel> clients) {
        this.clients = clients;
    }

    @Override
    protected void channelRead0(ChannelHandlerContext ctx, ChatMessage msg) {
        switch (msg.getType()) {
            case ChatMessage.JOIN:
                handleJoin(ctx, msg);
                break;
            case ChatMessage.LEAVE:
                handleLeave(ctx, msg);
                break;
            case ChatMessage.CHAT:
                handleChat(msg);
                break;
        }
    }

    private void handleJoin(ChannelHandlerContext ctx, ChatMessage msg) {
        String username = msg.getUsername();
        clients.put(username, ctx.channel());
        System.out.println(username + " 加入聊天室");

        // 广播加入消息
        ChatMessage joinMsg = new ChatMessage(
            ChatMessage.JOIN,
            "系统",
            username + " 加入了聊天室"
        );
        broadcast(joinMsg);
    }

    private void handleLeave(ChannelHandlerContext ctx, ChatMessage msg) {
        String username = msg.getUsername();
        clients.remove(username);
        System.out.println(username + " 离开聊天室");

        // 广播离开消息
        ChatMessage leaveMsg = new ChatMessage(
            ChatMessage.LEAVE,
            "系统",
            username + " 离开了聊天室"
        );
        broadcast(leaveMsg);
    }

    private void handleChat(ChatMessage msg) {
        System.out.println(msg.getUsername() + ": " + msg.getContent());
        broadcast(msg);
    }

    private void broadcast(ChatMessage msg) {
        clients.forEach((username, channel) -> {
            if (channel.isActive()) {
                channel.writeAndFlush(msg);
            }
        });
    }

    @Override
    public void channelInactive(ChannelHandlerContext ctx) {
        // 连接断开时，移除客户端
        clients.forEach((username, channel) -> {
            if (channel == ctx.channel()) {
                clients.remove(username);
                System.out.println(username + " 连接断开");
            }
        });
    }

    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
        cause.printStackTrace();
        ctx.close();
    }
}
```

### 自定义编解码器

```java
public class ChatMessageEncoder extends MessageToByteEncoder<ChatMessage> {
    @Override
    protected void encode(ChannelHandlerContext ctx, ChatMessage msg, ByteBuf out) {
        byte[] usernameBytes = msg.getUsername().getBytes(CharsetUtil.UTF_8);
        byte[] contentBytes = msg.getContent().getBytes(CharsetUtil.UTF_8);

        int dataLength = 1 + 4 + usernameBytes.length + 4 + contentBytes.length;
        out.writeInt(dataLength);
        out.writeByte(msg.getType());
        out.writeInt(usernameBytes.length);
        out.writeBytes(usernameBytes);
        out.writeInt(contentBytes.length);
        out.writeBytes(contentBytes);
    }
}

public class ChatMessageDecoder extends ByteToMessageDecoder {
    @Override
    protected void decode(ChannelHandlerContext ctx, ByteBuf in, List<Object> out) {
        if (in.readableBytes() < 13) { // 最小长度
            return;
        }

        in.markReaderIndex();
        int dataLength = in.readInt();

        if (in.readableBytes() < dataLength) {
            in.resetReaderIndex();
            return;
        }

        byte type = in.readByte();

        int usernameLength = in.readInt();
        byte[] usernameBytes = new byte[usernameLength];
        in.readBytes(usernameBytes);

        int contentLength = in.readInt();
        byte[] contentBytes = new byte[contentLength];
        in.readBytes(contentBytes);

        String username = new String(usernameBytes, CharsetUtil.UTF_8);
        String content = new String(contentBytes, CharsetUtil.UTF_8);

        ChatMessage msg = new ChatMessage(type, username, content);
        out.add(msg);
    }
}
```

## 案例 3：HTTP 服务器

使用 Netty 实现简单的 HTTP 服务器。

```java
public class HttpServer {
    private int port;

    public HttpServer(int port) {
        this.port = port;
    }

    public void start() throws InterruptedException {
        EventLoopGroup bossGroup = new NioEventLoopGroup(1);
        EventLoopGroup workerGroup = new NioEventLoopGroup();

        try {
            ServerBootstrap bootstrap = new ServerBootstrap();
            bootstrap.group(bossGroup, workerGroup)
                .channel(NioServerSocketChannel.class)
                .childHandler(new ChannelInitializer<SocketChannel>() {
                    @Override
                    protected void initChannel(SocketChannel ch) {
                        ChannelPipeline pipeline = ch.pipeline();

                        // HTTP 编解码器
                        pipeline.addLast(new HttpServerCodec());

                        // HTTP 内容聚合器
                        pipeline.addLast(new HttpObjectAggregator(512 * 1024));

                        // 业务处理
                        pipeline.addLast(new HttpServerHandler());
                    }
                });

            ChannelFuture future = bootstrap.bind(port).sync();
            System.out.println("HTTP Server started on port " + port);
            future.channel().closeFuture().sync();
        } finally {
            bossGroup.shutdownGracefully();
            workerGroup.shutdownGracefully();
        }
    }

    public static void main(String[] args) throws InterruptedException {
        new HttpServer(8080).start();
    }
}

public class HttpServerHandler extends SimpleChannelInboundHandler<FullHttpRequest> {
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest request) {
        String uri = request.uri();
        String method = request.method().name();

        System.out.println("请求方法: " + method);
        System.out.println("请求路径: " + uri);

        // 构建响应
        String content = "Hello World from Netty!\n" +
                         "Method: " + method + "\n" +
                         "URI: " + uri;

        ByteBuf buf = Unpooled.copiedBuffer(content, CharsetUtil.UTF_8);

        FullHttpResponse response = new DefaultFullHttpResponse(
            HttpVersion.HTTP_1_1,
            HttpResponseStatus.OK,
            buf
        );

        response.headers()
            .set(HttpHeaderNames.CONTENT_TYPE, "text/plain; charset=utf-8")
            .set(HttpHeaderNames.CONTENT_LENGTH, buf.readableBytes());

        ctx.writeAndFlush(response);
    }

    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
        cause.printStackTrace();
        ctx.close();
    }
}
```

使用浏览器访问 `http://localhost:8080` 即可看到响应。

## 最佳实践总结

1. **合理配置线程池** - 使用多个 Worker EventLoop 处理连接
2. **避免阻塞 EventLoop** - 耗时操作提交到独立线程池
3. **及时释放资源** - 释放 ByteBuf，关闭连接
4. **异常处理** - 在 exceptionCaught 中处理异常
5. **内存管理** - 使用对象池重用内存，避免频繁分配
6. **协议设计** - 使用长度字段或分界符便于解析
7. **连接管理** - 设置合理的超时和心跳机制

---

[下一章：高级特性](/docs/netty/advanced)
