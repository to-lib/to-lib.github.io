---
sidebar_position: 6.4
title: Netty 测试（EmbeddedChannel）
---

# Netty 测试（EmbeddedChannel）

很多 Netty 代码写起来很快，但验证起来很痛：

- 启一个端口跑起来再手工测？慢。
- 需要真实网络环境、线程调度、端口占用？不稳定。

Netty 提供了 `EmbeddedChannel`，可以在**不打开真实 Socket** 的情况下，把 `ChannelPipeline` 当作一个可测试的“纯逻辑处理链”来测试。

## EmbeddedChannel 能测什么

- 编解码器（`ByteToMessageDecoder` / `MessageToByteEncoder`）
- Handler 的入站/出站逻辑
- 粘包/拆包处理
- 业务 Handler 的状态机（例如登录/鉴权/心跳）

不适合测：

- OS 层面的网络行为
- `epoll/kqueue` 的差异
- 真实 TLS 握手（可以测 Handler 组合，但真实证书链/兼容性需要集成测试）

## 示例：测试 LengthFieldBasedFrameDecoder + 自定义 Decoder

下面以一个“长度字段 + UTF-8 字符串”的协议为例：

- 帧格式：`[len:int32][payload:bytes]`

### 1) Pipeline 组装

```java
import io.netty.channel.embedded.EmbeddedChannel;
import io.netty.handler.codec.LengthFieldBasedFrameDecoder;
import io.netty.handler.codec.string.StringDecoder;
import io.netty.util.CharsetUtil;

EmbeddedChannel ch = new EmbeddedChannel(
  new LengthFieldBasedFrameDecoder(1024 * 1024, 0, 4, 0, 4),
  new StringDecoder(CharsetUtil.UTF_8)
);
```

### 2) 构造输入 ByteBuf 并写入 inbound

```java
import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;

byte[] payload = "hello".getBytes(CharsetUtil.UTF_8);
ByteBuf buf = Unpooled.buffer();
buf.writeInt(payload.length);
buf.writeBytes(payload);

boolean accepted = ch.writeInbound(buf);
```

### 3) 断言输出

```java
Object msg = ch.readInbound();
// 这里应当是 StringDecoder 的输出
assert "hello".equals(msg);

assert ch.readInbound() == null;
ch.finish();
```

> [!IMPORTANT]
> `EmbeddedChannel#finish()` 会触发收尾逻辑并释放资源。测试结束务必调用。

## 示例：测试粘包（两个帧一次写入）

```java
ByteBuf merged = Unpooled.buffer();

byte[] p1 = "a".getBytes(CharsetUtil.UTF_8);
merged.writeInt(p1.length).writeBytes(p1);

byte[] p2 = "bb".getBytes(CharsetUtil.UTF_8);
merged.writeInt(p2.length).writeBytes(p2);

ch.writeInbound(merged);

assert "a".equals(ch.readInbound());
assert "bb".equals(ch.readInbound());
assert ch.readInbound() == null;
```

## 示例：测试拆包（半包分两次写入）

```java
byte[] payload = "hello".getBytes(CharsetUtil.UTF_8);

ByteBuf part1 = Unpooled.buffer();
part1.writeInt(payload.length);
part1.writeBytes(payload, 0, 2); // 只写前 2 字节

ByteBuf part2 = Unpooled.wrappedBuffer(payload, 2, payload.length - 2);

// 第一次写入不应产生完整消息
ch.writeInbound(part1);
assert ch.readInbound() == null;

// 第二次补齐后，才产出完整消息
ch.writeInbound(part2);
assert "hello".equals(ch.readInbound());
```

## 出站测试：writeOutbound

如果你要测试 encoder：

```java
import io.netty.handler.codec.string.StringEncoder;

EmbeddedChannel outCh = new EmbeddedChannel(new StringEncoder(CharsetUtil.UTF_8));

outCh.writeOutbound("hello");
ByteBuf encoded = outCh.readOutbound();

assert "hello".equals(encoded.toString(CharsetUtil.UTF_8));
encoded.release();
outCh.finish();
```

## 常见坑

### ByteBuf 的释放

- `EmbeddedChannel` 里读出来的 `ByteBuf`，如果你在测试里拿出来做断言，通常需要你自己 `release()`。
- 如果你把 `ByteBuf` 缓存到队列里（模拟背压），要考虑 `retain()` / `release()`。

### 把时间相关逻辑抽象掉

很多 Handler 会 `schedule()` 定时任务（心跳/超时）。纯单测里更推荐：

- 抽象时间源
- 或把定时任务触发写成可直接调用的方法

## 小结

- `EmbeddedChannel` 是 Netty 单元测试的核心工具。
- 先用它把 Pipeline 的逻辑测稳，再用少量集成测试验证真实网络行为。

---

你可以结合：

- [编解码](/docs/netty/codec)
- [背压与可写性控制](/docs/netty/backpressure)
