---
sidebar_position: 6.3
title: 背压与可写性（Backpressure）
---

# 背压与可写性（Backpressure）

在高并发/大吞吐场景里，“写得出去”往往比“读得到”更难。

- 如果对端读得慢、网络抖动、下游处理变慢，你的 `writeAndFlush()` 会越来越快地把数据塞进 Netty 的出站缓冲。
- 出站缓冲持续膨胀会导致：
  - 直接内存/堆内存上升
  - GC 压力变大
  - 延迟抖动、甚至 OOM

Netty 提供了相对完善的背压手段：**可写性（writability）** 与 **写缓冲水位线（WriteBufferWaterMark）**。

## 关键概念

### ChannelOutboundBuffer

每个 `Channel` 都有自己的出站缓冲（内部为 `ChannelOutboundBuffer`）。调用 `ctx.write(...)` 并不会立刻写到 socket，而是先进入缓冲区；`flush()` 才会尝试真正写到内核。

### 可写性：`channel.isWritable()`

当出站缓冲超过高水位线时，Channel 会变为 **不可写**；当缓冲回落到低水位线以下，Channel 会恢复 **可写**。

你可以用：

- `channel.isWritable()`：查询当前状态
- `channelWritabilityChanged(...)`：监听状态变化

## 配置写缓冲水位线

```java
import io.netty.channel.ChannelOption;
import io.netty.channel.WriteBufferWaterMark;

ServerBootstrap b = new ServerBootstrap();

b.childOption(
  ChannelOption.WRITE_BUFFER_WATER_MARK,
  new WriteBufferWaterMark(
    32 * 1024,      // low: 32KB
    1024 * 1024     // high: 1MB
  )
);
```

> [!IMPORTANT]
> 水位线不是“越大越好”。
>
> - 太小：频繁触发不可写，吞吐受限
> - 太大：更容易堆积导致内存上涨、尾延迟恶化

建议基于压测调整，并在监控中观察：

- `channel.isWritable()` 变化频率
- pending outbound bytes（待发送字节）
- P95/P99 延迟

## 监听可写性变化

```java
import io.netty.channel.ChannelDuplexHandler;
import io.netty.channel.ChannelHandlerContext;

public class WritabilityAwareHandler extends ChannelDuplexHandler {

  @Override
  public void channelWritabilityChanged(ChannelHandlerContext ctx) {
    if (ctx.channel().isWritable()) {
      // 从不可写恢复
      // 适合：继续发送、恢复读、释放积压队列
    } else {
      // 进入不可写
      // 适合：暂停发送、降级、丢弃低优先级消息
    }
    ctx.fireChannelWritabilityChanged();
  }
}
```

## 常见背压策略

### 1) “读 → 写”桥接：暂停 autoRead

当 Channel 不可写时，你往往也不希望继续读取更多请求（否则会积压更多业务对象/ByteBuf）。

```java
import io.netty.channel.ChannelConfig;

ChannelConfig cfg = ctx.channel().config();

if (!ctx.channel().isWritable()) {
  cfg.setAutoRead(false);
}

// 在 channelWritabilityChanged 中恢复
if (ctx.channel().isWritable()) {
  cfg.setAutoRead(true);
}
```

> [!NOTE]
> `autoRead=false` 不是“立刻停止所有 read 回调”，但会抑制后续读事件触发（具体取决于 transport 与当前循环）。

### 2) 应用层发送队列（限长）

如果你必须持续接受业务请求（例如来自内部队列），可采用“限长队列 + 丢弃/拒绝”策略：

- 达到上限就拒绝请求
- 只保留高优先级消息
- 对低优先级消息做 sampling

### 3) 分批写入 + 合并 flush

减少系统调用次数，降低压力：

- `ctx.write(...)` 多次
- 在批次末尾 `ctx.flush()` 一次

不过它不能替代背压，只是优化。

## 典型坑

### 在不可写时仍疯狂 write

`writeAndFlush()` 不会“阻塞”，它只会把对象放进出站缓冲；因此不可写时继续 write 会更快把内存打爆。

### Handler 中做阻塞 I/O

当业务阻塞 EventLoop 时，写出也会变慢，背压更容易触发。

### ByteBuf 引用计数

当你把消息缓存到应用层队列（准备稍后写出）时，需要考虑引用计数（retain/release），否则要么泄漏要么提前释放。

## 小结

- 背压核心：**控制写入速度**，让系统能在下游变慢时自我保护。
- 最常用组合：
  - `WRITE_BUFFER_WATER_MARK`
  - `channelWritabilityChanged`
  - 不可写时 `autoRead=false`

---

下一篇建议阅读：

- [Netty 测试：EmbeddedChannel](/docs/netty/testing)
