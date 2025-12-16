---
sidebar_position: 6.2
title: Native Transport（Epoll / KQueue）
---

# Native Transport（Epoll / KQueue）

Netty 在不同操作系统上支持不同的底层 Transport：

- Linux：`epoll`（`EpollEventLoopGroup` / `EpollServerSocketChannel`）
- macOS / BSD：`kqueue`（`KQueueEventLoopGroup` / `KQueueServerSocketChannel`）
- 跨平台兜底：`nio`（`NioEventLoopGroup` / `NioServerSocketChannel`）

合理使用 Native Transport 通常可以获得更低的延迟与更高的吞吐，但也会带来平台依赖、打包复杂度提升等成本。

## 什么时候值得用 Native Transport

- 你有明确的性能目标（P99 延迟、吞吐、CPU 使用率）并且已经建立了压测基线。
- 你的部署环境固定（例如生产只跑 Linux）。
- 你需要更好的 I/O 事件驱动能力与更成熟的事件通知机制（尤其是 Linux 上的 `epoll`）。

如果你的项目需要跨平台运行（开发机 macOS + 生产 Linux），建议：

- 代码层面做**运行时探测** + 自动降级；
- 依赖层面同时引入 `epoll` 与 `kqueue`（或按 profile 引入）。

## 依赖引入

### Maven（示例）

> [!IMPORTANT]
> 下面展示的是“思路”。不同 CPU 架构（x86_64 / aarch64）需要不同 classifier，你需要按实际环境选择。

```xml
<!-- Linux epoll -->
<dependency>
  <groupId>io.netty</groupId>
  <artifactId>netty-transport-native-epoll</artifactId>
  <version>4.1.100.Final</version>
  <!-- 示例：linux-x86_64 / linux-aarch_64 等 -->
  <classifier>linux-x86_64</classifier>
</dependency>

<!-- macOS / BSD kqueue -->
<dependency>
  <groupId>io.netty</groupId>
  <artifactId>netty-transport-native-kqueue</artifactId>
  <version>4.1.100.Final</version>
  <classifier>osx-x86_64</classifier>
</dependency>
```

## 运行时探测与自动降级

Netty 提供了能力检测方法：

- `io.netty.channel.epoll.Epoll.isAvailable()`
- `io.netty.channel.kqueue.KQueue.isAvailable()`

你可以在启动时根据平台选择合适的 `EventLoopGroup` 与 `Channel` 类型。

```java
import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.ServerChannel;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;

import io.netty.channel.epoll.Epoll;
import io.netty.channel.epoll.EpollEventLoopGroup;
import io.netty.channel.epoll.EpollServerSocketChannel;

import io.netty.channel.kqueue.KQueue;
import io.netty.channel.kqueue.KQueueEventLoopGroup;
import io.netty.channel.kqueue.KQueueServerSocketChannel;

public class TransportSelector {

  public static EventLoopGroup newBossGroup(int threads) {
    if (Epoll.isAvailable()) {
      return new EpollEventLoopGroup(threads);
    }
    if (KQueue.isAvailable()) {
      return new KQueueEventLoopGroup(threads);
    }
    return new NioEventLoopGroup(threads);
  }

  public static EventLoopGroup newWorkerGroup(int threads) {
    if (Epoll.isAvailable()) {
      return new EpollEventLoopGroup(threads);
    }
    if (KQueue.isAvailable()) {
      return new KQueueEventLoopGroup(threads);
    }
    return new NioEventLoopGroup(threads);
  }

  @SuppressWarnings("unchecked")
  public static Class<? extends ServerChannel> serverChannelClass() {
    if (Epoll.isAvailable()) {
      return EpollServerSocketChannel.class;
    }
    if (KQueue.isAvailable()) {
      return KQueueServerSocketChannel.class;
    }
    return NioServerSocketChannel.class;
  }

  public static ServerBootstrap configure(ServerBootstrap bootstrap,
                                         EventLoopGroup boss,
                                         EventLoopGroup worker) {
    return bootstrap.group(boss, worker).channel(serverChannelClass());
  }
}
```

## 常见坑

### 依赖 classifier 不匹配

- 本机是 Apple Silicon（aarch64），但是你引入了 `osx-x86_64`，运行时会加载失败。
- 生产是 Linux aarch64，但你只打了 `linux-x86_64`。

### 容器/裁剪镜像导致 native library 加载失败

- 某些极简镜像（distroless）可能缺少需要的系统库。
- 你需要通过实际部署环境验证 `Epoll.isAvailable()` 的原因（`Epoll.unavailabilityCause()`）。

### 不要把“用了 epoll”当成性能银弹

Native Transport 只是 I/O 模型的一部分：

- 业务 Handler 阻塞 EventLoop
- 写队列堆积导致背压
- 编解码不合理导致对象频繁创建

这些问题比是否 epoll/kqueue 更常见。

## 小结

- Native Transport 的价值在于：更贴近 OS 的事件通知机制，通常性能更好。
- 推荐做法：**运行时探测 + 自动降级**，同时建立压测基线验证收益。

---

下一篇建议阅读：

- [背压与可写性控制](/docs/netty/backpressure)
