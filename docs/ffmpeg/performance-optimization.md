---
sidebar_position: 17
title: 性能优化
description: 提升转码速度、降低资源占用、定位性能瓶颈的实用方法
---

# 性能优化

FFmpeg 性能问题通常来自：

- 解码/编码开销（软件编码最常见）
- 滤镜链太重（例如降噪、复杂叠加）
- 磁盘 IO / 网络 IO 不足
- 并发任务过多导致争抢资源

这页提供一套通用的优化思路与参数模板。

## 1) 先确认瓶颈在哪里

- CPU 满：常见于 `libx264/libx265` 或复杂滤镜
- GPU 满：硬件编码/滤镜（若使用）
- 磁盘满：高码率素材读写、图片序列
- 网络满：拉流/推流

## 2) 选择合适的编码器与 preset

```bash
# 更快编码（牺牲部分压缩效率）
ffmpeg -i input.mp4 -c:v libx264 -preset veryfast -crf 23 -c:a aac out.mp4

# 极致速度（适合临时预览）
ffmpeg -i input.mp4 -c:v libx264 -preset ultrafast -crf 23 -c:a aac out.mp4
```

## 3) 降低分辨率/帧率往往更有效

```bash
# 降到 720p
ffmpeg -i input.mp4 -vf scale=-2:720 -c:v libx264 -crf 23 -preset medium -c:a aac out.mp4

# 降到 30fps
ffmpeg -i input.mp4 -vf fps=30 -c:v libx264 -crf 23 -preset medium -c:a aac out.mp4
```

## 4) 尽量减少滤镜链成本

- 能用 `-c copy` 就别重编码
- 多个滤镜尽量合并为一条 `-vf`（避免不必要的格式转换）
- 重滤镜（降噪、插帧）先在低分辨率上测试

```bash
# 先缩放再降噪，往往更快
ffmpeg -i input.mp4 -vf "scale=-2:720,hqdn3d=4:4:6:6" -c:v libx264 -crf 23 -c:a aac out.mp4
```

## 5) 多线程与线程数

多数场景 FFmpeg 会自动使用多线程，但在并发跑多个任务时，反而可能导致整体吞吐下降。

```bash
# 限制线程数（并发跑多个转码任务时很有用）
ffmpeg -threads 4 -i input.mp4 -c:v libx264 -crf 23 -c:a aac out.mp4
```

## 6) 使用硬件编码（实时/批量时常用）

硬件编码优势在速度，但同体积/同码率下画质通常不如 `libx264/libx265`。

```bash
# NVIDIA
ffmpeg -i input.mp4 -c:v h264_nvenc -preset p4 -c:a aac out.mp4

# macOS
ffmpeg -i input.mp4 -c:v h264_videotoolbox -b:v 4M -c:a aac out.mp4
```

## 7) IO 优化建议

- 图片序列转码：尽量用 `-pattern_type glob` 或保证连续编号，避免频繁 stat
- 大文件转码：输出到本地 SSD，再拷贝到网络盘
- 拉流/推流：先在本地落盘验证参数

## 8) 排查“看似慢但其实在等 IO”

如果你观察到 CPU 并不高但速度仍很慢：

- 检查输入是否在网络盘
- 检查输出磁盘是否写满/写入速度不足
- 检查是否在生成大量小文件（例如帧提取）
