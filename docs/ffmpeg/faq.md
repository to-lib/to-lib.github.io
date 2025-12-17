---
sidebar_position: 9
title: 常见问题
description: FFmpeg 常见问题与解决方案
---

# FFmpeg 常见问题

本文汇总 FFmpeg 使用过程中的常见问题和解决方案。

## 安装问题

### Q: 提示 "ffmpeg: command not found"？

**A:** FFmpeg 未正确安装或未添加到 PATH。

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# 验证安装
which ffmpeg
ffmpeg -version
```

### Q: 缺少某些编解码器？

**A:** 安装完整版或从源码编译。

```bash
# Ubuntu 安装额外编解码器
sudo apt install libavcodec-extra

# macOS 完整安装
brew install ffmpeg

# 查看支持的编解码器
ffmpeg -codecs
```

## 编码问题

### Q: "Unknown encoder 'libx264'"？

**A:** 系统未安装 x264 库。

```bash
# Ubuntu
sudo apt install libx264-dev
# 重新编译 FFmpeg

# macOS（Homebrew 版本默认包含）
brew reinstall ffmpeg
```

### Q: 输出文件体积过大？

**A:** 调整编码参数。

```bash
# 使用 CRF 控制质量（推荐 23-28）
ffmpeg -i input.mp4 -c:v libx264 -crf 28 output.mp4

# 降低比特率
ffmpeg -i input.mp4 -c:v libx264 -b:v 1M output.mp4

# 降低分辨率
ffmpeg -i input.mp4 -vf scale=1280:-1 -c:v libx264 -crf 23 output.mp4
```

### Q: 编码速度太慢？

**A:** 使用更快的预设或硬件加速。

```bash
# 使用更快的预设
ffmpeg -i input.mp4 -c:v libx264 -preset ultrafast output.mp4

# 使用硬件加速（NVIDIA）
ffmpeg -hwaccel cuda -i input.mp4 -c:v h264_nvenc output.mp4

# 使用硬件加速（macOS）
ffmpeg -i input.mp4 -c:v h264_videotoolbox output.mp4
```

## 格式问题

### Q: "Could not find codec parameters"？

**A:** 输入文件可能损坏或格式不支持。

```bash
# 检查文件信息
ffprobe input.mp4

# 尝试修复
ffmpeg -err_detect ignore_err -i input.mp4 -c copy output.mp4
```

### Q: 输出视频无法播放？

**A:** 容器格式与编码不兼容。

```bash
# 确保 MP4 使用 H.264 + AAC
ffmpeg -i input.avi -c:v libx264 -c:a aac output.mp4

# 添加 faststart 标志（网络播放）
ffmpeg -i input.avi -c:v libx264 -c:a aac -movflags +faststart output.mp4
```

### Q: 音视频不同步？

**A:** 使用适当的同步选项。

```bash
# 重新编码以修复同步
ffmpeg -i input.mp4 -c:v libx264 -c:a aac -async 1 output.mp4

# 使用 -vsync 参数
ffmpeg -i input.mp4 -c:v libx264 -vsync cfr output.mp4
```

## 裁剪问题

### Q: 裁剪后视频开头有问题？

**A:** 将 `-ss` 放在输入之前进行输入端 seeking。

```bash
# 推荐方式：输入端 seeking（更精确）
ffmpeg -ss 00:01:00 -i input.mp4 -t 00:00:30 -c copy output.mp4

# 如果还有问题，重新编码
ffmpeg -ss 00:01:00 -i input.mp4 -t 00:00:30 -c:v libx264 -c:a aac output.mp4
```

### Q: 裁剪不精确？

**A:** 使用关键帧或重新编码。

```bash
# 精确裁剪（需要重新编码）
ffmpeg -i input.mp4 -ss 00:01:00 -to 00:02:00 \
    -c:v libx264 -c:a aac output.mp4

# 复制流模式只能从关键帧开始
```

## 滤镜问题

### Q: "No such filter: xxx"？

**A:** FFmpeg 版本不支持该滤镜。

```bash
# 列出所有可用滤镜
ffmpeg -filters

# 更新 FFmpeg 到最新版本
brew upgrade ffmpeg  # macOS
sudo apt update && sudo apt upgrade ffmpeg  # Ubuntu
```

### Q: 滤镜执行速度慢？

**A:** 优化滤镜链或降低输出分辨率。

```bash
# 先缩放再处理
ffmpeg -i input.mp4 -vf "scale=640:-1,滤镜" output.mp4

# 使用硬件加速滤镜（如果支持）
ffmpeg -hwaccel cuda -i input.mp4 -vf "scale_cuda=1280:720" output.mp4
```

## 流媒体问题

### Q: RTMP 推流失败？

**A:** 检查 URL 和网络连接。

```bash
# 测试连接
ffmpeg -re -i input.mp4 -c copy -f flv rtmp://server/live/stream

# 增加超时时间
ffmpeg -re -i input.mp4 -c copy -timeout 10000000 -f flv rtmp://server/live/stream
```

### Q: 直播延迟高？

**A:** 使用低延迟参数。

```bash
ffmpeg -re -i input.mp4 \
    -c:v libx264 -preset ultrafast -tune zerolatency \
    -g 30 -keyint_min 30 \
    -f flv rtmp://server/live/stream
```

## 性能问题

### Q: CPU 占用过高？

**A:** 使用硬件加速或降低质量。

```bash
# 使用硬件编码
ffmpeg -i input.mp4 -c:v h264_nvenc output.mp4  # NVIDIA
ffmpeg -i input.mp4 -c:v h264_videotoolbox output.mp4  # macOS

# 使用多线程
ffmpeg -threads 4 -i input.mp4 -c:v libx264 output.mp4
```

### Q: 内存占用过高？

**A:** 分段处理或降低缓冲区。

```bash
# 限制线程数
ffmpeg -threads 2 -i input.mp4 output.mp4

# 处理超大文件时分段
ffmpeg -ss 00:00:00 -t 00:30:00 -i large.mp4 -c copy part1.mp4
ffmpeg -ss 00:30:00 -t 00:30:00 -i large.mp4 -c copy part2.mp4
```

## 其他问题

### Q: 如何查看详细错误信息？

**A:** 使用详细日志级别。

```bash
# 显示详细信息
ffmpeg -v verbose -i input.mp4 output.mp4

# 显示所有信息
ffmpeg -v debug -i input.mp4 output.mp4

# 隐藏版本信息
ffmpeg -hide_banner -i input.mp4 output.mp4
```

### Q: 如何批量处理文件？

**A:** 使用 shell 脚本。

```bash
# 批量转换
for f in *.avi; do
    ffmpeg -i "$f" -c:v libx264 -c:a aac "${f%.avi}.mp4"
done

# 使用 find
find . -name "*.mov" -exec ffmpeg -i {} -c:v libx264 {}.mp4 \;
```
