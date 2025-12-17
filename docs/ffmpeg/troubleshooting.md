---
sidebar_position: 16
title: 排错指南
description: 常见错误定位思路、日志阅读、时间戳/音画同步/兼容性排查
---

# 排错指南

FFmpeg 的报错信息很多，但大部分都能归类到：

- 输入源问题（损坏、时间戳异常、编码参数缺失）
- 编码器/封装不兼容
- 过滤器链参数错误
- 资源不足（CPU/GPU/内存/磁盘 IO）

## 基本排错流程

- 先用 `ffprobe` 看清楚输入
- 再用最小参数重现（减少滤镜/参数）
- 最后逐步添加参数，定位是哪一段导致失败

## 日志与常用调试参数

```bash
# 更详细日志
ffmpeg -v verbose -i input.mp4 out.mp4

# 最详细
ffmpeg -v debug -i input.mp4 out.mp4

# 隐藏 banner，让输出更干净
ffmpeg -hide_banner -i input.mp4 out.mp4

# 只输出错误
ffmpeg -v error -i input.mp4 out.mp4
```

## 输入文件排查

### 1) 输入是否损坏 / 是否缺少关键参数

```bash
ffprobe -v error -show_format -show_streams input.mp4
```

### 2) 时间戳异常（常见于录屏/拉流保存）

```bash
# 尝试生成新的时间戳
ffmpeg -fflags +genpts -i input.mp4 -c copy out.mp4
```

## “Unknown encoder …” / 编码器不可用

原因：你的 FFmpeg 构建没有带该编码器，或环境缺依赖。

```bash
# 列出所有编码器
ffmpeg -hide_banner -encoders

# 查找特定编码器
ffmpeg -hide_banner -encoders | grep x264
```

解决：安装/替换带该编码器的 FFmpeg，或改用可用编码器。

## 输出无法播放 / 兼容性问题

优先保证：

- MP4：H.264 + AAC + `yuv420p`
- Web 播放：加 `-movflags +faststart`

```bash
ffmpeg -i input.mkv \
  -c:v libx264 -pix_fmt yuv420p -crf 23 -preset medium \
  -c:a aac -b:a 128k \
  -movflags +faststart \
  out.mp4
```

## 音画不同步

常见触发点：变速/拼接/切片/拉流。

```bash
# 尝试重新编码音频并启用异步采样
ffmpeg -i input.mp4 -c:v copy -c:a aac -af aresample=async=1 out.mp4

# 以 CFR 输出（让视频帧率恒定）
ffmpeg -i input.mp4 -vsync cfr -c:v libx264 -crf 23 -c:a aac out.mp4
```

如果是“无损切片”导致的不同步，优先考虑重编码。

## “Non-monotonous DTS”

原因：时间戳不单调递增，常见于拼接、流转封装。

```bash
# 尝试生成/修复时间戳
ffmpeg -fflags +genpts -i input.mp4 -c copy out.mp4

# 或尝试让输出时间戳从 0 开始
ffmpeg -i input.mp4 -c copy -reset_timestamps 1 out.mp4
```

## 滤镜报错（No such filter / Invalid argument）

```bash
# 查看滤镜是否存在
ffmpeg -hide_banner -filters | grep drawtext

# 查看滤镜帮助
ffmpeg -h filter=drawtext
```

## 性能问题

- 确认瓶颈：CPU / GPU / 磁盘 IO / 网络
- 尝试：降低分辨率、降低帧率、使用更快的 `-preset`、硬件编码

```bash
# 更快 preset
ffmpeg -i input.mp4 -c:v libx264 -preset veryfast -crf 23 -c:a aac out.mp4
```
