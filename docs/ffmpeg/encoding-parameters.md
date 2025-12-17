---
sidebar_position: 13
title: 编码参数与码控
description: CRF/ABR/2pass、GOP、像素格式、faststart 等常用编码策略
---

# 编码参数与码控

这页聚焦“怎么把视频压得更小、同时保持可接受画质”，以及分发常见兼容性参数。

## 质量优先：CRF（推荐）

以 `libx264` 为例：

```bash
# 常见通用配置
ffmpeg -i input.mp4 \
  -c:v libx264 -crf 23 -preset medium \
  -c:a aac -b:a 128k \
  out.mp4
```

- `-crf`：0-51，越小质量越好、体积越大
- `-preset`：速度/压缩率权衡，越慢压得越小

常用经验：

- 视觉无明显损失：`crf 18-20`
- 综合平衡：`crf 21-24`
- 明显压缩：`crf 25-28`

## 码率优先：ABR / CBR

### 目标码率

```bash
ffmpeg -i input.mp4 -c:v libx264 -b:v 2M -maxrate 2.5M -bufsize 5M -c:a aac -b:a 128k out.mp4
```

- `-b:v`：平均目标码率
- `-maxrate` / `-bufsize`：VBV，限制瞬时码率，适合直播/严格带宽

### 两遍编码（2-pass）

适用于：给定文件大小/码率预算时提升质量。

```bash
# pass 1
ffmpeg -y -i input.mp4 -c:v libx264 -b:v 2M -pass 1 -an -f null /dev/null

# pass 2
ffmpeg -i input.mp4 -c:v libx264 -b:v 2M -pass 2 -c:a aac -b:a 128k out.mp4
```

## GOP 与关键帧

- `-g`：关键帧间隔（单位：帧）
- 对直播/快进拖动/切片都很重要

```bash
# 30fps 下每 2 秒一个关键帧
ffmpeg -i input.mp4 -c:v libx264 -g 60 -keyint_min 60 -sc_threshold 0 -c:a aac out.mp4
```

## 像素格式与兼容性

最通用的像素格式通常是 `yuv420p`。

```bash
# 强制 yuv420p，提升老设备兼容性
ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p -c:a aac out.mp4
```

## Web 分发：faststart

MP4 默认可能把 moov 放在文件尾，导致边下边播体验差。

```bash
ffmpeg -i input.mp4 -c copy -movflags +faststart out.mp4
```

## 分辨率/帧率与体积

降低分辨率通常比盲目提高 `crf` 更有效。

```bash
# 降到 720p
ffmpeg -i input.mp4 -vf scale=-2:720 -c:v libx264 -crf 23 -preset medium -c:a aac out.mp4
```

## 音频参数建议

- 一般分发：AAC `128k` / `160k`
- 语音为主：`96k` 也可

```bash
ffmpeg -i input.mp4 -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k out.mp4
```
