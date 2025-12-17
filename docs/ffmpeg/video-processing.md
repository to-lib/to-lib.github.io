---
sidebar_position: 4
title: 视频处理
description: FFmpeg 视频处理高级操作
---

# FFmpeg 视频处理

本文介绍 FFmpeg 的视频处理高级操作，包括编码、裁剪、合并等。

## 视频编码

### H.264 编码

```bash
# 基础 H.264 编码
ffmpeg -i input.mp4 -c:v libx264 -preset medium -crf 23 output.mp4

# 高质量编码
ffmpeg -i input.mp4 -c:v libx264 -preset slow -crf 18 output.mp4

# 快速编码（牺牲质量换速度）
ffmpeg -i input.mp4 -c:v libx264 -preset ultrafast -crf 23 output.mp4
```

**preset 选项**（从快到慢）：

- `ultrafast`, `superfast`, `veryfast`, `faster`, `fast`
- `medium`（默认）
- `slow`, `slower`, `veryslow`

**CRF 值**：0-51，越小质量越高，推荐 18-28

### H.265/HEVC 编码

```bash
# 基础 H.265 编码
ffmpeg -i input.mp4 -c:v libx265 -preset medium -crf 28 output.mp4

# 10-bit 编码
ffmpeg -i input.mp4 -c:v libx265 -pix_fmt yuv420p10le output.mp4
```

### 硬件加速编码

```bash
# NVIDIA NVENC（H.264）
ffmpeg -i input.mp4 -c:v h264_nvenc -preset fast output.mp4

# NVIDIA NVENC（H.265）
ffmpeg -i input.mp4 -c:v hevc_nvenc output.mp4

# macOS VideoToolbox
ffmpeg -i input.mp4 -c:v h264_videotoolbox output.mp4

# Intel QSV
ffmpeg -i input.mp4 -c:v h264_qsv output.mp4
```

## 视频画面处理

### 裁剪画面（Crop）

```bash
# 裁剪中心 640x480 区域
ffmpeg -i input.mp4 -vf "crop=640:480" output.mp4

# 指定起始位置裁剪
ffmpeg -i input.mp4 -vf "crop=640:480:100:50" output.mp4
# 格式：crop=宽:高:x偏移:y偏移

# 裁剪去除黑边
ffmpeg -i input.mp4 -vf "cropdetect" -f null - 2>&1 | grep crop
ffmpeg -i input.mp4 -vf "crop=1920:800:0:140" output.mp4
```

### 填充画面（Pad）

```bash
# 添加黑边到 16:9
ffmpeg -i input.mp4 -vf "pad=1920:1080:(ow-iw)/2:(oh-ih)/2" output.mp4

# 添加彩色边框
ffmpeg -i input.mp4 -vf "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:red" output.mp4
```

### 旋转和翻转

```bash
# 顺时针旋转 90 度
ffmpeg -i input.mp4 -vf "transpose=1" output.mp4

# 逆时针旋转 90 度
ffmpeg -i input.mp4 -vf "transpose=2" output.mp4

# 旋转 180 度
ffmpeg -i input.mp4 -vf "transpose=1,transpose=1" output.mp4

# 水平翻转
ffmpeg -i input.mp4 -vf "hflip" output.mp4

# 垂直翻转
ffmpeg -i input.mp4 -vf "vflip" output.mp4
```

### 调整速度

```bash
# 视频加速 2 倍
ffmpeg -i input.mp4 -vf "setpts=0.5*PTS" output.mp4

# 视频减速 2 倍
ffmpeg -i input.mp4 -vf "setpts=2.0*PTS" output.mp4

# 同时调整音频速度
ffmpeg -i input.mp4 -vf "setpts=0.5*PTS" -af "atempo=2.0" output.mp4
```

## 视频合成

### 画中画

```bash
# 右下角添加小视频
ffmpeg -i main.mp4 -i overlay.mp4 \
    -filter_complex "[0:v][1:v]overlay=W-w-10:H-h-10" \
    output.mp4

# 左上角添加 Logo
ffmpeg -i input.mp4 -i logo.png \
    -filter_complex "[0:v][1:v]overlay=10:10" \
    output.mp4
```

### 多视频并排

```bash
# 左右并排
ffmpeg -i left.mp4 -i right.mp4 \
    -filter_complex "[0:v]scale=640:480[v0];[1:v]scale=640:480[v1];[v0][v1]hstack" \
    output.mp4

# 上下并排
ffmpeg -i top.mp4 -i bottom.mp4 \
    -filter_complex "[0:v]scale=1280:360[v0];[1:v]scale=1280:360[v1];[v0][v1]vstack" \
    output.mp4

# 2x2 网格
ffmpeg -i 1.mp4 -i 2.mp4 -i 3.mp4 -i 4.mp4 \
    -filter_complex "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack" \
    output.mp4
```

### 添加水印

```bash
# 图片水印
ffmpeg -i input.mp4 -i watermark.png \
    -filter_complex "[0:v][1:v]overlay=10:10" \
    output.mp4

# 文字水印
ffmpeg -i input.mp4 \
    -vf "drawtext=text='版权所有':fontsize=24:fontcolor=white:x=10:y=10" \
    output.mp4

# 动态文字水印（显示时间）
ffmpeg -i input.mp4 \
    -vf "drawtext=text='%{localtime\:%Y-%m-%d %H\\\:%M\\\:%S}':fontsize=20:fontcolor=white:x=10:y=10" \
    output.mp4
```

## 视频剪辑

### 精确剪辑

```bash
# 从第 10 秒开始剪辑 30 秒
ffmpeg -ss 00:00:10 -i input.mp4 -t 00:00:30 -c copy output.mp4

# 精确模式（可能需要重新编码）
ffmpeg -i input.mp4 -ss 00:00:10 -t 00:00:30 -c:v libx264 -c:a aac output.mp4
```

### 多段剪辑合并

```bash
# 剪辑多段并合并
ffmpeg -i input.mp4 -filter_complex \
    "[0:v]trim=start=10:end=20,setpts=PTS-STARTPTS[v1]; \
     [0:v]trim=start=30:end=40,setpts=PTS-STARTPTS[v2]; \
     [0:a]atrim=start=10:end=20,asetpts=PTS-STARTPTS[a1]; \
     [0:a]atrim=start=30:end=40,asetpts=PTS-STARTPTS[a2]; \
     [v1][a1][v2][a2]concat=n=2:v=1:a=1[v][a]" \
    -map "[v]" -map "[a]" output.mp4
```

## 格式转换最佳实践

### 转换为 Web 友好格式

```bash
# 转换为 MP4（H.264 + AAC）
ffmpeg -i input.mov \
    -c:v libx264 -preset slow -crf 22 \
    -c:a aac -b:a 128k \
    -movflags +faststart \
    output.mp4
```

### 生成预览图

```bash
# 生成缩略图网格
ffmpeg -i input.mp4 -vf "fps=1/10,scale=320:-1,tile=4x4" thumbnail.png

# 生成视频预览 GIF
ffmpeg -i input.mp4 -vf "fps=10,scale=320:-1:flags=lanczos" -t 5 preview.gif
```
