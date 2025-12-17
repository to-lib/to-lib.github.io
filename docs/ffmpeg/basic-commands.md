---
sidebar_position: 3
title: 基础命令
description: FFmpeg 基础命令语法和常用操作
---

# FFmpeg 基础命令

本文介绍 FFmpeg 的基础命令语法和常用操作。

## 命令基本结构

```bash
ffmpeg [全局选项] [输入选项] -i 输入文件 [输出选项] 输出文件
```

### 常用全局选项

| 选项           | 说明           |
| -------------- | -------------- |
| `-y`           | 覆盖输出文件   |
| `-n`           | 不覆盖输出文件 |
| `-v level`     | 设置日志级别   |
| `-hide_banner` | 隐藏版本信息   |

## 格式转换

### 基本转换

```bash
# MP4 转 AVI
ffmpeg -i input.mp4 output.avi

# MP4 转 MKV
ffmpeg -i input.mp4 output.mkv

# 视频转 GIF
ffmpeg -i input.mp4 output.gif
```

### 指定编解码器

```bash
# 使用 H.264 编码
ffmpeg -i input.mp4 -c:v libx264 output.mp4

# 使用 H.265/HEVC 编码
ffmpeg -i input.mp4 -c:v libx265 output.mp4

# 复制流（不重新编码）
ffmpeg -i input.mp4 -c copy output.mkv
```

## 视频基本操作

### 调整分辨率

```bash
# 指定分辨率
ffmpeg -i input.mp4 -vf scale=1920:1080 output.mp4

# 保持宽高比缩放
ffmpeg -i input.mp4 -vf scale=1280:-1 output.mp4

# 缩放到一半
ffmpeg -i input.mp4 -vf scale=iw/2:ih/2 output.mp4
```

### 调整帧率

```bash
# 设置为 30fps
ffmpeg -i input.mp4 -r 30 output.mp4

# 设置为 24fps
ffmpeg -i input.mp4 -filter:v fps=24 output.mp4
```

### 视频裁剪

```bash
# 裁剪视频片段（从第 10 秒开始，持续 30 秒）
ffmpeg -i input.mp4 -ss 00:00:10 -t 00:00:30 -c copy output.mp4

# 从第 1 分钟开始到第 2 分钟
ffmpeg -i input.mp4 -ss 00:01:00 -to 00:02:00 -c copy output.mp4
```

### 视频拼接

```bash
# 创建文件列表
echo "file 'video1.mp4'" > list.txt
echo "file 'video2.mp4'" >> list.txt
echo "file 'video3.mp4'" >> list.txt

# 拼接视频
ffmpeg -f concat -safe 0 -i list.txt -c copy output.mp4
```

## 音频基本操作

### 提取音频

```bash
# 提取为 MP3
ffmpeg -i input.mp4 -vn -acodec mp3 output.mp3

# 提取为 AAC
ffmpeg -i input.mp4 -vn -acodec aac output.aac

# 保持原始音频格式
ffmpeg -i input.mp4 -vn -acodec copy output.aac
```

### 音频转换

```bash
# WAV 转 MP3
ffmpeg -i input.wav -acodec mp3 -ab 192k output.mp3

# 调整采样率
ffmpeg -i input.mp3 -ar 44100 output.mp3

# 调整比特率
ffmpeg -i input.mp3 -ab 320k output.mp3
```

### 调整音量

```bash
# 音量加倍
ffmpeg -i input.mp4 -af "volume=2.0" output.mp4

# 音量减半
ffmpeg -i input.mp4 -af "volume=0.5" output.mp4

# 音量增加 6dB
ffmpeg -i input.mp4 -af "volume=6dB" output.mp4
```

## 媒体信息查看

### 使用 ffprobe

```bash
# 查看基本信息
ffprobe input.mp4

# JSON 格式输出
ffprobe -v quiet -print_format json -show_format -show_streams input.mp4

# 只显示流信息
ffprobe -show_streams input.mp4

# 只显示格式信息
ffprobe -show_format input.mp4
```

## 图片处理

### 视频转图片序列

```bash
# 每秒提取一帧
ffmpeg -i input.mp4 -r 1 output_%04d.png

# 提取所有帧
ffmpeg -i input.mp4 output_%04d.png

# 提取特定时间的帧
ffmpeg -i input.mp4 -ss 00:00:10 -frames:v 1 output.png
```

### 图片序列转视频

```bash
# 图片序列转视频
ffmpeg -framerate 24 -i image_%04d.png -c:v libx264 output.mp4

# 指定开始编号
ffmpeg -framerate 24 -start_number 100 -i image_%04d.png output.mp4
```

## 屏幕录制

### macOS

```bash
# 录制屏幕
ffmpeg -f avfoundation -i "1" output.mp4

# 列出可用设备
ffmpeg -f avfoundation -list_devices true -i ""
```

### Linux

```bash
# 录制屏幕
ffmpeg -f x11grab -s 1920x1080 -i :0.0 output.mp4

# 录制指定区域
ffmpeg -f x11grab -s 800x600 -i :0.0+100,200 output.mp4
```

### Windows

```bash
# 使用 GDI 录制屏幕
ffmpeg -f gdigrab -i desktop output.mp4

# 录制特定窗口
ffmpeg -f gdigrab -i title="窗口标题" output.mp4
```

## 常用参数速查

| 参数   | 说明       | 示例           |
| ------ | ---------- | -------------- |
| `-c:v` | 视频编码器 | `-c:v libx264` |
| `-c:a` | 音频编码器 | `-c:a aac`     |
| `-b:v` | 视频比特率 | `-b:v 5M`      |
| `-b:a` | 音频比特率 | `-b:a 192k`    |
| `-r`   | 帧率       | `-r 30`        |
| `-s`   | 分辨率     | `-s 1920x1080` |
| `-ss`  | 开始时间   | `-ss 00:01:00` |
| `-t`   | 持续时间   | `-t 00:00:30`  |
| `-to`  | 结束时间   | `-to 00:02:00` |
| `-vn`  | 禁用视频   | -              |
| `-an`  | 禁用音频   | -              |
