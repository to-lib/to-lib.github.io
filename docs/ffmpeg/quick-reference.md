---
sidebar_position: 8
title: 快速参考
description: FFmpeg 常用命令速查表
---

# FFmpeg 快速参考

本文提供 FFmpeg 常用命令的速查表。

## 格式转换

| 操作       | 命令                                                       |
| ---------- | ---------------------------------------------------------- |
| MP4 转 AVI | `ffmpeg -i input.mp4 output.avi`                           |
| MP4 转 MKV | `ffmpeg -i input.mp4 output.mkv`                           |
| 视频转 GIF | `ffmpeg -i input.mp4 -vf "fps=10,scale=320:-1" output.gif` |
| WAV 转 MP3 | `ffmpeg -i input.wav -c:a libmp3lame -b:a 320k output.mp3` |

## 视频操作

| 操作       | 命令                                                              |
| ---------- | ----------------------------------------------------------------- |
| 调整分辨率 | `ffmpeg -i input.mp4 -vf scale=1280:720 output.mp4`               |
| 调整帧率   | `ffmpeg -i input.mp4 -r 30 output.mp4`                            |
| 裁剪时间   | `ffmpeg -ss 00:00:10 -i input.mp4 -t 00:00:30 -c copy output.mp4` |
| 去除音频   | `ffmpeg -i input.mp4 -an output.mp4`                              |
| 旋转 90°   | `ffmpeg -i input.mp4 -vf "transpose=1" output.mp4`                |
| 水平翻转   | `ffmpeg -i input.mp4 -vf "hflip" output.mp4`                      |
| 压缩视频   | `ffmpeg -i input.mp4 -c:v libx264 -crf 28 output.mp4`             |

## 音频操作

| 操作           | 命令                                                  |
| -------------- | ----------------------------------------------------- |
| 提取音频       | `ffmpeg -i video.mp4 -vn -c:a copy audio.aac`         |
| 调整音量       | `ffmpeg -i input.mp3 -af "volume=2.0" output.mp3`     |
| 立体声转单声道 | `ffmpeg -i input.mp3 -ac 1 output.mp3`                |
| 调整采样率     | `ffmpeg -i input.mp3 -ar 44100 output.mp3`            |
| 音频淡入       | `ffmpeg -i input.mp3 -af "afade=t=in:d=3" output.mp3` |

## 视频合成

| 操作     | 命令                                                                               |
| -------- | ---------------------------------------------------------------------------------- |
| 添加水印 | `ffmpeg -i input.mp4 -i logo.png -filter_complex "overlay=10:10" output.mp4`       |
| 画中画   | `ffmpeg -i main.mp4 -i pip.mp4 -filter_complex "overlay=W-w-10:H-h-10" output.mp4` |
| 左右并排 | `ffmpeg -i left.mp4 -i right.mp4 -filter_complex "hstack" output.mp4`              |
| 上下并排 | `ffmpeg -i top.mp4 -i bottom.mp4 -filter_complex "vstack" output.mp4`              |

## 文字水印

```bash
# 添加文字
ffmpeg -i input.mp4 -vf "drawtext=text='Hello':fontsize=48:fontcolor=white:x=10:y=10" output.mp4

# 添加时间戳
ffmpeg -i input.mp4 -vf "drawtext=text='%{localtime}':fontsize=24:fontcolor=white:x=10:y=10" output.mp4
```

## 图片处理

| 操作       | 命令                                                                 |
| ---------- | -------------------------------------------------------------------- |
| 视频截图   | `ffmpeg -i input.mp4 -ss 00:00:10 -frames:v 1 output.png`            |
| 每秒截图   | `ffmpeg -i input.mp4 -r 1 output_%04d.png`                           |
| 图片转视频 | `ffmpeg -framerate 24 -i image_%04d.png output.mp4`                  |
| 缩略图网格 | `ffmpeg -i input.mp4 -vf "fps=1/10,scale=160:-1,tile=5x5" thumb.png` |

## 流媒体

| 操作      | 命令                                                               |
| --------- | ------------------------------------------------------------------ |
| RTMP 推流 | `ffmpeg -re -i input.mp4 -c copy -f flv rtmp://server/live/stream` |
| 拉流保存  | `ffmpeg -i rtmp://server/live/stream -c copy output.mp4`           |
| 生成 HLS  | `ffmpeg -i input.mp4 -f hls -hls_time 4 output.m3u8`               |

## 媒体信息

```bash
# 查看文件信息
ffprobe input.mp4

# JSON 格式输出
ffprobe -v quiet -print_format json -show_format -show_streams input.mp4

# 只看视频流
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,duration -of csv=p=0 input.mp4
```

## 常用编解码器

| 类型 | 编码器       | 说明             |
| ---- | ------------ | ---------------- |
| 视频 | `libx264`    | H.264 编码       |
| 视频 | `libx265`    | H.265/HEVC 编码  |
| 视频 | `libvpx-vp9` | VP9 编码         |
| 音频 | `aac`        | AAC 编码         |
| 音频 | `libmp3lame` | MP3 编码         |
| 音频 | `libopus`    | Opus 编码        |
| 音频 | `copy`       | 复制流（不编码） |

## 常用参数

| 参数      | 说明       | 示例                  |
| --------- | ---------- | --------------------- |
| `-c:v`    | 视频编码器 | `-c:v libx264`        |
| `-c:a`    | 音频编码器 | `-c:a aac`            |
| `-b:v`    | 视频比特率 | `-b:v 5M`             |
| `-b:a`    | 音频比特率 | `-b:a 192k`           |
| `-r`      | 帧率       | `-r 30`               |
| `-s`      | 分辨率     | `-s 1920x1080`        |
| `-ss`     | 起始时间   | `-ss 00:01:00`        |
| `-t`      | 持续时间   | `-t 00:00:30`         |
| `-to`     | 结束时间   | `-to 00:02:00`        |
| `-vn`     | 禁用视频   |                       |
| `-an`     | 禁用音频   |                       |
| `-y`      | 覆盖输出   |                       |
| `-vf`     | 视频滤镜   | `-vf "scale=1280:-1"` |
| `-af`     | 音频滤镜   | `-af "volume=2.0"`    |
| `-crf`    | 质量因子   | `-crf 23`             |
| `-preset` | 编码预设   | `-preset fast`        |

## CRF 质量参考

| CRF 值 | 质量     | 适用场景  |
| ------ | -------- | --------- |
| 0      | 无损     | 存档      |
| 18-20  | 高质量   | 专业制作  |
| 21-23  | 良好质量 | 一般用途  |
| 24-28  | 较低质量 | 网络分发  |
| 29+    | 低质量   | 预览/测试 |

## 硬件加速

```bash
# 查看支持的硬件加速
ffmpeg -hwaccels

# NVIDIA CUDA 加速
ffmpeg -hwaccel cuda -i input.mp4 output.mp4

# macOS VideoToolbox
ffmpeg -hwaccel videotoolbox -i input.mp4 output.mp4

# Intel QSV
ffmpeg -hwaccel qsv -i input.mp4 output.mp4
```
