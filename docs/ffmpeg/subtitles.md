---
sidebar_position: 10
title: 字幕处理
description: 提取、转换、烧录字幕以及多字幕轨道处理
---

# 字幕处理

FFmpeg 可以处理常见字幕格式（SRT/ASS/WebVTT/PGS 等），支持：

- 提取字幕轨
- 转换字幕格式
- 将字幕“烧录”（hardcode）到视频
- 将字幕作为独立轨道封装进容器（softsub）

## 查看字幕轨道

```bash
# 列出所有字幕流（mkv 常见）
ffprobe -v error -select_streams s -show_streams input.mkv

# 查看字幕语言/标题
ffprobe -v error -show_entries stream=index,codec_type,codec_name:stream_tags=language,title \
  -of default=nw=1 input.mkv
```

## 提取字幕

### 提取为 SRT（文本字幕）

```bash
# 提取第 1 条字幕流
ffmpeg -i input.mkv -map 0:s:0 out.srt
```

如果是图形字幕（例如 PGS/HDMV），无法直接转成 SRT，需要 OCR 工具。

### 提取为 ASS

```bash
ffmpeg -i input.mkv -map 0:s:0 out.ass
```

## 字幕格式转换

```bash
# SRT -> WebVTT
ffmpeg -i in.srt out.vtt

# SRT -> ASS
ffmpeg -i in.srt out.ass

# ASS -> SRT（可能丢失样式）
ffmpeg -i in.ass out.srt
```

## 软字幕：封装字幕轨到容器

### MP4 + mov_text

MP4 容器通常使用 `mov_text` 字幕轨（不是 SRT 原样封装）。

```bash
# 将 srt 作为字幕轨写入 mp4
ffmpeg -i input.mp4 -i sub.srt \
  -c:v copy -c:a copy \
  -c:s mov_text \
  -metadata:s:s:0 language=chi \
  out.mp4
```

### MKV + SRT/ASS

MKV 对字幕支持更好，可以直接封装 `srt` / `ass`。

```bash
ffmpeg -i input.mp4 -i sub.ass \
  -c:v copy -c:a copy -c:s copy \
  -metadata:s:s:0 language=chi \
  out.mkv
```

## 硬字幕：烧录字幕到画面（Hardcode）

适合：目标播放器不支持字幕轨，或你需要固定字幕样式。

### 烧录 SRT

```bash
# Windows 路径注意反斜杠转义
ffmpeg -i input.mp4 -vf "subtitles=sub.srt" -c:a copy out.mp4
```

### 烧录 ASS（保留样式）

```bash
ffmpeg -i input.mp4 -vf "ass=sub.ass" -c:a copy out.mp4
```

### 字体与样式

字幕滤镜依赖字体。Linux 上建议安装常用字体并配置 fontconfig；macOS/Windows 通常会自动发现系统字体。

当你需要强制字体时，可以：

- 在 ASS 文件里指定字体
- 或在 `subtitles`/`ass` 滤镜参数中指定 fontsdir

```bash
ffmpeg -i input.mp4 -vf "subtitles=sub.srt:fontsdir=./fonts" out.mp4
```

## 多字幕轨与默认字幕

```bash
# 添加两条字幕轨
ffmpeg -i input.mp4 -i zh.srt -i en.srt \
  -map 0 -map 1 -map 2 \
  -c copy -c:s mov_text \
  -metadata:s:s:0 language=chi \
  -metadata:s:s:1 language=eng \
  out.mp4

# 设置默认字幕（依播放器支持而定）
ffmpeg -i input.mp4 -i zh.srt \
  -c copy -c:s mov_text \
  -disposition:s:0 default \
  out.mp4
```

## 常见问题

- 软字幕不显示：检查播放器是否支持该容器/字幕编码（例如 MP4+SRT 直接封装通常不可行）。
- 烧录字幕报错：确认文件编码（建议 UTF-8）、路径是否包含空格、字体是否存在。
