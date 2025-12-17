---
sidebar_position: 12
title: ffprobe 分析
description: 使用 ffprobe 解析媒体流、提取元信息与排查编码问题
---

# ffprobe 分析

`ffprobe` 是 FFmpeg 套件中的媒体分析工具，用于读取容器/码流信息、列出音视频/字幕轨道、打印关键元数据，适合用于排障和写自动化脚本。

## 常用参数

```bash
# 隐藏 banner，减少噪音
ffprobe -hide_banner input.mp4

# 更安静的输出（只在出错时输出）
ffprobe -v error input.mp4

# 帮助
ffprobe -h
ffprobe -h full
```

## 查看容器与轨道信息

```bash
# 展示 format + streams
ffprobe -hide_banner -show_format -show_streams input.mp4

# 只展示 streams
ffprobe -hide_banner -show_streams input.mp4

# 只展示 format
ffprobe -hide_banner -show_format input.mp4
```

### 以 JSON 输出（推荐用于脚本）

```bash
ffprobe -v error \
  -print_format json \
  -show_format -show_streams \
  input.mp4
```

## 精确选择流（select_streams）

```bash
# 选择第 1 路视频流
ffprobe -v error -select_streams v:0 -show_streams input.mp4

# 选择所有音频流
ffprobe -v error -select_streams a -show_streams input.mkv

# 选择字幕流
ffprobe -v error -select_streams s -show_streams input.mkv
```

## 提取关键字段（show_entries）

```bash
# 只要宽高、帧率、像素格式
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,avg_frame_rate,pix_fmt,profile,level \
  -of default=noprint_wrappers=1 input.mp4

# 只要时长与码率
ffprobe -v error \
  -show_entries format=duration,bit_rate,size,format_name \
  -of default=noprint_wrappers=1 input.mp4
```

### 输出为 CSV / TSV

```bash
# CSV
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,codec_name \
  -of csv=p=0 input.mp4

# TSV
ffprobe -v error -select_streams a:0 \
  -show_entries stream=sample_rate,channels,codec_name \
  -of tsv=p=0 input.mp4
```

## GOP / 关键帧分析

```bash
# 列出关键帧时间戳（CSV）
ffprobe -v error -select_streams v:0 \
  -show_frames \
  -show_entries frame=pkt_pts_time,pict_type,key_frame \
  -of csv=p=0 input.mp4 | grep ",I,"
```

## 旋转信息（手机视频常见）

某些手机视频是通过 metadata 标记旋转角度。

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream_tags=rotate \
  -of default=nw=1 input.mp4
```

## 常见排障用法

### 1) 播放器无法播放 / 兼容性问题

优先确认：

- 容器：`format_name`
- 视频编码：`codec_name`
- 音频编码：`codec_name`
- 像素格式：`pix_fmt`

```bash
ffprobe -v error -show_entries format=format_name -of default=nw=1 input.mp4
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,pix_fmt,profile -of default=nw=1 input.mp4
ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,channels,sample_rate -of default=nw=1 input.mp4
```

### 2) 多音轨 / 多字幕轨，确认轨道语言

```bash
ffprobe -v error -show_entries stream=index,codec_type,codec_name:stream_tags=language,title \
  -of default=nw=1 input.mkv
```

## 与 ffmpeg 的衔接

- 先用 `ffprobe` 确定流序号（index）和轨道类型
- 再用 `ffmpeg -map` 精确选择输出

```bash
# 只导出第 1 路视频 + 第 2 路音频
ffmpeg -i input.mkv -map 0:v:0 -map 0:a:1 -c copy out.mkv
```
