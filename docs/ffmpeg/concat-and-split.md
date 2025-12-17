---
sidebar_position: 11
title: 拼接与切片
description: concat 拼接、无损切片、分段、关键帧与常见坑
---

# 拼接与切片

日常处理里，“拼接/切片”是最容易踩坑的地方：

- **无损剪切/拼接**（`-c copy`）速度快，但受关键帧限制
- **精确剪切/拼接** 通常需要重新编码

## 无损切片（快速）

### 输入端 seeking（更快）

```bash
# 从 10s 开始，截取 30s（无损）
ffmpeg -ss 00:00:10 -i input.mp4 -t 00:00:30 -c copy out.mp4
```

### 输出端 seeking（更精确但更慢）

```bash
ffmpeg -i input.mp4 -ss 00:00:10 -t 00:00:30 -c copy out.mp4
```

注意：无损模式只能从关键帧附近开始，可能导致开头不精确、花屏或音画问题。

## 精确切片（推荐用于严格时间点）

```bash
ffmpeg -i input.mp4 -ss 00:00:10 -t 00:00:30 \
  -c:v libx264 -crf 23 -preset medium \
  -c:a aac -b:a 128k \
  out.mp4
```

## 拼接：concat demuxer（同编码参数下无损）

适用于：同分辨率、同编码器、同参数、同 timebase 的文件。

1) 创建列表文件：

```bash
cat > list.txt << EOF
file 'part1.mp4'
file 'part2.mp4'
file 'part3.mp4'
EOF
```

2) 拼接：

```bash
ffmpeg -f concat -safe 0 -i list.txt -c copy out.mp4
```

如果失败，通常是“参数不一致”。可以考虑统一转码后再拼。

## 拼接：concat filter（可处理不同参数，但要重编码）

```bash
ffmpeg -i a.mp4 -i b.mp4 \
  -filter_complex "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" \
  -c:v libx264 -crf 23 -preset medium -c:a aac \
  out.mp4
```

## 分段：按时长切 HLS/VOD 风格小文件

```bash
# 每 10 秒切一段（不保证关键帧对齐，可能触发重封装问题）
ffmpeg -i input.mp4 -c copy -f segment -segment_time 10 -reset_timestamps 1 out_%03d.mp4
```

更稳妥的方式：先强制关键帧再分段（需要重编码）。

```bash
ffmpeg -i input.mp4 \
  -c:v libx264 -crf 23 -preset medium \
  -force_key_frames "expr:gte(t,n_forced*10)" \
  -c:a aac \
  -f segment -segment_time 10 -reset_timestamps 1 out_%03d.mp4
```

## 常见坑

- **花屏/绿屏**：常见于无损剪切切到非关键帧附近。
- **音画不同步**：切片后容器时间戳异常，可尝试重编码或 `-reset_timestamps 1`。
- **拼接失败**：文件参数不一致；用 `ffprobe` 检查宽高、编码器、profile、time_base。
