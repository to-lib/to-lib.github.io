---
sidebar_position: 5
title: 音频处理
description: FFmpeg 音频处理完整指南
---

# FFmpeg 音频处理

本文介绍 FFmpeg 的音频处理功能，包括格式转换、编辑和效果处理。

## 音频格式转换

### 常见格式转换

```bash
# WAV 转 MP3
ffmpeg -i input.wav -c:a libmp3lame -b:a 320k output.mp3

# MP3 转 AAC
ffmpeg -i input.mp3 -c:a aac -b:a 256k output.aac

# FLAC 转 MP3
ffmpeg -i input.flac -c:a libmp3lame -b:a 320k output.mp3

# 任意格式转 WAV
ffmpeg -i input.mp3 output.wav
```

### 音频质量设置

```bash
# MP3 质量（-q:a 0-9，0 最高）
ffmpeg -i input.wav -c:a libmp3lame -q:a 0 output.mp3

# AAC 比特率
ffmpeg -i input.wav -c:a aac -b:a 256k output.aac

# Opus 编码（推荐用于网络传输）
ffmpeg -i input.wav -c:a libopus -b:a 128k output.opus
```

## 从视频提取音频

```bash
# 提取为 MP3
ffmpeg -i video.mp4 -vn -c:a libmp3lame -b:a 192k audio.mp3

# 提取为原始格式（无需重新编码）
ffmpeg -i video.mp4 -vn -c:a copy audio.aac

# 提取所有音轨
ffmpeg -i video.mkv -map 0:a:0 audio1.mp3 -map 0:a:1 audio2.mp3
```

## 音频参数调整

### 调整采样率

```bash
# 转换为 44.1kHz
ffmpeg -i input.wav -ar 44100 output.wav

# 转换为 48kHz
ffmpeg -i input.wav -ar 48000 output.wav
```

### 调整声道

```bash
# 立体声转单声道
ffmpeg -i input.mp3 -ac 1 output.mp3

# 单声道转立体声
ffmpeg -i input.mp3 -ac 2 output.mp3

# 5.1 声道转立体声
ffmpeg -i input.ac3 -ac 2 output.mp3
```

### 调整音量

```bash
# 调整为原来的 2 倍
ffmpeg -i input.mp3 -af "volume=2.0" output.mp3

# 降低 50%
ffmpeg -i input.mp3 -af "volume=0.5" output.mp3

# 增加 6dB
ffmpeg -i input.mp3 -af "volume=6dB" output.mp3

# 音量标准化
ffmpeg -i input.mp3 -af "loudnorm" output.mp3
```

## 音频剪辑

### 裁剪音频

```bash
# 从第 30 秒开始，持续 60 秒
ffmpeg -i input.mp3 -ss 00:00:30 -t 00:01:00 -c copy output.mp3

# 从开始到第 2 分钟
ffmpeg -i input.mp3 -to 00:02:00 -c copy output.mp3

# 去掉开头 10 秒
ffmpeg -i input.mp3 -ss 00:00:10 -c copy output.mp3
```

### 音频合并

```bash
# 创建文件列表
cat > list.txt << EOF
file 'audio1.mp3'
file 'audio2.mp3'
file 'audio3.mp3'
EOF

# 合并音频
ffmpeg -f concat -safe 0 -i list.txt -c copy output.mp3
```

### 音频混合

```bash
# 混合两个音频（同时播放）
ffmpeg -i audio1.mp3 -i audio2.mp3 \
    -filter_complex "[0:a][1:a]amix=inputs=2:duration=longest" \
    output.mp3

# 音频叠加（背景音乐）
ffmpeg -i voice.mp3 -i bgm.mp3 \
    -filter_complex "[1:a]volume=0.3[bgm];[0:a][bgm]amix=inputs=2:duration=first" \
    output.mp3
```

## 音频效果

### 淡入淡出

```bash
# 淡入 3 秒
ffmpeg -i input.mp3 -af "afade=t=in:ss=0:d=3" output.mp3

# 淡出 3 秒
ffmpeg -i input.mp3 -af "afade=t=out:st=57:d=3" output.mp3

# 同时淡入淡出
ffmpeg -i input.mp3 -af "afade=t=in:ss=0:d=3,afade=t=out:st=57:d=3" output.mp3
```

### 降噪处理

```bash
# 高通滤波器（去除低频噪音）
ffmpeg -i input.mp3 -af "highpass=f=200" output.mp3

# 低通滤波器（去除高频噪音）
ffmpeg -i input.mp3 -af "lowpass=f=3000" output.mp3

# 组合使用
ffmpeg -i input.mp3 -af "highpass=f=200,lowpass=f=3000" output.mp3
```

### 均衡器

```bash
# 增强低音
ffmpeg -i input.mp3 -af "bass=g=10" output.mp3

# 增强高音
ffmpeg -i input.mp3 -af "treble=g=5" output.mp3

# 多频段均衡
ffmpeg -i input.mp3 \
    -af "equalizer=f=100:t=h:w=200:g=5,equalizer=f=1000:t=h:w=200:g=-3" \
    output.mp3
```

### 变速变调

```bash
# 加速播放（不改变音调）
ffmpeg -i input.mp3 -af "atempo=1.5" output.mp3

# 减速播放
ffmpeg -i input.mp3 -af "atempo=0.75" output.mp3

# 大幅加速（需要串联）
ffmpeg -i input.mp3 -af "atempo=2.0,atempo=2.0" output.mp3
```

## 音频与视频合成

### 替换视频音轨

```bash
# 替换音频
ffmpeg -i video.mp4 -i audio.mp3 \
    -c:v copy -c:a aac \
    -map 0:v:0 -map 1:a:0 \
    output.mp4

# 删除原音频，添加新音频
ffmpeg -i video.mp4 -i audio.mp3 \
    -c:v copy -c:a aac \
    -map 0:v -map 1:a \
    -shortest output.mp4
```

### 添加背景音乐

```bash
# 视频加背景音乐（保留原声）
ffmpeg -i video.mp4 -i bgm.mp3 \
    -filter_complex "[0:a]volume=1.0[a0];[1:a]volume=0.3[a1];[a0][a1]amix=inputs=2:duration=first" \
    -c:v copy output.mp4
```

## 音频分析

### 生成波形图

```bash
# 生成波形图
ffmpeg -i input.mp3 -filter_complex \
    "showwavespic=s=1920x480:colors=blue" \
    -frames:v 1 waveform.png

# 生成频谱图
ffmpeg -i input.mp3 -filter_complex \
    "showspectrumpic=s=1920x480" \
    -frames:v 1 spectrum.png
```

### 检测音频信息

```bash
# 音量检测
ffmpeg -i input.mp3 -af "volumedetect" -f null -

# 静音检测
ffmpeg -i input.mp3 -af "silencedetect=n=-50dB:d=1" -f null -
```
