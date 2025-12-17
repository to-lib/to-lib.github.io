---
sidebar_position: 6
title: 滤镜使用
description: FFmpeg 滤镜系统详解
---

# FFmpeg 滤镜使用

FFmpeg 的滤镜系统非常强大，可以实现各种音视频效果处理。

## 滤镜基础

### 滤镜语法

```bash
# 单个滤镜
ffmpeg -i input.mp4 -vf "滤镜名=参数" output.mp4

# 多个滤镜（逗号分隔）
ffmpeg -i input.mp4 -vf "滤镜1,滤镜2,滤镜3" output.mp4

# 音频滤镜
ffmpeg -i input.mp4 -af "滤镜名=参数" output.mp4

# 复杂滤镜图
ffmpeg -i input.mp4 -filter_complex "滤镜图表达式" output.mp4
```

### 查看可用滤镜

```bash
# 列出所有滤镜
ffmpeg -filters

# 查看特定滤镜帮助
ffmpeg -h filter=scale
```

## 常用视频滤镜

### 缩放（scale）

```bash
# 指定尺寸
ffmpeg -i input.mp4 -vf "scale=1280:720" output.mp4

# 保持宽高比
ffmpeg -i input.mp4 -vf "scale=1280:-1" output.mp4
ffmpeg -i input.mp4 -vf "scale=-1:720" output.mp4

# 使用表达式
ffmpeg -i input.mp4 -vf "scale=iw/2:ih/2" output.mp4

# 高质量缩放算法
ffmpeg -i input.mp4 -vf "scale=1280:720:flags=lanczos" output.mp4
```

### 裁剪（crop）

```bash
# 裁剪中心区域
ffmpeg -i input.mp4 -vf "crop=640:480" output.mp4

# 指定偏移
ffmpeg -i input.mp4 -vf "crop=640:480:100:50" output.mp4

# 裁剪居中
ffmpeg -i input.mp4 -vf "crop=640:480:(iw-640)/2:(ih-480)/2" output.mp4
```

### 填充（pad）

```bash
# 添加黑边
ffmpeg -i input.mp4 -vf "pad=1920:1080:(ow-iw)/2:(oh-ih)/2" output.mp4

# 指定颜色
ffmpeg -i input.mp4 -vf "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:white" output.mp4
```

### 覆盖（overlay）

```bash
# 右下角覆盖
ffmpeg -i main.mp4 -i overlay.png \
    -filter_complex "[0:v][1:v]overlay=W-w-10:H-h-10" output.mp4

# 透明度调节
ffmpeg -i main.mp4 -i overlay.png \
    -filter_complex "[1:v]format=rgba,colorchannelmixer=aa=0.5[ov];[0:v][ov]overlay=10:10" \
    output.mp4
```

### 文字绘制（drawtext）

```bash
# 基础文字
ffmpeg -i input.mp4 -vf \
    "drawtext=text='Hello World':fontsize=48:fontcolor=white:x=10:y=10" \
    output.mp4

# 使用字体文件
ffmpeg -i input.mp4 -vf \
    "drawtext=text='你好世界':fontfile=/path/to/font.ttf:fontsize=48:fontcolor=white:x=10:y=10" \
    output.mp4

# 添加背景框
ffmpeg -i input.mp4 -vf \
    "drawtext=text='Hello':fontsize=48:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:x=10:y=10" \
    output.mp4

# 动态时间戳
ffmpeg -i input.mp4 -vf \
    "drawtext=text='%{localtime}':fontsize=24:fontcolor=white:x=10:y=10" \
    output.mp4
```

## 颜色调整滤镜

### 色彩调节（eq）

```bash
# 调整亮度、对比度、饱和度
ffmpeg -i input.mp4 -vf "eq=brightness=0.1:contrast=1.2:saturation=1.5" output.mp4

# 调整伽马值
ffmpeg -i input.mp4 -vf "eq=gamma=1.5" output.mp4
```

### 色调调节（hue）

```bash
# 调整色相
ffmpeg -i input.mp4 -vf "hue=h=90" output.mp4

# 调整饱和度
ffmpeg -i input.mp4 -vf "hue=s=2" output.mp4
```

### 颜色曲线（curves）

```bash
# 增加对比度
ffmpeg -i input.mp4 -vf "curves=preset=increase_contrast" output.mp4

# 复古效果
ffmpeg -i input.mp4 -vf "curves=preset=vintage" output.mp4

# 自定义曲线
ffmpeg -i input.mp4 -vf "curves=r='0/0 0.5/0.4 1/1':g='0/0 0.5/0.5 1/1':b='0/0 0.5/0.6 1/1'" output.mp4
```

### 颜色映射（colorchannelmixer）

```bash
# 灰度效果
ffmpeg -i input.mp4 -vf "colorchannelmixer=.3:.4:.3:0:.3:.4:.3:0:.3:.4:.3" output.mp4

# 棕褐色调
ffmpeg -i input.mp4 -vf "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131" output.mp4
```

## 特效滤镜

### 模糊效果

```bash
# 方框模糊
ffmpeg -i input.mp4 -vf "boxblur=5:1" output.mp4

# 高斯模糊
ffmpeg -i input.mp4 -vf "gblur=sigma=10" output.mp4
```

### 锐化效果

```bash
# 锐化
ffmpeg -i input.mp4 -vf "unsharp=5:5:1.0:5:5:0.0" output.mp4
```

### 降噪

```bash
# 时域降噪
ffmpeg -i input.mp4 -vf "hqdn3d=4:4:6:6" output.mp4

# 空域降噪
ffmpeg -i input.mp4 -vf "nlmeans=s=3:p=7:r=15" output.mp4
```

### 边缘检测

```bash
# Sobel 边缘检测
ffmpeg -i input.mp4 -vf "edgedetect=mode=colormix:high=0" output.mp4
```

### 老电影效果

```bash
# 添加噪点和划痕
ffmpeg -i input.mp4 -vf \
    "noise=alls=20:allf=t+u,curves=preset=vintage,vignette" \
    output.mp4
```

## 复杂滤镜图

### 标签系统

```bash
# 使用标签
ffmpeg -i input1.mp4 -i input2.mp4 \
    -filter_complex "[0:v]scale=640:480[v0];[1:v]scale=640:480[v1];[v0][v1]hstack[out]" \
    -map "[out]" output.mp4
```

### 分裂和合并

```bash
# 分裂流
ffmpeg -i input.mp4 \
    -filter_complex "[0:v]split=2[v1][v2];[v1]crop=iw/2:ih:0:0[left];[v2]crop=iw/2:ih:iw/2:0[right];[left][right]hstack" \
    output.mp4
```

### 条件滤镜

```bash
# 根据时间应用效果
ffmpeg -i input.mp4 \
    -vf "drawtext=text='开始':enable='between(t,0,5)':fontsize=48:fontcolor=white:x=10:y=10" \
    output.mp4
```

## 常用音频滤镜

### 混音（amix）

```bash
ffmpeg -i audio1.mp3 -i audio2.mp3 \
    -filter_complex "[0:a][1:a]amix=inputs=2:duration=longest" \
    output.mp3
```

### 音量调节

```bash
# 静态调节
ffmpeg -i input.mp3 -af "volume=1.5" output.mp3

# 动态调节
ffmpeg -i input.mp3 -af "volume='if(lt(t,5),t/5,1)':eval=frame" output.mp3
```

### 重采样（aresample）

```bash
ffmpeg -i input.mp3 -af "aresample=44100" output.mp3
```

### 声道映射（channelmap）

```bash
# 左声道复制到右声道
ffmpeg -i input.mp3 -af "channelmap=0|0" output.mp3
```
