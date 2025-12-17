---
sidebar_position: 14
title: 硬件加速
description: 硬件解码/编码与常见平台（NVIDIA/Intel/macOS）实践
---

# 硬件加速

FFmpeg 的“硬件加速”主要分两类：

- **硬件解码（hwaccel）**：降低解码开销
- **硬件编码（hw encoder）**：提高编码速度（画质与码控能力与软件编码不同）

不同平台支持能力差异较大，建议先确认你本机 FFmpeg 的编译选项与可用加速后端。

## 查看可用硬件加速后端

```bash
ffmpeg -hide_banner -hwaccels
```

## NVIDIA（CUDA / NVENC / NVDEC）

### 1) 硬件解码

```bash
ffmpeg -hwaccel cuda -i input.mp4 -f null -
```

### 2) 硬件编码（H.264/H.265）

```bash
# H.264 NVENC
ffmpeg -i input.mp4 -c:v h264_nvenc -preset p4 -c:a copy out.mp4

# H.265 NVENC
ffmpeg -i input.mp4 -c:v hevc_nvenc -preset p4 -c:a copy out.mp4
```

常见参数（随 FFmpeg/NVENC 版本略有差异）：

- `-preset`：速度/质量平衡
- `-rc`：码率控制模式（CBR/VBR 等）
- `-b:v` / `-maxrate` / `-bufsize`：码率与 VBV

```bash
# 示例：VBR 码控
ffmpeg -i input.mp4 -c:v h264_nvenc -rc vbr -b:v 4M -maxrate 6M -bufsize 12M -c:a aac out.mp4
```

## Intel（QSV）

```bash
# QSV 编码（H.264）
ffmpeg -i input.mp4 -c:v h264_qsv -c:a copy out.mp4

# QSV 编码（HEVC）
ffmpeg -i input.mp4 -c:v hevc_qsv -c:a copy out.mp4
```

## macOS（VideoToolbox）

```bash
# 硬件编码（H.264）
ffmpeg -i input.mp4 -c:v h264_videotoolbox -c:a copy out.mp4

# 硬件编码（HEVC）
ffmpeg -i input.mp4 -c:v hevc_videotoolbox -c:a copy out.mp4
```

### 码率控制示例

```bash
ffmpeg -i input.mp4 -c:v h264_videotoolbox -b:v 4M -maxrate 6M -bufsize 12M -c:a aac out.mp4
```

## 常用注意事项

## 1) 不是所有滤镜都能“全程上 GPU”

很多滤镜仍在 CPU 上执行。即便使用了硬件解码/编码，中间滤镜链仍可能把帧拷回 CPU，性能未必线性提升。

## 2) 硬件编码的质量/体积与软件编码不同

同等码率下，`libx264/libx265` 往往能更好压缩；硬件编码优势在速度。对存档或高质量分发，常用软件编码；对实时转码/直播，常用硬件编码。

## 3) 先确认你要的是“解码”还是“编码”

- `-hwaccel ...` 只影响解码
- `-c:v h264_nvenc/h264_qsv/h264_videotoolbox` 才是硬件编码

## 4) 诊断方式

```bash
# 打印详细日志，看是否启用了硬件路径
ffmpeg -v verbose -hwaccel cuda -i input.mp4 -f null -

# 查看编译信息
ffmpeg -buildconf
```
