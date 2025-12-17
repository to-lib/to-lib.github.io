---
sidebar_position: 2
title: 安装配置
description: FFmpeg 在各平台的安装与配置指南
---

# FFmpeg 安装配置

本文介绍 FFmpeg 在 Windows、macOS 和 Linux 平台的安装方法。

## macOS 安装

### 使用 Homebrew（推荐）

```bash
# 安装 Homebrew（如未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装 FFmpeg
brew install ffmpeg

# 验证安装
ffmpeg -version
```

### 完整版安装（包含所有编解码器）

```bash
brew install ffmpeg --with-fdk-aac --with-sdl2 --with-freetype --with-libass --with-libvorbis --with-opus --with-x265
```

## Linux 安装

### Ubuntu/Debian

```bash
# 更新软件包列表
sudo apt update

# 安装 FFmpeg
sudo apt install ffmpeg

# 验证安装
ffmpeg -version
```

### CentOS/RHEL

```bash
# 启用 EPEL 仓库
sudo yum install epel-release

# 启用 RPM Fusion 仓库
sudo yum install https://download1.rpmfusion.org/free/el/rpmfusion-free-release-$(rpm -E %rhel).noarch.rpm

# 安装 FFmpeg
sudo yum install ffmpeg ffmpeg-devel

# 验证安装
ffmpeg -version
```

### Arch Linux

```bash
sudo pacman -S ffmpeg
```

## Windows 安装

### 方法一：使用包管理器

```powershell
# 使用 Chocolatey
choco install ffmpeg

# 使用 Scoop
scoop install ffmpeg

# 使用 winget
winget install FFmpeg
```

### 方法二：手动安装

1. 访问 [FFmpeg 官方下载页](https://ffmpeg.org/download.html)
2. 下载 Windows 构建版本
3. 解压到指定目录（如 `C:\ffmpeg`）
4. 添加到系统环境变量 `PATH`

```powershell
# 添加到系统 PATH
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\ffmpeg\bin", "Machine")
```

## 从源码编译

### 编译依赖

```bash
# Ubuntu/Debian
sudo apt install build-essential yasm nasm pkg-config \
    libx264-dev libx265-dev libvpx-dev libfdk-aac-dev \
    libmp3lame-dev libopus-dev libass-dev

# 下载源码
git clone https://git.ffmpeg.org/ffmpeg.git
cd ffmpeg

# 配置编译选项
./configure --enable-gpl --enable-nonfree \
    --enable-libx264 --enable-libx265 \
    --enable-libvpx --enable-libfdk-aac \
    --enable-libmp3lame --enable-libopus

# 编译安装
make -j$(nproc)
sudo make install
```

## 验证安装

```bash
# 查看版本信息
ffmpeg -version

# 查看支持的编解码器
ffmpeg -codecs

# 查看支持的格式
ffmpeg -formats

# 查看支持的滤镜
ffmpeg -filters
```

## Docker 使用

```bash
# 使用官方镜像
docker run -v $(pwd):/data jrottenberg/ffmpeg:latest -i /data/input.mp4 /data/output.avi

# 创建别名方便使用
alias ffmpeg='docker run -v $(pwd):/data jrottenberg/ffmpeg:latest'
```

## 常见问题

### 找不到编解码器

如果遇到 `Unknown encoder` 错误，可能需要安装额外的编解码器：

```bash
# Ubuntu
sudo apt install libavcodec-extra

# macOS
brew reinstall ffmpeg
```

### 硬件加速支持

查看硬件加速支持：

```bash
# 查看可用的硬件加速方式
ffmpeg -hwaccels

# NVIDIA GPU 支持
ffmpeg -hwaccel cuda -i input.mp4 output.mp4

# macOS VideoToolbox 支持
ffmpeg -hwaccel videotoolbox -i input.mp4 output.mp4
```
