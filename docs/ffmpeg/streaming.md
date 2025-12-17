---
sidebar_position: 7
title: 流媒体
description: FFmpeg 流媒体推流与直播
---

# FFmpeg 流媒体处理

本文介绍 FFmpeg 的流媒体功能，包括推流、直播和 HLS 生成。

## RTMP 推流

### 推流到 RTMP 服务器

```bash
# 推送本地视频文件
ffmpeg -re -i input.mp4 -c copy -f flv rtmp://server/live/stream

# 推送摄像头（macOS）
ffmpeg -f avfoundation -framerate 30 -i "0" \
    -c:v libx264 -preset ultrafast -tune zerolatency \
    -f flv rtmp://server/live/stream

# 推送摄像头（Linux）
ffmpeg -f v4l2 -framerate 30 -video_size 1280x720 -i /dev/video0 \
    -c:v libx264 -preset ultrafast -tune zerolatency \
    -f flv rtmp://server/live/stream
```

### 推流到常见平台

```bash
# 推流到 Bilibili
ffmpeg -re -i input.mp4 \
    -c:v libx264 -preset fast -b:v 3000k \
    -c:a aac -b:a 128k \
    -f flv "rtmp://live-push.bilivideo.com/live-bvc/?streamname=xxx"

# 推流到 YouTube
ffmpeg -re -i input.mp4 \
    -c:v libx264 -preset fast -b:v 4500k -maxrate 4500k -bufsize 9000k \
    -c:a aac -b:a 128k \
    -f flv "rtmp://a.rtmp.youtube.com/live2/xxxx-xxxx-xxxx-xxxx"

# 推流到 Twitch
ffmpeg -re -i input.mp4 \
    -c:v libx264 -preset fast -b:v 3000k \
    -c:a aac -b:a 160k \
    -f flv "rtmp://live.twitch.tv/app/live_xxxxxxxx"
```

## 屏幕直播

### macOS 屏幕推流

```bash
# 屏幕 + 系统音频
ffmpeg -f avfoundation -framerate 30 -i "1:0" \
    -c:v libx264 -preset ultrafast -tune zerolatency -b:v 3000k \
    -c:a aac -b:a 128k \
    -f flv rtmp://server/live/stream
```

### Linux 屏幕推流

```bash
# 屏幕 + 音频
ffmpeg -f x11grab -framerate 30 -video_size 1920x1080 -i :0.0 \
    -f pulse -i default \
    -c:v libx264 -preset ultrafast -tune zerolatency \
    -c:a aac \
    -f flv rtmp://server/live/stream
```

### Windows 屏幕推流

```bash
# 屏幕推流
ffmpeg -f gdigrab -framerate 30 -i desktop \
    -c:v libx264 -preset ultrafast -tune zerolatency \
    -f flv rtmp://server/live/stream
```

## HLS 直播流

### 生成 HLS 流

```bash
# 基础 HLS 生成
ffmpeg -i input.mp4 \
    -c:v libx264 -c:a aac \
    -hls_time 10 \
    -hls_list_size 6 \
    -hls_flags delete_segments \
    -f hls output.m3u8

# 多码率 HLS
ffmpeg -i input.mp4 \
    -map 0:v -map 0:a -c:v libx264 -c:a aac \
    -b:v:0 5000k -s:v:0 1920x1080 \
    -b:v:1 3000k -s:v:1 1280x720 \
    -b:v:2 1500k -s:v:2 854x480 \
    -var_stream_map "v:0,a:0 v:1,a:1 v:2,a:2" \
    -master_pl_name master.m3u8 \
    -f hls -hls_time 4 -hls_playlist_type vod \
    stream_%v.m3u8
```

### HLS 参数说明

| 参数                         | 说明                 |
| ---------------------------- | -------------------- |
| `-hls_time`                  | 每个分片时长（秒）   |
| `-hls_list_size`             | 播放列表中的分片数量 |
| `-hls_flags delete_segments` | 删除旧分片           |
| `-hls_playlist_type vod`     | VOD 类型（完整列表） |
| `-hls_playlist_type event`   | 事件类型（追加模式） |

## DASH 流

```bash
# 生成 DASH 流
ffmpeg -i input.mp4 \
    -c:v libx264 -c:a aac \
    -f dash \
    -seg_duration 4 \
    -init_seg_name 'init-$RepresentationID$.m4s' \
    -media_seg_name 'chunk-$RepresentationID$-$Number%05d$.m4s' \
    output.mpd
```

## 拉流与转码

### 拉取 RTMP 流

```bash
# 拉流保存
ffmpeg -i rtmp://server/live/stream -c copy output.mp4

# 拉流转码
ffmpeg -i rtmp://server/live/stream \
    -c:v libx264 -c:a aac \
    output.mp4
```

### 拉取 HLS 流

```bash
# 保存 HLS 流
ffmpeg -i "https://example.com/stream.m3u8" -c copy output.mp4

# 转码 HLS 流
ffmpeg -i "https://example.com/stream.m3u8" \
    -c:v libx264 -c:a aac \
    output.mp4
```

### 流转发

```bash
# RTMP 转发
ffmpeg -i rtmp://source/live/stream \
    -c copy \
    -f flv rtmp://destination/live/stream

# RTMP 转 HLS
ffmpeg -i rtmp://server/live/stream \
    -c:v libx264 -c:a aac \
    -f hls -hls_time 4 \
    output.m3u8
```

## 本地流媒体服务器

### 使用 FFmpeg + nginx-rtmp

```nginx
# nginx.conf
rtmp {
    server {
        listen 1935;
        chunk_size 4096;

        application live {
            live on;
            record off;

            # HLS
            hls on;
            hls_path /tmp/hls;
            hls_fragment 3;
            hls_playlist_length 60;
        }
    }
}
```

### 推流命令

```bash
ffmpeg -re -i input.mp4 -c copy -f flv rtmp://localhost/live/stream
```

## 直播优化参数

```bash
# 低延迟直播
ffmpeg -re -i input.mp4 \
    -c:v libx264 -preset ultrafast -tune zerolatency \
    -g 30 -keyint_min 30 \
    -c:a aac -b:a 128k \
    -f flv rtmp://server/live/stream
```

### 关键参数说明

| 参数                | 说明                   |
| ------------------- | ---------------------- |
| `-re`               | 按原速率读取输入       |
| `-preset ultrafast` | 最快编码速度           |
| `-tune zerolatency` | 零延迟调优             |
| `-g`                | GOP 大小（关键帧间隔） |
| `-keyint_min`       | 最小关键帧间隔         |
