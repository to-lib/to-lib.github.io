---
sidebar_position: 12
title: 选流与映射（-map）
description: 用 -map 精确选择视频/音频/字幕流，处理多音轨、多字幕与默认轨
---

# 选流与映射（-map）

当输入文件包含多路视频/音频/字幕（例如 MKV 多音轨、多字幕、含 commentary 音轨），或者你只想导出其中某一路时，**`-map` 是最可靠的方式**。

如果你不使用 `-map`，FFmpeg 会按默认规则自动挑选流（通常是“第一路视频 + 第一/最佳音频”），这在多音轨场景很容易“选错”。

## 先用 ffprobe 看清楚输入

```bash
# 列出所有轨道（含 index、语言等）
ffprobe -v error \
  -show_entries stream=index,codec_type,codec_name:stream_tags=language,title \
  -of default=nw=1 input.mkv

# 只看音频轨
ffprobe -v error -select_streams a -show_entries stream=index,codec_name,channels:stream_tags=language,title -of default=nw=1 input.mkv

# 只看字幕轨
ffprobe -v error -select_streams s -show_entries stream=index,codec_name:stream_tags=language,title -of default=nw=1 input.mkv
```

## `-map` 的基本用法

### 1) 只导出第一路视频 + 第一路音频

```bash
ffmpeg -i input.mkv \
  -map 0:v:0 -map 0:a:0 \
  -c copy \
  out.mkv
```

说明：

- `0` 表示第 1 个输入（如果你有多个 `-i`，输入序号会递增）
- `v` / `a` / `s` 分别表示视频/音频/字幕
- `:0` 表示该类型的第 1 路流（不是 stream index，和 `ffprobe` 输出的 `index` 不同概念）

### 2) 保留所有流（最安全的“原样封装”）

```bash
ffmpeg -i input.mkv -map 0 -c copy out.mkv
```

这比“直接 `-c copy` 不写 `-map`”更明确：它表示把输入 0 的所有 stream 都映射到输出。

### 3) 排除某些流

```bash
# 保留所有流，但去掉字幕
ffmpeg -i input.mkv -map 0 -map -0:s -c copy out.mkv

# 保留所有流，但去掉第二路音频
ffmpeg -i input.mkv -map 0 -map -0:a:1 -c copy out.mkv
```

规则：

- 先 `-map 0` 选全量
- 再用 `-map -...` 做“负映射”排除

## 多输入的映射（最常见：换音轨/加字幕）

### 1) 替换音轨（保留视频，使用外部音频）

```bash
ffmpeg -i input.mp4 -i new_audio.m4a \
  -map 0:v:0 -map 1:a:0 \
  -c:v copy -c:a aac \
  -shortest \
  out.mp4
```

要点：

- `-shortest` 避免音频/视频时长不一致导致输出超长
- 若 `new_audio.m4a` 已是目标编码，也可以 `-c:a copy`

### 2) 给 MP4 添加软字幕（SRT -> mov_text）

```bash
ffmpeg -i input.mp4 -i sub_zh.srt \
  -map 0 -map 1 \
  -c:v copy -c:a copy -c:s mov_text \
  -metadata:s:s:0 language=chi \
  out.mp4
```

提示：MP4 容器通常使用 `mov_text` 字幕轨，直接 `-c:s copy` 往往不可行（取决于输入字幕类型）。

### 3) MKV 添加 ASS 字幕（尽量不重编码）

```bash
ffmpeg -i input.mkv -i sub.ass \
  -map 0 -map 1 \
  -c copy \
  -metadata:s:s:0 language=chi \
  out.mkv
```

## 默认轨道与语言标记

### 1) 设置默认音轨/字幕

```bash
# 将第二路音频设为 default（其余保持不变）
ffmpeg -i input.mkv -map 0 -c copy \
  -disposition:a:1 default \
  -disposition:a:0 0 \
  out.mkv

# 设置默认字幕
ffmpeg -i input.mkv -map 0 -c copy \
  -disposition:s:0 default \
  out.mkv
```

注意：具体播放器是否遵循 `disposition` 取决于播放器实现。

### 2) 设置语言/标题 metadata

```bash
ffmpeg -i input.mkv -map 0 -c copy \
  -metadata:s:a:0 language=chi \
  -metadata:s:a:0 title="Chinese" \
  -metadata:s:a:1 language=eng \
  -metadata:s:a:1 title="English" \
  out.mkv
```

## 常见坑与排查

- **选流序号混淆**：`ffprobe` 的 `stream index` 与 `-map 0:v:0` 的“类型序号”不是一个概念。拿不准就用 `-map 0` 先保全量。
- **MP4 字幕兼容性**：MP4 对字幕支持有限，推荐用 MKV 承载 SRT/ASS，或在 MP4 使用 `mov_text`。
- **“导出后没声音/语言不对”**：十有八九是 `-map` 选错或默认规则选错，先用 ffprobe 确认每路音轨语言。
- **复制流失败**：`-c copy` 需要目标容器支持该编码/该 stream 类型；不支持就会报错，需要转码或换容器。
