---
sidebar_position: 15
title: 脚本自动化
description: 用 shell/Python 批量调用 FFmpeg，参数模板化与安全实践
---

# 脚本自动化

当你需要批量转码、生成缩略图、统一音轨/字幕轨、批处理目录时，把命令写成脚本会更可靠。

## 目标与原则

- **可重复**：同一份脚本在不同机器/不同目录结构下尽量可运行
- **可观测**：输出清晰日志，失败能定位
- **安全**：避免覆盖原文件，优先输出到新目录

## Shell 批量转码（示例）

### 1) 批量把 MOV 转成 MP4（H.264 + AAC）

```bash
mkdir -p out
for f in *.mov; do
  ffmpeg -hide_banner -y -i "$f" \
    -c:v libx264 -crf 23 -preset medium \
    -c:a aac -b:a 128k \
    -movflags +faststart \
    "out/${f%.*}.mp4"
done
```

### 2) 批量抽帧生成缩略图

```bash
mkdir -p thumbs
for f in *.mp4; do
  ffmpeg -hide_banner -y -i "$f" \
    -vf "fps=1/10,scale=320:-1,tile=4x4" \
    -frames:v 1 \
    "thumbs/${f%.*}.png"
done
```

## 先用 ffprobe 再做决策

常见需求：不同输入素材有不同分辨率/旋转信息/音轨数量，脚本要先分析再转码。

```bash
# 获取宽高（CSV）
ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 input.mp4

# 获取旋转角度
ffprobe -v error -select_streams v:0 -show_entries stream_tags=rotate -of default=nw=1 input.mp4
```

## Python 调用（示例）

用 Python 的好处是：

- 更容易处理路径、并发、重试
- 更容易 parse JSON（ffprobe 输出）

```python
import json
import subprocess
from pathlib import Path


def ffprobe_json(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    out = subprocess.check_output(cmd)
    return json.loads(out)


def transcode_to_mp4(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    src = Path("input.mp4")
    meta = ffprobe_json(src)
    print(meta["format"].get("format_name"))
    transcode_to_mp4(src, Path("out.mp4"))
```

## 常见脚本坑

- **空格/特殊字符路径**：shell 一定要用引号包裹 `"$f"`
- **覆盖原文件**：避免输出到同名文件；或显式要求 `-n`（不覆盖）
- **并发过高**：多个 ffmpeg 进程会争抢 IO/CPU/GPU，建议限制并发数
- **参数漂移**：把 `crf/preset/音频码率` 做成变量或配置项
