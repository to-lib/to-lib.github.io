---
sidebar_position: 15
title: 🖼️ 多模态 AI
---

# 多模态 AI (Multimodal AI)

多模态 AI 是指能够处理和理解多种类型数据（文本、图像、音频、视频）的人工智能系统。现代 LLM 如 GPT-4o、Claude 3.5、Gemini 都具备强大的多模态能力。

## 多模态能力概览

| 模态       | 输入 | 输出 | 典型应用                   |
| ---------- | ---- | ---- | -------------------------- |
| **文本**   | ✅   | ✅   | 对话、写作、翻译           |
| **图像**   | ✅   | ✅   | 图像理解、图像生成         |
| **音频**   | ✅   | ✅   | 语音识别、语音合成         |
| **视频**   | ✅   | ⚠️   | 视频理解（生成能力有限）   |

## Vision（图像理解）

### OpenAI Vision API

```python
from openai import OpenAI
import base64

client = OpenAI()

def encode_image(image_path: str) -> str:
    """将图片编码为 base64"""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

def analyze_image(image_path: str, prompt: str = "描述这张图片") -> str:
    """分析图片内容"""
    base64_image = encode_image(image_path)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"  # low, high, auto
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    
    return response.choices[0].message.content

# 使用示例
result = analyze_image("screenshot.png", "这个界面有什么问题？请给出改进建议。")
print(result)
```

### 使用 URL 图片

```python
def analyze_image_url(image_url: str, prompt: str) -> str:
    """分析网络图片"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ]
    )
    
    return response.choices[0].message.content

# 分析网络图片
result = analyze_image_url(
    "https://example.com/chart.png",
    "请分析这个图表的数据趋势"
)
```

### 多图片分析

```python
def compare_images(image_paths: list[str], prompt: str) -> str:
    """比较多张图片"""
    content = [{"type": "text", "text": prompt}]
    
    for path in image_paths:
        base64_image = encode_image(path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=1500
    )
    
    return response.choices[0].message.content

# 比较两个设计稿
result = compare_images(
    ["design_v1.png", "design_v2.png"],
    "比较这两个设计稿的差异，哪个更好？"
)
```

### Anthropic Vision

```python
import anthropic
import base64

client = anthropic.Anthropic()

def analyze_with_claude(image_path: str, prompt: str) -> str:
    """使用 Claude 分析图片"""
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    
    # 获取图片类型
    import mimetypes
    media_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    )
    
    return message.content[0].text
```

## 图像生成

### DALL-E 3

```python
def generate_image(prompt: str, size: str = "1024x1024", quality: str = "standard") -> str:
    """使用 DALL-E 3 生成图片"""
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,  # 1024x1024, 1792x1024, 1024x1792
        quality=quality,  # standard, hd
        n=1
    )
    
    return response.data[0].url

# 生成图片
image_url = generate_image(
    "一只可爱的机器人在花园里浇花，皮克斯风格，温暖的阳光",
    size="1024x1024",
    quality="hd"
)
print(f"生成的图片: {image_url}")
```

### 图片编辑

```python
def edit_image(image_path: str, mask_path: str, prompt: str) -> str:
    """编辑图片（需要 DALL-E 2）"""
    response = client.images.edit(
        model="dall-e-2",
        image=open(image_path, "rb"),
        mask=open(mask_path, "rb"),  # 透明区域表示要编辑的部分
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    
    return response.data[0].url
```

### 图片变体

```python
def create_variation(image_path: str) -> str:
    """创建图片变体"""
    response = client.images.create_variation(
        model="dall-e-2",
        image=open(image_path, "rb"),
        n=1,
        size="1024x1024"
    )
    
    return response.data[0].url
```

## 语音处理

### 语音转文字 (STT)

```python
def transcribe_audio(audio_path: str, language: str = None) -> dict:
    """语音转文字"""
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,  # 可选，如 "zh", "en"
            response_format="verbose_json",  # 包含时间戳
            timestamp_granularities=["word", "segment"]
        )
    
    return {
        "text": transcript.text,
        "segments": transcript.segments,
        "words": transcript.words
    }

# 转录音频
result = transcribe_audio("meeting.mp3", language="zh")
print(result["text"])

# 带时间戳的转录
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
```

### 语音翻译

```python
def translate_audio(audio_path: str) -> str:
    """将音频翻译为英文"""
    with open(audio_path, "rb") as audio_file:
        translation = client.audio.translations.create(
            model="whisper-1",
            file=audio_file
        )
    
    return translation.text

# 将中文音频翻译为英文
english_text = translate_audio("chinese_speech.mp3")
```

### 文字转语音 (TTS)

```python
from pathlib import Path

def text_to_speech(
    text: str, 
    output_path: str,
    voice: str = "alloy",
    model: str = "tts-1"
) -> str:
    """文字转语音"""
    # 可用声音: alloy, echo, fable, onyx, nova, shimmer
    # 模型: tts-1 (快速), tts-1-hd (高质量)
    
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text
    )
    
    response.stream_to_file(output_path)
    return output_path

# 生成语音
text_to_speech(
    "你好，欢迎使用我们的服务！",
    "welcome.mp3",
    voice="nova"
)
```

### 流式语音生成

```python
def stream_speech(text: str):
    """流式生成语音"""
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=text
    ) as response:
        for chunk in response.iter_bytes(chunk_size=1024):
            # 实时播放或处理音频块
            yield chunk
```

## 实战应用

### 1. 文档 OCR 与理解

```python
def extract_document_info(image_path: str) -> dict:
    """从文档图片中提取结构化信息"""
    
    prompt = """分析这张文档图片，提取以下信息并以 JSON 格式返回：
    - document_type: 文档类型（发票/合同/身份证/其他）
    - key_fields: 关键字段及其值
    - summary: 文档摘要
    
    只返回 JSON，不要其他内容。
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

# 提取发票信息
invoice_info = extract_document_info("invoice.jpg")
print(invoice_info)
```

### 2. 会议录音总结

```python
async def summarize_meeting(audio_path: str) -> dict:
    """会议录音转录与总结"""
    
    # 1. 转录音频
    transcript = transcribe_audio(audio_path, language="zh")
    
    # 2. 使用 LLM 总结
    summary_prompt = f"""请总结以下会议内容：

{transcript['text']}

请提供：
1. 会议主题
2. 主要讨论点（3-5 个）
3. 决策事项
4. 待办事项（包括负责人和截止日期，如果提到的话）
5. 下一步行动

以 JSON 格式输出。
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": summary_prompt}],
        response_format={"type": "json_object"}
    )
    
    return {
        "transcript": transcript,
        "summary": json.loads(response.choices[0].message.content)
    }
```

### 3. 产品图片分析

```python
def analyze_product_image(image_path: str) -> dict:
    """分析产品图片，生成描述和标签"""
    
    prompt = """分析这张产品图片，提供：
    1. 产品描述（适合电商使用，100字以内）
    2. 产品特点（3-5个要点）
    3. 建议标签（5-10个关键词）
    4. 目标受众
    5. 建议定价区间（如果能判断的话）
    
    以 JSON 格式输出。
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}
                    }
                ]
            }
        ],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)
```

### 4. 智能客服（多模态）

```python
class MultimodalAssistant:
    """多模态智能助手"""
    
    def __init__(self):
        self.client = OpenAI()
        self.conversation_history = []
    
    def process_message(self, text: str = None, image_path: str = None, audio_path: str = None) -> str:
        """处理多模态输入"""
        content = []
        
        # 处理音频输入
        if audio_path:
            transcript = transcribe_audio(audio_path)
            text = transcript["text"]
        
        # 构建消息内容
        if text:
            content.append({"type": "text", "text": text})
        
        if image_path:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}
            })
        
        if not content:
            return "请提供文字、图片或语音输入"
        
        # 添加到对话历史
        self.conversation_history.append({"role": "user", "content": content})
        
        # 调用模型
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一个友好的多模态助手，可以理解文字、图片和语音。"},
                *self.conversation_history
            ]
        )
        
        assistant_message = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message

# 使用示例
assistant = MultimodalAssistant()

# 文字对话
print(assistant.process_message(text="你好！"))

# 图片理解
print(assistant.process_message(
    text="这是什么产品？",
    image_path="product.jpg"
))

# 语音输入
print(assistant.process_message(audio_path="question.mp3"))
```

## 成本与限制

### Vision 成本

| 图片大小   | 低细节 (low) | 高细节 (high) |
| ---------- | ------------ | ------------- |
| 512x512    | 85 tokens    | 85 tokens     |
| 1024x1024  | 85 tokens    | 765 tokens    |
| 2048x2048  | 85 tokens    | 1105 tokens   |

### 音频成本

| 服务      | 价格              |
| --------- | ----------------- |
| Whisper   | $0.006 / 分钟     |
| TTS       | $15.00 / 1M 字符  |
| TTS-HD    | $30.00 / 1M 字符  |

### 图像生成成本

| 模型     | 分辨率    | 质量     | 价格      |
| -------- | --------- | -------- | --------- |
| DALL-E 3 | 1024x1024 | standard | $0.040    |
| DALL-E 3 | 1024x1024 | hd       | $0.080    |
| DALL-E 3 | 1792x1024 | standard | $0.080    |
| DALL-E 3 | 1792x1024 | hd       | $0.120    |

## 延伸阅读

- [OpenAI Vision 文档](https://platform.openai.com/docs/guides/vision)
- [OpenAI Audio 文档](https://platform.openai.com/docs/guides/speech-to-text)
- [DALL-E 文档](https://platform.openai.com/docs/guides/images)
- [Anthropic Vision 文档](https://docs.anthropic.com/claude/docs/vision)
