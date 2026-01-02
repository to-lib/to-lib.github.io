---
sidebar_position: 30
title: 🎙️ 语音交互
---

# 语音交互

语音交互让 AI 应用能够听和说，包括语音识别（STT）、语音合成（TTS）和实时语音对话。

## 技术栈

```
语音输入 ──> STT ──> 文本 ──> LLM ──> 文本 ──> TTS ──> 语音输出
                              │
                    Realtime API（端到端）
```

## OpenAI 语音 API

### 语音合成（TTS）

```python
from openai import OpenAI
from pathlib import Path

client = OpenAI()

def text_to_speech(text: str, output_file: str = "output.mp3"):
    """文本转语音"""
    response = client.audio.speech.create(
        model="tts-1",  # tts-1 或 tts-1-hd
        voice="alloy",  # alloy, echo, fable, onyx, nova, shimmer
        input=text
    )
    
    response.stream_to_file(Path(output_file))
    return output_file

# 使用
text_to_speech("你好，我是 AI 助手。", "greeting.mp3")
```

### 流式 TTS

```python
def stream_tts(text: str):
    """流式语音合成"""
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
        response_format="pcm"  # 流式需要 pcm 格式
    )
    
    # 流式播放
    for chunk in response.iter_bytes(chunk_size=1024):
        yield chunk

# 配合音频播放库使用
import pyaudio

def play_stream(text: str):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    
    for chunk in stream_tts(text):
        stream.write(chunk)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
```

### 语音识别（STT）

```python
def speech_to_text(audio_file: str) -> str:
    """语音转文本"""
    with open(audio_file, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text"
        )
    return transcript

# 带时间戳
def speech_to_text_with_timestamps(audio_file: str) -> dict:
    """语音转文本（带时间戳）"""
    with open(audio_file, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"]
        )
    return transcript

# 翻译（任意语言转英语）
def translate_audio(audio_file: str) -> str:
    """语音翻译"""
    with open(audio_file, "rb") as f:
        translation = client.audio.translations.create(
            model="whisper-1",
            file=f
        )
    return translation.text
```

## OpenAI Realtime API

Realtime API 提供端到端的实时语音对话能力。

### WebSocket 连接

```python
import asyncio
import websockets
import json
import base64

async def realtime_conversation():
    """实时语音对话"""
    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "OpenAI-Beta": "realtime=v1"
    }
    
    async with websockets.connect(url, extra_headers=headers) as ws:
        # 配置会话
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": "你是一个友好的助手，用中文回答。",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",  # 服务端语音活动检测
                    "threshold": 0.5
                }
            }
        }))
        
        # 处理消息
        async for message in ws:
            event = json.loads(message)
            await handle_event(event)

async def handle_event(event: dict):
    """处理 Realtime API 事件"""
    event_type = event.get("type")
    
    if event_type == "response.audio.delta":
        # 收到音频数据
        audio_data = base64.b64decode(event["delta"])
        # 播放音频...
        
    elif event_type == "response.text.delta":
        # 收到文本
        print(event["delta"], end="", flush=True)
        
    elif event_type == "response.done":
        print("\n--- 回复完成 ---")
```

### 发送音频

```python
async def send_audio(ws, audio_data: bytes):
    """发送音频数据"""
    await ws.send(json.dumps({
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(audio_data).decode()
    }))

async def commit_audio(ws):
    """提交音频缓冲区"""
    await ws.send(json.dumps({
        "type": "input_audio_buffer.commit"
    }))
    
    # 请求响应
    await ws.send(json.dumps({
        "type": "response.create"
    }))
```

### 完整示例

```python
import sounddevice as sd
import numpy as np

class RealtimeVoiceChat:
    """实时语音聊天"""
    
    def __init__(self):
        self.sample_rate = 24000
        self.channels = 1
        self.ws = None
    
    async def connect(self):
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
        headers = {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "OpenAI-Beta": "realtime=v1"
        }
        self.ws = await websockets.connect(url, extra_headers=headers)
        
        # 配置
        await self.ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": "nova",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16"
            }
        }))
    
    async def start(self):
        await self.connect()
        
        # 启动录音和播放
        asyncio.create_task(self.record_audio())
        asyncio.create_task(self.receive_messages())
    
    async def record_audio(self):
        """录制音频并发送"""
        def callback(indata, frames, time, status):
            audio_bytes = indata.tobytes()
            asyncio.create_task(self.send_audio(audio_bytes))
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.int16,
            callback=callback
        ):
            await asyncio.sleep(float('inf'))
    
    async def send_audio(self, audio_data: bytes):
        if self.ws:
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_data).decode()
            }))
    
    async def receive_messages(self):
        """接收并处理消息"""
        audio_buffer = []
        
        async for message in self.ws:
            event = json.loads(message)
            
            if event["type"] == "response.audio.delta":
                audio_data = base64.b64decode(event["delta"])
                audio_buffer.append(audio_data)
                
            elif event["type"] == "response.audio.done":
                # 播放完整音频
                full_audio = b"".join(audio_buffer)
                self.play_audio(full_audio)
                audio_buffer = []
    
    def play_audio(self, audio_data: bytes):
        """播放音频"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        sd.play(audio_array, self.sample_rate)
        sd.wait()
```


## 其他语音服务

### Azure Speech

```python
import azure.cognitiveservices.speech as speechsdk

# 配置
speech_config = speechsdk.SpeechConfig(
    subscription="your_key",
    region="eastus"
)
speech_config.speech_synthesis_voice_name = "zh-CN-XiaoxiaoNeural"

# TTS
synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
result = synthesizer.speak_text_async("你好").get()

# STT
audio_config = speechsdk.AudioConfig(filename="audio.wav")
recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config,
    audio_config=audio_config
)
result = recognizer.recognize_once()
print(result.text)
```

### ElevenLabs

```python
from elevenlabs import generate, play, set_api_key

set_api_key("your_api_key")

# 生成语音
audio = generate(
    text="Hello, this is a test.",
    voice="Rachel",
    model="eleven_multilingual_v2"
)

play(audio)
```

## 语音助手架构

```python
class VoiceAssistant:
    """语音助手"""
    
    def __init__(self):
        self.client = OpenAI()
        self.conversation_history = []
    
    def listen(self, audio_file: str) -> str:
        """听取用户输入"""
        with open(audio_file, "rb") as f:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return transcript.text
    
    def think(self, user_input: str) -> str:
        """处理并生成回复"""
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一个语音助手，回复要简洁自然。"},
                *self.conversation_history
            ]
        )
        
        assistant_message = response.choices[0].message.content
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def speak(self, text: str, output_file: str = "response.mp3"):
        """语音输出"""
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        response.stream_to_file(output_file)
        return output_file
    
    def process(self, audio_file: str) -> str:
        """完整处理流程"""
        # 1. 语音转文本
        user_input = self.listen(audio_file)
        print(f"用户：{user_input}")
        
        # 2. 生成回复
        response = self.think(user_input)
        print(f"助手：{response}")
        
        # 3. 文本转语音
        output_file = self.speak(response)
        
        return output_file
```

## 实时语音翻译

```python
class RealtimeTranslator:
    """实时语音翻译"""
    
    def __init__(self, source_lang: str = "zh", target_lang: str = "en"):
        self.client = OpenAI()
        self.source_lang = source_lang
        self.target_lang = target_lang
    
    def translate(self, audio_file: str) -> tuple[str, str]:
        """翻译音频"""
        # 1. 语音识别
        with open(audio_file, "rb") as f:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=self.source_lang
            )
        
        original_text = transcript.text
        
        # 2. 文本翻译
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"将以下文本翻译成{self.target_lang}，只返回翻译结果。"
                },
                {"role": "user", "content": original_text}
            ]
        )
        
        translated_text = response.choices[0].message.content
        
        # 3. 语音合成
        audio_response = self.client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=translated_text
        )
        
        output_file = "translated.mp3"
        audio_response.stream_to_file(output_file)
        
        return original_text, translated_text
```

## 价格参考

| 服务 | 价格 |
|------|------|
| Whisper (STT) | $0.006/分钟 |
| TTS-1 | $15/1M 字符 |
| TTS-1-HD | $30/1M 字符 |
| Realtime API | $0.06/分钟（音频）+ Token 费用 |

## 最佳实践

1. **降噪处理**：输入音频先做降噪
2. **分段处理**：长音频分段识别
3. **流式输出**：TTS 使用流式提升体验
4. **错误处理**：处理网络中断和识别失败
5. **隐私保护**：敏感音频本地处理

## 延伸阅读

- [OpenAI Audio API](https://platform.openai.com/docs/guides/speech-to-text)
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
- [Azure Speech](https://azure.microsoft.com/en-us/products/ai-services/speech-services)