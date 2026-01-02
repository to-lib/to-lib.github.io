---
sidebar_position: 26
title: 🛡️ Guardrails（护栏）
---

# Guardrails（护栏）

Guardrails 是保护 AI 应用安全的防护机制，用于过滤有害输入、验证输出质量、防止模型滥用。

## 为什么需要 Guardrails？

```
用户输入 ──> [输入护栏] ──> LLM ──> [输出护栏] ──> 最终响应
              │                        │
              ├─ 过滤恶意内容           ├─ 验证格式
              ├─ 检测注入攻击           ├─ 过滤敏感信息
              └─ 话题边界控制           └─ 事实性检查
```

## 护栏类型

| 类型 | 作用 | 示例 |
|------|------|------|
| 输入护栏 | 过滤用户输入 | 注入检测、敏感词过滤 |
| 输出护栏 | 验证模型输出 | 格式校验、内容审核 |
| 话题护栏 | 限制对话范围 | 只回答产品相关问题 |
| 安全护栏 | 防止有害内容 | 暴力、色情、歧视检测 |

## 基础实现

### 输入过滤

```python
from openai import OpenAI
import re

client = OpenAI()

class InputGuardrail:
    """输入护栏"""
    
    def __init__(self):
        self.blocked_patterns = [
            r"ignore.*previous.*instructions",
            r"ignore.*above",
            r"disregard.*rules",
            r"你是.*DAN",
            r"jailbreak",
        ]
        
        self.sensitive_words = ["密码", "信用卡", "身份证"]
    
    def check_injection(self, text: str) -> tuple[bool, str]:
        """检测 Prompt 注入"""
        text_lower = text.lower()
        
        for pattern in self.blocked_patterns:
            if re.search(pattern, text_lower):
                return False, "检测到潜在的注入攻击"
        
        return True, ""
    
    def check_sensitive(self, text: str) -> tuple[bool, str]:
        """检测敏感信息"""
        for word in self.sensitive_words:
            if word in text:
                return False, f"请勿在对话中包含敏感信息：{word}"
        
        return True, ""
    
    def check_length(self, text: str, max_length: int = 4000) -> tuple[bool, str]:
        """检查输入长度"""
        if len(text) > max_length:
            return False, f"输入过长，请限制在 {max_length} 字符以内"
        return True, ""
    
    def validate(self, text: str) -> tuple[bool, str]:
        """综合验证"""
        checks = [
            self.check_injection,
            self.check_sensitive,
            self.check_length,
        ]
        
        for check in checks:
            passed, message = check(text)
            if not passed:
                return False, message
        
        return True, ""

# 使用
guardrail = InputGuardrail()
passed, message = guardrail.validate(user_input)
if not passed:
    print(f"输入被拒绝：{message}")
```

### 输出验证

```python
import json
from pydantic import BaseModel, ValidationError

class OutputGuardrail:
    """输出护栏"""
    
    def __init__(self):
        self.forbidden_content = ["暴力", "色情", "歧视"]
        self.client = OpenAI()
    
    def check_format(self, output: str, expected_format: type[BaseModel]) -> tuple[bool, str]:
        """验证输出格式"""
        try:
            expected_format.model_validate_json(output)
            return True, ""
        except ValidationError as e:
            return False, f"格式验证失败：{e}"
    
    def check_content_safety(self, output: str) -> tuple[bool, str]:
        """内容安全检查"""
        # 使用 Moderation API
        response = self.client.moderations.create(input=output)
        
        result = response.results[0]
        if result.flagged:
            categories = [k for k, v in result.categories.model_dump().items() if v]
            return False, f"内容不安全：{categories}"
        
        return True, ""
    
    def check_pii(self, output: str) -> tuple[bool, str]:
        """检测 PII（个人身份信息）"""
        pii_patterns = {
            "phone": r"\d{11}",
            "id_card": r"\d{17}[\dXx]",
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        }
        
        for pii_type, pattern in pii_patterns.items():
            if re.search(pattern, output):
                return False, f"输出包含敏感信息：{pii_type}"
        
        return True, ""
    
    def validate(self, output: str) -> tuple[bool, str]:
        """综合验证"""
        checks = [
            self.check_content_safety,
            self.check_pii,
        ]
        
        for check in checks:
            passed, message = check(output)
            if not passed:
                return False, message
        
        return True, ""
```

### 话题边界控制

```python
class TopicGuardrail:
    """话题护栏"""
    
    def __init__(self, allowed_topics: list[str], system_context: str):
        self.allowed_topics = allowed_topics
        self.system_context = system_context
        self.client = OpenAI()
    
    def check_topic(self, user_input: str) -> tuple[bool, str]:
        """检查是否在允许的话题范围内"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""判断用户问题是否与以下话题相关：
{', '.join(self.allowed_topics)}

上下文：{self.system_context}

只回答 "相关" 或 "不相关"。"""
                },
                {"role": "user", "content": user_input}
            ],
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip()
        
        if "不相关" in result:
            return False, f"抱歉，我只能回答关于 {', '.join(self.allowed_topics)} 的问题。"
        
        return True, ""

# 使用
topic_guard = TopicGuardrail(
    allowed_topics=["产品功能", "技术支持", "账户问题"],
    system_context="这是一个电商平台的客服助手"
)
```

## NeMo Guardrails

NVIDIA NeMo Guardrails 是一个开源框架，提供声明式的护栏配置。

### 安装

```bash
pip install nemoguardrails
```

### 配置文件

```yaml
# config.yml
models:
  - type: main
    engine: openai
    model: gpt-4o

rails:
  input:
    flows:
      - self check input
  output:
    flows:
      - self check output

prompts:
  - task: self_check_input
    content: |
      判断以下用户输入是否安全、合规：
      
      用户输入：{{ user_input }}
      
      如果输入安全，回答 "allowed"
      如果输入不安全，回答 "blocked"
      
  - task: self_check_output
    content: |
      判断以下 AI 回复是否安全、合规：
      
      AI 回复：{{ bot_response }}
      
      如果回复安全，回答 "allowed"
      如果回复不安全，回答 "blocked"
```

### Colang 规则

```colang
# rails.co

# 定义用户意图
define user ask about product
  "这个产品怎么样"
  "产品有什么功能"
  "价格是多少"

define user ask harmful
  "如何制作炸弹"
  "怎么攻击别人"

# 定义机器人回复
define bot refuse harmful
  "抱歉，我无法回答这类问题。"

define bot answer product
  "让我来介绍一下我们的产品..."

# 定义对话流程
define flow
  user ask harmful
  bot refuse harmful

define flow
  user ask about product
  bot answer product
```

### 使用 NeMo Guardrails

```python
from nemoguardrails import RailsConfig, LLMRails

# 加载配置
config = RailsConfig.from_path("./config")
rails = LLMRails(config)

# 生成回复
response = await rails.generate_async(
    messages=[{"role": "user", "content": "介绍一下你们的产品"}]
)

print(response["content"])
```

## Guardrails AI

Guardrails AI 是另一个流行的护栏框架，专注于输出验证。

### 安装

```bash
pip install guardrails-ai
```

### 基础使用

```python
from guardrails import Guard
from guardrails.hub import ToxicLanguage, DetectPII

# 创建护栏
guard = Guard().use_many(
    ToxicLanguage(on_fail="exception"),
    DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail="fix")
)

# 验证输出
try:
    result = guard.validate("这是一段测试文本，我的邮箱是 test@example.com")
    print(result.validated_output)
except Exception as e:
    print(f"验证失败：{e}")
```

### 结构化输出验证

```python
from guardrails import Guard
from pydantic import BaseModel, Field

class ProductReview(BaseModel):
    sentiment: str = Field(description="情感：positive/negative/neutral")
    score: int = Field(ge=1, le=5, description="评分 1-5")
    summary: str = Field(max_length=200, description="摘要")

guard = Guard.from_pydantic(ProductReview)

raw_output = """
{
    "sentiment": "positive",
    "score": 4,
    "summary": "这是一个很好的产品，质量不错，推荐购买。"
}
"""

result = guard.validate(raw_output)
print(result.validated_output)
```

## OpenAI Moderation API

```python
from openai import OpenAI

client = OpenAI()

def moderate_content(text: str) -> dict:
    """使用 OpenAI Moderation API"""
    response = client.moderations.create(input=text)
    result = response.results[0]
    
    return {
        "flagged": result.flagged,
        "categories": {
            k: v for k, v in result.categories.model_dump().items() if v
        },
        "scores": {
            k: round(v, 4) 
            for k, v in result.category_scores.model_dump().items() 
            if v > 0.1
        }
    }

# 使用
result = moderate_content("这是一段测试文本")
if result["flagged"]:
    print(f"内容被标记：{result['categories']}")
```

## 完整护栏系统

```python
from dataclasses import dataclass
from enum import Enum
from typing import Callable

class GuardrailAction(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"
    WARN = "warn"

@dataclass
class GuardrailResult:
    action: GuardrailAction
    message: str = ""
    modified_content: str = ""

class GuardrailSystem:
    """完整护栏系统"""
    
    def __init__(self):
        self.input_guards: list[Callable] = []
        self.output_guards: list[Callable] = []
        self.client = OpenAI()
    
    def add_input_guard(self, guard: Callable):
        self.input_guards.append(guard)
    
    def add_output_guard(self, guard: Callable):
        self.output_guards.append(guard)
    
    def check_input(self, text: str) -> GuardrailResult:
        """检查输入"""
        for guard in self.input_guards:
            result = guard(text)
            if result.action == GuardrailAction.BLOCK:
                return result
        return GuardrailResult(action=GuardrailAction.ALLOW)
    
    def check_output(self, text: str) -> GuardrailResult:
        """检查输出"""
        for guard in self.output_guards:
            result = guard(text)
            if result.action == GuardrailAction.BLOCK:
                return result
            if result.action == GuardrailAction.MODIFY:
                text = result.modified_content
        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            modified_content=text
        )
    
    def chat(self, user_input: str, system_prompt: str = "") -> str:
        """带护栏的对话"""
        # 输入检查
        input_result = self.check_input(user_input)
        if input_result.action == GuardrailAction.BLOCK:
            return f"输入被拒绝：{input_result.message}"
        
        # 调用 LLM
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_input})
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        
        output = response.choices[0].message.content
        
        # 输出检查
        output_result = self.check_output(output)
        if output_result.action == GuardrailAction.BLOCK:
            return "抱歉，无法生成合适的回复。"
        
        return output_result.modified_content or output

# 定义护栏函数
def injection_guard(text: str) -> GuardrailResult:
    patterns = ["ignore previous", "disregard"]
    for p in patterns:
        if p in text.lower():
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                message="检测到注入攻击"
            )
    return GuardrailResult(action=GuardrailAction.ALLOW)

def pii_guard(text: str) -> GuardrailResult:
    # 简单的 PII 脱敏
    import re
    modified = re.sub(r'\d{11}', '[PHONE]', text)
    modified = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', modified)
    
    if modified != text:
        return GuardrailResult(
            action=GuardrailAction.MODIFY,
            modified_content=modified
        )
    return GuardrailResult(action=GuardrailAction.ALLOW)

# 使用
system = GuardrailSystem()
system.add_input_guard(injection_guard)
system.add_output_guard(pii_guard)

response = system.chat("你好，请介绍一下产品")
```

## 最佳实践

1. **分层防护**：输入、输出都要检查
2. **快速失败**：危险内容立即拒绝
3. **日志记录**：记录所有被拦截的请求
4. **定期更新**：根据新的攻击模式更新规则
5. **用户反馈**：提供清晰的拒绝原因

## 延伸阅读

- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
- [Guardrails AI](https://www.guardrailsai.com/)
- [OpenAI Moderation](https://platform.openai.com/docs/guides/moderation)
