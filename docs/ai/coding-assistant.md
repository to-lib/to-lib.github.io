---
sidebar_position: 33
title: 💻 AI 编码助手开发
---

# AI 编码助手开发

构建类似 GitHub Copilot 的 AI 编码助手，包括代码补全、代码生成、代码解释等功能。

## 核心功能

```
┌─────────────────────────────────────────────────────────┐
│                   AI 编码助手                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  代码补全：根据上下文自动补全代码                        │
│  代码生成：根据注释/描述生成代码                        │
│  代码解释：解释代码功能和逻辑                           │
│  代码重构：优化和重构现有代码                           │
│  Bug 修复：识别并修复代码问题                           │
│  测试生成：自动生成单元测试                             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 代码补全

### 基础实现

````python
from openai import OpenAI

client = OpenAI()

def code_completion(
    prefix: str,
    suffix: str = "",
    language: str = "python",
    max_tokens: int = 150
) -> str:
    """代码补全"""
    prompt = f"""Complete the following {language} code. Only return the completion, no explanation.

```{language}
{prefix}<CURSOR>{suffix}
```

Completion:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0
    )

    return response.choices[0].message.content


# 使用 Fill-in-the-Middle (FIM)
def fim_completion(prefix: str, suffix: str) -> str:
    """FIM 模式补全"""
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>",
        max_tokens=150,
        temperature=0
    )
    return response.choices[0].text
````

### 上下文收集

```python
import os
from pathlib import Path

class CodeContext:
    """代码上下文收集器"""

    def __init__(self, workspace_path: str):
        self.workspace = Path(workspace_path)

    def get_file_content(self, file_path: str) -> str:
        """获取文件内容"""
        full_path = self.workspace / file_path
        if full_path.exists():
            return full_path.read_text()
        return ""

    def get_related_files(self, current_file: str, max_files: int = 5) -> list:
        """获取相关文件"""
        current = Path(current_file)
        related = []

        # 同目录文件
        for f in current.parent.glob(f"*{current.suffix}"):
            if f != current and len(related) < max_files:
                related.append(str(f))

        return related

    def build_context(
        self,
        current_file: str,
        cursor_line: int,
        cursor_col: int
    ) -> dict:
        """构建补全上下文"""
        content = self.get_file_content(current_file)
        lines = content.split("\n")

        # 分割前缀和后缀
        prefix_lines = lines[:cursor_line]
        suffix_lines = lines[cursor_line:]

        if prefix_lines:
            prefix_lines[-1] = prefix_lines[-1][:cursor_col]
        if suffix_lines:
            suffix_lines[0] = suffix_lines[0][cursor_col:]

        prefix = "\n".join(prefix_lines)
        suffix = "\n".join(suffix_lines)

        # 获取相关文件作为额外上下文
        related = self.get_related_files(current_file)
        related_content = []
        for f in related[:3]:
            related_content.append({
                "file": f,
                "content": self.get_file_content(f)[:2000]  # 限制长度
            })

        return {
            "prefix": prefix,
            "suffix": suffix,
            "related_files": related_content,
            "language": self._detect_language(current_file)
        }

    def _detect_language(self, file_path: str) -> str:
        """检测编程语言"""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust"
        }
        ext = Path(file_path).suffix
        return ext_map.get(ext, "text")
```

## 代码生成

```python
def generate_code(
    description: str,
    language: str = "python",
    context: str = ""
) -> str:
    """根据描述生成代码"""
    system_prompt = f"""你是一个专业的 {language} 开发者。
根据用户描述生成高质量代码。

要求：
1. 代码简洁、高效
2. 添加必要的注释
3. 遵循 {language} 最佳实践
4. 处理边界情况"""

    user_prompt = f"描述：{description}"
    if context:
        user_prompt = f"上下文：\n{context}\n\n{user_prompt}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

def generate_from_comment(code_with_comment: str) -> str:
    """根据注释生成代码"""
    prompt = f"""根据注释生成代码实现：

```

{code_with_comment}

```

只返回完整的代码，包含注释。"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content
```

## 代码解释

````python
def explain_code(code: str, detail_level: str = "medium") -> str:
    """解释代码"""
    detail_prompts = {
        "brief": "用一两句话简要说明这段代码的功能。",
        "medium": "解释这段代码的功能、主要逻辑和关键步骤。",
        "detailed": "详细解释这段代码，包括每个函数、变量的作用，算法逻辑，以及可能的改进点。"
    }

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"你是一个代码解释专家。{detail_prompts[detail_level]}"
            },
            {"role": "user", "content": f"```\n{code}\n```"}
        ]
    )

    return response.choices[0].message.content

def explain_error(code: str, error_message: str) -> str:
    """解释错误"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "分析代码错误，解释原因并提供修复方案。"
            },
            {
                "role": "user",
                "content": f"代码：\n```\n{code}\n```\n\n错误信息：\n{error_message}"
            }
        ]
    )

    return response.choices[0].message.content
````

## 代码重构

````python
def refactor_code(
    code: str,
    refactor_type: str = "general"
) -> str:
    """代码重构"""
    refactor_prompts = {
        "general": "优化代码结构、可读性和性能",
        "performance": "专注于性能优化",
        "readability": "提高代码可读性和可维护性",
        "security": "修复安全问题",
        "modern": "使用现代语法和最佳实践重写"
    }

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"你是代码重构专家。任务：{refactor_prompts[refactor_type]}。返回重构后的代码和改动说明。"
            },
            {"role": "user", "content": f"```\n{code}\n```"}
        ]
    )

    return response.choices[0].message.content
````

## 测试生成

````python
def generate_tests(
    code: str,
    framework: str = "pytest"
) -> str:
    """生成单元测试"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""为代码生成全面的单元测试。
使用 {framework} 框架。
包含：正常情况、边界情况、异常情况。"""
            },
            {"role": "user", "content": f"```\n{code}\n```"}
        ]
    )

    return response.choices[0].message.content
````

## 完整编码助手

```python
class CodingAssistant:
    """AI 编码助手"""

    def __init__(self, workspace: str = "."):
        self.client = OpenAI()
        self.context = CodeContext(workspace)
        self.conversation = []

    def complete(self, file_path: str, line: int, col: int) -> str:
        """代码补全"""
        ctx = self.context.build_context(file_path, line, col)

        # 构建带上下文的提示
        related_context = ""
        for f in ctx["related_files"]:
            related_context += f"\n// {f['file']}\n{f['content'][:500]}\n"

        prompt = f"""Language: {ctx['language']}
Related files:{related_context}

Complete the code at <CURSOR>:
```

{ctx['prefix']}<CURSOR>{ctx['suffix']}

````"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0
        )

        return response.choices[0].message.content

    def chat(self, message: str, code_context: str = "") -> str:
        """对话式编程助手"""
        self.conversation.append({"role": "user", "content": message})

        system = """你是一个专业的编程助手。
帮助用户编写、调试、优化代码。
回答要简洁、准确、实用。"""

        if code_context:
            system += f"\n\n当前代码上下文：\n```\n{code_context}\n```"

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                *self.conversation[-10:]  # 保留最近 10 轮
            ]
        )

        assistant_msg = response.choices[0].message.content
        self.conversation.append({"role": "assistant", "content": assistant_msg})

        return assistant_msg

    def fix_bug(self, code: str, bug_description: str) -> str:
        """修复 Bug"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "你是 Bug 修复专家。分析问题，提供修复方案和修复后的代码。"
                },
                {
                    "role": "user",
                    "content": f"代码：\n```\n{code}\n```\n\n问题描述：{bug_description}"
                }
            ]
        )

        return response.choices[0].message.content

# 使用
assistant = CodingAssistant("./my_project")
completion = assistant.complete("src/main.py", 10, 0)
answer = assistant.chat("如何优化这个函数的性能？", code_context)
````

## 最佳实践

1. **上下文很重要**：提供足够的代码上下文
2. **流式输出**：补全时使用流式提升体验
3. **缓存结果**：相似请求复用结果
4. **本地模型**：考虑使用本地模型降低延迟
5. **安全过滤**：过滤敏感代码和凭证

## 延伸阅读

- [GitHub Copilot](https://github.com/features/copilot)
- [Continue.dev](https://continue.dev/)
- [Cursor](https://cursor.sh/)
