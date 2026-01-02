---
sidebar_position: 24
title: 🧠 对话记忆管理
---

# 对话记忆管理

对话记忆是让 AI 助手能够记住上下文、保持连贯对话的关键技术。本文介绍各种记忆管理策略。

## 记忆类型

```
┌─────────────────────────────────────────────────────────┐
│                     记忆系统                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  短期记忆 (Short-term)                                  │
│  └─> 当前对话的消息历史                                 │
│                                                         │
│  长期记忆 (Long-term)                                   │
│  ├─> 向量记忆：语义检索历史信息                         │
│  ├─> 摘要记忆：压缩历史对话                             │
│  └─> 实体记忆：记住关键实体信息                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 短期记忆：消息历史

### 基础实现

```python
from openai import OpenAI
from typing import List, Dict

class ConversationMemory:
    """基础对话记忆"""
    
    def __init__(self, system_prompt: str = "你是一个有帮助的助手。"):
        self.client = OpenAI()
        self.system_prompt = system_prompt
        self.messages: List[Dict] = []
    
    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
    
    def get_messages(self) -> List[Dict]:
        return [{"role": "system", "content": self.system_prompt}] + self.messages
    
    def chat(self, user_input: str) -> str:
        self.add_user_message(user_input)
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=self.get_messages()
        )
        
        assistant_message = response.choices[0].message.content
        self.add_assistant_message(assistant_message)
        
        return assistant_message
    
    def clear(self):
        self.messages = []

# 使用
memory = ConversationMemory()
print(memory.chat("我叫张三"))
print(memory.chat("我叫什么名字？"))  # 能记住名字
```

### 窗口记忆（限制消息数量）

```python
class WindowMemory:
    """滑动窗口记忆"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.messages: List[Dict] = []
        self.client = OpenAI()
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        # 保持窗口大小
        if len(self.messages) > self.window_size * 2:  # user + assistant
            self.messages = self.messages[-self.window_size * 2:]
    
    def chat(self, user_input: str) -> str:
        self.add_message("user", user_input)
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一个助手。"},
                *self.messages
            ]
        )
        
        assistant_message = response.choices[0].message.content
        self.add_message("assistant", assistant_message)
        
        return assistant_message
```

### Token 限制记忆

```python
import tiktoken

class TokenLimitMemory:
    """基于 Token 限制的记忆"""
    
    def __init__(self, max_tokens: int = 4000, model: str = "gpt-4o"):
        self.max_tokens = max_tokens
        self.model = model
        self.messages: List[Dict] = []
        self.client = OpenAI()
        self.encoder = tiktoken.encoding_for_model(model)
    
    def _count_tokens(self, messages: List[Dict]) -> int:
        """计算消息的 token 数"""
        total = 0
        for msg in messages:
            total += len(self.encoder.encode(msg["content"])) + 4  # 角色标记
        return total
    
    def _trim_messages(self):
        """裁剪消息以符合 token 限制"""
        while self._count_tokens(self.messages) > self.max_tokens and len(self.messages) > 2:
            # 保留最新的消息，删除最旧的
            self.messages.pop(0)
    
    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        self._trim_messages()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个助手。"},
                *self.messages
            ]
        )
        
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        self._trim_messages()
        
        return assistant_message
```

## 摘要记忆

当对话过长时，自动生成摘要来压缩历史。

```python
class SummaryMemory:
    """摘要记忆"""
    
    def __init__(self, summary_threshold: int = 10):
        self.summary_threshold = summary_threshold
        self.messages: List[Dict] = []
        self.summary: str = ""
        self.client = OpenAI()
    
    def _generate_summary(self) -> str:
        """生成对话摘要"""
        conversation = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.messages
        ])
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "请简洁地总结以下对话的要点，保留关键信息（人名、数字、决定等）。"
                },
                {"role": "user", "content": conversation}
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def _maybe_summarize(self):
        """检查是否需要生成摘要"""
        if len(self.messages) >= self.summary_threshold:
            # 生成摘要
            new_summary = self._generate_summary()
            
            # 合并旧摘要
            if self.summary:
                self.summary = f"之前的对话摘要：{self.summary}\n\n最近的对话摘要：{new_summary}"
            else:
                self.summary = new_summary
            
            # 清空消息，只保留最近几条
            self.messages = self.messages[-4:]
    
    def get_context(self) -> str:
        """获取完整上下文"""
        context_parts = []
        
        if self.summary:
            context_parts.append(f"对话历史摘要：\n{self.summary}")
        
        if self.messages:
            recent = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in self.messages
            ])
            context_parts.append(f"最近的对话：\n{recent}")
        
        return "\n\n".join(context_parts)
    
    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        self._maybe_summarize()
        
        system_prompt = "你是一个助手。"
        if self.summary:
            system_prompt += f"\n\n{self.get_context()}"
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                *self.messages
            ]
        )
        
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
```

## 向量记忆

使用向量数据库存储和检索历史对话。

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from datetime import datetime
import uuid

class VectorMemory:
    """向量记忆"""
    
    def __init__(self, collection_name: str = "conversation_memory"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./memory_db"
        )
        self.client = OpenAI()
        self.session_id = str(uuid.uuid4())
    
    def _store_exchange(self, user_input: str, assistant_response: str):
        """存储对话交换"""
        exchange = f"用户: {user_input}\n助手: {assistant_response}"
        
        self.vectorstore.add_texts(
            texts=[exchange],
            metadatas=[{
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input
            }]
        )
    
    def _retrieve_relevant(self, query: str, k: int = 3) -> List[str]:
        """检索相关历史"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def chat(self, user_input: str) -> str:
        # 检索相关历史
        relevant_history = self._retrieve_relevant(user_input)
        
        # 构建上下文
        context = ""
        if relevant_history:
            context = "相关的历史对话：\n" + "\n---\n".join(relevant_history)
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"你是一个助手。{context}"
                },
                {"role": "user", "content": user_input}
            ]
        )
        
        assistant_message = response.choices[0].message.content
        
        # 存储这次对话
        self._store_exchange(user_input, assistant_message)
        
        return assistant_message
```

## 实体记忆

记住对话中提到的关键实体。

```python
from typing import Dict, Any
import json

class EntityMemory:
    """实体记忆"""
    
    def __init__(self):
        self.entities: Dict[str, Any] = {}
        self.client = OpenAI()
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """从文本中提取实体"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """从文本中提取关键实体信息，返回 JSON 格式：
{
  "人物": {"姓名": "...", "特征": "..."},
  "地点": ["..."],
  "时间": ["..."],
  "事件": ["..."],
  "偏好": {"...": "..."}
}
只返回 JSON，不要其他内容。如果没有相关信息，对应字段为空。"""
                },
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {}
    
    def _update_entities(self, new_entities: Dict[str, Any]):
        """更新实体存储"""
        for key, value in new_entities.items():
            if key not in self.entities:
                self.entities[key] = value
            elif isinstance(value, dict) and isinstance(self.entities[key], dict):
                self.entities[key].update(value)
            elif isinstance(value, list) and isinstance(self.entities[key], list):
                self.entities[key].extend(value)
                self.entities[key] = list(set(self.entities[key]))  # 去重
            else:
                self.entities[key] = value
    
    def get_entity_context(self) -> str:
        """获取实体上下文"""
        if not self.entities:
            return ""
        return f"已知信息：\n{json.dumps(self.entities, ensure_ascii=False, indent=2)}"
    
    def chat(self, user_input: str) -> str:
        # 提取实体
        new_entities = self._extract_entities(user_input)
        self._update_entities(new_entities)
        
        # 构建上下文
        entity_context = self.get_entity_context()
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"你是一个助手。\n\n{entity_context}"
                },
                {"role": "user", "content": user_input}
            ]
        )
        
        assistant_message = response.choices[0].message.content
        
        # 从回复中也提取实体
        response_entities = self._extract_entities(assistant_message)
        self._update_entities(response_entities)
        
        return assistant_message

# 使用
memory = EntityMemory()
print(memory.chat("我叫张三，今年 30 岁，住在北京"))
print(memory.chat("我喜欢吃川菜"))
print(memory.chat("你还记得我的信息吗？"))
print(f"存储的实体：{memory.entities}")
```

## 组合记忆

结合多种记忆类型。

```python
class CombinedMemory:
    """组合记忆系统"""
    
    def __init__(self):
        self.short_term = []  # 短期：最近消息
        self.summary = ""     # 中期：摘要
        self.entities = {}    # 长期：实体
        self.vector_memory = VectorMemory()  # 长期：向量检索
        self.client = OpenAI()
        self.message_count = 0
    
    def _build_context(self, user_input: str) -> str:
        """构建完整上下文"""
        parts = []
        
        # 实体信息
        if self.entities:
            parts.append(f"用户信息：{json.dumps(self.entities, ensure_ascii=False)}")
        
        # 历史摘要
        if self.summary:
            parts.append(f"对话摘要：{self.summary}")
        
        # 相关历史（向量检索）
        relevant = self.vector_memory._retrieve_relevant(user_input, k=2)
        if relevant:
            parts.append(f"相关历史：\n" + "\n".join(relevant))
        
        return "\n\n".join(parts)
    
    def chat(self, user_input: str) -> str:
        self.message_count += 1
        
        # 构建上下文
        context = self._build_context(user_input)
        
        # 短期记忆
        self.short_term.append({"role": "user", "content": user_input})
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"你是一个助手。\n\n{context}"},
                *self.short_term[-10:]  # 最近 10 条
            ]
        )
        
        assistant_message = response.choices[0].message.content
        self.short_term.append({"role": "assistant", "content": assistant_message})
        
        # 更新各种记忆
        self._update_memories(user_input, assistant_message)
        
        return assistant_message
    
    def _update_memories(self, user_input: str, response: str):
        """更新各种记忆"""
        # 存储到向量记忆
        self.vector_memory._store_exchange(user_input, response)
        
        # 每 10 轮更新摘要
        if self.message_count % 10 == 0:
            self._update_summary()
        
        # 提取实体
        self._extract_and_update_entities(user_input)
    
    def _update_summary(self):
        """更新摘要"""
        if len(self.short_term) < 4:
            return
        
        conversation = "\n".join([
            f"{m['role']}: {m['content']}" for m in self.short_term
        ])
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "简洁总结对话要点。"},
                {"role": "user", "content": conversation}
            ],
            max_tokens=300
        )
        
        new_summary = response.choices[0].message.content
        
        if self.summary:
            self.summary = f"{self.summary}\n\n{new_summary}"
        else:
            self.summary = new_summary
        
        # 清理短期记忆
        self.short_term = self.short_term[-4:]
    
    def _extract_and_update_entities(self, text: str):
        """提取并更新实体"""
        # 简化版实体提取
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "提取文本中的关键信息（姓名、偏好等），返回 JSON。"
                },
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        
        try:
            new_entities = json.loads(response.choices[0].message.content)
            self.entities.update(new_entities)
        except:
            pass
```

## LangChain 记忆

```python
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    VectorStoreRetrieverMemory
)
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

llm = ChatOpenAI(model="gpt-4o")

# 缓冲记忆
buffer_memory = ConversationBufferMemory()

# 窗口记忆
window_memory = ConversationBufferWindowMemory(k=5)

# 摘要记忆
summary_memory = ConversationSummaryMemory(llm=llm)

# 摘要缓冲记忆（结合两者）
summary_buffer_memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=2000
)

# 使用记忆的对话链
conversation = ConversationChain(
    llm=llm,
    memory=summary_buffer_memory,
    verbose=True
)

response = conversation.predict(input="你好，我叫张三")
response = conversation.predict(input="我叫什么名字？")
```

## 最佳实践

1. **选择合适的记忆类型**：简单场景用窗口记忆，复杂场景用组合记忆
2. **控制上下文长度**：避免超出模型限制
3. **定期清理**：防止记忆无限增长
4. **持久化存储**：重要信息存入数据库
5. **隐私保护**：敏感信息脱敏处理

## 延伸阅读

- [LangChain Memory](https://python.langchain.com/docs/modules/memory/)
- [LlamaIndex Chat Engine](https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/)
