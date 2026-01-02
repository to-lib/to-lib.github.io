---
sidebar_position: 23
title: 📜 长上下文处理
---

# 长上下文处理

处理超长文档是 LLM 应用的常见挑战。本文介绍各种长上下文处理策略。

## 模型上下文长度

| 模型              | 上下文长度 | 约等于         |
| ----------------- | ---------- | -------------- |
| GPT-4o            | 128K       | ~300 页文档    |
| GPT-4o-mini       | 128K       | ~300 页文档    |
| Claude 3.5 Sonnet | 200K       | ~500 页文档    |
| Gemini 1.5 Pro    | 2M         | ~5000 页文档   |
| Qwen2.5           | 128K       | ~300 页文档    |

## 处理策略概览

```
┌─────────────────────────────────────────────────────────┐
│                   长文档处理策略                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  文档长度 < 上下文窗口                                   │
│  └─> 直接处理                                           │
│                                                         │
│  文档长度 > 上下文窗口                                   │
│  ├─> 策略1: 切分 + RAG 检索                             │
│  ├─> 策略2: Map-Reduce 摘要                             │
│  ├─> 策略3: Refine 迭代精炼                             │
│  └─> 策略4: 层次化摘要                                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 策略 1: 直接处理（短文档）

```python
from openai import OpenAI

client = OpenAI()

def process_short_document(document: str, question: str) -> str:
    """直接处理短文档"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "你是一个文档分析助手。请根据提供的文档回答问题。"
            },
            {
                "role": "user",
                "content": f"文档内容：\n\n{document}\n\n问题：{question}"
            }
        ]
    )
    return response.choices[0].message.content
```

## 策略 2: Map-Reduce

将长文档切分，分别处理后合并结果。适合摘要、信息提取等任务。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
import tiktoken

class MapReduceProcessor:
    """Map-Reduce 文档处理器"""
    
    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._count_tokens
        )
        self.client = OpenAI()
    
    def _count_tokens(self, text: str) -> int:
        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(text))
    
    def _map_chunk(self, chunk: str, task: str) -> str:
        """Map 阶段：处理单个块"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Map 阶段用小模型
            messages=[
                {"role": "system", "content": f"请对以下文本执行任务：{task}"},
                {"role": "user", "content": chunk}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    
    def _reduce(self, results: list[str], task: str) -> str:
        """Reduce 阶段：合并结果"""
        combined = "\n\n---\n\n".join(results)
        
        response = self.client.chat.completions.create(
            model="gpt-4o",  # Reduce 阶段用大模型
            messages=[
                {
                    "role": "system",
                    "content": f"以下是对文档各部分执行'{task}'的结果。请综合这些结果，生成最终输出。"
                },
                {"role": "user", "content": combined}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    
    def process(self, document: str, task: str, parallel: bool = True) -> str:
        """处理长文档"""
        # 切分文档
        chunks = self.splitter.split_text(document)
        print(f"文档被切分为 {len(chunks)} 个块")
        
        # Map 阶段
        if parallel:
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(
                    lambda c: self._map_chunk(c, task),
                    chunks
                ))
        else:
            results = [self._map_chunk(c, task) for c in chunks]
        
        # Reduce 阶段
        if len(results) == 1:
            return results[0]
        
        return self._reduce(results, task)

# 使用示例
processor = MapReduceProcessor()
summary = processor.process(
    long_document,
    task="提取关键信息并生成摘要"
)
```

## 策略 3: Refine（迭代精炼）

逐块处理，每次基于前一次的结果进行精炼。适合需要连贯性的任务。

```python
class RefineProcessor:
    """Refine 迭代精炼处理器"""
    
    def __init__(self, chunk_size: int = 4000):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200
        )
        self.client = OpenAI()
    
    def process(self, document: str, task: str) -> str:
        """迭代精炼处理"""
        chunks = self.splitter.split_text(document)
        
        # 处理第一个块
        current_result = self._initial_process(chunks[0], task)
        
        # 迭代精炼后续块
        for i, chunk in enumerate(chunks[1:], 2):
            print(f"处理第 {i}/{len(chunks)} 块...")
            current_result = self._refine(current_result, chunk, task)
        
        return current_result
    
    def _initial_process(self, chunk: str, task: str) -> str:
        """处理第一个块"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"请对以下文本执行任务：{task}"},
                {"role": "user", "content": chunk}
            ]
        )
        return response.choices[0].message.content
    
    def _refine(self, current_result: str, new_chunk: str, task: str) -> str:
        """基于新内容精炼结果"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""你之前对文档部分内容执行了任务：{task}
                    
当前结果：
{current_result}

现在有新的文档内容。请根据新内容更新和完善你的结果。
如果新内容包含重要信息，请添加到结果中。
如果新内容与现有结果矛盾，请进行修正。"""
                },
                {"role": "user", "content": f"新内容：\n{new_chunk}"}
            ]
        )
        return response.choices[0].message.content

# 使用
refiner = RefineProcessor()
result = refiner.process(long_document, "生成详细摘要")
```

## 策略 4: 层次化摘要

构建摘要树，适合超长文档。

```python
class HierarchicalSummarizer:
    """层次化摘要"""
    
    def __init__(self, leaf_size: int = 2000, branch_factor: int = 4):
        self.leaf_size = leaf_size
        self.branch_factor = branch_factor
        self.client = OpenAI()
    
    def summarize(self, document: str) -> dict:
        """生成层次化摘要"""
        # 切分为叶子节点
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.leaf_size)
        chunks = splitter.split_text(document)
        
        # 构建摘要树
        tree = self._build_tree(chunks)
        
        return {
            "final_summary": tree["summary"],
            "tree": tree
        }
    
    def _build_tree(self, chunks: list[str], level: int = 0) -> dict:
        """递归构建摘要树"""
        if len(chunks) == 1:
            summary = self._summarize_chunk(chunks[0])
            return {"level": level, "summary": summary, "children": None}
        
        # 分组
        groups = [
            chunks[i:i + self.branch_factor]
            for i in range(0, len(chunks), self.branch_factor)
        ]
        
        # 递归处理每组
        children = []
        child_summaries = []
        
        for group in groups:
            if len(group) == 1:
                child_summary = self._summarize_chunk(group[0])
            else:
                # 先合并组内内容再摘要
                combined = "\n\n".join(group)
                child_summary = self._summarize_chunk(combined)
            
            children.append({
                "level": level + 1,
                "summary": child_summary,
                "original_chunks": group
            })
            child_summaries.append(child_summary)
        
        # 如果子摘要数量仍然很多，继续递归
        if len(child_summaries) > self.branch_factor:
            return self._build_tree(child_summaries, level)
        
        # 合并子摘要
        final_summary = self._merge_summaries(child_summaries)
        
        return {
            "level": level,
            "summary": final_summary,
            "children": children
        }
    
    def _summarize_chunk(self, text: str) -> str:
        """摘要单个块"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "请生成简洁的摘要，保留关键信息。"},
                {"role": "user", "content": text}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    
    def _merge_summaries(self, summaries: list[str]) -> str:
        """合并多个摘要"""
        combined = "\n\n".join([f"部分 {i+1}：{s}" for i, s in enumerate(summaries)])
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "请将以下多个摘要合并为一个连贯、全面的摘要。"
                },
                {"role": "user", "content": combined}
            ]
        )
        return response.choices[0].message.content
```

## 策略 5: RAG 检索

对于问答场景，只检索相关部分。

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

class LongDocumentQA:
    """长文档问答系统"""
    
    def __init__(self, document: str):
        # 切分文档
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(document)
        
        # 创建向量索引
        self.vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=OpenAIEmbeddings(model="text-embedding-3-small")
        )
        
        self.client = OpenAI()
    
    def query(self, question: str, k: int = 5) -> str:
        """查询"""
        # 检索相关块
        docs = self.vectorstore.similarity_search(question, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 生成回答
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "根据提供的上下文回答问题。如果上下文中没有相关信息，请说明。"
                },
                {
                    "role": "user",
                    "content": f"上下文：\n{context}\n\n问题：{question}"
                }
            ]
        )
        
        return response.choices[0].message.content
```

## 策略 6: 滑动窗口

适合需要处理整个文档但上下文有限的场景。

```python
class SlidingWindowProcessor:
    """滑动窗口处理器"""
    
    def __init__(self, window_size: int = 4000, stride: int = 2000):
        self.window_size = window_size
        self.stride = stride
        self.client = OpenAI()
    
    def process(self, document: str, task: str) -> list[dict]:
        """滑动窗口处理"""
        results = []
        
        # 按字符滑动（实际应用中应按 token）
        for i in range(0, len(document), self.stride):
            window = document[i:i + self.window_size]
            
            if len(window) < 100:  # 跳过太短的窗口
                continue
            
            result = self._process_window(window, task, i)
            results.append({
                "start": i,
                "end": i + len(window),
                "result": result
            })
        
        return results
    
    def _process_window(self, window: str, task: str, position: int) -> str:
        """处理单个窗口"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"这是文档的一部分（位置：{position}）。请执行任务：{task}"
                },
                {"role": "user", "content": window}
            ]
        )
        return response.choices[0].message.content
```

## LangChain 集成

```python
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Map-Reduce 摘要
def langchain_map_reduce(text: str) -> str:
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000)
    docs = [Document(page_content=t) for t in splitter.split_text(text)]
    
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)

# Refine 摘要
def langchain_refine(text: str) -> str:
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000)
    docs = [Document(page_content=t) for t in splitter.split_text(text)]
    
    chain = load_summarize_chain(llm, chain_type="refine")
    return chain.run(docs)
```

## 策略选择指南

| 场景           | 推荐策略       | 原因                     |
| -------------- | -------------- | ------------------------ |
| 文档摘要       | Map-Reduce     | 并行处理，速度快         |
| 详细分析       | Refine         | 保持连贯性               |
| 超长文档摘要   | 层次化摘要     | 处理任意长度             |
| 问答           | RAG            | 只检索相关部分           |
| 信息提取       | 滑动窗口       | 不遗漏任何部分           |
| 文档 < 上下文  | 直接处理       | 最简单                   |

## 最佳实践

1. **先评估文档长度**：选择合适的策略
2. **合理设置 chunk_size**：太小丢失上下文，太大超出限制
3. **保留 overlap**：避免信息在边界丢失
4. **并行处理**：Map-Reduce 可以并行加速
5. **使用小模型处理中间步骤**：降低成本

## 延伸阅读

- [LangChain Summarization](https://python.langchain.com/docs/tutorials/summarization/)
- [LlamaIndex Document Summary](https://docs.llamaindex.ai/en/stable/examples/index_structs/doc_summary/)
