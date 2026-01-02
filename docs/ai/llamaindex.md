---
sidebar_position: 21
title: 🦙 LlamaIndex 框架
---

# LlamaIndex 框架

LlamaIndex 是专注于数据索引和检索的 LLM 应用框架，特别擅长 RAG 场景。相比 LangChain，它在数据处理和检索优化方面更加专业。

## LlamaIndex vs LangChain

| 特性         | LlamaIndex           | LangChain            |
| ------------ | -------------------- | -------------------- |
| **定位**     | 数据索引与检索       | 通用 LLM 应用框架    |
| **RAG 优化** | 深度优化，开箱即用   | 需要更多配置         |
| **数据连接** | 丰富的数据加载器     | 相对较少             |
| **索引类型** | 多种专业索引         | 主要依赖向量索引     |
| **学习曲线** | 相对简单             | 概念较多             |
| **灵活性**   | RAG 场景灵活         | 通用场景更灵活       |

## 安装

```bash
pip install llama-index
pip install llama-index-llms-openai
pip install llama-index-embeddings-openai
```

## 快速开始

### 5 分钟构建 RAG

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# 配置模型
Settings.llm = OpenAI(model="gpt-4o", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 创建索引
index = VectorStoreIndex.from_documents(documents)

# 查询
query_engine = index.as_query_engine()
response = query_engine.query("文档的主要内容是什么？")
print(response)
```

### 持久化索引

```python
from llama_index.core import StorageContext, load_index_from_storage

# 保存索引
index.storage_context.persist(persist_dir="./storage")

# 加载索引
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

## 核心概念

### 1. Document 和 Node

```python
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# 创建文档
doc = Document(
    text="这是文档内容...",
    metadata={"source": "wiki", "author": "张三"}
)

# 切分为节点
parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
nodes = parser.get_nodes_from_documents([doc])

for node in nodes:
    print(f"Node ID: {node.node_id}")
    print(f"Content: {node.text[:100]}...")
    print(f"Metadata: {node.metadata}")
```

### 2. 索引类型

```python
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    TreeIndex,
    KeywordTableIndex
)

# 向量索引 - 语义搜索
vector_index = VectorStoreIndex.from_documents(documents)

# 摘要索引 - 遍历所有节点
summary_index = SummaryIndex.from_documents(documents)

# 树形索引 - 层次化摘要
tree_index = TreeIndex.from_documents(documents)

# 关键词索引 - 关键词匹配
keyword_index = KeywordTableIndex.from_documents(documents)
```

### 3. 检索器

```python
from llama_index.core.retrievers import VectorIndexRetriever

# 基础检索器
retriever = index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve("查询内容")

# 自定义检索器
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
    filters={"source": "wiki"}  # 元数据过滤
)
```

### 4. 查询引擎

```python
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

# 自定义查询引擎
retriever = index.as_retriever(similarity_top_k=5)
response_synthesizer = get_response_synthesizer(
    response_mode="compact"  # tree_summarize, refine, compact, simple
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer
)

response = query_engine.query("问题")
```

## 数据加载器

### 内置加载器

```python
from llama_index.core import SimpleDirectoryReader

# 自动识别文件类型
documents = SimpleDirectoryReader(
    input_dir="./data",
    recursive=True,
    required_exts=[".pdf", ".docx", ".txt", ".md"]
).load_data()
```

### LlamaHub 加载器

```python
# 安装特定加载器
# pip install llama-index-readers-web

from llama_index.readers.web import SimpleWebPageReader

# 网页加载
loader = SimpleWebPageReader()
documents = loader.load_data(urls=["https://example.com"])

# 更多加载器：https://llamahub.ai/
# - NotionPageReader
# - SlackReader
# - GoogleDocsReader
# - DatabaseReader
```

### 自定义加载器

```python
from llama_index.core.readers.base import BaseReader
from llama_index.core import Document

class CustomReader(BaseReader):
    def load_data(self, file_path: str) -> list[Document]:
        with open(file_path, 'r') as f:
            content = f.read()
        
        return [Document(
            text=content,
            metadata={"source": file_path}
        )]
```

## 高级 RAG 技术

### 1. 查询转换

```python
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

# HyDE: 假设文档嵌入
hyde = HyDEQueryTransform(include_original=True)
query_engine = index.as_query_engine()
hyde_query_engine = TransformQueryEngine(query_engine, hyde)

response = hyde_query_engine.query("什么是机器学习？")
```

### 2. 重排序

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

# 使用重排序模型
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=3
)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker]
)
```

### 3. 混合检索

```python
from llama_index.core.retrievers import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

# BM25 检索器
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=5
)

# 向量检索器
vector_retriever = index.as_retriever(similarity_top_k=5)

# 融合检索器
fusion_retriever = QueryFusionRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    similarity_top_k=5,
    num_queries=1,
    mode="reciprocal_rerank"  # 倒数排名融合
)
```

### 4. 子问题查询

```python
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import SubQuestionQueryEngine

# 创建多个查询引擎工具
tools = [
    QueryEngineTool.from_defaults(
        query_engine=index1.as_query_engine(),
        name="financial_data",
        description="财务数据查询"
    ),
    QueryEngineTool.from_defaults(
        query_engine=index2.as_query_engine(),
        name="product_info",
        description="产品信息查询"
    )
]

# 子问题查询引擎
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=tools
)

# 复杂问题会被分解为子问题
response = query_engine.query("比较产品A和产品B的财务表现")
```

### 5. 递归检索

```python
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# 文档 -> 摘要 -> 详细内容的递归检索
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    node_dict=all_nodes_dict,
    verbose=True
)

query_engine = RetrieverQueryEngine.from_args(recursive_retriever)
```

## Chat Engine

```python
from llama_index.core.memory import ChatMemoryBuffer

# 创建聊天引擎
memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

chat_engine = index.as_chat_engine(
    chat_mode="context",  # context, condense_question, react
    memory=memory,
    system_prompt="你是一个有帮助的助手。"
)

# 多轮对话
response = chat_engine.chat("你好")
print(response)

response = chat_engine.chat("继续上面的话题")
print(response)

# 重置对话
chat_engine.reset()
```

## Agent

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

# 定义工具
def multiply(a: int, b: int) -> int:
    """两数相乘"""
    return a * b

def add(a: int, b: int) -> int:
    """两数相加"""
    return a + b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

# 创建 Agent
agent = ReActAgent.from_tools(
    tools=[multiply_tool, add_tool],
    llm=Settings.llm,
    verbose=True
)

response = agent.chat("计算 (3 + 5) * 2")
print(response)
```

## 向量数据库集成

### Chroma

```python
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

# 创建 Chroma 客户端
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("documents")

# 创建向量存储
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 创建索引
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)
```

### Milvus

```python
from llama_index.vector_stores.milvus import MilvusVectorStore

vector_store = MilvusVectorStore(
    uri="http://localhost:19530",
    collection_name="documents",
    dim=1536
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
```

## 评估

```python
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator
)

# 忠实度评估
faithfulness_evaluator = FaithfulnessEvaluator()
result = faithfulness_evaluator.evaluate_response(
    query="问题",
    response=response
)
print(f"忠实度: {result.passing}")

# 相关性评估
relevancy_evaluator = RelevancyEvaluator()
result = relevancy_evaluator.evaluate_response(
    query="问题",
    response=response
)
print(f"相关性: {result.passing}")
```

## 最佳实践

1. **选择合适的索引**：简单场景用 VectorStoreIndex，复杂场景考虑组合索引
2. **优化切分策略**：根据文档类型调整 chunk_size 和 overlap
3. **使用重排序**：提高检索精度
4. **添加元数据**：便于过滤和追溯
5. **持久化索引**：避免重复构建

## 延伸阅读

- [LlamaIndex 官方文档](https://docs.llamaindex.ai/)
- [LlamaHub 数据加载器](https://llamahub.ai/)
- [LlamaIndex GitHub](https://github.com/run-llama/llama_index)
