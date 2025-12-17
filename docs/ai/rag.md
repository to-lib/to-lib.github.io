---
sidebar_position: 4
title: 📚 RAG (检索增强生成)
---

# RAG (检索增强生成)

**RAG (Retrieval-Augmented Generation)** 是一种结合了**检索 (Retrieval)** 和 **生成 (Generation)** 的 AI 技术架构。它通过在生成回答之前先从外部知识库中检索相关信息，并将其作为上下文输入给大型语言模型 (LLM)，从而显著提升回答的准确性和时效性。

## 为什么需要 RAG？

LLM (如 GPT-4) 存在以下局限性：

- **知识截止**：模型训练数据是静态的，无法获知最新的时事。
- **幻觉 (Hallucination)**：在不知道答案时可能会一本正经地胡说八道。
- **私有数据缺失**：模型从未见过企业的内部文档和私有数据。

RAG 通过外挂知识库通过解决了这些问题。

## RAG 的工作流程

RAG 的典型流程包含三个阶段：

### 1. 索引 (Indexing) - 准备阶段

将文档转换为向量并存入数据库。

- **加载 (Load)**：读取 PDF, Word, Markdown, HTML 等文件。
- **切分 (Split)**：将长文档切分为较小的文本块 (Chunks)。
- **嵌入 (Embed)**：使用 Embedding 模型将文本块转换为向量 (Vectors)。
- **存储 (Store)**：将向量存储到向量数据库 (Vector DB) 中。

### 2. 检索 (Retrieval) - 运行阶段

- **查询编码**：将用户的自然语言问题转换为向量。
- **相似度搜索**：在向量数据库中查找与问题向量最相似的文本块 (Top-K)。

### 3. 生成 (Generation) - 运行阶段

- **构建 Prompt**：将检索到的文本块作为“上下文 (Context)”填入 Prompt 模板。
- **LLM 回答**：LLM 基于提供的上下文回答用户的问题。

## 核心技术栈

### 向量数据库 (Vector Database)

- **Pinecone**: 托管型向量数据库，易于使用。
- **Milvus**: 开源高性能向量数据库。
- **Chroma**: 轻量级开源向量数据库，适合本地开发。
- **Elasticsearch / pgvector**: 传统数据库的向量扩展。

### 开发框架

- **LangChain**: 最流行的 LLM 应用开发框架，提供了丰富的 RAG 组件。
- **LlamaIndex**: 专注于数据索引和检索的框架，对 RAG 优化极佳。

## 高级 RAG 技巧

- **混合检索 (Hybrid Search)**：结合关键词检索 (BM25) 和向量检索，提高召回率。
- **重排序 (Re-ranking)**：检索出较多结果后，使用专门的 Re-rank 模型进行精细排序。
- **元数据过滤 (Metadata Filtering)**：在检索前通过时间、作者等标签过滤数据。
- **查询重写 (Query Rewriting)**：将用户模糊的问题改写为更适合检索的形式。

## 代码实现示例

### 使用 LangChain 构建 RAG

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 加载文档
loader = DirectoryLoader("./docs", glob="**/*.md", loader_cls=TextLoader)
documents = loader.load()

# 2. 切分文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "！", "？", "；", " "]
)
chunks = text_splitter.split_documents(documents)

# 3. 创建向量存储
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# 4. 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 5. 创建 RAG 链
llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_template("""
请根据以下上下文回答问题。如果上下文中没有相关信息，请说"我没有找到相关信息"。

上下文：
{context}

问题：{question}

回答：
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. 使用
answer = rag_chain.invoke("什么是 Spring Boot 的自动配置？")
print(answer)
```

### 使用 LlamaIndex 构建 RAG

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# 配置模型
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o", temperature=0)

# 加载文档并创建索引
documents = SimpleDirectoryReader("./docs").load_data()
index = VectorStoreIndex.from_documents(documents)

# 创建查询引擎
query_engine = index.as_query_engine(similarity_top_k=3)

# 查询
response = query_engine.query("什么是 RAG？")
print(response)
```

### 添加混合检索 (Hybrid Search)

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# 向量检索器
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# BM25 关键词检索器
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

# 集成检索器 (权重可调)
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.6, 0.4]  # 60% 向量 + 40% 关键词
)

# 使用
docs = hybrid_retriever.invoke("Spring Boot 配置文件")
```

## Embedding 模型选择指南

| 使用场景   | 推荐模型                 | 说明               |
| ---------- | ------------------------ | ------------------ |
| 英文通用   | `text-embedding-3-small` | 经济实惠           |
| 英文高精度 | `text-embedding-3-large` | 更高准确率         |
| 中文通用   | `bge-large-zh-v1.5`      | 开源最佳           |
| 多语言     | `bge-m3`                 | 支持 100+ 语言     |
| 本地部署   | `m3e-base`               | 轻量，适合边缘设备 |

### 使用开源 Embedding 模型

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

# 使用方式与 OpenAI Embeddings 相同
vectorstore = Chroma.from_documents(chunks, embeddings)
```

## 最佳实践

1. **文档预处理**：清理无关内容，保留结构化信息
2. **合理切分**：chunk_size 500-1500，overlap 10-20%
3. **元数据丰富**：添加来源、时间、类别等元数据
4. **混合检索**：结合向量和关键词，提高召回
5. **结果重排序**：使用 Re-ranker 提升精度
6. **定期更新**：保持知识库时效性

## 延伸阅读

- [LangChain RAG 教程](https://python.langchain.com/docs/tutorials/rag/)
- [LlamaIndex 文档](https://docs.llamaindex.ai/)
- [RAG 论文](https://arxiv.org/abs/2005.11401)
