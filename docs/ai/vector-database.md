---
sidebar_position: 20
title: 🗄️ 向量数据库实战
---

# 向量数据库实战

向量数据库是 RAG 和语义搜索的核心组件，用于存储和检索高维向量。本文介绍主流向量数据库的使用方法。

## 向量数据库对比

| 数据库       | 类型   | 特点                     | 适用场景         |
| ------------ | ------ | ------------------------ | ---------------- |
| **Chroma**   | 嵌入式 | 轻量、易用、Python 原生  | 开发测试、小规模 |
| **Milvus**   | 分布式 | 高性能、可扩展           | 生产环境、大规模 |
| **Pinecone** | 托管   | 全托管、免运维           | 快速上线、企业级 |
| **Qdrant**   | 开源   | Rust 实现、高性能        | 生产环境         |
| **pgvector** | 扩展   | PostgreSQL 插件          | 已有 PG 基础设施 |
| **Weaviate** | 开源   | GraphQL API、多模态      | 复杂查询需求     |

## Chroma

Chroma 是最简单的向量数据库，适合快速开发和原型验证。

### 安装

```bash
pip install chromadb
```

### 基础使用

```python
import chromadb
from chromadb.utils import embedding_functions

# 创建客户端
client = chromadb.Client()  # 内存模式
# client = chromadb.PersistentClient(path="./chroma_db")  # 持久化

# 使用 OpenAI Embedding
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-api-key",
    model_name="text-embedding-3-small"
)

# 创建集合
collection = client.create_collection(
    name="documents",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
)

# 添加文档
collection.add(
    documents=["机器学习是人工智能的一个分支", "深度学习使用神经网络", "自然语言处理处理文本数据"],
    metadatas=[{"source": "wiki"}, {"source": "book"}, {"source": "paper"}],
    ids=["doc1", "doc2", "doc3"]
)

# 查询
results = collection.query(
    query_texts=["什么是 AI？"],
    n_results=2,
    where={"source": "wiki"}  # 元数据过滤
)

print(results["documents"])
print(results["distances"])
```

### 更新和删除

```python
# 更新
collection.update(
    ids=["doc1"],
    documents=["更新后的内容"],
    metadatas=[{"source": "updated"}]
)

# 删除
collection.delete(ids=["doc1"])

# 按条件删除
collection.delete(where={"source": "wiki"})
```

### 与 LangChain 集成

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 从文档创建
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 检索
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("查询内容")
```

## Milvus

Milvus 是高性能的分布式向量数据库，适合生产环境。

### 安装

```bash
# 使用 Docker
docker run -d --name milvus \
    -p 19530:19530 \
    -p 9091:9091 \
    milvusdb/milvus:latest

# Python SDK
pip install pymilvus
```

### 基础使用

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# 连接
connections.connect("default", host="localhost", port="19530")

# 定义 Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256)
]

schema = CollectionSchema(fields, description="文档集合")

# 创建集合
collection = Collection("documents", schema)

# 创建索引
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
collection.create_index("embedding", index_params)

# 插入数据
data = [
    ["文档1内容", "文档2内容"],  # text
    [[0.1] * 1536, [0.2] * 1536],  # embedding
    ["wiki", "book"]  # source
]
collection.insert(data)

# 加载到内存
collection.load()

# 搜索
search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
results = collection.search(
    data=[[0.1] * 1536],  # 查询向量
    anns_field="embedding",
    param=search_params,
    limit=5,
    expr="source == 'wiki'",  # 过滤条件
    output_fields=["text", "source"]
)

for hits in results:
    for hit in hits:
        print(f"ID: {hit.id}, Distance: {hit.distance}")
        print(f"Text: {hit.entity.get('text')}")
```

### 高级功能

```python
# 混合搜索（向量 + 标量）
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=10,
    expr="source in ['wiki', 'book'] and length(text) > 100",
    output_fields=["text", "source"]
)

# 批量插入
from pymilvus import utility

# 分批插入大量数据
batch_size = 1000
for i in range(0, len(all_data), batch_size):
    batch = all_data[i:i+batch_size]
    collection.insert(batch)
    collection.flush()

# 删除
collection.delete(expr="id in [1, 2, 3]")
```

### 与 LangChain 集成

```python
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

vectorstore = Milvus.from_documents(
    documents=docs,
    embedding=embeddings,
    connection_args={"host": "localhost", "port": "19530"},
    collection_name="langchain_docs"
)

# 检索
docs = vectorstore.similarity_search("查询", k=3)
```

## Pinecone

Pinecone 是全托管的向量数据库服务。

### 安装

```bash
pip install pinecone-client
```

### 基础使用

```python
from pinecone import Pinecone, ServerlessSpec

# 初始化
pc = Pinecone(api_key="your-api-key")

# 创建索引
pc.create_index(
    name="documents",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# 获取索引
index = pc.Index("documents")

# 插入向量
index.upsert(
    vectors=[
        {
            "id": "doc1",
            "values": [0.1] * 1536,
            "metadata": {"text": "文档内容", "source": "wiki"}
        }
    ],
    namespace="default"
)

# 查询
results = index.query(
    vector=[0.1] * 1536,
    top_k=5,
    include_metadata=True,
    filter={"source": {"$eq": "wiki"}}
)

for match in results["matches"]:
    print(f"ID: {match['id']}, Score: {match['score']}")
    print(f"Metadata: {match['metadata']}")
```

### 批量操作

```python
# 批量插入
vectors = [
    {"id": f"doc{i}", "values": embeddings[i], "metadata": {"text": texts[i]}}
    for i in range(len(texts))
]

# 分批上传
batch_size = 100
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i+batch_size]
    index.upsert(vectors=batch)

# 删除
index.delete(ids=["doc1", "doc2"])
index.delete(filter={"source": {"$eq": "wiki"}})
```

## pgvector

pgvector 是 PostgreSQL 的向量扩展，适合已有 PostgreSQL 基础设施的场景。

### 安装

```sql
-- PostgreSQL 中安装扩展
CREATE EXTENSION vector;
```

```bash
pip install pgvector psycopg2-binary
```

### 基础使用

```python
import psycopg2
from pgvector.psycopg2 import register_vector

# 连接
conn = psycopg2.connect("postgresql://user:pass@localhost/db")
register_vector(conn)

cur = conn.cursor()

# 创建表
cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        text TEXT,
        embedding vector(1536),
        source VARCHAR(256),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

# 创建索引
cur.execute("""
    CREATE INDEX ON documents 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100)
""")

# 插入
cur.execute(
    "INSERT INTO documents (text, embedding, source) VALUES (%s, %s, %s)",
    ("文档内容", [0.1] * 1536, "wiki")
)

# 查询（余弦相似度）
cur.execute("""
    SELECT id, text, source, 1 - (embedding <=> %s) as similarity
    FROM documents
    WHERE source = %s
    ORDER BY embedding <=> %s
    LIMIT 5
""", ([0.1] * 1536, "wiki", [0.1] * 1536))

results = cur.fetchall()
for row in results:
    print(f"ID: {row[0]}, Similarity: {row[3]:.4f}")

conn.commit()
```

### 使用 SQLAlchemy

```python
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True)
    text = Column(Text)
    embedding = Column(Vector(1536))
    source = Column(String(256))

engine = create_engine("postgresql://user:pass@localhost/db")
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

# 插入
doc = Document(text="内容", embedding=[0.1] * 1536, source="wiki")
session.add(doc)
session.commit()

# 查询
from sqlalchemy import select

query_embedding = [0.1] * 1536
results = session.scalars(
    select(Document)
    .order_by(Document.embedding.cosine_distance(query_embedding))
    .limit(5)
).all()
```

## Qdrant

Qdrant 是用 Rust 编写的高性能向量数据库。

### 安装

```bash
# Docker
docker run -p 6333:6333 qdrant/qdrant

# Python SDK
pip install qdrant-client
```

### 基础使用

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 连接
client = QdrantClient(host="localhost", port=6333)

# 创建集合
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# 插入
client.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=1,
            vector=[0.1] * 1536,
            payload={"text": "文档内容", "source": "wiki"}
        )
    ]
)

# 搜索
results = client.search(
    collection_name="documents",
    query_vector=[0.1] * 1536,
    query_filter={
        "must": [{"key": "source", "match": {"value": "wiki"}}]
    },
    limit=5
)

for result in results:
    print(f"ID: {result.id}, Score: {result.score}")
    print(f"Payload: {result.payload}")
```

## 最佳实践

### 1. 索引选择

| 索引类型   | 特点                 | 适用场景       |
| ---------- | -------------------- | -------------- |
| Flat       | 精确搜索，慢         | 小数据集       |
| IVF        | 近似搜索，快         | 中等数据集     |
| HNSW       | 高召回率，内存占用大 | 高精度需求     |
| PQ         | 压缩存储，速度快     | 大规模数据     |

### 2. 元数据设计

```python
# 好的元数据设计
metadata = {
    "source": "wiki",           # 来源
    "category": "technology",   # 分类
    "created_at": "2024-01-01", # 时间
    "author": "张三",           # 作者
    "access_level": "public",   # 权限
    "chunk_index": 0,           # 分块索引
    "doc_id": "doc_123"         # 原文档 ID
}
```

### 3. 分片策略

```python
# 按租户分片
collection_name = f"tenant_{tenant_id}_documents"

# 按时间分片
collection_name = f"documents_{year}_{month}"
```

### 4. 性能优化

```python
# 批量操作
batch_size = 1000
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i+batch_size]
    collection.insert(batch)

# 预热索引
collection.load()

# 调整索引参数
index_params = {
    "index_type": "IVF_PQ",
    "params": {
        "nlist": 2048,
        "m": 16,
        "nbits": 8
    }
}
```

## 延伸阅读

- [Chroma 文档](https://docs.trychroma.com/)
- [Milvus 文档](https://milvus.io/docs)
- [Pinecone 文档](https://docs.pinecone.io/)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [Qdrant 文档](https://qdrant.tech/documentation/)
