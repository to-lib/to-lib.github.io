---
sidebar_position: 29
title: 🕸️ GraphRAG
---

# GraphRAG（知识图谱增强检索）

GraphRAG 结合知识图谱和向量检索，提供更准确、更有上下文的检索结果。特别适合需要理解实体关系的复杂查询。

## 传统 RAG vs GraphRAG

| 特性 | 传统 RAG | GraphRAG |
|------|---------|----------|
| 检索方式 | 向量相似度 | 图遍历 + 向量 |
| 关系理解 | 弱 | 强 |
| 多跳推理 | 困难 | 自然支持 |
| 全局摘要 | 不支持 | 支持 |
| 构建成本 | 低 | 高 |

## 工作原理

```
文档 ──> 实体提取 ──> 关系抽取 ──> 知识图谱
                                    │
查询 ──> 实体识别 ──> 图检索 ──> 子图 ──> LLM ──> 答案
                        │
                    向量检索 ──> 相关文档
```

## 微软 GraphRAG

### 安装

```bash
pip install graphrag
```

### 初始化项目

```bash
# 创建项目目录
mkdir my_graphrag
cd my_graphrag

# 初始化
python -m graphrag.index --init --root .
```

### 配置

```yaml
# settings.yaml
llm:
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o
  
embeddings:
  api_key: ${OPENAI_API_KEY}
  model: text-embedding-3-small

chunks:
  size: 1200
  overlap: 100

entity_extraction:
  max_gleanings: 1

community_reports:
  max_length: 2000
```

### 构建索引

```bash
# 将文档放入 input 目录
cp documents/*.txt ./input/

# 构建索引
python -m graphrag.index --root .
```

### 查询

```bash
# 全局查询（适合摘要性问题）
python -m graphrag.query --root . --method global "文档的主要主题是什么？"

# 局部查询（适合具体问题）
python -m graphrag.query --root . --method local "张三和李四是什么关系？"
```

## 手动实现 GraphRAG

### 1. 实体和关系提取

```python
from openai import OpenAI
import json
from dataclasses import dataclass

client = OpenAI()

@dataclass
class Entity:
    name: str
    type: str
    description: str

@dataclass
class Relationship:
    source: str
    target: str
    relation: str
    description: str

def extract_entities_and_relations(text: str) -> tuple[list[Entity], list[Relationship]]:
    """从文本中提取实体和关系"""
    prompt = f"""
从以下文本中提取实体和关系。

文本：
{text}

返回 JSON 格式：
{{
    "entities": [
        {{"name": "实体名", "type": "类型(人物/组织/地点/概念)", "description": "描述"}}
    ],
    "relationships": [
        {{"source": "源实体", "target": "目标实体", "relation": "关系类型", "description": "关系描述"}}
    ]
}}
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    data = json.loads(response.choices[0].message.content)
    
    entities = [Entity(**e) for e in data.get("entities", [])]
    relationships = [Relationship(**r) for r in data.get("relationships", [])]
    
    return entities, relationships
```

### 2. 构建知识图谱

```python
import networkx as nx
from langchain_openai import OpenAIEmbeddings
import numpy as np

class KnowledgeGraph:
    """知识图谱"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.entity_embeddings = {}
    
    def add_entity(self, entity: Entity):
        """添加实体"""
        self.graph.add_node(
            entity.name,
            type=entity.type,
            description=entity.description
        )
        # 计算嵌入
        self.entity_embeddings[entity.name] = self.embeddings.embed_query(
            f"{entity.name}: {entity.description}"
        )
    
    def add_relationship(self, rel: Relationship):
        """添加关系"""
        self.graph.add_edge(
            rel.source,
            rel.target,
            relation=rel.relation,
            description=rel.description
        )
    
    def build_from_documents(self, documents: list[str]):
        """从文档构建图谱"""
        for doc in documents:
            entities, relationships = extract_entities_and_relations(doc)
            
            for entity in entities:
                self.add_entity(entity)
            
            for rel in relationships:
                if rel.source in self.graph and rel.target in self.graph:
                    self.add_relationship(rel)
    
    def find_similar_entities(self, query: str, top_k: int = 5) -> list[str]:
        """找到与查询相似的实体"""
        query_embedding = self.embeddings.embed_query(query)
        
        similarities = []
        for entity, embedding in self.entity_embeddings.items():
            sim = np.dot(query_embedding, embedding)
            similarities.append((entity, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [e[0] for e in similarities[:top_k]]
    
    def get_subgraph(self, entities: list[str], hops: int = 2) -> nx.DiGraph:
        """获取实体周围的子图"""
        nodes = set(entities)
        
        for _ in range(hops):
            new_nodes = set()
            for node in nodes:
                if node in self.graph:
                    new_nodes.update(self.graph.predecessors(node))
                    new_nodes.update(self.graph.successors(node))
            nodes.update(new_nodes)
        
        return self.graph.subgraph(nodes)
    
    def subgraph_to_text(self, subgraph: nx.DiGraph) -> str:
        """将子图转换为文本描述"""
        lines = []
        
        # 实体描述
        lines.append("实体：")
        for node in subgraph.nodes():
            data = self.graph.nodes[node]
            lines.append(f"- {node} ({data.get('type', '未知')}): {data.get('description', '')}")
        
        # 关系描述
        lines.append("\n关系：")
        for source, target, data in subgraph.edges(data=True):
            lines.append(f"- {source} --[{data.get('relation', '')}]--> {target}")
        
        return "\n".join(lines)
```


### 3. GraphRAG 查询

```python
class GraphRAG:
    """GraphRAG 查询系统"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.client = OpenAI()
    
    def query(self, question: str) -> str:
        """查询"""
        # 1. 找到相关实体
        relevant_entities = self.kg.find_similar_entities(question, top_k=5)
        
        # 2. 获取子图
        subgraph = self.kg.get_subgraph(relevant_entities, hops=2)
        
        # 3. 转换为上下文
        context = self.kg.subgraph_to_text(subgraph)
        
        # 4. 生成答案
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"根据以下知识图谱信息回答问题。\n\n{context}"
                },
                {"role": "user", "content": question}
            ]
        )
        
        return response.choices[0].message.content

# 使用
kg = KnowledgeGraph()
kg.build_from_documents(documents)

rag = GraphRAG(kg)
answer = rag.query("张三和李四是什么关系？")
```

## 混合检索

结合向量检索和图检索。

```python
from langchain_community.vectorstores import Chroma

class HybridGraphRAG:
    """混合 GraphRAG"""
    
    def __init__(self, documents: list[str]):
        # 向量存储
        self.vectorstore = Chroma.from_texts(
            texts=documents,
            embedding=OpenAIEmbeddings(model="text-embedding-3-small")
        )
        
        # 知识图谱
        self.kg = KnowledgeGraph()
        self.kg.build_from_documents(documents)
        
        self.client = OpenAI()
    
    def query(self, question: str) -> str:
        # 向量检索
        vector_docs = self.vectorstore.similarity_search(question, k=3)
        vector_context = "\n".join([d.page_content for d in vector_docs])
        
        # 图检索
        entities = self.kg.find_similar_entities(question, top_k=3)
        subgraph = self.kg.get_subgraph(entities, hops=1)
        graph_context = self.kg.subgraph_to_text(subgraph)
        
        # 合并上下文
        combined_context = f"""
文档片段：
{vector_context}

知识图谱：
{graph_context}
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"根据以下信息回答问题。\n\n{combined_context}"
                },
                {"role": "user", "content": question}
            ]
        )
        
        return response.choices[0].message.content
```

## LlamaIndex GraphRAG

```python
from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.graph_stores.neo4j import Neo4jGraphStore

# 配置 Neo4j
graph_store = Neo4jGraphStore(
    username="neo4j",
    password="password",
    url="bolt://localhost:7687",
    database="neo4j"
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

# 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 构建知识图谱索引
index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=10,
    include_embeddings=True
)

# 查询
query_engine = index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize"
)

response = query_engine.query("主要人物之间的关系是什么？")
```

## 社区检测与摘要

```python
from networkx.algorithms import community

def detect_communities(kg: KnowledgeGraph) -> list[set]:
    """检测社区"""
    # 转换为无向图进行社区检测
    undirected = kg.graph.to_undirected()
    communities = community.louvain_communities(undirected)
    return communities

def summarize_community(kg: KnowledgeGraph, nodes: set) -> str:
    """生成社区摘要"""
    subgraph = kg.graph.subgraph(nodes)
    context = kg.subgraph_to_text(subgraph)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "根据以下知识图谱片段，生成一个简洁的摘要。"
            },
            {"role": "user", "content": context}
        ],
        max_tokens=500
    )
    
    return response.choices[0].message.content

# 全局查询：使用社区摘要
def global_query(kg: KnowledgeGraph, question: str) -> str:
    communities = detect_communities(kg)
    summaries = [summarize_community(kg, c) for c in communities[:5]]
    
    combined = "\n\n".join([f"主题 {i+1}：{s}" for i, s in enumerate(summaries)])
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"根据以下主题摘要回答问题。\n\n{combined}"
            },
            {"role": "user", "content": question}
        ]
    )
    
    return response.choices[0].message.content
```

## 最佳实践

1. **选择合适的场景**：关系密集的数据更适合 GraphRAG
2. **控制图规模**：大图需要分区或采样
3. **混合使用**：结合向量检索和图检索
4. **增量更新**：支持图谱的增量构建
5. **质量控制**：验证提取的实体和关系

## 延伸阅读

- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [LlamaIndex Knowledge Graph](https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/)
- [Neo4j + LLM](https://neo4j.com/developer/genai/)