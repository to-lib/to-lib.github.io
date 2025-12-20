---
sidebar_position: 4
title: 🧩 Embeddings（向量表示）
---

# Embeddings（向量表示）

Embedding（向量表示）是把文本/图片等内容映射到高维向量空间的技术，使得“语义相近”的内容在向量空间里距离更近。它是 RAG、语义搜索、相似度匹配、聚类与推荐的基础能力。

## 什么时候需要 Embeddings

- **语义检索**：在文档库中用自然语言找到最相关片段。
- **相似度去重/聚类**：合并重复问题、对工单/评论自动聚类。
- **召回阶段**：与 BM25 等关键词检索组合形成混合检索。
- **长期记忆**：为 Agent 存储长期记忆（对话摘要、事件、偏好）。

## Embedding 模型如何选

关注几个核心维度：

- **语言与领域**：中文优先选中文效果强的模型（如 `bge-large-zh`、`bge-m3`）。
- **精度 vs 成本**：商业模型通常效果稳定、成本更高；开源模型可本地部署。
- **维度**：维度越大通常越精确，但存储/计算成本也更高。
- **最大输入长度**：要覆盖你的 chunk 大小（按 token 计）。

## 向量相似度与距离

常见相似度：

- **Cosine Similarity（余弦相似度）**：最常见，通常需要向量归一化。
- **Dot Product（点积）**：与归一化后的余弦相似度等价。
- **L2 Distance（欧氏距离）**：一些向量库支持。

:::tip 归一化建议
如果你的模型/库没有保证归一化，建议在入库前统一对向量做 L2 normalization，以便余弦/点积表现稳定。
:::

## 文档切分（Chunking）实践

- **chunk_size**：500-1500 token（取决于模型上下文与文档结构）。
- **chunk_overlap**：10%-20%（减少“句子被切断”导致召回失败）。
- **按结构切分**：优先用标题/段落边界；必要时再退化到句子/字符切分。
- **保留元数据**：来源、路径、更新时间、权限标签（非常重要）。

## 代码示例

### OpenAI Embeddings

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

resp = client.embeddings.create(
    model="text-embedding-3-small",
    input=["今天北京天气怎么样？", "上海的天气如何？"],
)

vectors = [item.embedding for item in resp.data]
print(len(vectors), len(vectors[0]))
```

### Hugging Face / BGE（本地推理）

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
texts = ["向量检索", "语义搜索"]
embeddings = model.encode(texts, normalize_embeddings=True)
print(embeddings.shape)
```

## 常见坑

- **把太长的 chunk 直接 embedding**：会丢信息且召回不稳定；需要切分。
- **混用不同 embedding 模型**：同一向量库里应保证同一模型/同一版本产出的向量。
- **忽略权限与租户隔离**：RAG 的安全问题经常来自元数据过滤缺失。
- **只看 Top-K，不做重排序**：召回质量到一定规模后，Re-rank 往往是质变点。

## 延伸阅读

- https://platform.openai.com/docs/guides/embeddings
- https://github.com/FlagOpen/FlagEmbedding
