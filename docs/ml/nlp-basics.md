---
sidebar_position: 21
title: 📝 NLP 基础
---

# 自然语言处理基础

NLP 让计算机理解和处理人类语言。

## 文本预处理

```python
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

def preprocess_text(text):
    # 小写化
    text = text.lower()
    # 去除特殊字符
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 分词
    tokens = word_tokenize(text)
    # 去停用词
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    # 词形还原
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens
```

## 文本表示

### TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(texts)
```

### Word2Vec

```python
from gensim.models import Word2Vec

# 训练词向量
sentences = [['I', 'love', 'NLP'], ['NLP', 'is', 'fun']]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# 获取词向量
vector = model.wv['NLP']

# 相似词
similar = model.wv.most_similar('NLP', topn=5)
```

### 预训练词向量

```python
import gensim.downloader as api

# 加载预训练模型
glove = api.load('glove-wiki-gigaword-100')
vector = glove['king']
```

## 文本分类

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

text_clf.fit(X_train, y_train)
```

### 使用 BERT

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

inputs = tokenizer("I love this movie!", return_tensors='pt', padding=True, truncation=True)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=1)
```

## 命名实体识别 (NER)

```python
from transformers import pipeline

ner = pipeline('ner', grouped_entities=True)
result = ner("Apple was founded by Steve Jobs in California.")
# [{'entity_group': 'ORG', 'word': 'Apple'},
#  {'entity_group': 'PER', 'word': 'Steve Jobs'},
#  {'entity_group': 'LOC', 'word': 'California'}]
```

## 文本相似度

```python
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF 相似度
tfidf = TfidfVectorizer()
vectors = tfidf.fit_transform(texts)
similarity = cosine_similarity(vectors[0:1], vectors[1:2])

# Sentence Transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(['text1', 'text2'])
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
```

## 常用任务

| 任务     | 描述               | 模型          |
| -------- | ------------------ | ------------- |
| 文本分类 | 情感分析、主题分类 | BERT, RoBERTa |
| NER      | 识别实体           | BERT-NER      |
| 问答     | 阅读理解           | BERT-QA       |
| 文本生成 | 写作、对话         | GPT           |
| 翻译     | 机器翻译           | MarianMT      |
| 摘要     | 文本摘要           | BART, T5      |
