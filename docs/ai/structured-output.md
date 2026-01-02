---
sidebar_position: 17
title: 📐 结构化输出
---

# 结构化输出 (Structured Output)

结构化输出是让 LLM 按照指定的格式（如 JSON、XML）返回数据的技术。这对于需要程序解析 LLM 输出的场景至关重要。

## 为什么需要结构化输出？

| 问题           | 说明                                   |
| -------------- | -------------------------------------- |
| **解析困难**   | 自由文本难以可靠地提取结构化信息       |
| **格式不稳定** | 同样的 prompt 可能返回不同格式         |
| **类型不安全** | 无法保证字段类型正确                   |
| **缺失字段**   | 模型可能遗漏必要字段                   |

## OpenAI JSON Mode

### 基础用法

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "你是一个数据提取助手。请以 JSON 格式返回结果。"
        },
        {
            "role": "user",
            "content": "提取以下文本中的人物信息：张三，男，30岁，北京人，软件工程师"
        }
    ],
    response_format={"type": "json_object"}
)

import json
data = json.loads(response.choices[0].message.content)
print(data)
# {"name": "张三", "gender": "男", "age": 30, "city": "北京", "occupation": "软件工程师"}
```

### 指定 JSON Schema

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": """提取人物信息，严格按照以下 JSON Schema 返回：
{
  "name": "string",
  "age": "number",
  "city": "string",
  "skills": ["string"]
}"""
        },
        {"role": "user", "content": "李四，25岁，上海，会 Python 和 Java"}
    ],
    response_format={"type": "json_object"}
)
```

## OpenAI Structured Outputs（推荐）

OpenAI 的 Structured Outputs 功能可以保证输出严格符合指定的 JSON Schema。

### 使用 Pydantic 定义 Schema

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional

client = OpenAI()

# 定义数据模型
class Person(BaseModel):
    name: str
    age: int
    city: str
    skills: List[str]
    email: Optional[str] = None

class ExtractionResult(BaseModel):
    people: List[Person]
    summary: str

# 使用 parse 方法
completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "从文本中提取人物信息"},
        {"role": "user", "content": "张三30岁在北京会Python；李四25岁在上海会Java和Go"}
    ],
    response_format=ExtractionResult
)

result = completion.choices[0].message.parsed
print(result.people[0].name)  # 张三
print(result.summary)
```

### 复杂嵌套结构

```python
from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Task(BaseModel):
    title: str = Field(description="任务标题")
    description: str = Field(description="任务描述")
    priority: Priority = Field(description="优先级")
    estimated_hours: float = Field(ge=0, le=100, description="预估工时")

class ProjectPlan(BaseModel):
    project_name: str
    tasks: List[Task]
    total_hours: float
    risks: List[str]

completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是项目规划助手"},
        {"role": "user", "content": "帮我规划一个电商网站项目，包含用户系统、商品管理、订单系统"}
    ],
    response_format=ProjectPlan
)

plan = completion.choices[0].message.parsed
for task in plan.tasks:
    print(f"[{task.priority.value}] {task.title}: {task.estimated_hours}h")
```

## Anthropic 结构化输出

### 使用 Tool Use 实现

```python
import anthropic
import json

client = anthropic.Anthropic()

# 定义工具作为输出格式
tools = [
    {
        "name": "extract_person",
        "description": "提取人物信息",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "姓名"},
                "age": {"type": "integer", "description": "年龄"},
                "city": {"type": "string", "description": "城市"},
                "occupation": {"type": "string", "description": "职业"}
            },
            "required": ["name", "age", "city"]
        }
    }
]

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "tool", "name": "extract_person"},  # 强制使用工具
    messages=[
        {"role": "user", "content": "提取信息：王五，28岁，深圳，产品经理"}
    ]
)

# 获取结构化输出
tool_use = next(block for block in message.content if block.type == "tool_use")
result = tool_use.input
print(result)
# {"name": "王五", "age": 28, "city": "深圳", "occupation": "产品经理"}
```

## LangChain 结构化输出

### 使用 with_structured_output

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

class MovieReview(BaseModel):
    """电影评论分析结果"""
    movie_name: str = Field(description="电影名称")
    sentiment: str = Field(description="情感倾向：positive/negative/neutral")
    score: float = Field(ge=0, le=10, description="评分")
    keywords: List[str] = Field(description="关键词")
    summary: str = Field(description="一句话总结")

llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm = llm.with_structured_output(MovieReview)

result = structured_llm.invoke("这部电影太精彩了！剧情紧凑，演员演技在线，特效震撼，强烈推荐！")
print(f"情感: {result.sentiment}, 评分: {result.score}")
```

### 使用 PydanticOutputParser

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

parser = PydanticOutputParser(pydantic_object=MovieReview)

prompt = PromptTemplate(
    template="分析以下电影评论：\n{review}\n\n{format_instructions}",
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

result = chain.invoke({"review": "剧情拖沓，演技尴尬，浪费时间"})
```

## 实战应用

### 1. 信息抽取

```python
from pydantic import BaseModel
from typing import List, Optional

class ContactInfo(BaseModel):
    name: str
    phone: Optional[str] = None
    email: Optional[str] = None
    company: Optional[str] = None
    position: Optional[str] = None

class ExtractionResult(BaseModel):
    contacts: List[ContactInfo]
    raw_text: str

def extract_contacts(text: str) -> ExtractionResult:
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "从文本中提取联系人信息"},
            {"role": "user", "content": text}
        ],
        response_format=ExtractionResult
    )
    return completion.choices[0].message.parsed

# 使用
text = """
请联系张经理（手机：13800138000，邮箱：zhang@example.com）
或者李总（ABC公司CEO，电话：13900139000）
"""
result = extract_contacts(text)
for contact in result.contacts:
    print(f"{contact.name}: {contact.phone}")
```

### 2. 分类任务

```python
from pydantic import BaseModel
from typing import Literal, List

class ClassificationResult(BaseModel):
    category: Literal["bug", "feature", "question", "other"]
    confidence: float
    reasoning: str
    suggested_labels: List[str]

def classify_issue(title: str, description: str) -> ClassificationResult:
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "对 GitHub Issue 进行分类"},
            {"role": "user", "content": f"标题: {title}\n描述: {description}"}
        ],
        response_format=ClassificationResult
    )
    return completion.choices[0].message.parsed

# 使用
result = classify_issue(
    "登录按钮点击无响应",
    "在 Chrome 浏览器上点击登录按钮没有任何反应，控制台显示 TypeError"
)
print(f"分类: {result.category}, 置信度: {result.confidence}")
```

### 3. 数据转换

```python
from pydantic import BaseModel
from typing import List

class TableRow(BaseModel):
    date: str
    product: str
    quantity: int
    price: float
    total: float

class TableData(BaseModel):
    headers: List[str]
    rows: List[TableRow]

def text_to_table(text: str) -> TableData:
    """将非结构化文本转换为表格数据"""
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "将文本中的数据转换为表格格式"},
            {"role": "user", "content": text}
        ],
        response_format=TableData
    )
    return completion.choices[0].message.parsed

# 使用
text = """
1月5日卖了10个苹果，单价5元；
1月6日卖了20个橙子，单价3元；
1月7日卖了15个香蕉，单价2元。
"""
table = text_to_table(text)
for row in table.rows:
    print(f"{row.date}: {row.product} x {row.quantity} = {row.total}")
```

### 4. API 响应格式化

```python
from pydantic import BaseModel
from typing import List, Optional, Generic, TypeVar
from datetime import datetime

T = TypeVar('T')

class APIResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    timestamp: str

class UserProfile(BaseModel):
    id: int
    username: str
    email: str
    created_at: str

def generate_api_response(query: str) -> APIResponse[UserProfile]:
    """生成符合 API 规范的响应"""
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "生成模拟的用户数据 API 响应"},
            {"role": "user", "content": query}
        ],
        response_format=APIResponse[UserProfile]
    )
    return completion.choices[0].message.parsed
```

## 错误处理

```python
from openai import OpenAI
from pydantic import BaseModel, ValidationError

client = OpenAI()

class Output(BaseModel):
    name: str
    age: int

def safe_parse(text: str) -> Output | None:
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": text}],
            response_format=Output
        )
        
        # 检查是否有 refusal
        if completion.choices[0].message.refusal:
            print(f"模型拒绝: {completion.choices[0].message.refusal}")
            return None
        
        return completion.choices[0].message.parsed
        
    except ValidationError as e:
        print(f"验证错误: {e}")
        return None
    except Exception as e:
        print(f"其他错误: {e}")
        return None
```

## 最佳实践

1. **Schema 设计**
   - 使用 Pydantic 的 Field 添加描述
   - 设置合理的约束（ge、le、max_length 等）
   - 使用 Optional 标记可选字段

2. **Prompt 优化**
   - 在 system prompt 中说明输出格式要求
   - 提供示例帮助模型理解

3. **错误处理**
   - 处理 refusal 情况
   - 添加重试逻辑
   - 验证输出数据

4. **性能考虑**
   - 结构化输出可能增加延迟
   - 复杂 Schema 可能影响准确性

## 延伸阅读

- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [LangChain Output Parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- [Pydantic 文档](https://docs.pydantic.dev/)
