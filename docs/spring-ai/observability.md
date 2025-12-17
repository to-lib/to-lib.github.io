---
sidebar_position: 20
title: 可观测性（Observability）
---

# 可观测性（Observability）

AI 功能上线后，最容易遇到的不是“能不能调用模型”，而是：

- 为什么回答质量突然变差？
- 为什么成本飙升？
- 哪些问题命中率低？
- 哪些文档/分块经常被召回？

可观测性要覆盖：**日志（Logs）**、**指标（Metrics）**、（可选）**链路追踪（Tracing）**。

## 建议的观测维度

- **请求维度**
  - `requestId` / `conversationId`
  - 模型提供商 / 模型名
  - 是否启用工具/函数调用
  - 是否启用 RAG

- **成本维度**
  - prompt tokens / completion tokens / total tokens
  - 每次请求耗时

- **质量维度（可量化的部分）**
  - RAG 检索条数、相似度阈值命中情况
  - 命中文档来源统计（按 `metadata.source` 聚合）

## 用 Advisor 做埋点

Spring AI 的 `Advisor` 是做可观测性最自然的切入点：

- 请求前：记录入参摘要、开始时间
- 响应后：记录耗时、token usage、错误类型

你可以参考：

- [Advisor 机制](/docs/spring-ai/advisors)
- [最佳实践](/docs/spring-ai/best-practices)

### 示例：记录请求耗时与 token

下面示例展示了一个思路：

- 用 `context` 传递 `startTime`
- 在响应阶段读取 `Usage` 计数

```java
public class TokenMetricsAdvisor implements RequestResponseAdvisor {

    private final MeterRegistry meterRegistry;

    public TokenMetricsAdvisor(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
    }

    @Override
    public AdvisedRequest adviseRequest(AdvisedRequest request, Map<String, Object> context) {
        context.put("startTime", System.nanoTime());
        return request;
    }

    @Override
    public ChatResponse adviseResponse(ChatResponse response, Map<String, Object> context) {
        long startTime = (Long) context.get("startTime");
        long durationNanos = System.nanoTime() - startTime;

        Usage usage = response.getMetadata().getUsage();
        meterRegistry.counter("ai.tokens.prompt").increment(usage.getPromptTokens());
        meterRegistry.counter("ai.tokens.completion").increment(usage.getGenerationTokens());
        meterRegistry.counter("ai.tokens.total").increment(usage.getTotalTokens());

        Timer.builder("ai.chat.duration").register(meterRegistry)
            .record(durationNanos, TimeUnit.NANOSECONDS);

        return response;
    }
}
```

## 日志规范建议

- **不要直接记录完整用户输入**：至少要截断、或做脱敏
- **用 requestId 做关联**：将“应用日志”和“AI 调用日志”串起来
- **区分错误类型**：`TransientAiException`（可重试） vs `NonTransientAiException`（配置/输入问题）

## RAG 场景的观测建议

建议额外记录：

- `topK`、`similarityThreshold`
- 实际召回文档数量
- 召回的 `source`/`page` 等元数据（可截断）

这样遇到“胡说/不基于知识库”时，能快速判断：

- 检索没命中？
- 命中但分块质量差？
- 命中但 prompt 约束不足？

## 下一步

- [评测与回归测试](/docs/spring-ai/evaluation)
- [RAG 应用](/docs/spring-ai/rag)
- [最佳实践](/docs/spring-ai/best-practices)
