import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...);

/**
 * Creating a sidebar enables you to:
 * - create an ordered group of docs
 * - render a sidebar for each doc of that group
 * - provide next/previous navigation
 *
 * The sidebars can be generated from the filesystem, or explicitly defined here.
 *
 * Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // AI Development sidebar
  ai: [
    {
      type: "category",
      label: "🤖 AI 开发",
      collapsed: false,
      items: [
        "ai/index",
        "ai/llm-fundamentals",
        "ai/prompt-engineering",
        "ai/embeddings",
      ],
    },
    {
      type: "category",
      label: "🎯 核心技术",
      collapsed: true,
      items: ["ai/rag", "ai/function-calling", "ai/agent", "ai/mcp"],
    },
    {
      type: "category",
      label: "🛠️ 工程实践",
      collapsed: true,
      items: [
        "ai/fine-tuning",
        "ai/evaluation",
        "ai/production",
        "ai/security",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: ["ai/quick-reference", "ai/faq"],
    },
  ],

  // Java Design Patterns sidebar
  designPatterns: [
    {
      type: "category",
      label: "📖 概览与参考",
      collapsed: false,
      items: [
        "java-design-patterns/index",
        "java-design-patterns/environment-setup",
        "java-design-patterns/overview",
        "java-design-patterns/quick-reference",
        "java-design-patterns/frameworks-in-practice",
        "java-design-patterns/best-practices",
        "java-design-patterns/scenarios",
        "java-design-patterns/selection-guide",
        "java-design-patterns/pattern-comparisons",
        "java-design-patterns/exercises",
        "java-design-patterns/faq",
        "java-design-patterns/interview-questions",
      ],
    },
    {
      type: "category",
      label: "创建型模式 (5)",
      collapsed: true,
      items: [
        "java-design-patterns/singleton-pattern",
        "java-design-patterns/factory-pattern",
        "java-design-patterns/abstract-factory-pattern",
        "java-design-patterns/builder-pattern",
        "java-design-patterns/prototype-pattern",
      ],
    },
    {
      type: "category",
      label: "结构型模式 (7)",
      collapsed: true,
      items: [
        "java-design-patterns/proxy-pattern",
        "java-design-patterns/adapter-pattern",
        "java-design-patterns/decorator-pattern",
        "java-design-patterns/facade-pattern",
        "java-design-patterns/composite-pattern",
        "java-design-patterns/flyweight-pattern",
        "java-design-patterns/bridge-pattern",
      ],
    },
    {
      type: "category",
      label: "行为型模式 (11)",
      collapsed: true,
      items: [
        "java-design-patterns/observer-pattern",
        "java-design-patterns/strategy-pattern",
        "java-design-patterns/template-method-pattern",
        "java-design-patterns/command-pattern",
        "java-design-patterns/iterator-pattern",
        "java-design-patterns/state-pattern",
        "java-design-patterns/chain-of-responsibility-pattern",
        "java-design-patterns/mediator-pattern",
        "java-design-patterns/memento-pattern",
        "java-design-patterns/visitor-pattern",
        "java-design-patterns/interpreter-pattern",
      ],
    },
  ],

  // Spring sidebar
  spring: [
    {
      type: "category",
      label: "📖 概览与基础",
      collapsed: false,
      items: [
        "spring/spring-index",
        "spring/core-concepts",
        "spring/dependency-injection",
        "spring/bean-management",
        "spring/configuration",
      ],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: true,
      items: [
        "spring/aop",
        "spring/transactions",
        "spring/events",
        "spring/resource-management",
        "spring/spel",
        "spring/caching",
      ],
    },
    {
      type: "category",
      label: "🌐 Web 开发",
      collapsed: true,
      items: ["spring/spring-mvc"],
    },
    {
      type: "category",
      label: "💾 数据访问",
      collapsed: true,
      items: ["spring/spring-data"],
    },
    {
      type: "category",
      label: "🔒 安全基础",
      collapsed: true,
      items: ["spring/security-basics"],
    },
    {
      type: "category",
      label: "🧪 测试",
      collapsed: true,
      items: ["spring/testing"],
    },
    {
      type: "category",
      label: "✅ 校验",
      collapsed: true,
      items: ["spring/validation"],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "spring/quick-reference",
        "spring/best-practices",
        "spring/faq",
        "spring/interview-questions",
      ],
    },
  ],

  // Spring Boot sidebar
  springboot: [
    {
      type: "category",
      label: "📖 快速入门",
      collapsed: false,
      items: [
        "springboot/index",
        "springboot/quick-start",
        "springboot/core-concepts",
      ],
    },
    {
      type: "category",
      label: "🎓 核心特性",
      collapsed: true,
      items: [
        "springboot/auto-configuration",
        "springboot/dependency-management",
        "springboot/project-structure-config",
        "springboot/transaction",
        "springboot/aop",
        "springboot/events",
      ],
    },
    {
      type: "category",
      label: "🌐 Web 开发",
      collapsed: true,
      items: [
        "springboot/web-development",
        "springboot/websocket",
        "springboot/file-upload",
        "springboot/i18n",
      ],
    },
    {
      type: "category",
      label: "💾 数据访问",
      collapsed: true,
      items: ["springboot/data-access", "springboot/cache-management"],
    },
    {
      type: "category",
      label: "🚀 进阶特性",
      collapsed: true,
      items: [
        "springboot/async",
        "springboot/scheduling",
        "springboot/message-queue",
        "springboot/testing",
      ],
    },
    {
      type: "category",
      label: "🔒 生产级特性",
      collapsed: true,
      items: [
        "springboot/security",
        "springboot/health-monitoring",
        "springboot/observability",
        "springboot/performance-optimization",
        "springboot/deployment",
        "springboot/docker",
      ],
    },
    {
      type: "category",
      label: "📚 开发指南",
      collapsed: true,
      items: [
        "springboot/best-practices",
        "springboot/devtools",
        "springboot/quick-reference",
        "springboot/faq",
      ],
    },
  ],

  // Netty sidebar
  netty: [
    {
      type: "category",
      label: "📖 基础入门",
      collapsed: false,
      items: [
        "netty/index",
        "netty/overview",
        "netty/basics",
        "netty/core-components",
      ],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: true,
      items: ["netty/bytebuf", "netty/codec", "netty/websocket"],
    },
    {
      type: "category",
      label: "🚀 进阶实战",
      collapsed: true,
      items: [
        "netty/practical-examples",
        "netty/native-transport",
        "netty/backpressure",
        "netty/testing",
        "netty/advanced",
        "netty/best-practices",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "netty/quick-reference",
        "netty/troubleshooting",
        "netty/faq",
        "netty/interview-questions",
      ],
    },
  ],

  // Java Programming sidebar
  java: [
    {
      type: "category",
      label: "📖 概览与基础",
      collapsed: false,
      items: [
        "java/index",
        "java/environment-setup",
        "java/build-tools",
        "java/basic-syntax",
        "java/oop",
        "java/common-classes",
      ],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: true,
      items: [
        "java/collections",
        "java/generics",
        "java/exception-handling",
        "java/regex",
        "java/io-streams",
        "java/date-time",
        "java/network-programming",
      ],
    },
    {
      type: "category",
      label: "🚀 高级主题",
      collapsed: true,
      items: [
        "java/multithreading",
        "java/functional-programming",
        "java/reflection-annotation",
        "java/jvm-basics",
        "java/performance",
      ],
    },
    {
      type: "category",
      label: "✨ JDK 新特性",
      collapsed: true,
      items: [
        "java/jdk8-features",
        "java/jdk11-features",
        "java/jdk17-features",
        "java/jdk21-features",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "java/best-practices",
        "java/quick-reference",
        "java/faq",
        "java/interview-questions",
        "java/senior-interview-questions",
      ],
    },
  ],

  // Rust Programming sidebar
  rust: [
    {
      type: "category",
      label: "📖 概览与基础",
      collapsed: false,
      items: [
        "rust/index",
        "rust/environment-setup",
        "rust/basic-syntax",
        "rust/ownership",
        "rust/structs-enums",
        "rust/collections",
        "rust/project-structure",
        "rust/cargo-guide",
      ],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: false,
      items: [
        "rust/error-handling",
        "rust/generics-traits",
        "rust/lifetimes",
        "rust/closures-iterators",
        "rust/modules-packages",
      ],
    },
    {
      type: "category",
      label: "🚀 高级主题",
      collapsed: true,
      items: [
        "rust/smart-pointers",
        "rust/concurrency",
        "rust/async-programming",
        "rust/macros",
        "rust/unsafe-rust",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "rust/testing",
        "rust/best-practices",
        "rust/quick-reference",
        "rust/faq",
        "rust/interview-questions",
        "rust/practical-projects",
      ],
    },
  ],

  // Linux sidebar
  linux: [
    {
      type: "category",
      label: "📖 基础知识",
      collapsed: false,
      items: [
        "linux/index",
        "linux/basic-commands",
        "linux/file-system",
        "linux/permissions",
        "linux/users-groups",
        "linux/package-management",
        "linux/text-editors",
      ],
    },
    {
      type: "category",
      label: "🎯 系统管理",
      collapsed: true,
      items: [
        "linux/process-management",
        "linux/system-admin",
        "linux/service-management",
        "linux/cron-scheduling",
        "linux/log-management",
        "linux/disk-management",
        "linux/backup-and-recovery",
      ],
    },
    {
      type: "category",
      label: "🚀 网络与安全",
      collapsed: true,
      items: ["linux/networking", "linux/nftables-firewall", "linux/security"],
    },
    {
      type: "category",
      label: "💡 进阶主题",
      collapsed: true,
      items: [
        "linux/shell-scripting",
        "linux/performance-tuning",
        "linux/troubleshooting",
        "linux/docker-basics",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "linux/quick-reference",
        "linux/best-practices",
        "linux/faq",
        "linux/interview-questions",
      ],
    },
  ],

  // React sidebar
  react: [
    {
      type: "category",
      label: "📖 概览与基础",
      collapsed: false,
      items: [
        "react/index",
        "react/getting-started",
        "react/components",
        "react/jsx-syntax",
        "react/props-and-state",
        "react/lifecycle",
      ],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: true,
      items: [
        "react/hooks",
        "react/data-fetching",
        "react/suspense-data-fetching",
        "react/event-handling",
        "react/conditional-rendering",
        "react/lists-and-keys",
        "react/forms",
        "react/context",
        "react/refs-dom",
        "react/error-boundaries",
        "react/composition-patterns",
      ],
    },
    {
      type: "category",
      label: "🚀 高级主题",
      collapsed: true,
      items: [
        "react/performance-optimization",
        "react/react-compiler",
        "react/react19-features",
        "react/react-router",
        "react/state-management",
        "react/typescript",
        "react/ssr-nextjs",
        "react/testing",
        "react/code-splitting",
        "react/styling-solutions",
        "react/project-structure",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "react/quick-reference",
        "react/best-practices",
        "react/accessibility",
        "react/faq",
        "react/practical-projects",
        "react/interview-questions",
      ],
    },
  ],

  // DSA sidebar
  dsa: [
    {
      type: "category",
      label: "📖 概览与基础",
      collapsed: false,
      items: [
        "dsa/index",
        "dsa/complexity",
        "dsa/array-linkedlist",
        "dsa/stack-queue",
        "dsa/hash-table",
      ],
    },
    {
      type: "category",
      label: "🧩 常用技巧",
      collapsed: true,
      items: [
        "dsa/two-pointers",
        "dsa/sliding-window",
        "dsa/prefix-sum-diff",
        "dsa/bit-manipulation",
      ],
    },
    {
      type: "category",
      label: "🏗️ 高级数据结构",
      collapsed: true,
      items: [
        "dsa/tree",
        "dsa/heap",
        "dsa/graph",
        "dsa/union-find",
        "dsa/fenwick-tree",
        "dsa/segment-tree",
      ],
    },
    {
      type: "category",
      label: "🎯 经典算法",
      collapsed: true,
      items: [
        "dsa/sorting",
        "dsa/searching",
        "dsa/recursion-divide",
        "dsa/dynamic-programming",
        "dsa/greedy",
        "dsa/backtracking",
        "dsa/string-algorithms",
      ],
    },
    {
      type: "category",
      label: "📚 参考与刷题",
      collapsed: true,
      items: ["dsa/quick-reference", "dsa/faq", "dsa/interview-questions"],
    },
  ],

  // MySQL sidebar
  mysql: [
    {
      type: "category",
      label: "📖 基础知识",
      collapsed: false,
      items: [
        "mysql/index",
        "mysql/installation-and-connection",
        "mysql/basic-concepts",
        "mysql/data-types",
        "mysql/sql-syntax",
      ],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: true,
      items: [
        "mysql/indexes",
        "mysql/transactions",
        "mysql/locks",
        "mysql/security-and-accounts",
        "mysql/stored-procedures",
        "mysql/views-triggers",
      ],
    },
    {
      type: "category",
      label: "🚀 高级应用",
      collapsed: true,
      items: [
        "mysql/performance-optimization",
        "mysql/monitoring-and-troubleshooting",
        "mysql/partitioning",
        "mysql/replication",
        "mysql/high-availability",
        "mysql/backup-recovery",
        "mysql/best-practices",
        "mysql/practical-examples",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "mysql/quick-reference",
        "mysql/faq",
        "mysql/interview-questions",
      ],
    },
  ],

  // Redis sidebar
  redis: [
    {
      type: "category",
      label: "📖 基础知识",
      collapsed: false,
      items: [
        "redis/index",
        "redis/quick-start",
        "redis/introduction",
        "redis/data-types",
      ],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: true,
      items: [
        "redis/persistence",
        "redis/replication",
        "redis/sentinel",
        "redis/cluster",
        "redis/transactions",
        "redis/pubsub",
        "redis/streams",
        "redis/pipeline",
        "redis/lua-scripting",
        "redis/geo",
      ],
    },
    {
      type: "category",
      label: "🚀 高级应用",
      collapsed: true,
      items: [
        "redis/cache-strategies",
        "redis/memory-management",
        "redis/performance-optimization",
        "redis/security",
        "redis/best-practices",
      ],
    },
    {
      type: "category",
      label: "🛠️ 运维与排障",
      collapsed: true,
      items: [
        "redis/configuration",
        "redis/monitoring-and-troubleshooting",
        "redis/backup-and-recovery",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "redis/quick-reference",
        "redis/faq",
        "redis/interview-questions",
        "redis/practical-examples",
      ],
    },
  ],

  // PostgreSQL sidebar
  postgres: [
    {
      type: "category",
      label: "📖 基础知识",
      collapsed: false,
      items: [
        "postgres/index",
        "postgres/installation-and-connection",
        "postgres/basic-concepts",
        "postgres/data-types",
        "postgres/sql-syntax",
      ],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: true,
      items: [
        "postgres/indexes",
        "postgres/transactions",
        "postgres/locks",
        "postgres/stored-procedures",
        "postgres/views-triggers",
      ],
    },
    {
      type: "category",
      label: "🚀 高级应用",
      collapsed: true,
      items: [
        "postgres/performance-optimization",
        "postgres/monitoring-and-troubleshooting",
        "postgres/partitioning",
        "postgres/replication",
        "postgres/high-availability",
        "postgres/backup-recovery",
        "postgres/security",
        "postgres/best-practices",
        "postgres/practical-examples",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "postgres/quick-reference",
        "postgres/faq",
        "postgres/interview-questions",
      ],
    },
  ],

  // Spring Cloud sidebar
  springcloud: [
    {
      type: "category",
      label: "📖 概览与基础",
      collapsed: false,
      items: [
        "springcloud/index",
        "springcloud/core-concepts",
        "springcloud/eureka",
        "springcloud/consul",
        "springcloud/config",
      ],
    },
    {
      type: "category",
      label: "🔧 核心组件",
      collapsed: true,
      items: [
        "springcloud/gateway",
        "springcloud/feign",
        "springcloud/ribbon",
        "springcloud/bus",
      ],
    },
    {
      type: "category",
      label: "🌟 现代组件 (推荐)",
      collapsed: true,
      items: [
        "springcloud/loadbalancer",
        "springcloud/resilience4j",
        "springcloud/stream",
      ],
    },
    {
      type: "category",
      label: "🛠 高级与维护",
      collapsed: true,
      items: [
        "springcloud/hystrix",
        "springcloud/sleuth",
        "springcloud/micrometer-tracing",
        "springcloud/zipkin",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "springcloud/quick-reference",
        "springcloud/best-practices",
        "springcloud/troubleshooting",
        "springcloud/faq",
        "springcloud/interview-questions",
      ],
    },
  ],

  // Spring Cloud Alibaba sidebar
  springcloudAlibaba: [
    {
      type: "category",
      label: "📖 概览与入门",
      collapsed: false,
      items: [
        "springcloud-alibaba/springcloud-alibaba-index",
        "springcloud-alibaba/core-concepts",
        "springcloud-alibaba/getting-started",
      ],
    },
    {
      type: "category",
      label: "🔧 核心组件",
      collapsed: true,
      items: [
        "springcloud-alibaba/nacos",
        "springcloud-alibaba/sentinel",
        "springcloud-alibaba/seata",
        "springcloud-alibaba/rocketmq",
        "springcloud-alibaba/dubbo",
        "springcloud-alibaba/gateway",
      ],
    },
    {
      type: "category",
      label: "🚀 进阶主题",
      collapsed: true,
      items: [
        "springcloud-alibaba/config-advanced",
        "springcloud-alibaba/service-governance",
        "springcloud-alibaba/best-practices",
        "springcloud-alibaba/monitoring",
      ],
    },
    {
      type: "category",
      label: "📚 实战与参考",
      collapsed: true,
      items: [
        "springcloud-alibaba/quick-reference",
        "springcloud-alibaba/practical-project",
        "springcloud-alibaba/migration-guide",
        "springcloud-alibaba/security-and-access",
        "springcloud-alibaba/faq",
        "springcloud-alibaba/interview-questions",
      ],
    },
  ],

  // RocketMQ sidebar
  rocketmq: [
    {
      type: "category",
      label: "📖 基础入门",
      collapsed: false,
      items: [
        "rocketmq/index",
        "rocketmq/introduction",
        "rocketmq/core-concepts",
        "rocketmq/quick-start",
      ],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: true,
      items: [
        "rocketmq/producer",
        "rocketmq/consumer",
        "rocketmq/message-types",
      ],
    },
    {
      type: "category",
      label: "🚀 高级应用",
      collapsed: true,
      items: [
        "rocketmq/cluster-management",
        "rocketmq/performance-optimization",
        "rocketmq/best-practices",
        "rocketmq/monitoring",
        "rocketmq/security",
        "rocketmq/troubleshooting",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "rocketmq/quick-reference",
        "rocketmq/faq",
        "rocketmq/interview-questions",
      ],
    },
  ],

  // Kafka sidebar
  kafka: [
    {
      type: "category",
      label: "📖 基础知识",
      collapsed: false,
      items: [
        "kafka/index",
        "kafka/introduction",
        "kafka/core-concepts",
        "kafka/quick-start",
      ],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: true,
      items: ["kafka/producer-api", "kafka/consumer-api"],
    },
    {
      type: "category",
      label: "🚀 高级应用",
      collapsed: true,
      items: [
        "kafka/cluster-management",
        "kafka/message-storage",
        "kafka/performance-optimization",
        "kafka/best-practices",
        "kafka/monitoring",
        "kafka/security",
        "kafka/kafka-connect",
        "kafka/kafka-streams",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "kafka/quick-reference",
        "kafka/faq",
        "kafka/interview-questions",
      ],
    },
  ],

  // Flink sidebar
  flink: [
    {
      type: "category",
      label: "📖 基础入门",
      collapsed: false,
      items: [
        "flink/index",
        "flink/introduction",
        "flink/core-concepts",
        "flink/quick-start",
      ],
    },
    {
      type: "category",
      label: "💻 核心 API",
      collapsed: true,
      items: ["flink/datastream-api", "flink/table-sql"],
    },
    {
      type: "category",
      label: "🎯 高级主题",
      collapsed: true,
      items: [
        "flink/state-management",
        "flink/cep",
        "flink/connectors",
        "flink/flink-cdc",
      ],
    },
    {
      type: "category",
      label: "🚀 生产部署",
      collapsed: true,
      items: [
        "flink/deployment",
        "flink/monitoring",
        "flink/performance-optimization",
        "flink/best-practices",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "flink/quick-reference",
        "flink/faq",
        "flink/interview-questions",
      ],
    },
    {
      type: "category",
      label: "🔥 实战案例",
      collapsed: true,
      items: ["flink/practical-examples"],
    },
  ],
  // RabbitMQ sidebar
  rabbitmq: [
    {
      type: "category",
      label: "📖 基础知识",
      collapsed: false,
      items: [
        "rabbitmq/index",
        "rabbitmq/introduction",
        "rabbitmq/core-concepts",
        "rabbitmq/exchanges",
        "rabbitmq/queues",
        "rabbitmq/quick-start",
        "rabbitmq/java-client",
      ],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: true,
      items: [
        "rabbitmq/producer",
        "rabbitmq/consumer",
        "rabbitmq/message-types",
      ],
    },
    {
      type: "category",
      label: "🚀 高级应用",
      collapsed: true,
      items: [
        "rabbitmq/advanced-features",
        "rabbitmq/advanced-config",
        "rabbitmq/cluster-management",
        "rabbitmq/performance-optimization",
        "rabbitmq/best-practices",
        "rabbitmq/monitoring",
        "rabbitmq/security",
        "rabbitmq/spring-integration",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "rabbitmq/quick-reference",

        "rabbitmq/faq",
        "rabbitmq/interview-questions",
      ],
    },
  ],
  // Spring AI sidebar
  springAi: [
    {
      type: "category",
      label: "📖 快速入门",
      collapsed: false,
      items: [
        "spring-ai/index",
        "spring-ai/quick-start",
        "spring-ai/core-concepts",
      ],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: true,
      items: [
        "spring-ai/chat-client",
        "spring-ai/prompts",
        "spring-ai/output-parsing",
        "spring-ai/function-calling",
        "spring-ai/embedding",
        "spring-ai/image-generation",
        "spring-ai/advisors",
      ],
    },
    {
      type: "category",
      label: "🚀 高级主题",
      collapsed: true,
      items: [
        "spring-ai/rag",
        "spring-ai/document-ingestion",
        "spring-ai/chat-memory",
        "spring-ai/observability",
        "spring-ai/evaluation",
        "spring-ai/model-providers",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "spring-ai/api-reference",
        "spring-ai/quick-reference",
        "spring-ai/best-practices",
        "spring-ai/faq",
        "spring-ai/interview-questions",
      ],
    },
  ],
  // FFmpeg sidebar
  ffmpeg: [
    {
      type: "category",
      label: "📖 基础入门",
      collapsed: false,
      items: ["ffmpeg/index", "ffmpeg/installation", "ffmpeg/basic-commands"],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: true,
      items: [
        "ffmpeg/video-processing",
        "ffmpeg/audio-processing",
        "ffmpeg/filters",
        "ffmpeg/streaming",
        "ffmpeg/subtitles",
      ],
    },
    {
      type: "category",
      label: "🚀 进阶与优化",
      collapsed: true,
      items: [
        "ffmpeg/concat-and-split",
        "ffmpeg/stream-mapping",
        "ffmpeg/ffprobe",
        "ffmpeg/encoding-parameters",
        "ffmpeg/hardware-acceleration",
        "ffmpeg/performance-optimization",
        "ffmpeg/scripting",
      ],
    },
    {
      type: "category",
      label: "🩺 排错与参考",
      collapsed: true,
      items: ["ffmpeg/troubleshooting", "ffmpeg/quick-reference", "ffmpeg/faq"],
    },
  ],
  // Docker sidebar
  docker: [
    {
      type: "category",
      label: "📖 基础入门",
      collapsed: false,
      items: ["docker/index", "docker/installation", "docker/basic-commands"],
    },
    {
      type: "category",
      label: "🏗️ 镜像与构建",
      collapsed: true,
      items: ["docker/dockerfile", "docker/buildkit", "docker/compose", "docker/registry"],
    },
    {
      type: "category",
      label: "🎯 网络与存储",
      collapsed: true,
      items: ["docker/networking", "docker/volumes"],
    },
    {
      type: "category",
      label: "🚀 进阶主题",
      collapsed: true,
      items: ["docker/swarm", "docker/security", "docker/monitoring"],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "docker/best-practices",
        "docker/quick-reference",
        "docker/troubleshooting",
        "docker/faq",
        "docker/interview-questions",
      ],
    },
  ],
  // Podman sidebar
  podman: [
    {
      type: "category",
      label: "📖 基础入门",
      collapsed: false,
      items: ["podman/index", "podman/installation", "podman/basic-commands"],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: true,
      items: [
        "podman/pods",
        "podman/rootless",
        "podman/systemd",
        "podman/networking",
        "podman/image-building",
      ],
    },
    {
      type: "category",
      label: "🔒 安全与运维",
      collapsed: true,
      items: ["podman/security", "podman/best-practices"],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "podman/docker-migration",
        "podman/quick-reference",
        "podman/faq",
        "podman/interview-questions",
      ],
    },
  ],
  // Kubernetes sidebar
  kubernetes: [
    {
      type: "category",
      label: "📖 基础入门",
      collapsed: false,
      items: [
        "kubernetes/index",
        "kubernetes/installation",
        "kubernetes/basic-commands",
      ],
    },
    {
      type: "category",
      label: "🎯 核心资源",
      collapsed: true,
      items: [
        "kubernetes/pods",
        "kubernetes/deployments",
        "kubernetes/statefulset",
        "kubernetes/services",
      ],
    },
    {
      type: "category",
      label: "⚙️ 配置与存储",
      collapsed: true,
      items: ["kubernetes/configmap-secret", "kubernetes/storage"],
    },
    {
      type: "category",
      label: "🌐 网络",
      collapsed: true,
      items: ["kubernetes/networking"],
    },
    {
      type: "category",
      label: "🚀 进阶主题",
      collapsed: true,
      items: ["kubernetes/helm", "kubernetes/rbac", "kubernetes/monitoring"],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "kubernetes/quick-reference",
        "kubernetes/best-practices",
        "kubernetes/troubleshooting",
        "kubernetes/faq",
        "kubernetes/interview-questions",
      ],
    },
  ],

  // Nginx sidebar
  nginx: [
    {
      type: "category",
      label: "📖 基础入门",
      collapsed: false,
      items: ["nginx/index", "nginx/installation", "nginx/basic-config"],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: true,
      items: [
        "nginx/static-files",
        "nginx/reverse-proxy",
        "nginx/load-balancing",
        "nginx/ssl-tls",
      ],
    },
    {
      type: "category",
      label: "🚀 进阶主题",
      collapsed: true,
      items: ["nginx/security", "nginx/performance", "nginx/logging"],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "nginx/quick-reference",
        "nginx/faq",
        "nginx/interview-questions",
      ],
    },
  ],
  // Networking sidebar
  networking: [
    {
      type: "category",
      label: "📖 基础知识",
      collapsed: false,
      items: [
        "networking/index",
        "networking/osi-tcp-ip",
        "networking/data-link-layer",
        "networking/network-layer",
      ],
    },
    {
      type: "category",
      label: "🔧 传输层协议",
      collapsed: true,
      items: [
        "networking/tcp",
        "networking/udp",
        "networking/socket-programming",
      ],
    },
    {
      type: "category",
      label: "🌐 应用层协议",
      collapsed: true,
      items: ["networking/http", "networking/dns", "networking/websocket"],
    },
    {
      type: "category",
      label: "🔒 网络安全",
      collapsed: true,
      items: ["networking/tls-ssl", "networking/security"],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "networking/quick-reference",
        "networking/faq",
        "networking/interview-questions",
      ],
    },
  ],

  // Microservices sidebar
  microservices: [
    {
      type: "category",
      label: "📖 基础入门",
      collapsed: false,
      items: [
        "microservices/index",
        "microservices/core-concepts",
        "microservices/design-patterns",
      ],
    },
    {
      type: "category",
      label: "🔧 服务治理",
      collapsed: true,
      items: [
        "microservices/service-governance",
        "microservices/observability",
      ],
    },
    {
      type: "category",
      label: "🚀 部署与安全",
      collapsed: true,
      items: [
        "microservices/deployment",
        "microservices/security",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "microservices/best-practices",
        "microservices/faq",
        "microservices/interview-questions",
        "microservices/quick-reference",
      ],
    },
  ],
};

export default sidebars;
