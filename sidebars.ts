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
  // Interview Questions sidebar
  interview: [
    {
      type: "category",
      label: "📝 面试题库",
      collapsed: false,
      items: ["interview/index"],
    },
    {
      type: "category",
      label: "☕ Java 技术栈",
      collapsed: false,
      items: [
        "interview/java-interview-questions",
        {
          type: "category",
          label: "🎯 Java 高级面试题",
          collapsed: true,
          items: [
            "interview/java-senior/index",
            "interview/java-senior/jvm",
            "interview/java-senior/concurrency",
            "interview/java-senior/performance",
            "interview/java-senior/architecture",
            "interview/java-senior/framework-source",
            "interview/java-senior/distributed",
            "interview/java-senior/design-patterns",
            "interview/java-senior/system-design",
          ],
        },
        "interview/java-design-patterns-interview-questions",
      ],
    },
    {
      type: "category",
      label: "🍃 Spring 生态",
      collapsed: true,
      items: [
        "interview/spring-interview-questions",
        "interview/springboot-interview-questions",
        "interview/springcloud-interview-questions",
        "interview/springcloud-alibaba-interview-questions",
        "interview/spring-ai-interview-questions",
      ],
    },
    {
      type: "category",
      label: "💾 数据库",
      collapsed: true,
      items: [
        "interview/mysql-interview-questions",
        "interview/mybatis-interview-questions",
        "interview/redis-interview-questions",
        "interview/postgres-interview-questions",
      ],
    },
    {
      type: "category",
      label: "📨 消息队列",
      collapsed: true,
      items: [
        "interview/kafka-interview-questions",
        "interview/rocketmq-interview-questions",
        "interview/rabbitmq-interview-questions",
      ],
    },
    {
      type: "category",
      label: "🐳 容器与运维",
      collapsed: true,
      items: [
        "interview/linux-interview-questions",
        "interview/docker-interview-questions",
        "interview/kubernetes-interview-questions",
        "interview/podman-interview-questions",
      ],
    },
    {
      type: "category",
      label: "🔧 框架与中间件",
      collapsed: true,
      items: [
        "interview/netty-interview-questions",
        "interview/flink-interview-questions",
        "interview/microservices-interview-questions",
        "interview/nginx-interview-questions",
        "interview/networking-interview-questions",
      ],
    },
    {
      type: "category",
      label: "🌐 前端与其他",
      collapsed: true,
      items: [
        "interview/react-interview-questions",
        "interview/dsa-interview-questions",
        "interview/rust-interview-questions",
      ],
    },
    {
      type: "category",
      label: "🏛️ 架构与软技能",
      collapsed: false,
      items: [
        "interview/system-design-interview-questions",
        "interview/behavioral-interview-questions",
      ],
    },
  ],

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
        "ai/lora-fine-tuning",
        "ai/evaluation",
        "ai/production",
        "ai/security",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "ai/quick-reference",
        "ai/gemini-for-google-workspace-prompting-guide-101",
        "ai/faq",
      ],
    },
  ],

  // Machine Learning sidebar
  ml: [
    {
      type: "category",
      label: "🧠 机器学习",
      collapsed: false,
      items: [
        "ml/index",
        "ml/fundamentals",
        "ml/math-basics",
        "ml/data-preprocessing",
      ],
    },
    {
      type: "category",
      label: "🎯 核心算法",
      collapsed: true,
      items: [
        "ml/supervised-learning",
        "ml/unsupervised-learning",
        "ml/ensemble-learning",
        "ml/bayesian-methods",
        "ml/reinforcement-learning",
      ],
    },
    {
      type: "category",
      label: "🧠 深度学习",
      collapsed: true,
      items: ["ml/neural-networks", "ml/deep-learning"],
    },
    {
      type: "category",
      label: "📊 应用领域",
      collapsed: true,
      items: [
        "ml/time-series",
        "ml/recommendation-system",
        "ml/nlp-basics",
        "ml/computer-vision",
      ],
    },
    {
      type: "category",
      label: "🛠️ 工程实践",
      collapsed: true,
      items: [
        "ml/model-evaluation",
        "ml/model-tuning",
        "ml/interpretability",
        "ml/automl",
        "ml/model-deployment",
        "ml/practical-projects",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: ["ml/quick-reference", "ml/faq"],
    },
    {
      type: "category",
      label: "🚀 进阶主题",
      collapsed: true,
      items: [
        "ml/graph-neural-networks",
        "ml/generative-models",
        "ml/contrastive-learning",
        "ml/multi-task-learning",
        "ml/meta-learning",
        "ml/federated-learning",
        "ml/causal-inference",
        "ml/multimodal-learning",
        "ml/active-learning",
        "ml/online-learning",
        "ml/continual-learning",
        "ml/model-compression",
        "ml/adversarial-robustness",
        "ml/uncertainty-quantification",
        "ml/anomaly-detection",
        "ml/advanced-rl",
        "ml/advanced-feature-engineering",
        "ml/distributed-training",
        "ml/mlops-tools",
        "ml/data-labeling",
        "ml/domain-specific-ml",
        "ml/transfer-learning",
        "ml/domain-adaptation",
        "ml/privacy-computing",
        "ml/evolutionary-algorithms",
        "ml/neuro-symbolic",
        "ml/trustworthy-ai",
      ],
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
      items: ["spring/quick-reference", "spring/best-practices", "spring/faq"],
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
      items: ["netty/quick-reference", "netty/troubleshooting", "netty/faq"],
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
      items: ["java/best-practices", "java/quick-reference", "java/faq"],
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
        "rust/learning-path",
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
        "rust/advanced-internals",
      ],
    },
    {
      type: "category",
      label: "🏗️ 工程化与实战",
      collapsed: true,
      items: [
        "rust/engineering",
        "rust/web-development",
        "rust/testing",
        "rust/practical-projects",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: ["rust/best-practices", "rust/quick-reference", "rust/faq"],
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
      items: ["linux/quick-reference", "linux/best-practices", "linux/faq"],
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
      items: ["dsa/quick-reference", "dsa/faq"],
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
      items: ["mysql/quick-reference", "mysql/faq"],
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
      items: ["redis/quick-reference", "redis/faq", "redis/practical-examples"],
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
        "springcloud/config",
      ],
    },
    {
      type: "category",
      label: "🎯 核心组件",
      collapsed: true,
      items: [
        "springcloud/eureka",
        "springcloud/ribbon",
        "springcloud/feign",
        "springcloud/hystrix",
        "springcloud/resilience4j",
        "springcloud/gateway",
        "springcloud/loadbalancer",
      ],
    },
    {
      type: "category",
      label: "🚀 分布式追踪与监控",
      collapsed: true,
      items: [
        "springcloud/sleuth",
        "springcloud/zipkin",
        "springcloud/micrometer-tracing",
      ],
    },
    {
      type: "category",
      label: "⚡ 消息驱动",
      collapsed: true,
      items: ["springcloud/stream", "springcloud/bus"],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "springcloud/consul",
        "springcloud/best-practices",
        "springcloud/troubleshooting",
        "springcloud/quick-reference",
        "springcloud/faq",
      ],
    },
  ],

  // Spring Cloud Alibaba sidebar
  springcloudAlibaba: [
    {
      type: "category",
      label: "📖 概览与基础",
      collapsed: false,
      items: [
        "springcloud-alibaba/index",
        "springcloud-alibaba/getting-started",
        "springcloud-alibaba/core-concepts",
        "springcloud-alibaba/migration-guide",
      ],
    },
    {
      type: "category",
      label: "🎯 核心组件",
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
      label: "🚀 高级特性",
      collapsed: true,
      items: [
        "springcloud-alibaba/config-advanced",
        "springcloud-alibaba/service-governance",
        "springcloud-alibaba/monitoring",
        "springcloud-alibaba/security-and-access",
      ],
    },
    {
      type: "category",
      label: "📚 实战与参考",
      collapsed: true,
      items: [
        "springcloud-alibaba/practical-project",
        "springcloud-alibaba/best-practices",
        "springcloud-alibaba/quick-reference",
        "springcloud-alibaba/faq",
      ],
    },
  ],

  // Spring AI sidebar
  springAi: [
    {
      type: "category",
      label: "📖 概览与基础",
      collapsed: false,
      items: [
        "spring-ai/index",
        "spring-ai/quick-start",
        "spring-ai/core-concepts",
        "spring-ai/api-reference",
      ],
    },
    {
      type: "category",
      label: "🤖 模型集成",
      collapsed: true,
      items: [
        "spring-ai/chat-client",
        "spring-ai/model-providers",
        "spring-ai/embedding",
        "spring-ai/image-generation",
      ],
    },
    {
      type: "category",
      label: "🎯 核心功能",
      collapsed: true,
      items: [
        "spring-ai/prompts",
        "spring-ai/output-parsing",
        "spring-ai/function-calling",
        "spring-ai/rag",
        "spring-ai/chat-memory",
        "spring-ai/document-ingestion",
      ],
    },
    {
      type: "category",
      label: "🚀 工程化",
      collapsed: true,
      items: [
        "spring-ai/advisors",
        "spring-ai/observability",
        "spring-ai/evaluation",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "spring-ai/best-practices",
        "spring-ai/quick-reference",
        "spring-ai/faq",
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
        "kafka/quick-start",
        "kafka/introduction",
        "kafka/core-concepts",
      ],
    },
    {
      type: "category",
      label: "🎯 核心架构",
      collapsed: true,
      items: [
        "kafka/producer-api",
        "kafka/consumer-api",
        "kafka/message-storage",
        "kafka/cluster-management",
      ],
    },
    {
      type: "category",
      label: "🚀 高级特性",
      collapsed: true,
      items: [
        "kafka/kafka-connect",
        "kafka/kafka-streams",
        "kafka/security",
        "kafka/monitoring",
        "kafka/performance-optimization",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: ["kafka/best-practices", "kafka/quick-reference", "kafka/faq"],
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
        "rabbitmq/quick-start",
        "rabbitmq/introduction",
        "rabbitmq/core-concepts",
      ],
    },
    {
      type: "category",
      label: "🎯 核心概念",
      collapsed: true,
      items: [
        "rabbitmq/exchanges",
        "rabbitmq/queues",
        "rabbitmq/message-types",
        "rabbitmq/producer",
        "rabbitmq/consumer",
      ],
    },
    {
      type: "category",
      label: "🚀 高级特性",
      collapsed: true,
      items: [
        "rabbitmq/advanced-features",
        "rabbitmq/advanced-config",
        "rabbitmq/cluster-management",
        "rabbitmq/monitoring",
        "rabbitmq/security",
        "rabbitmq/performance-optimization",
      ],
    },
    {
      type: "category",
      label: "⚡ 集成开发",
      collapsed: true,
      items: ["rabbitmq/java-client", "rabbitmq/spring-integration"],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "rabbitmq/best-practices",
        "rabbitmq/quick-reference",
        "rabbitmq/faq",
      ],
    },
  ],

  // RocketMQ sidebar
  rocketmq: [
    {
      type: "category",
      label: "📖 基础知识",
      collapsed: false,
      items: [
        "rocketmq/index",
        "rocketmq/introduction",
        "rocketmq/quick-start",
        "rocketmq/core-concepts",
      ],
    },
    {
      type: "category",
      label: "🎯 核心机制",
      collapsed: true,
      items: [
        "rocketmq/message-types",
        "rocketmq/producer",
        "rocketmq/consumer",
        "rocketmq/cluster-management",
      ],
    },
    {
      type: "category",
      label: "🚀 高级特性",
      collapsed: true,
      items: [
        "rocketmq/monitoring",
        "rocketmq/performance-optimization",
        "rocketmq/security",
        "rocketmq/troubleshooting",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "rocketmq/best-practices",
        "rocketmq/quick-reference",
        "rocketmq/faq",
      ],
    },
  ],

  // Flink sidebar
  flink: [
    {
      type: "category",
      label: "📖 基础知识",
      collapsed: false,
      items: [
        "flink/index",
        "flink/introduction",
        "flink/quick-start",
        "flink/core-concepts",
      ],
    },
    {
      type: "category",
      label: "🎯 核心 API",
      collapsed: true,
      items: [
        "flink/datastream-api",
        "flink/table-sql",
        "flink/state-management",
        "flink/connectors",
        "flink/cep",
        "flink/flink-cdc",
      ],
    },
    {
      type: "category",
      label: "🚀 运维与优化",
      collapsed: true,
      items: [
        "flink/deployment",
        "flink/monitoring",
        "flink/performance-optimization",
      ],
    },
    {
      type: "category",
      label: "📚 实战与参考",
      collapsed: true,
      items: [
        "flink/practical-examples",
        "flink/best-practices",
        "flink/quick-reference",
        "flink/faq",
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
        "postgres/views-triggers",
        "postgres/stored-procedures",
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
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "postgres/best-practices",
        "postgres/quick-reference",
        "postgres/faq",
      ],
    },
  ],

  // Docker sidebar
  docker: [
    {
      type: "category",
      label: "📖 基础知识",
      collapsed: false,
      items: [
        "docker/index",
        "docker/installation",
        "docker/basic-commands",
        "docker/dockerfile",
        "docker/compose",
      ],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: true,
      items: [
        "docker/volumes",
        "docker/networking",
        "docker/registry",
        "docker/buildkit",
        "docker/swarm",
      ],
    },
    {
      type: "category",
      label: "🚀 运维与安全",
      collapsed: true,
      items: ["docker/security", "docker/monitoring", "docker/troubleshooting"],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: ["docker/best-practices", "docker/quick-reference", "docker/faq"],
    },
  ],

  // Kubernetes sidebar
  kubernetes: [
    {
      type: "category",
      label: "📖 基础知识",
      collapsed: false,
      items: [
        "kubernetes/index",
        "kubernetes/installation",
        "kubernetes/basic-commands",
        "kubernetes/pods",
      ],
    },
    {
      type: "category",
      label: "🎯 核心组件",
      collapsed: true,
      items: [
        "kubernetes/deployments",
        "kubernetes/services",
        "kubernetes/configmap-secret",
        "kubernetes/statefulset",
        "kubernetes/storage",
        "kubernetes/networking",
        "kubernetes/rbac",
      ],
    },
    {
      type: "category",
      label: "🚀 高级特性",
      collapsed: true,
      items: [
        "kubernetes/helm",
        "kubernetes/monitoring",
        "kubernetes/troubleshooting",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "kubernetes/best-practices",
        "kubernetes/quick-reference",
        "kubernetes/faq",
      ],
    },
  ],

  // Podman sidebar
  podman: [
    {
      type: "category",
      label: "📖 基础知识",
      collapsed: false,
      items: [
        "podman/index",
        "podman/installation",
        "podman/basic-commands",
        "podman/pods",
      ],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: true,
      items: [
        "podman/image-building",
        "podman/networking",
        "podman/systemd",
        "podman/rootless",
        "podman/docker-migration",
      ],
    },
    {
      type: "category",
      label: "🚀 使用指南",
      collapsed: true,
      items: [
        "podman/security",
        "podman/best-practices",
        "podman/quick-reference",
        "podman/faq",
      ],
    },
  ],

  // Nginx sidebar
  nginx: [
    {
      type: "category",
      label: "📖 基础知识",
      collapsed: false,
      items: ["nginx/index", "nginx/installation", "nginx/basic-config"],
    },
    {
      type: "category",
      label: "🎯 核心功能",
      collapsed: true,
      items: [
        "nginx/static-files",
        "nginx/reverse-proxy",
        "nginx/load-balancing",
        "nginx/ssl-tls",
        "nginx/logging",
      ],
    },
    {
      type: "category",
      label: "🚀 高级配置",
      collapsed: true,
      items: [
        "nginx/security",
        "nginx/performance",
        "nginx/quick-reference",
        "nginx/faq",
      ],
    },
  ],

  // FFmpeg sidebar
  ffmpeg: [
    {
      type: "category",
      label: "📖 基础知识",
      collapsed: false,
      items: [
        "ffmpeg/index",
        "ffmpeg/installation",
        "ffmpeg/basic-commands",
        "ffmpeg/ffprobe",
      ],
    },
    {
      type: "category",
      label: "🎯 音视频处理",
      collapsed: true,
      items: [
        "ffmpeg/video-processing",
        "ffmpeg/audio-processing",
        "ffmpeg/encoding-parameters",
        "ffmpeg/concat-and-split",
        "ffmpeg/subtitles",
        "ffmpeg/filters",
        "ffmpeg/hardware-acceleration",
      ],
    },
    {
      type: "category",
      label: "🚀 流媒体与脚本",
      collapsed: true,
      items: ["ffmpeg/streaming", "ffmpeg/stream-mapping", "ffmpeg/scripting"],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "ffmpeg/troubleshooting",
        "ffmpeg/performance-optimization",
        "ffmpeg/quick-reference",
        "ffmpeg/faq",
      ],
    },
  ],

  // Microservices sidebar
  microservices: [
    {
      type: "category",
      label: "📖 架构基础",
      collapsed: false,
      items: [
        "microservices/index",
        "microservices/core-concepts",
        "microservices/design-patterns",
      ],
    },
    {
      type: "category",
      label: "🎯 治理与运维",
      collapsed: true,
      items: [
        "microservices/service-governance",
        "microservices/deployment",
        "microservices/observability",
        "microservices/security",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: [
        "microservices/best-practices",
        "microservices/quick-reference",
        "microservices/faq",
      ],
    },
  ],

  // MyBatis sidebar
  mybatis: [
    {
      type: "category",
      label: "📖 基础知识",
      collapsed: false,
      items: [
        "mybatis/index",
        "mybatis/core-concepts",
        "mybatis/configuration",
      ],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: true,
      items: [
        "mybatis/xml-mapping",
        "mybatis/dynamic-sql",
        "mybatis/annotations",
        "mybatis/caching",
      ],
    },
    {
      type: "category",
      label: "🚀 进阶应用",
      collapsed: true,
      items: [
        "mybatis/spring-integration",
        "mybatis/plugins",
        "mybatis/best-practices",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: ["mybatis/quick-reference", "mybatis/faq"],
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
        "networking/network-layer",
        "networking/data-link-layer",
      ],
    },
    {
      type: "category",
      label: "🎯 传输协议",
      collapsed: true,
      items: [
        "networking/tcp",
        "networking/udp",
        "networking/http",
        "networking/websocket",
        "networking/dns",
        "networking/tls-ssl",
      ],
    },
    {
      type: "category",
      label: "🚀 编程与安全",
      collapsed: true,
      items: ["networking/socket-programming", "networking/security"],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: true,
      items: ["networking/quick-reference", "networking/faq"],
    },
  ],
};

export default sidebars;
