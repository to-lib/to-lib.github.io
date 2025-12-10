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
  // Java Design Patterns sidebar
  designPatterns: [
    {
      type: "category",
      label: "📖 概览与参考",
      collapsed: false,
      items: [
        "java-design-patterns/index",
        "java-design-patterns/overview",
        "java-design-patterns/quick-reference",
        "java-design-patterns/best-practices",
        "java-design-patterns/scenarios",
        "java-design-patterns/selection-guide",
        "java-design-patterns/interview-questions",
      ],
    },
    {
      type: "category",
      label: "创建型模式 (5)",
      collapsed: false,
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
      collapsed: false,
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
      collapsed: false,
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
      ],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: false,
      items: [
        "spring/aop",
        "spring/transactions",
        "spring/events",
        "spring/resource-management",
      ],
    },
    {
      type: "category",
      label: "🌐 Web 开发",
      collapsed: false,
      items: ["spring/spring-mvc"],
    },
    {
      type: "category",
      label: "💾 数据访问",
      collapsed: false,
      items: ["spring/spring-data"],
    },
    {
      type: "category",
      label: "🔒 安全基础",
      collapsed: false,
      items: ["spring/security-basics"],
    },
    {
      type: "category",
      label: "🧪 测试",
      collapsed: false,
      items: ["spring/testing"],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: false,
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
      collapsed: false,
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
      collapsed: false,
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
      collapsed: false,
      items: ["springboot/data-access", "springboot/cache-management"],
    },
    {
      type: "category",
      label: "🚀 进阶特性",
      collapsed: false,
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
      collapsed: false,
      items: [
        "springboot/security",
        "springboot/health-monitoring",
        "springboot/performance-optimization",
        "springboot/deployment",
        "springboot/docker",
      ],
    },
    {
      type: "category",
      label: "📚 开发指南",
      collapsed: false,
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
      type: "autogenerated",
      dirName: "netty",
    },
  ],

  // Java Programming sidebar
  java: [
    {
      type: "autogenerated",
      dirName: "java",
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
      collapsed: false,
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
      collapsed: false,
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
      type: "autogenerated",
      dirName: "linux",
    },
  ],

  // React sidebar
  react: [
    {
      type: "autogenerated",
      dirName: "react",
    },
  ],

  // MySQL sidebar
  mysql: [
    {
      type: "autogenerated",
      dirName: "mysql",
    },
  ],

  // Redis sidebar
  redis: [
    {
      type: "category",
      label: "📖 基础知识",
      collapsed: false,
      items: ["redis/index", "redis/introduction", "redis/data-types"],
    },
    {
      type: "category",
      label: "🎯 核心特性",
      collapsed: false,
      items: [
        "redis/persistence",
        "redis/replication",
        "redis/sentinel",
        "redis/cluster",
        "redis/transactions",
        "redis/pubsub",
        "redis/streams",
        "redis/pipeline",
      ],
    },
    {
      type: "category",
      label: "🚀 高级应用",
      collapsed: false,
      items: [
        "redis/cache-strategies",
        "redis/performance-optimization",
        "redis/best-practices",
      ],
    },
    {
      type: "category",
      label: "📚 参考指南",
      collapsed: false,
      items: [
        "redis/quick-reference",
        "redis/faq",
        "redis/interview-questions",
        "redis/practical-examples",
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
        "springcloud/config",
      ],
    },
    {
      type: "category",
      label: "🔧 核心组件",
      collapsed: false,
      items: ["springcloud/gateway", "springcloud/feign", "springcloud/ribbon"],
    },
    {
      type: "category",
      label: "🌟 现代组件 (推荐)",
      collapsed: false,
      items: [
        "springcloud/loadbalancer",
        "springcloud/resilience4j",
        "springcloud/stream",
      ],
    },
    {
      type: "category",
      label: "🛠 高级与维护",
      collapsed: false,
      items: ["springcloud/hystrix", "springcloud/sleuth"],
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
      collapsed: false,
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
      collapsed: false,
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
      collapsed: false,
      items: [
        "springcloud-alibaba/practical-project",
        "springcloud-alibaba/migration-guide",
        "springcloud-alibaba/faq",
        "springcloud-alibaba/interview-questions",
      ],
    },
  ],
};

export default sidebars;
