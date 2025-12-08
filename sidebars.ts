import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // 设计模式侧边栏
  docsSidebar: [
    // 学习资源 - 顶部快速入口
    {
      type: "category",
      label: "📚 学习资源",
      collapsed: false,
      items: [
        "java-design-patterns/overview",
        "java-design-patterns/quick-reference",
        "java-design-patterns/best-practices",
      ],
    },

    // 创建型模式 (5个)
    {
      type: "category",
      label: "🎨 创建型模式 (5)",
      collapsed: false,
      link: {
        type: "generated-index",
        title: "创建型模式",
        description:
          "创建型模式关注对象的创建机制，以合适的方式创建对象。包括单例、工厂方法、抽象工厂、建造者和原型模式。",
        slug: "/category/creational-patterns",
      },
      items: [
        {
          type: "doc",
          id: "java-design-patterns/singleton-pattern",
          label: "📌 单例模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/factory-pattern",
          label: "🏭 工厂方法模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/abstract-factory-pattern",
          label: "🏢 抽象工厂模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/builder-pattern",
          label: "🔨 建造者模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/prototype-pattern",
          label: "🐑 原型模式",
        },
      ],
    },

    // 结构型模式 (7个)
    {
      type: "category",
      label: "🏗️ 结构型模式 (7)",
      collapsed: false,
      link: {
        type: "generated-index",
        title: "结构型模式",
        description:
          "结构型模式关注类和对象的组合，通过继承和组合来获得更灵活的结构。包括代理、适配器、装饰器、外观、组合、享元和桥接模式。",
        slug: "/category/structural-patterns",
      },
      items: [
        {
          type: "doc",
          id: "java-design-patterns/proxy-pattern",
          label: "🎭 代理模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/adapter-pattern",
          label: "🔌 适配器模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/decorator-pattern",
          label: "🎁 装饰器模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/facade-pattern",
          label: "🏛️ 外观模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/composite-pattern",
          label: "🌳 组合模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/flyweight-pattern",
          label: "♻️ 享元模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/bridge-pattern",
          label: "🌉 桥接模式",
        },
      ],
    },

    // 行为型模式 (11个)
    {
      type: "category",
      label: "⚡ 行为型模式 (11)",
      collapsed: false,
      link: {
        type: "generated-index",
        title: "行为型模式",
        description:
          "行为型模式关注对象之间的通信和职责分配。包括观察者、策略、模板方法、命令、迭代器、状态、责任链、中介者、备忘录、访问者和解释器模式。",
        slug: "/category/behavioral-patterns",
      },
      items: [
        {
          type: "doc",
          id: "java-design-patterns/observer-pattern",
          label: "👀 观察者模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/strategy-pattern",
          label: "🎲 策略模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/template-method-pattern",
          label: "📋 模板方法模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/command-pattern",
          label: "⚡ 命令模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/iterator-pattern",
          label: "🔄 迭代器模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/state-pattern",
          label: "🔀 状态模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/chain-of-responsibility-pattern",
          label: "⛓️ 责任链模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/mediator-pattern",
          label: "🤝 中介者模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/memento-pattern",
          label: "💾 备忘录模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/visitor-pattern",
          label: "🚶 访问者模式",
        },
        {
          type: "doc",
          id: "java-design-patterns/interpreter-pattern",
          label: "🔤 解释器模式",
        },
      ],
    },
  ],

  // Netty 侧边栏
  nettySidebar: [
    {
      type: "doc",
      id: "netty/index",
      label: "🏠 Netty 首页",
    },
    {
      type: "category",
      label: "📖 基础入门",
      collapsed: false,
      items: [
        {
          type: "doc",
          id: "netty/overview",
          label: "📘 Netty 概述",
        },
        {
          type: "doc",
          id: "netty/basics",
          label: "🔰 基础知识",
        },
        {
          type: "doc",
          id: "netty/core-components",
          label: "🧩 核心组件",
        },
      ],
    },
    {
      type: "category",
      label: "🔧 核心功能",
      collapsed: false,
      items: [
        {
          type: "doc",
          id: "netty/bytebuf",
          label: "📦 ByteBuf 详解",
        },
        {
          type: "doc",
          id: "netty/codec",
          label: "🔄 编解码器",
        },
      ],
    },
    {
      type: "category",
      label: "🚀 进阶实战",
      collapsed: false,
      items: [
        {
          type: "doc",
          id: "netty/advanced",
          label: "⚡ 高级特性",
        },
        {
          type: "doc",
          id: "netty/practical-examples",
          label: "💡 实战案例",
        },
      ],
    },
    {
      type: "category",
      label: "📚 参考资料",
      collapsed: false,
      items: [
        {
          type: "doc",
          id: "netty/quick-reference",
          label: "⚡ 快速参考",
        },
        {
          type: "doc",
          id: "netty/troubleshooting",
          label: "🔧 故障排查",
        },
      ],
    },
  ],

  // Spring Framework 侧边栏
  springSidebar: [
    {
      type: "doc",
      id: "spring/index",
      label: "🏠 Spring 首页",
    },
    {
      type: "category",
      label: "📖 核心概念",
      collapsed: false,
      items: [
        {
          type: "doc",
          id: "spring/core-concepts",
          label: "🧩 核心概念",
        },
        {
          type: "doc",
          id: "spring/dependency-injection",
          label: "💉 依赖注入",
        },
        {
          type: "doc",
          id: "spring/bean-management",
          label: "🫘 Bean 管理",
        },
      ],
    },
    {
      type: "category",
      label: "🔧 核心功能",
      collapsed: false,
      items: [
        {
          type: "doc",
          id: "spring/aop",
          label: "🎯 AOP 切面",
        },
        {
          type: "doc",
          id: "spring/spring-mvc",
          label: "🌐 Spring MVC",
        },
        {
          type: "doc",
          id: "spring/transactions",
          label: "💳 事务管理",
        },
      ],
    },
    {
      type: "category",
      label: "📚 参考资料",
      collapsed: false,
      items: [
        {
          type: "doc",
          id: "spring/quick-reference",
          label: "⚡ 快速参考",
        },
        {
          type: "doc",
          id: "spring/best-practices",
          label: "✨ 最佳实践",
        },
        {
          type: "doc",
          id: "spring/faq",
          label: "❓ 常见问题",
        },
      ],
    },
  ],

  // Spring Boot 侧边栏
  springBootSidebar: [
    {
      type: "doc",
      id: "springboot/index",
      label: "🏠 Spring Boot 首页",
    },
    {
      type: "category",
      label: "🚀 快速开始",
      collapsed: false,
      items: [
        {
          type: "doc",
          id: "springboot/quick-start",
          label: "⚡ 快速开始",
        },
        {
          type: "doc",
          id: "springboot/core-concepts",
          label: "🧩 核心概念",
        },
        {
          type: "doc",
          id: "springboot/project-structure-config",
          label: "📁 项目结构与配置",
        },
      ],
    },
    {
      type: "category",
      label: "🔧 核心功能",
      collapsed: false,
      items: [
        {
          type: "doc",
          id: "springboot/auto-configuration",
          label: "⚙️ 自动配置",
        },
        {
          type: "doc",
          id: "springboot/dependency-management",
          label: "📦 依赖管理",
        },
        {
          type: "doc",
          id: "springboot/web-development",
          label: "🌐 Web 开发",
        },
        {
          type: "doc",
          id: "springboot/data-access",
          label: "💾 数据访问",
        },
      ],
    },
    {
      type: "category",
      label: "⚡ 高级特性",
      collapsed: false,
      items: [
        {
          type: "doc",
          id: "springboot/cache-management",
          label: "🗄️ 缓存管理",
        },
        {
          type: "doc",
          id: "springboot/message-queue",
          label: "📨 消息队列",
        },
        {
          type: "doc",
          id: "springboot/scheduling",
          label: "⏰ 任务调度",
        },
        {
          type: "doc",
          id: "springboot/security",
          label: "🔒 安全认证",
        },
      ],
    },
    {
      type: "category",
      label: "🚀 运维部署",
      collapsed: false,
      items: [
        {
          type: "doc",
          id: "springboot/health-monitoring",
          label: "💊 健康监控",
        },
        {
          type: "doc",
          id: "springboot/performance-optimization",
          label: "⚡ 性能优化",
        },
        {
          type: "doc",
          id: "springboot/deployment",
          label: "🚢 部署发布",
        },
      ],
    },
    {
      type: "category",
      label: "📚 参考资料",
      collapsed: false,
      items: [
        {
          type: "doc",
          id: "springboot/quick-reference",
          label: "⚡ 快速参考",
        },
        {
          type: "doc",
          id: "springboot/best-practices",
          label: "✨ 最佳实践",
        },
        {
          type: "doc",
          id: "springboot/faq",
          label: "❓ 常见问题",
        },
      ],
    },
  ],
};

export default sidebars;
