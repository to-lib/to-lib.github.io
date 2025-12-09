import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: "TechLib - 开发者知识库",
  tagline: "专业的技术学习与开发工具集",
  favicon: "img/favicon.ico",

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: "https://to-lib.github.io",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/",

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "to-lib", // Usually your GitHub org/user name.
  projectName: "to-lib.github.io", // Usually your repo name.
  trailingSlash: false,

  onBrokenLinks: "warn",

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl: "https://github.com/to-lib/to-lib.github.io/tree/main/",
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ["rss", "atom"],
            xslt: true,
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl: "https://github.com/to-lib/to-lib.github.io/tree/main/",
          // Useful options to enforce blogging best practices
          onInlineTags: "warn",
          onInlineAuthors: "warn",
          onUntruncatedBlogPosts: "warn",
        },
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themes: [
    [
      "@easyops-cn/docusaurus-search-local",
      /** @type {import("@easyops-cn/docusaurus-search-local").PluginOptions} */
      {
        hashed: true,
      },
    ],
    "@docusaurus/theme-mermaid",
  ],
  markdown: {
    mermaid: true,
  },

  themeConfig: {
    // Replace with your project's social card
    image: "img/docusaurus-social-card.jpg",
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: "to-lib",
      logo: {
        alt: "to-lib Logo",
        src: "img/logo.svg",
      },
      items: [
        {
          type: "dropdown",
          label: "☕ Java 编程",
          position: "left",
          items: [
            {
              label: "📖 Java 概述",
              to: "/docs/java",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #1976d2;'>基础知识</div>",
            },
            {
              label: "📝 基础语法",
              to: "/docs/java/basic-syntax",
            },
            {
              label: "🎯 面向对象",
              to: "/docs/java/oop",
            },
            {
              label: "❌ 异常处理",
              to: "/docs/java/exception-handling",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #388e3c;'>核心特性</div>",
            },
            {
              label: "📦 集合框架",
              to: "/docs/java/collections",
            },
            {
              label: "🔤 泛型编程",
              to: "/docs/java/generics",
            },
            {
              label: "💾 IO 流",
              to: "/docs/java/io-streams",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #d32f2f;'>高级主题</div>",
            },
            {
              label: "🧵 多线程",
              to: "/docs/java/multithreading",
            },
            {
              label: "⚡ 函数式编程",
              to: "/docs/java/functional-programming",
            },
            {
              label: "🖥️ JVM 基础",
              to: "/docs/java/jvm-basics",
            },
            {
              label: "🚀 性能优化",
              to: "/docs/java/performance",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #ff9800;'>版本新特性</div>",
            },
            {
              label: "📦 JDK 17 新特性",
              to: "/docs/java/jdk17-features",
            },
            {
              label: "🎯 JDK 21 新特性",
              to: "/docs/java/jdk21-features",
            },
          ],
        },
        {
          type: "dropdown",
          label: "🦀 Rust 编程",
          position: "left",
          items: [
            {
              label: "📖 Rust 概述",
              to: "/docs/rust",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #1976d2;'>基础知识</div>",
            },
            {
              label: "📝 基础语法",
              to: "/docs/rust/basic-syntax",
            },
            {
              label: "🔑 所有权系统",
              to: "/docs/rust/ownership",
            },
            {
              label: "📦 结构体和枚举",
              to: "/docs/rust/structs-enums",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #388e3c;'>核心特性</div>",
            },
            {
              label: "❌ 错误处理",
              to: "/docs/rust/error-handling",
            },
            {
              label: "🔤 泛型和 Trait",
              to: "/docs/rust/generics-traits",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #d32f2f;'>高级主题</div>",
            },
            {
              label: "🧵 并发编程",
              to: "/docs/rust/concurrency",
            },
            {
              label: "📌 智能指针",
              to: "/docs/rust/smart-pointers",
            },
          ],
        },
        {
          type: "dropdown",
          label: "🐧 Linux",
          position: "left",
          items: [
            {
              label: "📖 Linux 概述",
              to: "/docs/linux",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #1976d2;'>基础知识</div>",
            },
            {
              label: "💻 基础命令",
              to: "/docs/linux/basic-commands",
            },
            {
              label: "📁 文件系统",
              to: "/docs/linux/file-system",
            },
            {
              label: "🔐 权限管理",
              to: "/docs/linux/permissions",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #388e3c;'>系统管理</div>",
            },
            {
              label: "⚙️ 进程管理",
              to: "/docs/linux/process-management",
            },
            {
              label: "🌐 网络配置",
              to: "/docs/linux/networking",
            },
            {
              label: "📜 Shell 脚本",
              to: "/docs/linux/shell-scripting",
            },
          ],
        },
        {
          type: "dropdown",
          label: "📚 设计模式",
          position: "left",
          items: [
            {
              label: "📘 模式概览",
              to: "/docs/java-design-patterns/overview",
            },
            {
              label: "⚡ 快速参考",
              to: "/docs/java-design-patterns/quick-reference",
            },
            {
              label: "✨ 最佳实践",
              to: "/docs/java-design-patterns/best-practices",
            },
            {
              label: "🎯 使用场景对比",
              to: "/docs/java-design-patterns/scenarios",
            },
            {
              label: "🔍 模式选择指南",
              to: "/docs/java-design-patterns/selection-guide",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #1976d2;'>创建型模式 (5)</div>",
            },
            {
              label: "📌 单例模式",
              to: "/docs/java-design-patterns/singleton-pattern",
            },
            {
              label: "🏭 工厂方法模式",
              to: "/docs/java-design-patterns/factory-pattern",
            },
            {
              label: "🏢 抽象工厂模式",
              to: "/docs/java-design-patterns/abstract-factory-pattern",
            },
            {
              label: "🔨 建造者模式",
              to: "/docs/java-design-patterns/builder-pattern",
            },
            {
              label: "🐑 原型模式",
              to: "/docs/java-design-patterns/prototype-pattern",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #388e3c;'>结构型模式 (7)</div>",
            },
            {
              label: "🎭 代理模式",
              to: "/docs/java-design-patterns/proxy-pattern",
            },
            {
              label: "🔌 适配器模式",
              to: "/docs/java-design-patterns/adapter-pattern",
            },
            {
              label: "🎁 装饰器模式",
              to: "/docs/java-design-patterns/decorator-pattern",
            },
            {
              label: "🏛️ 外观模式",
              to: "/docs/java-design-patterns/facade-pattern",
            },
            {
              label: "🌳 组合模式",
              to: "/docs/java-design-patterns/composite-pattern",
            },
            {
              label: "♻️ 享元模式",
              to: "/docs/java-design-patterns/flyweight-pattern",
            },
            {
              label: "🌉 桥接模式",
              to: "/docs/java-design-patterns/bridge-pattern",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #d32f2f;'>行为型模式 (11)</div>",
            },
            {
              label: "👀 观察者模式",
              to: "/docs/java-design-patterns/observer-pattern",
            },
            {
              label: "🎲 策略模式",
              to: "/docs/java-design-patterns/strategy-pattern",
            },
            {
              label: "📋 模板方法模式",
              to: "/docs/java-design-patterns/template-method-pattern",
            },
            {
              label: "⚡ 命令模式",
              to: "/docs/java-design-patterns/command-pattern",
            },
            {
              label: "🔄 迭代器模式",
              to: "/docs/java-design-patterns/iterator-pattern",
            },
            {
              label: "🔀 状态模式",
              to: "/docs/java-design-patterns/state-pattern",
            },
            {
              label: "⛓️ 责任链模式",
              to: "/docs/java-design-patterns/chain-of-responsibility-pattern",
            },
            {
              label: "🤝 中介者模式",
              to: "/docs/java-design-patterns/mediator-pattern",
            },
            {
              label: "💾 备忘录模式",
              to: "/docs/java-design-patterns/memento-pattern",
            },
            {
              label: "🚶 访问者模式",
              to: "/docs/java-design-patterns/visitor-pattern",
            },
            {
              label: "🔤 解释器模式",
              to: "/docs/java-design-patterns/interpreter-pattern",
            },
          ],
        },
        {
          type: "dropdown",
          label: "🛠️ 框架应用",
          position: "left",
          items: [
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #388e3c;'>Spring 生态</div>",
            },
            {
              label: "🍃 Spring Framework",
              to: "/docs/spring",
            },
            {
              label: "🚀 Spring Boot",
              to: "/docs/springboot",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #d32f2f;'>网络框架</div>",
            },
            {
              label: "⚡ Netty",
              to: "/docs/netty",
            },
          ],
        },
        {
          type: "dropdown",
          label: "📝 面试题库",
          position: "left",
          items: [
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #1976d2;'>Java 核心</div>",
            },
            {
              label: "☕ Java 基础知识",
              to: "/docs/java",
            },
            {
              label: "📋 Java 面试题精选",
              to: "/docs/java/interview-questions",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #388e3c;'>框架应用</div>",
            },
            {
              label: "🍃 Spring 框架",
              to: "/docs/spring",
            },
            {
              label: "📋 Spring 面试题精选",
              to: "/docs/spring/interview-questions",
            },
            {
              label: "� Spring Boot",
              to: "/docs/springboot",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #d32f2f;'>进阶主题</div>",
            },
            {
              label: "�📚 设计模式",
              to: "/docs/java-design-patterns/overview",
            },
            {
              label: "📋 设计模式面试题精选",
              to: "/docs/java-design-patterns/interview-questions",
            },
            {
              label: "⚡ Netty 网络编程",
              to: "/docs/netty",
            },
            {
              label: "📋 Netty 面试题精选",
              to: "/docs/netty/interview-questions",
            },
          ],
        },
        {
          href: "https://github.com/to-lib/to-lib.github.io",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "📚 设计模式",
          items: [
            {
              label: "模式概览",
              to: "/docs/java-design-patterns/overview",
            },
            {
              label: "创建型模式 (5)",
              to: "/docs/java-design-patterns/singleton-pattern",
            },
            {
              label: "结构型模式 (7)",
              to: "/docs/java-design-patterns/proxy-pattern",
            },
            {
              label: "行为型模式 (11)",
              to: "/docs/java-design-patterns/observer-pattern",
            },
          ],
        },
        {
          title: "☕ Java 编程",
          items: [
            {
              label: "Java 概述",
              to: "/docs/java",
            },
            {
              label: "基础语法",
              to: "/docs/java/basic-syntax",
            },
            {
              label: "集合框架",
              to: "/docs/java/collections",
            },
            {
              label: "多线程",
              to: "/docs/java/multithreading",
            },
            {
              label: "JVM 基础",
              to: "/docs/java/jvm-basics",
            },
            {
              label: "性能优化",
              to: "/docs/java/performance",
            },
          ],
        },
        {
          title: "🦀 Rust 编程",
          items: [
            {
              label: "Rust 概述",
              to: "/docs/rust",
            },
            {
              label: "所有权系统",
              to: "/docs/rust/ownership",
            },
            {
              label: "错误处理",
              to: "/docs/rust/error-handling",
            },
            {
              label: "并发编程",
              to: "/docs/rust/concurrency",
            },
          ],
        },
        {
          title: "🐧 Linux",
          items: [
            {
              label: "Linux 概述",
              to: "/docs/linux",
            },
            {
              label: "基础命令",
              to: "/docs/linux/basic-commands",
            },
            {
              label: "权限管理",
              to: "/docs/linux/permissions",
            },
            {
              label: "Shell 脚本",
              to: "/docs/linux/shell-scripting",
            },
          ],
        },
        {
          title: "📖 学习资源",
          items: [
            {
              label: "快速参考",
              to: "/docs/java-design-patterns/quick-reference",
            },
            {
              label: "最佳实践",
              to: "/docs/java-design-patterns/best-practices",
            },
            {
              label: "选择指南",
              to: "/docs/java-design-patterns/selection-guide",
            },
          ],
        },
        {
          title: "🛠️ 框架应用",
          items: [
            {
              label: "Spring Framework",
              to: "/docs/spring",
            },
            {
              label: "Spring Boot",
              to: "/docs/springboot",
            },
            {
              label: "Netty",
              to: "/docs/netty/overview",
            },
          ],
        },
        {
          title: "🔗 链接",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/to-lib/to-lib.github.io",
            },
            {
              label: "问题反馈",
              href: "https://github.com/to-lib/to-lib.github.io/issues",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} to-lib 开发者知识库. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
