import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

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
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
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
    format: "detect",
    mdx1Compat: {
      comments: false,
      admonitions: false,
      headingIds: false,
    },
  },

  stylesheets: [
    {
      href: "https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css",
      type: "text/css",
      integrity:
        "sha384-GvrOXuhMATgEsSwCs4smOFZETl1RojAnj1Q3LqyqZP/EaGzz0YsZvs0jTfnXADWY",
      crossorigin: "anonymous",
    },
  ],

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
        { to: "/blog", label: "Blog", position: "left" },
        {
          type: "dropdown",
          label: "💻 编程语言",
          position: "left",
          items: [
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #1976d2;'>Java 编程</div>",
            },
            {
              label: "📖 Java 概述",
              to: "/docs/java",
            },
            {
              label: "📦 JDK 8-21 新特性",
              to: "/docs/java/jdk21-features",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #388e3c;'>Rust 编程</div>",
            },
            {
              label: "📖 Rust 概述",
              to: "/docs/rust",
            },
            {
              label: "📦 Cargo 指南",
              to: "/docs/rust/cargo-guide",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #555;'>C 语言编程</div>",
            },
            {
              label: "📖 C 语言概述",
              to: "/docs/c",
            },
            {
              label: "🔧 嵌入式开发",
              to: "/docs/c/embedded",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #d32f2f;'>前端开发</div>",
            },
            {
              label: "📖 前端概述",
              to: "/docs/frontend",
            },
            {
              label: "🌐 HTML",
              to: "/docs/frontend/html",
            },
            {
              label: "🎨 CSS",
              to: "/docs/frontend/css",
            },
            {
              label: "📜 JavaScript",
              to: "/docs/frontend/javascript",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #61dafb;'>框架</div>",
            },
            {
              label: "⚛️ React 19",
              to: "/docs/react",
            },
          ],
        },
        {
          type: "dropdown",
          label: "🏗️ 框架与中间件",
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
              label: "☁️ Spring Cloud",
              to: "/docs/springcloud",
            },
            {
              label: "🛍️ Spring Cloud Alibaba",
              to: "/docs/springcloud-alibaba",
            },
            {
              label: "🤖 Spring AI",
              to: "/docs/spring-ai",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #d32f2f;'>网络与消息</div>",
            },
            {
              label: "⚡ Netty",
              to: "/docs/netty",
            },
            {
              label: "🐰 RabbitMQ",
              to: "/docs/rabbitmq",
            },
            {
              label: "📊 Kafka",
              to: "/docs/kafka",
            },
            {
              label: "🚀 RocketMQ",
              to: "/docs/rocketmq",
            },
            {
              label: "⚡ Flink",
              to: "/docs/flink",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #1976d2;'>数据库</div>",
            },
            {
              label: "🐬 MySQL",
              to: "/docs/mysql",
            },
            {
              label: "💾 Redis",
              to: "/docs/redis",
            },
            {
              label: "🐘 PostgreSQL",
              to: "/docs/postgres",
            },
            {
              label: "🗃️ MyBatis",
              to: "/docs/mybatis",
            },
          ],
        },
        {
          type: "dropdown",
          label: "🐧 运维与工具",
          position: "left",
          items: [
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #000;'>Linux System</div>",
            },
            {
              label: "🐧 Linux",
              to: "/docs/linux",
            },
            {
              label: "🐚 Shell Scripting",
              to: "/docs/linux/shell-scripting",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #009688;'>Web Server & Media</div>",
            },
            {
              label: "🌐 Nginx",
              to: "/docs/nginx",
            },
            {
              label: "🎬 FFmpeg",
              to: "/docs/ffmpeg",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #1976d2;'>Containers</div>",
            },
            {
              label: "🐳 Docker",
              to: "/docs/docker",
            },
            {
              label: "🦭 Podman",
              to: "/docs/podman",
            },
            {
              label: "☸️ Kubernetes",
              to: "/docs/kubernetes",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #9c27b0;'>架构</div>",
            },
            {
              label: "🏛️ 微服务架构",
              to: "/docs/microservices",
            },
          ],
        },
        {
          type: "dropdown",
          label: "🧠 CS 基础",
          position: "left",
          items: [
            {
              label: "🧮 数据结构与算法",
              to: "/docs/dsa",
            },
            {
              label: "📋 DSA 快速参考",
              to: "/docs/dsa/quick-reference",
            },
            {
              label: "🧠 DSA 面试题",
              to: "/docs/interview/dsa-interview-questions",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              label: "🌐 计算机网络",
              to: "/docs/networking",
            },
            {
              label: "🎨 设计模式 (Java)",
              to: "/docs/java-design-patterns",
            },
          ],
        },
        {
          type: "dropdown",
          label: "📝 面试题库",
          position: "left",
          items: [
            {
              label: "📚 面试题库首页",
              to: "/docs/interview",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #d32f2f;'>Java 面试</div>",
            },
            {
              label: "☕ Java 基础面试题",
              to: "/docs/interview/java-interview-questions",
            },
            {
              label: "🎯 Java 高级面试题",
              to: "/docs/interview/java-senior",
            },
            {
              label: "🎨 设计模式面试题",
              to: "/docs/interview/java-design-patterns-interview-questions",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #388e3c;'>Spring 生态</div>",
            },
            {
              label: "🍃 Spring 面试题",
              to: "/docs/interview/spring-interview-questions",
            },
            {
              label: "🚀 Spring Boot 面试题",
              to: "/docs/interview/springboot-interview-questions",
            },
            {
              label: "☁️ Spring Cloud 面试题",
              to: "/docs/interview/springcloud-interview-questions",
            },
            {
              label: "🛍️ Spring Cloud Alibaba 面试题",
              to: "/docs/interview/springcloud-alibaba-interview-questions",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #1976d2;'>数据库 & 中间件</div>",
            },
            {
              label: "🐬 MySQL 面试题",
              to: "/docs/interview/mysql-interview-questions",
            },
            {
              label: "🗃️ MyBatis 面试题",
              to: "/docs/interview/mybatis-interview-questions",
            },
            {
              label: "💾 Redis 面试题",
              to: "/docs/interview/redis-interview-questions",
            },
            {
              label: "🐘 PostgreSQL 面试题",
              to: "/docs/interview/postgres-interview-questions",
            },
            {
              label: "📊 Kafka 面试题",
              to: "/docs/interview/kafka-interview-questions",
            },
            {
              label: "🚀 RocketMQ 面试题",
              to: "/docs/interview/rocketmq-interview-questions",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #9c27b0;'>其他</div>",
            },
            {
              label: "🐧 Linux 面试题",
              to: "/docs/interview/linux-interview-questions",
            },
            {
              label: "⚡ Netty 面试题",
              to: "/docs/interview/netty-interview-questions",
            },
            {
              label: "🐳 Docker 面试题",
              to: "/docs/interview/docker-interview-questions",
            },
            {
              label: "☸️ Kubernetes 面试题",
              to: "/docs/interview/kubernetes-interview-questions",
            },
            {
              label: "🐰 RabbitMQ 面试题",
              to: "/docs/interview/rabbitmq-interview-questions",
            },
            {
              label: "⚡ Flink 面试题",
              to: "/docs/interview/flink-interview-questions",
            },
            {
              label: "🏛️ 微服务面试题",
              to: "/docs/interview/microservices-interview-questions",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #61dafb;'>前端 & 其他</div>",
            },
            {
              label: "⚛️ React 面试题",
              to: "/docs/interview/react-interview-questions",
            },
            {
              label: "🧮 数据结构与算法面试题",
              to: "/docs/interview/dsa-interview-questions",
            },
            {
              label: "🦀 Rust 面试题",
              to: "/docs/interview/rust-interview-questions",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #ff9800;'>架构与软技能</div>",
            },
            {
              label: "🏗️ 系统设计面试题",
              to: "/docs/interview/system-design-interview-questions",
            },
            {
              label: "🗣️ 行为面试题 (BQ)",
              to: "/docs/interview/behavioral-interview-questions",
            },
          ],
        },
        {
          type: "dropdown",
          label: "🤖 AI 开发",
          position: "left",
          items: [
            {
              label: "📖 AI 概览",
              to: "/docs/ai",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #9c27b0;'>基础知识</div>",
            },
            {
              label: "🧠 LLM 基础知识",
              to: "/docs/ai/llm-fundamentals",
            },
            {
              label: "✨ 提示工程",
              to: "/docs/ai/prompt-engineering",
            },
            {
              label: "🧩 Embeddings（向量表示）",
              to: "/docs/ai/embeddings",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #1976d2;'>核心技术</div>",
            },
            {
              label: "🤖 AI Agent (智能体)",
              to: "/docs/ai/agent",
            },
            {
              label: "🔧 Function Calling",
              to: "/docs/ai/function-calling",
            },
            {
              label: "📚 RAG (检索增强生成)",
              to: "/docs/ai/rag",
            },
            {
              label: "🔌 MCP (模型上下文协议)",
              to: "/docs/ai/mcp",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #ff9800;'>工程实践</div>",
            },
            {
              label: "🧪 Fine-tuning（微调）",
              to: "/docs/ai/fine-tuning",
            },
            {
              label: "📏 Evaluation（评估与测试）",
              to: "/docs/ai/evaluation",
            },
            {
              label: "🚀 Production（生产化与部署）",
              to: "/docs/ai/production",
            },
            {
              label: "🔐 Security（安全与隐私）",
              to: "/docs/ai/security",
            },
            {
              type: "html",
              value: "<hr style='margin: 8px 0;'>",
            },
            {
              type: "html",
              value:
                "<div style='padding: 8px 12px; font-weight: bold; color: #388e3c;'>参考指南</div>",
            },
            {
              label: "📋 快速参考",
              to: "/docs/ai/quick-reference",
            },
            {
              label: "❓ 常见问题",
              to: "/docs/ai/faq",
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
          title: "� C 语言",
          items: [
            {
              label: "C 语言概述",
              to: "/docs/c",
            },
            {
              label: "指针详解",
              to: "/docs/c/pointers",
            },
            {
              label: "嵌入式编程",
              to: "/docs/c/embedded",
            },
            {
              label: "面试题汇总",
              to: "/docs/c/interview-questions",
            },
          ],
        },
        {
          title: "�🐧 Linux",
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
              label: "Spring Cloud",
              to: "/docs/springcloud",
            },
            {
              label: "Spring Cloud Alibaba",
              to: "/docs/springcloud-alibaba",
            },
            {
              label: "Spring Cloud Alibaba 安全与权限",
              to: "/docs/springcloud-alibaba/security-and-access",
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
