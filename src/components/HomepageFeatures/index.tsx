import type { ReactNode } from "react";
import clsx from "clsx";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

type FeatureItem = {
  title: string;
  emoji: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: "📚 系统化学习",
    emoji: "📚",
    description: (
      <>
        全面覆盖前后端开发技术栈：Java/Rust 编程、React 19 前端开发、Linux
        系统、 23种设计模式、Spring 生态、Netty 网络编程、MySQL/Redis 数据库。
        系统化的知识体系助你全面提升技术能力。
      </>
    ),
  },
  {
    title: "💡 实战导向",
    emoji: "💡",
    description: (
      <>
        不仅讲解理论知识，更注重实际应用。提供完整的代码示例、真实的应用场景、
        以及业界最佳实践，帮助你快速将知识应用到实际项目中。
      </>
    ),
  },
  {
    title: "🚀 持续更新",
    emoji: "🚀",
    description: (
      <>
        定期更新内容，紧跟技术发展趋势。从基础概念到高级应用，
        提供清晰的学习路径和快速参考指南，让学习更加高效。
      </>
    ),
  },
];

function Feature({ title, emoji, description }: FeatureItem) {
  return (
    <div className={clsx("col col--4", styles.featureCol)}>
      <div className={styles.featureCard}>
        <div className={styles.featureIcon}>{emoji}</div>
        <div className={styles.featureContent}>
          <Heading as="h3" className={styles.featureTitle}>
            {title}
          </Heading>
          <p className={styles.featureDescription}>{description}</p>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.featureContainer}>
          <Heading as="h2" className={styles.featuresTitle}>
            为什么选择 TechLib？
          </Heading>
          <div className="row">
            {FeatureList.map((props, idx) => (
              <Feature key={idx} {...props} />
            ))}
          </div>
        </div>
      </div>

      <div className={styles.statsSection}>
        <div className="container">
          <div className={styles.statsGrid}>
            <div className={styles.stat}>
              <div className={styles.statNumber}>100+</div>
              <div className={styles.statLabel}>技术文档</div>
            </div>
            <div className={styles.stat}>
              <div className={styles.statNumber}>9</div>
              <div className={styles.statLabel}>核心模块</div>
            </div>
            <div className={styles.stat}>
              <div className={styles.statNumber}>300+</div>
              <div className={styles.statLabel}>代码示例</div>
            </div>
            <div className={styles.stat}>
              <div className={styles.statNumber}>200K+</div>
              <div className={styles.statLabel}>字内容</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
