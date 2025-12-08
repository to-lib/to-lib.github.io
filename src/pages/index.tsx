import type { ReactNode } from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import HomepageFeatures from "@site/src/components/HomepageFeatures";
import Heading from "@theme/Heading";

import styles from "./index.module.css";

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx("hero", styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroContent}>
          <Heading as="h1" className={styles.heroTitle}>
            <span className={styles.titleGradient}>
              Java 设计模式完全指南
            </span>
          </Heading>
          <p className={styles.heroSubtitle}>
            掌握23种经典设计模式 • 提升代码质量 • 成为更优秀的开发者
          </p>
          <p className={styles.heroDescription}>
            详细讲解、完整代码示例、实际应用场景、最佳实践指导。
            无论你是初学者还是资深开发者，都能从这份完整指南中获益。
          </p>
          <div className={styles.buttons}>
            <Link className={clsx("button button--primary button--lg", styles.primaryBtn)} to="/docs/java-design-patterns">
              🚀 立即开始学习
            </Link>
            <Link className={clsx("button button--secondary button--lg", styles.secondaryBtn)} to="/docs/intro">
              📖 了解更多
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />"
    >
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
