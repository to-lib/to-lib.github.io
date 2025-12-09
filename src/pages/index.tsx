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
            <span className={styles.titleGradient}>TechLib - 开发者知识库</span>
          </Heading>
          <p className={styles.heroSubtitle}>
            设计模式 • Spring 框架 • Spring Boot • Netty 网络编程
          </p>
          <p className={styles.heroDescription}>
            系统化学习 Java 开发核心技术，涵盖设计模式、Spring
            生态和高性能网络编程。
            提供详细教程、完整代码示例、实战案例和最佳实践指导。
          </p>
          <div className={styles.buttons}>
            <Link
              className={clsx(
                "button button--primary button--lg",
                styles.primaryBtn
              )}
              to="/docs/intro"
            >
              🚀 开始探索
            </Link>
            <Link
              className={clsx(
                "button button--secondary button--lg",
                styles.secondaryBtn
              )}
              to="/docs/java-design-patterns/quick-reference"
            >
              ⚡ 快速参考
            </Link>
          </div>

          {/* Quick Navigation Cards */}
          <div className={styles.quickNav}>
            <div className={styles.navCard}>
              <span className={styles.navIcon}>🎨</span>
              <Link to="/docs/java-design-patterns/overview">
                <h4>设计模式</h4>
                <p className={styles.navDesc}>23种经典模式</p>
              </Link>
            </div>
            <div className={styles.navCard}>
              <span className={styles.navIcon}>🍃</span>
              <Link to="/docs/spring">
                <h4>Spring 框架</h4>
                <p className={styles.navDesc}>IoC & AOP 核心</p>
              </Link>
            </div>
            <div className={styles.navCard}>
              <span className={styles.navIcon}>🚀</span>
              <Link to="/docs/springboot">
                <h4>Spring Boot</h4>
                <p className={styles.navDesc}>快速开发指南</p>
              </Link>
            </div>
            <div className={styles.navCard}>
              <span className={styles.navIcon}>⚡</span>
              <Link to="/docs/netty">
                <h4>Netty</h4>
                <p className={styles.navDesc}>高性能网络框架</p>
              </Link>
            </div>
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
