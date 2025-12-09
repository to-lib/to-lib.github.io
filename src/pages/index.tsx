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
            Java • Rust • React 19 • Linux • 设计模式 • Spring • Spring Cloud •
            Netty • MySQL • Redis
          </p>
          <p className={styles.heroDescription}>
            全面覆盖前后端开发技术栈，从编程语言基础到框架应用实战。
            系统化学习路径，涵盖 Java/Rust 编程、React 19 前端开发、Linux
            运维、23种设计模式、Spring
            生态（含微服务）、高性能网络编程和主流数据库技术。
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
              <span className={styles.navIcon}>☕</span>
              <Link to="/docs/java">
                <h4>Java 编程</h4>
                <p className={styles.navDesc}>基础到高级</p>
              </Link>
            </div>
            <div className={styles.navCard}>
              <span className={styles.navIcon}>🦀</span>
              <Link to="/docs/rust">
                <h4>Rust 编程</h4>
                <p className={styles.navDesc}>系统编程语言</p>
              </Link>
            </div>
            <div className={styles.navCard}>
              <span className={styles.navIcon}>⚛️</span>
              <Link to="/docs/react">
                <h4>React 19</h4>
                <p className={styles.navDesc}>现代前端框架</p>
              </Link>
            </div>
            <div className={styles.navCard}>
              <span className={styles.navIcon}>🐧</span>
              <Link to="/docs/linux">
                <h4>Linux 系统</h4>
                <p className={styles.navDesc}>运维与脚本</p>
              </Link>
            </div>
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
              <span className={styles.navIcon}>☁️</span>
              <Link to="/docs/springcloud">
                <h4>Spring Cloud</h4>
                <p className={styles.navDesc}>微服务治理</p>
              </Link>
            </div>
            <div className={styles.navCard}>
              <span className={styles.navIcon}>☁️</span>
              <Link to="/docs/springcloud-alibaba">
                <h4>Spring Cloud Alibaba</h4>
                <p className={styles.navDesc}>阿里微服务</p>
              </Link>
            </div>
            <div className={styles.navCard}>
              <span className={styles.navIcon}>⚡</span>
              <Link to="/docs/netty">
                <h4>Netty</h4>
                <p className={styles.navDesc}>高性能网络框架</p>
              </Link>
            </div>
            <div className={styles.navCard}>
              <span className={styles.navIcon}>🐬</span>
              <Link to="/docs/mysql">
                <h4>MySQL</h4>
                <p className={styles.navDesc}>关系型数据库</p>
              </Link>
            </div>
            <div className={styles.navCard}>
              <span className={styles.navIcon}>🔴</span>
              <Link to="/docs/redis">
                <h4>Redis</h4>
                <p className={styles.navDesc}>高性能缓存</p>
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
      title={`${siteConfig.title} - 专业的技术学习与开发工具集`}
      description="全面覆盖前后端开发技术栈：Java、Rust、React 19、Linux、设计模式、Spring生态（Framework/Boot/Cloud）、Netty、MySQL、Redis。提供系统化学习路径、详细教程和最佳实践。"
    >
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
