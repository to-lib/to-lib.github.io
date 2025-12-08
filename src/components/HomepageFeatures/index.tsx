import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  emoji: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: '📚 完整覆盖',
    emoji: '📚',
    description: (
      <>
        包含全部23种经典设计模式，分为创建型、结构型和行为型三大类。
        每个模式都有详细的讲解、完整的代码示例和实际应用场景。
      </>
    ),
  },
  {
    title: '💡 实战应用',
    emoji: '💡',
    description: (
      <>
        不仅讲解理论，更重要的是展示如何在实际项目中应用这些模式。
        学习Spring、Hibernate等开源框架中的模式使用。
      </>
    ),
  },
  {
    title: '🚀 快速上手',
    emoji: '🚀',
    description: (
      <>
        提供快速参考表、决策树和学习路径。从初级到高级，循序渐进地掌握设计模式。
      </>
    ),
  },
];

function Feature({title, emoji, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4', styles.featureCol)}>
      <div className={styles.featureCard}>
        <div className={styles.featureIcon}>{emoji}</div>
        <div className={styles.featureContent}>
          <Heading as="h3" className={styles.featureTitle}>{title}</Heading>
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
            为什么选择这份指南？
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
              <div className={styles.statNumber}>23</div>
              <div className={styles.statLabel}>设计模式</div>
            </div>
            <div className={styles.stat}>
              <div className={styles.statNumber}>100+</div>
              <div className={styles.statLabel}>代码示例</div>
            </div>
            <div className={styles.stat}>
              <div className={styles.statNumber}>50+</div>
              <div className={styles.statLabel}>应用场景</div>
            </div>
            <div className={styles.stat}>
              <div className={styles.statNumber}>80K+</div>
              <div className={styles.statLabel}>字内容</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
