---
sidebar_position: 39
title: 🎮 强化学习进阶
---

# 强化学习进阶

深入介绍现代强化学习的核心算法。

## PPO (Proximal Policy Optimization)

```python
import torch
import torch.nn as nn

class PPO:
    def __init__(self, actor, critic, lr=3e-4, clip_ratio=0.2):
        self.actor = actor
        self.critic = critic
        self.clip_ratio = clip_ratio
        self.optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=lr
        )

    def compute_advantages(self, rewards, values, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages)

    def update(self, states, actions, old_log_probs, rewards, values):
        advantages = self.compute_advantages(rewards, values)
        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(10):  # PPO epochs
            new_log_probs = self.actor.log_prob(states, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            new_values = self.critic(states)
            critic_loss = nn.functional.mse_loss(new_values, returns)

            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
```

## SAC (Soft Actor-Critic)

```python
class SAC:
    def __init__(self, actor, critic1, critic2, alpha=0.2):
        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2
        self.alpha = alpha  # 熵系数

    def update_critic(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.target_critic1(next_states, next_actions)
            q2_next = self.target_critic2(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_next

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

        return critic_loss

    def update_actor(self, states):
        actions, log_probs = self.actor.sample(states)
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q = torch.min(q1, q2)

        actor_loss = (self.alpha * log_probs - q).mean()
        return actor_loss
```

## TD3 (Twin Delayed DDPG)

```python
class TD3:
    def __init__(self, actor, critic1, critic2, policy_delay=2):
        self.policy_delay = policy_delay
        self.update_count = 0

    def update(self, batch):
        # 更新 critic
        with torch.no_grad():
            noise = torch.randn_like(batch.actions) * 0.2
            next_actions = (self.target_actor(batch.next_states) + noise).clamp(-1, 1)

            q1_next = self.target_critic1(batch.next_states, next_actions)
            q2_next = self.target_critic2(batch.next_states, next_actions)
            target_q = batch.rewards + self.gamma * torch.min(q1_next, q2_next)

        q1 = self.critic1(batch.states, batch.actions)
        q2 = self.critic2(batch.states, batch.actions)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

        self.update_count += 1

        # 延迟更新 actor
        if self.update_count % self.policy_delay == 0:
            actor_loss = -self.critic1(batch.states, self.actor(batch.states)).mean()
            # 更新 actor
            # 软更新 target 网络
```

## 算法对比

| 算法 | 类型         | 动作空间 | 特点       |
| ---- | ------------ | -------- | ---------- |
| DQN  | Value-based  | 离散     | 经典入门   |
| PPO  | Policy       | 两者     | 简单鲁棒   |
| SAC  | Actor-Critic | 连续     | 样本效率高 |
| TD3  | Actor-Critic | 连续     | 稳定性好   |

## 常用库

```python
# Stable Baselines 3
from stable_baselines3 import PPO, SAC, TD3

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# 评估
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
```
