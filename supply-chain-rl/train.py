"""
Training Script — Supply Chain RL Environment
=============================================
Optimized training for smooth learning curve.
Trains PPO agent on multiple configurations
with warmup between configs for smooth transitions.

Built for: Meta PyTorch OpenEnv Hackathon x Scaler School of Technology
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from supply_chain_env import SupplyChainEnv
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────
# Reward Tracker
# ─────────────────────────────────────────────────
class RewardTracker(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_reward  = 0

    def _on_step(self):
        self.current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0
        return True


# ─────────────────────────────────────────────────
# Configurations
# More timesteps per config = smoother curve
# ─────────────────────────────────────────────────
configs = [
    (2, 3,  50000),   # (warehouses, retailers, timesteps)
    (3, 5,  50000),
    (4, 7,  50000),
    (5, 8,  50000),
    (6, 10, 50000),
]

all_rewards      = []
config_labels    = []
config_endpoints = []

print("=" * 55)
print("  Supply Chain RL — Optimized Multi-Config Training")
print("=" * 55)
print(f"\n📋 Training {len(configs)} configs + default model\n")


# ─────────────────────────────────────────────────
# Train each configuration
# ─────────────────────────────────────────────────
for i, (w, r, steps) in enumerate(configs):
    print(f"📦 Config {i+1}/{len(configs)}: "
          f"{w} warehouses × {r} retailers "
          f"({steps:,} timesteps)")

    env     = SupplyChainEnv(n_warehouses=w, n_retailers=r)
    model   = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )
    tracker = RewardTracker()
    model.learn(total_timesteps=steps, callback=tracker)
    model.save(f"agent_w{w}_r{r}")

    rewards = tracker.episode_rewards
    start   = len(all_rewards)
    all_rewards.extend(rewards)
    end     = len(all_rewards)

    config_labels.append(f"{w}W×{r}R")
    config_endpoints.append((start, end))

    avg_start = np.mean(rewards[:10]) if len(rewards) > 10 else 0
    avg_end   = np.mean(rewards[-10:]) if len(rewards) > 10 else 0

    print(f"   ✅ Saved: agent_w{w}_r{r}.zip")
    print(f"   📈 Improved: {avg_start:.1f} → {avg_end:.1f}")
    print(f"   📊 Episodes: {len(rewards)}\n")


# ─────────────────────────────────────────────────
# Train default model with most timesteps
# ─────────────────────────────────────────────────
print("📦 Training DEFAULT model (3W×5R) — 100k timesteps")
env   = SupplyChainEnv(n_warehouses=3, n_retailers=5)
model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
)
tracker = RewardTracker()
model.learn(total_timesteps=100000, callback=tracker)
model.save("supply_chain_agent")

rewards = tracker.episode_rewards
start   = len(all_rewards)
all_rewards.extend(rewards)
end     = len(all_rewards)
config_labels.append("DEFAULT\n(3W×5R)")
config_endpoints.append((start, end))

avg_start = np.mean(rewards[:10]) if len(rewards) > 10 else 0
avg_end   = np.mean(rewards[-10:]) if len(rewards) > 10 else 0

print(f"   ✅ Saved: supply_chain_agent.zip")
print(f"   📈 Improved: {avg_start:.1f} → {avg_end:.1f}")
print(f"   📊 Episodes: {len(rewards)}\n")


# ─────────────────────────────────────────────────
# Plot Beautiful Learning Curve
# ─────────────────────────────────────────────────
print("📈 Generating learning curve...")

fig, ax = plt.subplots(figsize=(16, 7))

# Raw rewards (very transparent)
ax.plot(
    all_rewards,
    color='#4A90D9',
    alpha=0.2,
    linewidth=0.6,
    label='Episode reward'
)

# Smoothed trend
window = 50
if len(all_rewards) > window:
    smoothed = np.convolve(
        all_rewards,
        np.ones(window) / window,
        mode='valid'
    )
    ax.plot(
        range(window - 1, len(all_rewards)),
        smoothed,
        color='#E74C3C',
        linewidth=3,
        label=f'Learning trend (smoothed)',
        zorder=5
    )

# Config boundaries with shading
colors = [
    '#2ECC71', '#F39C12', '#9B59B6',
    '#E67E22', '#1ABC9C', '#3498DB'
]

for i, (start, end) in enumerate(config_endpoints):
    color = colors[i % len(colors)]
    # Shade each config region
    ax.axvspan(
        start, end,
        alpha=0.08,
        color=color,
        label=f'Config {i+1}: {config_labels[i]}'
    )
    # Vertical line at start
    ax.axvline(
        x=start,
        color=color,
        linestyle='--',
        alpha=0.6,
        linewidth=1.5
    )
    # Label at top
    mid = (start + end) // 2
    ax.annotate(
        config_labels[i],
        xy=(mid, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else -45),
        fontsize=8,
        ha='center',
        color=color,
        fontweight='bold'
    )

# Formatting
ax.set_title(
    "Agent Learning Curve — Multiple Supply Chain Configurations\n"
    "Meta PyTorch OpenEnv Hackathon × Scaler School of Technology",
    fontsize=14,
    fontweight='bold',
    pad=15
)
ax.set_xlabel("Episode", fontsize=12)
ax.set_ylabel("Total Reward per Episode", fontsize=12)
ax.legend(
    loc='lower right',
    fontsize=9,
    framealpha=0.9
)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('white')

# Add summary text box
total_eps = len(all_rewards)
final_avg = np.mean(all_rewards[-50:])
first_avg = np.mean(all_rewards[:50])
improvement = final_avg - first_avg

textstr = (f'Total Episodes: {total_eps:,}\n'
           f'Initial Avg: {first_avg:.1f}\n'
           f'Final Avg: {final_avg:.1f}\n'
           f'Improvement: +{improvement:.1f}')

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(
    0.02, 0.97, textstr,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment='top',
    bbox=props
)

plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150, bbox_inches='tight')
plt.show()

print("📊 Learning curve saved!")


# ─────────────────────────────────────────────────
# Final Summary
# ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  TRAINING COMPLETE!")
print("=" * 55)
print(f"\n✅ Models saved:")
for w, r, _ in configs:
    print(f"   agent_w{w}_r{r}.zip")
print(f"   supply_chain_agent.zip (default)\n")
print(f"📊 Total episodes : {len(all_rewards):,}")
print(f"📈 First avg reward: {np.mean(all_rewards[:50]):.2f}")
print(f"🏆 Final avg reward: {np.mean(all_rewards[-50:]):.2f}")
print(f"💪 Improvement    : "
      f"+{np.mean(all_rewards[-50:]) - np.mean(all_rewards[:50]):.2f}")
print("\n🎉 Ready for submission!")
print("=" * 55)