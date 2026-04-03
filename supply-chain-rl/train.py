# train.py
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from supply_chain_env import SupplyChainEnv
import matplotlib.pyplot as plt

# --- Track rewards during training ---
class RewardTracker(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_reward = 0

    def _on_step(self):
        self.current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0
        return True

# --- Train the agent ---
print("🚀 Starting training...")
env = SupplyChainEnv()
tracker = RewardTracker()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000, callback=tracker)
model.save("supply_chain_agent")
print("✅ Agent trained and saved!")

# --- Plot learning curve ---
plt.figure(figsize=(10, 5))
plt.plot(tracker.episode_rewards, color='blue', alpha=0.6)
plt.title("Agent Learning Over Time")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.savefig("learning_curve.png")
plt.show()
print("📊 Learning curve saved!")