import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SupplyChainEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.n_warehouses = 3
        self.n_retailers = 5
        self.max_steps = 100
        self.timestep = 0
        self.inventory = None
        self.demand = None

        # Action: order 0-10 units per warehouse
        self.action_space = spaces.Box(
            low=0, high=10,
            shape=(self.n_warehouses,),
            dtype=np.float32
        )

        # Observation: inventory + demand + timestep
        self.observation_space = spaces.Box(
            low=0, high=100,
            shape=(self.n_warehouses + self.n_retailers + 1,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.inventory = np.ones(self.n_warehouses, dtype=np.float32) * 5
        self.demand = np.random.uniform(1, 5, self.n_retailers).astype(np.float32)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, 0, 10)

        # Disruption check
        can_restock, demand_multiplier = self._inject_disruption()

        # Restock
        if can_restock:
            self.inventory += action

        # Apply demand
        actual_demand = self.demand * demand_multiplier
        total_demand = actual_demand.sum()
        total_stock = self.inventory.sum()
        fulfilled = min(total_demand, total_stock)

        # Reduce inventory
        if total_stock > 0:
            ratio = fulfilled / (total_stock + 1e-9)
            self.inventory = np.maximum(self.inventory - self.inventory * ratio, 0)

        # New demand
        self.demand = np.random.uniform(1, 5, self.n_retailers).astype(np.float32)
        self.timestep += 1
        done = self.timestep >= self.max_steps

        reward = self._compute_reward(total_demand, fulfilled)
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return np.concatenate([
            np.clip(self.inventory, 0, 100),
            np.clip(self.demand, 0, 100),
            [self.timestep]
        ]).astype(np.float32)

    def _compute_reward(self, demand, fulfilled):
        # Fulfillment rate: 0 to 1
        fulfillment_rate = fulfilled / (demand + 1e-9)

        # Reward between -1 and +1
        reward = (fulfillment_rate * 2) - 1
        return float(reward)

    def _inject_disruption(self):
        event = np.random.choice(
            ["none", "demand_spike", "supplier_delay"],
            p=[0.7, 0.2, 0.1]
        )
        if event == "demand_spike":
            return True, 2.0   # demand doubles
        elif event == "supplier_delay":
            return False, 1.0  # no restock
        return True, 1.0


# --- Quick Test ---
if __name__ == "__main__":
    env = SupplyChainEnv()
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print("\nRunning 5 steps...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        print(f"Step {i+1} | Reward: {reward:.3f} | Done: {done}")
    print("\nEnvironment working correctly!")