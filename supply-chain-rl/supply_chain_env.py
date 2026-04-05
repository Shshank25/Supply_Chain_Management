# -*- coding: utf-8 -*-
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SupplyChainEnv(gym.Env):
    """
    Supply Chain RL Environment
    Built for: Meta PyTorch OpenEnv Hackathon x Scaler
    """
    metadata = {"render_modes": []}

    def __init__(self, n_warehouses=3, n_retailers=5, max_steps=100):
        super().__init__()
        self.n_warehouses = n_warehouses
        self.n_retailers  = n_retailers
        self.max_steps    = max_steps
        self.timestep     = 0
        self.inventory    = None
        self.demand       = None

        self.total_demand    = 0
        self.total_fulfilled = 0
        self.total_stockouts = 0
        self.total_overstock = 0
        self.episode_rewards = []

        self.action_space = spaces.Box(
            low=0, high=10,
            shape=(self.n_warehouses,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0, high=500,
            shape=(self.n_warehouses + self.n_retailers + 1,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.timestep        = 0
        self.total_demand    = 0
        self.total_fulfilled = 0
        self.total_stockouts = 0
        self.total_overstock = 0
        self.episode_rewards = []

        if seed is not None:
            self.action_space.seed(seed)

        # Start LOW so agent learns to actively restock
        self.inventory = np.ones(self.n_warehouses, dtype=np.float32) * 3
        self.demand    = self.np_random.uniform(1, 5, self.n_retailers).astype(np.float32)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, 0, 10)

        can_restock, demand_mult = self._inject_disruption()

        if can_restock:
            self.inventory += action

        actual_demand = self.demand * demand_mult
        total_demand  = float(actual_demand.sum())
        total_stock   = float(self.inventory.sum())
        fulfilled     = min(total_demand, total_stock)
        stockout      = max(0, total_demand - fulfilled)
        overstock     = max(0, total_stock - total_demand)

        self.total_demand    += total_demand
        self.total_fulfilled += fulfilled
        self.total_stockouts += stockout
        self.total_overstock += overstock

        if total_stock > 0:
            ratio = fulfilled / (total_stock + 1e-9)
            self.inventory = np.maximum(
                self.inventory - self.inventory * ratio, 0
            )

        self.demand    = self.np_random.uniform(1, 5, self.n_retailers).astype(np.float32)
        self.timestep += 1
        done           = self.timestep >= self.max_steps

        reward = self._compute_reward(total_demand, fulfilled, overstock)
        self.episode_rewards.append(reward)

        info = self._get_metrics(total_demand, fulfilled, stockout, overstock)
        return self._get_obs(), reward, done, False, info

    def _get_obs(self):
        return np.concatenate([
            np.clip(self.inventory, 0, 500),
            np.clip(self.demand,    0, 500),
            [self.timestep / self.max_steps]
        ]).astype(np.float32)

    def _compute_reward(self, demand, fulfilled, overstock):
        """
        Simple and clean reward.
        High fulfillment = high reward.
        Agent must learn to keep stock high to beat random.
        """
        rate = fulfilled / (demand + 1e-9)
        reward = (rate * 2) - 1

        # Tiny penalty only for extreme overstock
        if demand > 0 and overstock > demand * 5:
            reward -= 0.1

        return float(np.clip(reward, -1.0, 1.0))

    def _inject_disruption(self):
        """
        Random disruptions:
            70% Normal
            20% Demand spike x2
            10% Supplier delay
        """
        event = self.np_random.choice(
            ["none", "demand_spike", "supplier_delay"],
            p=[0.7, 0.2, 0.1]
        )
        if event == "demand_spike":
            return True, 2.0
        elif event == "supplier_delay":
            return False, 1.0
        return True, 1.0

    def _get_metrics(self, demand, fulfilled, stockout, overstock):
        rate    = fulfilled / (demand + 1e-9)
        ep_rate = self.total_fulfilled / (self.total_demand + 1e-9)
        return {
            "fulfillment_rate"    : round(float(rate), 3),
            "stockout_rate"       : round(float(stockout / (demand + 1e-9)), 3),
            "efficiency_score"    : round(float(rate * 100), 1),
            "episode_fulfillment" : round(float(ep_rate * 100), 1),
            "total_demand"        : round(self.total_demand, 1),
            "total_fulfilled"     : round(self.total_fulfilled, 1),
            "total_stockouts"     : round(self.total_stockouts, 1),
        }

    def get_episode_summary(self):
        if self.total_demand == 0:
            return {}
        return {
            "total_steps"        : self.timestep,
            "avg_reward"         : round(float(np.mean(self.episode_rewards)), 3),
            "fulfillment_rate_%" : round(self.total_fulfilled / self.total_demand * 100, 1),
            "stockout_rate_%"    : round(self.total_stockouts / self.total_demand * 100, 1),
            "total_demand"       : round(self.total_demand, 1),
            "total_fulfilled"    : round(self.total_fulfilled, 1),
        }


if __name__ == "__main__":
    env = SupplyChainEnv()
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")
    total = 0
    for _ in range(100):
        o, r, d, _, i = env.step(env.action_space.sample())
        total += r
        if d: break
    print(f"Total reward: {total:.2f}")
    print("Environment OK!")
