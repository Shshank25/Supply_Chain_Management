"""
Supply Chain RL Environment
============================
A Reinforcement Learning environment built using OpenEnv/Gymnasium.
Simulates a supply chain where an AI agent manages inventory across
3 warehouses and 5 retailers while handling real-world disruptions.

Built for: Meta PyTorch OpenEnv Hackathon x Scaler School of Technology
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SupplyChainEnv(gym.Env):
    """
    Supply Chain Management RL Environment.

    The agent controls ordering decisions across multiple warehouses
    to fulfill retailer demand while minimizing stockouts and overstock.

    Observation Space (9-dimensional):
        - Inventory levels at each warehouse (3 values)
        - Current demand at each retailer  (5 values)
        - Current timestep                 (1 value)

    Action Space (3-dimensional):
        - Order quantity per warehouse (0-10 units each)

    Reward:
        - Based on fulfillment rate
        - Range: -1.0 (worst) to +1.0 (best)

    Configurable Parameters:
        - n_warehouses: Number of warehouses (default: 3)
        - n_retailers:  Number of retailers  (default: 5)
        - max_steps:    Episode length       (default: 100)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_warehouses=3,
        n_retailers=5,
        max_steps=100
    ):
        super().__init__()

        # --- Configurable Parameters ---
        self.n_warehouses = n_warehouses
        self.n_retailers  = n_retailers
        self.max_steps    = max_steps
        self.timestep     = 0

        # --- Internal State ---
        self.inventory = None
        self.demand    = None

        # --- Performance Metrics ---
        self.total_demand    = 0
        self.total_fulfilled = 0
        self.total_stockouts = 0
        self.total_overstock = 0
        self.episode_rewards = []

        # --- Action Space ---
        # Agent orders 0-10 units per warehouse
        self.action_space = spaces.Box(
            low=0, high=10,
            shape=(self.n_warehouses,),
            dtype=np.float32
        )

        # --- Observation Space ---
        # inventory + demand + timestep
        self.observation_space = spaces.Box(
            low=0, high=100,
            shape=(self.n_warehouses + self.n_retailers + 1,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        Called at start of every new episode.
        Also resets all performance metrics.
        """
        self.timestep = 0

        # Reset metrics for new episode
        self.total_demand    = 0
        self.total_fulfilled = 0
        self.total_stockouts = 0
        self.total_overstock = 0
        self.episode_rewards = []

        # Start with moderate inventory
        self.inventory = np.ones(
            self.n_warehouses, dtype=np.float32
        ) * 5

        # Random initial demand
        self.demand = np.random.uniform(
            1, 5, self.n_retailers
        ).astype(np.float32)

        return self._get_obs(), {}

    def step(self, action):
        """
        Execute one timestep.

        Args:
            action: Order quantities per warehouse [shape: (n_warehouses,)]

        Returns:
            observation: New state
            reward: Score (-1.0 to +1.0)
            terminated: True if episode complete
            truncated: Always False
            info: Dict with performance metrics
        """
        action = np.clip(action, 0, 10)

        # Check disruptions
        can_restock, demand_multiplier = self._inject_disruption()

        # Restock warehouses
        if can_restock:
            self.inventory += action

        # Calculate demand and fulfillment
        actual_demand = self.demand * demand_multiplier
        total_demand  = actual_demand.sum()
        total_stock   = self.inventory.sum()
        fulfilled     = min(total_demand, total_stock)
        stockout      = max(0, total_demand - fulfilled)
        overstock     = max(0, total_stock - total_demand)

        # Update metrics
        self.total_demand    += total_demand
        self.total_fulfilled += fulfilled
        self.total_stockouts += stockout
        self.total_overstock += overstock

        # Reduce inventory after fulfillment
        if total_stock > 0:
            ratio = fulfilled / (total_stock + 1e-9)
            self.inventory = np.maximum(
                self.inventory - self.inventory * ratio, 0
            )

        # New demand for next step
        self.demand = np.random.uniform(
            1, 5, self.n_retailers
        ).astype(np.float32)

        self.timestep += 1
        done = self.timestep >= self.max_steps

        # Calculate reward
        reward = self._compute_reward(total_demand, fulfilled)
        self.episode_rewards.append(reward)

        # Build info dict with metrics
        info = self._get_metrics(total_demand, fulfilled,
                                  stockout, overstock)

        return self._get_obs(), reward, done, False, info

    def _get_obs(self):
        """
        Get current observation as flat numpy array.
        Shape: (n_warehouses + n_retailers + 1,)
        """
        return np.concatenate([
            np.clip(self.inventory, 0, 100),
            np.clip(self.demand,    0, 100),
            [self.timestep]
        ]).astype(np.float32)

    def _compute_reward(self, demand, fulfilled):
        """
        Reward based on fulfillment rate.

        Maps:
            100% fulfilled → +1.0 (perfect)
             50% fulfilled →  0.0 (neutral)
              0% fulfilled → -1.0 (stockout)
        """
        fulfillment_rate = fulfilled / (demand + 1e-9)
        reward = (fulfillment_rate * 2) - 1
        return float(reward)

    def _inject_disruption(self):
        """
        Random supply chain disruptions:
            70% → Normal day
            20% → Demand spike (demand × 2)
            10% → Supplier delay (no restock)
        """
        event = np.random.choice(
            ["none", "demand_spike", "supplier_delay"],
            p=[0.7, 0.2, 0.1]
        )
        if event == "demand_spike":
            return True, 2.0
        elif event == "supplier_delay":
            return False, 1.0
        return True, 1.0

    def _get_metrics(self, demand, fulfilled,
                     stockout, overstock):
        """
        Calculate performance metrics for this step.

        Returns dict with:
            - fulfillment_rate: % of demand met (0-1)
            - stockout_rate:    % of demand unfulfilled (0-1)
            - efficiency_score: overall performance (0-100)
            - episode_reward:   cumulative reward so far
        """
        fulfillment_rate = fulfilled / (demand + 1e-9)
        stockout_rate    = stockout  / (demand + 1e-9)

        # Overall efficiency 0-100
        efficiency_score = fulfillment_rate * 100

        # Episode-level metrics
        ep_fulfillment = (
            self.total_fulfilled /
            (self.total_demand + 1e-9)
        ) * 100

        return {
            "fulfillment_rate"    : round(fulfillment_rate, 3),
            "stockout_rate"       : round(stockout_rate, 3),
            "efficiency_score"    : round(efficiency_score, 1),
            "episode_fulfillment" : round(ep_fulfillment, 1),
            "total_demand"        : round(self.total_demand, 1),
            "total_fulfilled"     : round(self.total_fulfilled, 1),
            "total_stockouts"     : round(self.total_stockouts, 1),
        }

    def get_episode_summary(self):
        """
        Get full episode performance summary.
        Call this after episode ends (done=True).
        """
        if self.total_demand == 0:
            return {}

        return {
            "total_steps"         : self.timestep,
            "avg_reward"          : round(
                np.mean(self.episode_rewards), 3
            ),
            "fulfillment_rate_%"  : round(
                self.total_fulfilled /
                self.total_demand * 100, 1
            ),
            "stockout_rate_%"     : round(
                self.total_stockouts /
                self.total_demand * 100, 1
            ),
            "total_demand"        : round(self.total_demand, 1),
            "total_fulfilled"     : round(self.total_fulfilled, 1),
        }


# ── Quick Validation ──────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Supply Chain RL Environment - Validation")
    print("=" * 50)

    # Test default config
    env = SupplyChainEnv()
    obs, _ = env.reset()
    print(f"\n✅ Default config:")
    print(f"   Warehouses : {env.n_warehouses}")
    print(f"   Retailers  : {env.n_retailers}")
    print(f"   Max steps  : {env.max_steps}")
    print(f"   Obs shape  : {obs.shape}")

    # Test custom config (shows reusability)
    env2 = SupplyChainEnv(
        n_warehouses=5,
        n_retailers=10,
        max_steps=200
    )
    obs2, _ = env2.reset()
    print(f"\n✅ Custom config:")
    print(f"   Warehouses : {env2.n_warehouses}")
    print(f"   Retailers  : {env2.n_retailers}")
    print(f"   Max steps  : {env2.max_steps}")
    print(f"   Obs shape  : {obs2.shape}")

    # Run episode and show metrics
    print("\n▶️  Running full episode...")
    obs, _ = env.reset()
    total_reward = 0
    while True:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        if done:
            break

    summary = env.get_episode_summary()
    print("\n📊 Episode Summary:")
    for k, v in summary.items():
        print(f"   {k}: {v}")
    print("\n✅ Environment working correctly!")