"""
Supply Chain RL Environment
============================
A Reinforcement Learning environment built using the OpenEnv/Gymnasium framework.
Simulates a supply chain where an AI agent learns to manage inventory across
3 warehouses and 5 retailers while handling real-world disruptions.

Built for: Meta PyTorch OpenEnv Hackathon x Scaler School of Technology
Team: [Add your team name here]
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SupplyChainEnv(gym.Env):
    """
    Supply Chain Management Environment.

    The agent controls ordering decisions across multiple warehouses
    to fulfill retailer demand while minimizing stockouts and overstock.

    Observation Space:
        - Inventory levels at each warehouse (3 values)
        - Current demand at each retailer (5 values)
        - Current timestep (1 value)
        Total: 9-dimensional continuous observation

    Action Space:
        - Order quantity for each warehouse (3 values, range 0-10)
        Total: 3-dimensional continuous action

    Reward:
        - Based on fulfillment rate (fulfilled demand / total demand)
        - Range: -1.0 (worst) to +1.0 (best)
        - Positive when majority of demand is met
        - Negative when stockouts occur
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # --- Environment Configuration ---
        self.n_warehouses = 3      # Number of warehouses
        self.n_retailers  = 5      # Number of retailers
        self.max_steps    = 100    # Episode length
        self.timestep     = 0      # Current step counter

        # --- Internal State ---
        self.inventory = None      # Stock at each warehouse
        self.demand    = None      # Demand at each retailer

        # --- Action Space ---
        # Agent decides how many units to order per warehouse (0 to 10)
        self.action_space = spaces.Box(
            low=0,
            high=10,
            shape=(self.n_warehouses,),
            dtype=np.float32
        )

        # --- Observation Space ---
        # Agent observes: inventory + demand + timestep
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(self.n_warehouses + self.n_retailers + 1,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        Called at the start of every new episode.

        Returns:
            observation: Initial state of the environment
            info: Empty dict (required by Gymnasium API)
        """
        self.timestep  = 0

        # Start with moderate inventory levels
        self.inventory = np.ones(
            self.n_warehouses, dtype=np.float32
        ) * 5

        # Random initial demand across retailers
        self.demand = np.random.uniform(
            1, 5, self.n_retailers
        ).astype(np.float32)

        return self._get_obs(), {}

    def step(self, action):
        """
        Execute one timestep in the environment.

        Args:
            action: Array of order quantities per warehouse [shape: (3,)]

        Returns:
            observation: New state after action
            reward: Score for this action (-1.0 to +1.0)
            terminated: True if episode is complete
            truncated: Always False
            info: Empty dict
        """
        # Clip action to valid range
        action = np.clip(action, 0, 10)

        # Check for supply chain disruptions
        can_restock, demand_multiplier = self._inject_disruption()

        # Restock warehouses (unless supplier delay)
        if can_restock:
            self.inventory += action

        # Calculate demand and fulfillment
        actual_demand = self.demand * demand_multiplier
        total_demand  = actual_demand.sum()
        total_stock   = self.inventory.sum()
        fulfilled     = min(total_demand, total_stock)

        # Reduce inventory proportionally after fulfillment
        if total_stock > 0:
            ratio = fulfilled / (total_stock + 1e-9)
            self.inventory = np.maximum(
                self.inventory - self.inventory * ratio, 0
            )

        # Generate new demand for next step
        self.demand = np.random.uniform(
            1, 5, self.n_retailers
        ).astype(np.float32)

        # Advance timestep and check if episode is done
        self.timestep += 1
        done = self.timestep >= self.max_steps

        # Calculate reward
        reward = self._compute_reward(total_demand, fulfilled)

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        """
        Get current observation as a flat numpy array.

        Returns:
            Array of shape (9,) containing:
            [warehouse_inventory x3, retailer_demand x5, timestep x1]
        """
        return np.concatenate([
            np.clip(self.inventory, 0, 100),  # warehouse stock
            np.clip(self.demand,    0, 100),  # retailer demand
            [self.timestep]                    # current step
        ]).astype(np.float32)

    def _compute_reward(self, demand, fulfilled):
        """
        Calculate reward based on fulfillment rate.

        Reward formula:
            reward = (fulfillment_rate * 2) - 1

        This maps:
            100% fulfillment → reward = +1.0 (perfect)
             50% fulfillment → reward =  0.0 (neutral)
              0% fulfillment → reward = -1.0 (stockout)

        Args:
            demand: Total demand this step
            fulfilled: Total demand actually fulfilled

        Returns:
            reward: Float between -1.0 and +1.0
        """
        fulfillment_rate = fulfilled / (demand + 1e-9)
        reward = (fulfillment_rate * 2) - 1
        return float(reward)

    def _inject_disruption(self):
        """
        Randomly inject supply chain disruption events.

        Disruption probabilities:
            70% → Normal day (no disruption)
            20% → Demand spike (demand doubles)
            10% → Supplier delay (no restock this turn)

        Returns:
            can_restock: Whether warehouses can restock
            demand_multiplier: Multiplier applied to demand
        """
        event = np.random.choice(
            ["none", "demand_spike", "supplier_delay"],
            p=[0.7, 0.2, 0.1]
        )

        if event == "demand_spike":
            # Demand suddenly doubles — tests agent robustness
            return True, 2.0

        elif event == "supplier_delay":
            # Supplier can't deliver — no restocking this turn
            return False, 1.0

        # Normal day
        return True, 1.0


# ============================================================
# Quick validation test — run this file directly to verify
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Supply Chain RL Environment - Validation Test")
    print("=" * 50)

    env = SupplyChainEnv()

    print(f"\n📦 Warehouses : {env.n_warehouses}")
    print(f"🏪 Retailers  : {env.n_retailers}")
    print(f"⏱️  Max Steps  : {env.max_steps}")
    print(f"🎮 Action Space: {env.action_space}")
    print(f"👁️  Obs Space  : {env.observation_space}")

    print("\n🔄 Resetting environment...")
    obs, _ = env.reset()
    print(f"Initial obs shape: {obs.shape}")

    print("\n▶️  Running 5 random steps...")
    total_reward = 0
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        print(f"Step {i+1} | Reward: {reward:+.3f} | Done: {done}")

    print(f"\n🏆 Total reward over 5 steps: {total_reward:.3f}")
    print("\n✅ Environment working correctly!")