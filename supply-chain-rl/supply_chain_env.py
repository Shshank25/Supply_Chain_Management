# -*- coding: utf-8 -*-
"""
Supply Chain RL Environment
============================
Built for: Meta PyTorch OpenEnv Hackathon x Scaler School of Technology
"""
import numpy as np
from pydantic import BaseModel, Field

# OpenEnv compatibility — falls back to Gymnasium if OpenEnv unavailable
try:
    import openenv
    from openenv import spaces
    _BaseEnv = openenv.env
    _GYM_BACKEND = "openenv"
except Exception:
    import gymnasium as gym
    from gymnasium import spaces
    _BaseEnv = gym.Env
    _GYM_BACKEND = "gymnasium"

# ==========================================
# OpenEnv Pydantic Models
# ==========================================
class Observation(BaseModel):
    inventory: list[float] = Field(description="Inventory levels at each warehouse")
    demand: list[float] = Field(description="Current demand at each retailer")
    timestep: int = Field(description="Current timestep")
    max_steps: int = Field(description="Maximum steps in the episode")

class Action(BaseModel):
    restock_quantities: list[float] = Field(description="Restock quantity per warehouse (0-10)")

class Reward(BaseModel):
    value: float = Field(description="Fulfillment reward, ranges -1 to +1")
    incremental_fulfilled: float = Field(description="Incremental items fulfilled in this step")

class Info(BaseModel):
    fulfillment_rate: float
    stockout_rate: float
    efficiency_score: float
    episode_fulfillment: float
    total_demand: float
    total_fulfilled: float
    total_stockouts: float

class SupplyChainEnv(_BaseEnv):
    """
    Supply Chain RL Environment.

    The agent controls restocking decisions across multiple warehouses
    to fulfill retailer demand while handling real-world disruptions.
    """
    metadata = {"render_modes": []}

    def __init__(self, n_warehouses=3, n_retailers=5, max_steps=100, task="medium"):
        super().__init__()
        self.n_warehouses = n_warehouses
        self.n_retailers  = n_retailers
        self.max_steps    = max_steps
        self.task         = task  # easy, medium, hard
        self.timestep     = 0
        self.inventory    = None
        self.demand       = None

        self.total_demand    = 0
        self.total_fulfilled = 0
        self.total_stockouts = 0
        self.total_overstock = 0
        self.episode_rewards = []

        # Action: restock quantity per warehouse (0-10)
        self.action_space = spaces.Box(
            low=0, high=10,
            shape=(self.n_warehouses,),
            dtype=np.float32
        )

        # Observation: inventory + demand + timestep
        self.observation_space = spaces.Box(
            low=0, high=500,
            shape=(self.n_warehouses + self.n_retailers + 1,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None) -> Observation:
        if hasattr(super(), 'reset'):
            try:
                super().reset(seed=seed)
            except Exception:
                pass

        if seed is not None:
            np.random.seed(seed)

        self.timestep        = 0
        self.total_demand    = 0
        self.total_fulfilled = 0
        self.total_stockouts = 0
        self.total_overstock = 0
        self.episode_rewards = []

        # Start low so agent learns to actively restock
        self.inventory = np.ones(self.n_warehouses, dtype=np.float32) * 3
        self.demand    = np.random.uniform(1, 5, self.n_retailers).astype(np.float32)
        return self._get_obs()

    def state(self) -> Observation:
        """OpenEnv specification: returns the current state."""
        return self._get_obs()

    def step(self, action: Action):
        restock = np.array(action.restock_quantities, dtype=np.float32)
        restock = np.clip(restock, 0, 10)

        can_restock, demand_mult = self._inject_disruption()

        if can_restock:
            self.inventory += restock

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

        self.demand    = np.random.uniform(1, 5, self.n_retailers).astype(np.float32)
        self.timestep += 1
        done           = self.timestep >= self.max_steps

        rew_value = self._compute_reward(total_demand, fulfilled, overstock)
        self.episode_rewards.append(rew_value)
        
        reward = Reward(value=rew_value, incremental_fulfilled=fulfilled)
        info = Info(**self._get_metrics(actual_demand.sum(), fulfilled, stockout, overstock))

        return self._get_obs(), reward, done, info

    def grade(self) -> float:
        """
        OpenEnv programmatic grader: 
        Returns a score bounded between 0.0 and 1.0 reflecting how well the agent performed.
        """
        if self.total_demand == 0:
            return 0.0
        return float(np.clip(self.total_fulfilled / self.total_demand, 0.0, 1.0))

    def _get_obs(self) -> Observation:
        return Observation(
            inventory=np.clip(self.inventory, 0, 500).tolist(),
            demand=np.clip(self.demand, 0, 500).tolist(),
            timestep=self.timestep,
            max_steps=self.max_steps
        )

    def _compute_reward(self, demand, fulfilled, overstock):
        """
        Simple fulfillment reward.
        High fulfillment = high reward.
        Incremental progress rewarded.
        """
        rate   = fulfilled / (demand + 1e-9)
        reward = (rate * 2) - 1

        # Tiny penalty for extreme overstock only
        if demand > 0 and overstock > demand * 5:
            reward -= 0.1

        return float(np.clip(reward, -1.0, 1.0))

    def _inject_disruption(self):
        """
        Random supply chain disruptions based on task difficulty.
        """
        if self.task == "easy":
            p = [1.0, 0.0, 0.0]
        elif self.task == "medium":
            p = [0.7, 0.2, 0.1]
        elif self.task == "hard":
            p = [0.4, 0.35, 0.25]
        else:
            p = [1.0, 0.0, 0.0]
            
        event = np.random.choice(
            ["none", "demand_spike", "supplier_delay"],
            p=p
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
    print(f"Backend: {_GYM_BACKEND}")
    env = SupplyChainEnv()
    obs = env.reset()
    print(f"Obs: {obs}")
    total = 0
    for _ in range(100):
        # sample action
        sampled = env.action_space.sample()
        act = Action(restock_quantities=sampled.tolist())
        o, r, d, i = env.step(act)
        total += r.value
        if d: break
    print(f"Total reward: {total:.2f}")
    print(f"Grade: {env.grade():.2f}")
    print("Environment OK!")