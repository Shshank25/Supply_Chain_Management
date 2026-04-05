# -*- coding: utf-8 -*-
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from supply_chain_env import SupplyChainEnv
import numpy as np


TRAIN_SEED = 42
EVAL_SEEDS = [11, 22, 33, 44, 55]
CONFIGS = [
    (2, 3, 80_000),
    (3, 5, 80_000),
    (4, 7, 80_000),
    (5, 8, 80_000),
    (6, 10, 80_000),
]
DEFAULT_CONFIG = (3, 5, 120_000)


def build_env(n_warehouses, n_retailers):
    return SupplyChainEnv(n_warehouses=n_warehouses, n_retailers=n_retailers)


def evaluate_model(model, n_warehouses, n_retailers, seeds):
    total_rewards = []
    fulfillment_rates = []

    for seed in seeds:
        env = build_env(n_warehouses, n_retailers)
        obs, _ = env.reset(seed=seed)
        total_reward = 0.0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break

        summary = env.get_episode_summary()
        total_rewards.append(total_reward)
        fulfillment_rates.append(summary.get("fulfillment_rate_%", 0.0))

    return {
        "avg_total_reward": float(np.mean(total_rewards)),
        "avg_fulfillment_rate": float(np.mean(fulfillment_rates)),
    }


def train_and_save(n_warehouses, n_retailers, total_timesteps, output_name):
    print(f"Training A2C model for {n_warehouses}W x {n_retailers}R...")
    env = make_vec_env(
        lambda: build_env(n_warehouses, n_retailers),
        n_envs=8,
        seed=TRAIN_SEED,
    )
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=7e-4,
        gamma=0.99,
        ent_coef=0.0,
        seed=TRAIN_SEED,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model.save(output_name)

    metrics = evaluate_model(model, n_warehouses, n_retailers, EVAL_SEEDS)
    print(
        f"Saved {output_name}.zip | "
        f"avg_total_reward={metrics['avg_total_reward']:.3f} | "
        f"avg_fulfillment={metrics['avg_fulfillment_rate']:.1f}%"
    )


if __name__ == "__main__":
    for warehouses, retailers, steps in CONFIGS:
        train_and_save(
            n_warehouses=warehouses,
            n_retailers=retailers,
            total_timesteps=steps,
            output_name=f"agent_w{warehouses}_r{retailers}_a2c",
        )

    default_warehouses, default_retailers, default_steps = DEFAULT_CONFIG
    train_and_save(
        n_warehouses=default_warehouses,
        n_retailers=default_retailers,
        total_timesteps=default_steps,
        output_name="supply_chain_agent_a2c",
    )

    print("All A2C models trained.")
