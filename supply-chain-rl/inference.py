# -*- coding: utf-8 -*-
"""
inference.py — Required entry point for OpenEnv Hackathon evaluation.
Runs the trained A2C agent on the Supply Chain environment.
"""
import os
import numpy as np
from supply_chain_env import SupplyChainEnv

try:
    from stable_baselines3 import A2C, PPO
except ImportError:
    raise ImportError("stable_baselines3 required: pip install stable_baselines3")


def load_best_model(env):
    """Load best available model — A2C preferred over PPO."""
    candidates = [
        ("supply_chain_agent_a2c", A2C),
        ("supply_chain_agent",     PPO),
    ]
    for path, cls in candidates:
        if os.path.exists(path + ".zip"):
            try:
                return cls.load(path), path
            except Exception as e:
                print(f"Could not load {path}: {e}")
    return None, None


def run_inference(n_warehouses=3, n_retailers=5, seed=42, verbose=True):
    """
    Run one full episode with the trained agent.
    Returns episode summary dict.
    """
    env   = SupplyChainEnv(n_warehouses=n_warehouses, n_retailers=n_retailers)
    model, model_path = load_best_model(env)

    if model is None:
        print("No trained model found. Running random agent.")

    obs, _ = env.reset(seed=seed)
    total_reward = 0
    step_count   = 0

    if verbose:
        print("=" * 55)
        print("  Supply Chain RL — Inference")
        print(f"  Model     : {model_path or 'Random'}")
        print(f"  Warehouses: {n_warehouses} | Retailers: {n_retailers}")
        print("=" * 55)

    while True:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        step_count   += 1

        if verbose and step_count % 10 == 0:
            print(
                f"Step {step_count:03d} | "
                f"Reward: {reward:+.3f} | "
                f"Fulfilled: {info['fulfillment_rate']*100:.1f}%"
            )

        if done:
            break

    summary = env.get_episode_summary()

    if verbose:
        print("=" * 55)
        print("  EPISODE SUMMARY")
        print("=" * 55)
        print(f"  Fulfillment Rate : {summary.get('fulfillment_rate_%', 0):.1f}%")
        print(f"  Stockout Rate    : {summary.get('stockout_rate_%', 0):.1f}%")
        print(f"  Total Reward     : {total_reward:.3f}")
        print(f"  Total Steps      : {summary.get('total_steps', 0)}")
        print("=" * 55)

    return {
        "model"            : model_path or "random",
        "n_warehouses"     : n_warehouses,
        "n_retailers"      : n_retailers,
        "total_reward"     : round(total_reward, 3),
        "fulfillment_rate" : summary.get("fulfillment_rate_%", 0),
        "stockout_rate"    : summary.get("stockout_rate_%", 0),
        "total_steps"      : summary.get("total_steps", 0),
    }


if __name__ == "__main__":
    # Default evaluation
    result = run_inference(n_warehouses=3, n_retailers=5, seed=42)
    print(f"\nFinal Result: {result}")