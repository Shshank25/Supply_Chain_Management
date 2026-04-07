# -*- coding: utf-8 -*-
"""
Baseline Inference Script for OpenEnv
Evaluates an Agent on the Supply Chain RL Environment.
"""
import os
import sys
import json
from openai import OpenAI
from supply_chain_env import SupplyChainEnv, Action

def run_baseline():
    # 1. Read required environment variables with defaults
    api_base_url = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
    hf_token = os.environ.get("HF_TOKEN")
    
    # 2. Validate HF_TOKEN
    if not hf_token:
        print("Warning: HF_TOKEN environment variable is missing or empty.", file=sys.stderr)
        
    client = OpenAI(
        base_url=api_base_url,
        api_key=hf_token or "mock_token"
    )

    env = SupplyChainEnv(task="medium", max_steps=10, use_pydantic=True)

    # 3. Exactly print [START] before execution
    print("[START]")
    try:
        obs = env.reset(seed=42)
        done = False
        
        while not done:
            # 4. Exactly print [STEP] on every step
            print("[STEP]")
            
            # Simple dummy heuristics for agent action to ensure execution
            action_array = [3.0] * env.n_warehouses
            action = Action(restock_quantities=action_array)
            obs, reward, done, info = env.step(action)
            
            # 5. Format outputs correctly: 
            # - exactly 2 decimal places for reward
            # - lowercase booleans for done and success
            reward_val = f"{float(reward.value):.2f}"
            is_done = "true" if done else "false"
            is_success = "true" if env.grade() >= 0.8 else "false"
            
            print(f'{{"reward": {reward_val}, "done": {is_done}, "success": {is_success}}}')

    except Exception as e:
        print(f"Exception encountered: {e}", file=sys.stderr)
    finally:
        # 6. Exactly print [END] always, even on exception
        print("[END]")

if __name__ == "__main__":
    run_baseline()
