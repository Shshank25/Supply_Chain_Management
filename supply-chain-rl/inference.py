# -*- coding: utf-8 -*-
"""
Baseline Inference Script for OpenEnv
Evaluates an LLM Agent (using OpenAI) on the Supply Chain RL Environment across all tasks.
"""
import os
import json
from openai import OpenAI
from supply_chain_env import SupplyChainEnv, Action

def call_llm(client: OpenAI, obs_data: dict, n_warehouses: int) -> list[float]:
    """
    Calls the OpenAI API to determine the restock quantities based on current observation.
    """
    prompt = f"""You are a supply chain manager controlling {n_warehouses} warehouses.
Your goal is to fulfill as much demand as possible across all retailers, avoiding stockouts but preventing extreme overstocking.
Each step you can request restocks for each warehouse. The maximum restock per warehouse is 10 units.

Current State:
{json.dumps(obs_data, indent=2)}

Output a JSON array of exactly {n_warehouses} numbers representing the restock quantities (from 0 to 10) for each warehouse.
Example output:
[2, 5, 0]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    # Parse the response safely
    try:
        content = response.choices[0].message.content.strip()
        # strip markdown formatting if any
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        action_array = json.loads(content.strip())
        
        # Ensure correct length
        if len(action_array) != n_warehouses:
            action_array = [0]*n_warehouses
            
    except Exception as e:
        print(f"Error parsing LLM response: {e}. Defaulting to 0 restock.")
        action_array = [0] * n_warehouses
        
    return action_array

def run_baseline():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable is missing.")
        print("Returning mock scores for baseline demonstration.")
        # If no key is provided, we can either exit or mock it. Let's just exit.
        # But for Hackathon, it's polite to provide mock if key fails.
        pass
    
    # We will initialize the client. It will fail if api_key is really needed and missing, 
    # but the environment should provide it.
    client = OpenAI(api_key=api_key) if api_key else None
    
    tasks = ["easy", "medium", "hard"]
    results = {}
    
    for task in tasks:
        # Running with 20 max steps to keep API costs down for baseline testing
        env = SupplyChainEnv(task=task, max_steps=20)
        obs = env.reset(seed=42)
        done = False
        print(f"\n--- Running Task: {task.upper()} ---")
        
        step_count = 0
        while not done:
            if client:
                action_array = call_llm(client, obs.model_dump(), env.n_warehouses)
            else:
                # If no OpenAI key, we act naively (restock 3 per warehouse)
                action_array = [3.0] * env.n_warehouses
                
            action = Action(restock_quantities=action_array)
            obs, reward, done, info = env.step(action)
            step_count += 1
            
            if step_count % 5 == 0:
                print(f"Step {step_count}: Fulfillment Rate {info.fulfillment_rate*100:.1f}%, Reward: {reward.value:.2f}")
                
        final_grade = env.grade()
        results[task] = final_grade
        print(f"Task '{task}' completed. Grade: {final_grade:.2f}")

    print("\n===============================")
    print(" BASELINE INFERENCE RESULTS ")
    print("===============================")
    for task, score in results.items():
        print(f" {task.title():<8} ->  {score:.3f}")
    print("===============================")

if __name__ == "__main__":
    run_baseline()