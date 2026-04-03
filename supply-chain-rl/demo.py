import gradio as gr
import numpy as np
from stable_baselines3 import PPO
from supply_chain_env import SupplyChainEnv

model = PPO.load("supply_chain_agent")

def run_demo(mode):
    env = SupplyChainEnv()
    obs, _ = env.reset()
    total_reward = 0
    log = []

    for step in range(50):
        if mode == "🤖 Trained Agent":
            action, _ = model.predict(obs)
        else:
            action = env.action_space.sample()

        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        log.append(
            f"Step {step+1:02d} | Reward: {reward:.3f} | Total: {total_reward:.2f}"
        )
        if done:
            break

    log.append(f"\n🏆 Final Total Reward: {total_reward:.2f}")
    return "\n".join(log)

gr.Interface(
    fn=run_demo,
    inputs=gr.Radio(
        ["🤖 Trained Agent", "🎲 Random Agent"],
        label="Who makes decisions?",
        value="🤖 Trained Agent"
    ),
    outputs=gr.Textbox(label="Episode Log", lines=25),
    title="🏭 Supply Chain RL Environment",
    description="Compare trained AI agent vs random decisions!"
).launch()