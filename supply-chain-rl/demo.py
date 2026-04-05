import os

import gradio as gr
import numpy as np
from stable_baselines3 import A2C, PPO

from supply_chain_env import SupplyChainEnv


MATCHED_DEMO_SEED = 42
TRAINED_LABEL = "Trained Agent"
RANDOM_LABEL = "Random Agent"
MODEL_CANDIDATES = [
    ("supply_chain_agent_a2c", A2C, "A2C"),
    ("supply_chain_agent", PPO, "PPO"),
]


DEMO_CSS = """
:root {
    --bg-a: #081120;
    --bg-b: #111b31;
    --panel: rgba(16, 24, 43, 0.88);
    --border: rgba(116, 149, 255, 0.2);
    --ink: #edf4ff;
    --muted: #a9b7d6;
    --accent: #4cd4ff;
}

.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(76, 212, 255, 0.16), transparent 26%),
        linear-gradient(180deg, var(--bg-a) 0%, var(--bg-b) 100%);
    color: var(--ink);
    font-family: "Aptos", "Segoe UI Variable Text", "Trebuchet MS", sans-serif;
}

.quick-shell {
    max-width: 920px;
    margin: 0 auto;
}

.quick-hero,
.quick-note {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 18px 46px rgba(0, 0, 0, 0.18);
}

.quick-hero h1,
.quick-note h3 {
    margin: 0 0 8px;
    color: var(--ink);
}

.quick-hero p,
.quick-note p {
    margin: 0;
    color: var(--muted);
    line-height: 1.6;
}
"""


def load_model():
    for model_path, model_cls, algorithm_name in MODEL_CANDIDATES:
        if not os.path.exists(model_path + ".zip"):
            continue
        try:
            return model_cls.load(model_path), algorithm_name
        except Exception as exc:
            print(f"Model load error for {model_path}.zip: {exc}")
    return None, "Unavailable"


model, model_algorithm = load_model()


def predict_agent_action(agent, obs, env):
    action, _ = agent.predict(obs, deterministic=True)

    agent_low = np.asarray(agent.action_space.low, dtype=np.float32)
    agent_high = np.asarray(agent.action_space.high, dtype=np.float32)
    env_low = np.asarray(env.action_space.low, dtype=np.float32)
    env_high = np.asarray(env.action_space.high, dtype=np.float32)

    if not (
        np.allclose(agent_low, env_low) and np.allclose(agent_high, env_high)
    ):
        action = env_low + (action - agent_low) * (env_high - env_low) / (
            agent_high - agent_low + 1e-9
        )

    return np.clip(action, env_low, env_high)


def run_demo(mode):
    env = SupplyChainEnv()
    obs, _ = env.reset(seed=MATCHED_DEMO_SEED)
    total_reward = 0
    log = [
        f"Mode: {mode}",
        f"Matched evaluation seed: {MATCHED_DEMO_SEED}",
        f"Checkpoint type: {model_algorithm}",
        "-" * 48,
    ]

    for step in range(50):
        if mode == TRAINED_LABEL and model is not None:
            action = predict_agent_action(model, obs, env)
        else:
            action = env.action_space.sample()

        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        log.append(
            f"Step {step + 1:02d} | Reward: {reward:.3f} | Total: {total_reward:.2f}"
        )
        if done:
            break

    log.append(f"\nFinal Total Reward: {total_reward:.2f}")
    return "\n".join(log)


with gr.Blocks(title="Supply Chain RL Quick Demo") as demo:
    gr.HTML(
        """
        <div class="quick-shell">
          <div class="quick-hero">
            <h1>Supply Chain RL Quick Demo</h1>
            <p>
              Run a single matched episode with the default trained checkpoint or the random baseline.
              This view is meant for a fast sanity check; the main dashboard provides averaged multi-episode comparisons.
            </p>
          </div>
        </div>
        """
    )

    with gr.Row():
        mode = gr.Radio(
            [TRAINED_LABEL, RANDOM_LABEL],
            label="Decision maker",
            value=TRAINED_LABEL,
        )
        run_button = gr.Button("Run quick episode", variant="primary")

    gr.HTML(
        f"""
        <div class="quick-shell">
          <div class="quick-note">
            <h3>Evaluation note</h3>
            <p>
              The quick demo reuses the same scenario seed ({MATCHED_DEMO_SEED}) each run so switching between the two modes stays comparable.
            </p>
          </div>
        </div>
        """
    )

    output = gr.Textbox(label="Episode log", lines=25)

    run_button.click(run_demo, inputs=mode, outputs=output)


if __name__ == "__main__":
    demo.launch(css=DEMO_CSS)
