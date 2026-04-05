# -*- coding: utf-8 -*-
import gradio as gr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from supply_chain_env import SupplyChainEnv
from stable_baselines3 import A2C, PPO

EVAL_SEEDS = [11, 22, 33, 44, 55]

APP_CSS = """
:root {
    --bg-top: #050d1a;
    --bg-mid: #081325;
    --bg-bottom: #101a31;
    --panel: rgba(10, 18, 35, 0.78);
    --panel-strong: rgba(8, 16, 31, 0.92);
    --panel-soft: rgba(17, 30, 56, 0.76);
    --panel-border: rgba(110, 163, 255, 0.18);
    --panel-glow: rgba(76, 215, 255, 0.18);
    --ink: #edf5ff;
    --muted: #a6b6d7;
    --accent: #57d8ff;
    --accent-2: #8dffb2;
    --accent-3: #ffd27b;
    --warn: #ff8f7c;
    --button-a: #2f73ff;
    --button-b: #38d1ff;
    --shadow: 0 28px 80px rgba(0, 0, 0, 0.28);
    --shadow-soft: 0 18px 50px rgba(0, 0, 0, 0.18);
}

.gradio-container {
    position: relative;
    overflow-x: hidden;
    background:
        radial-gradient(circle at 12% 10%, rgba(87, 216, 255, 0.2), transparent 26%),
        radial-gradient(circle at 88% 12%, rgba(141, 255, 178, 0.12), transparent 18%),
        radial-gradient(circle at 50% 100%, rgba(255, 210, 123, 0.10), transparent 20%),
        linear-gradient(180deg, var(--bg-top) 0%, var(--bg-mid) 38%, var(--bg-bottom) 100%);
    color: var(--ink);
    font-family: "Bahnschrift", "Aptos", "Trebuchet MS", sans-serif;
}

.gradio-container::before,
.gradio-container::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
}

.gradio-container::before {
    background:
        linear-gradient(90deg, rgba(255, 255, 255, 0.045) 1px, transparent 1px),
        linear-gradient(rgba(255, 255, 255, 0.045) 1px, transparent 1px);
    background-size: 120px 120px;
    mask-image: radial-gradient(circle at center, black 10%, transparent 72%);
    opacity: 0.12;
}

.gradio-container::after {
    background:
        radial-gradient(circle at 18% 30%, rgba(87, 216, 255, 0.22), transparent 16%),
        radial-gradient(circle at 78% 62%, rgba(141, 255, 178, 0.16), transparent 14%);
    filter: blur(24px);
    opacity: 0.7;
}

.app-shell {
    position: relative;
    z-index: 1;
    max-width: 1180px;
    margin: 0 auto;
}

.hero-panel,
.control-card,
.insight-card,
.metric-card,
.plot-shell,
.summary-shell {
    background: linear-gradient(180deg, rgba(14, 24, 46, 0.92) 0%, rgba(10, 18, 34, 0.86) 100%);
    border: 1px solid var(--panel-border);
    border-radius: 24px;
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(18px);
    animation: rise-in 0.8s ease both;
}

.hero-panel {
    position: relative;
    overflow: hidden;
    padding: 36px 36px 30px;
    margin-bottom: 18px;
    box-shadow: var(--shadow);
}

.hero-panel::before,
.hero-panel::after {
    content: "";
    position: absolute;
    border-radius: 999px;
    pointer-events: none;
}

.hero-panel::before {
    width: 280px;
    height: 280px;
    right: -60px;
    top: -90px;
    background: radial-gradient(circle, rgba(87, 216, 255, 0.28) 0%, rgba(87, 216, 255, 0.02) 72%);
    filter: blur(6px);
}

.hero-panel::after {
    width: 240px;
    height: 240px;
    left: -90px;
    bottom: -130px;
    background: radial-gradient(circle, rgba(141, 255, 178, 0.18) 0%, rgba(141, 255, 178, 0.01) 74%);
}

.hero-grid {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(300px, 0.8fr);
    gap: 20px;
    align-items: end;
}

.hero-copy-block {
    max-width: 720px;
}

.hero-brow {
    display: inline-block;
    padding: 8px 12px;
    border-radius: 999px;
    background: rgba(87, 216, 255, 0.12);
    color: var(--accent);
    font-size: 12px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-weight: 700;
    border: 1px solid rgba(87, 216, 255, 0.18);
}

.hero-title {
    margin: 18px 0 12px;
    font-size: clamp(38px, 4vw, 64px);
    line-height: 0.95;
    font-weight: 900;
    letter-spacing: -0.04em;
    color: var(--ink);
}

.hero-accent {
    color: transparent;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
    background-clip: text;
}

.hero-copy {
    margin: 0;
    max-width: 760px;
    color: var(--muted);
    font-size: 17px;
    line-height: 1.7;
}

.hero-chip-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 20px;
}

.hero-chip {
    padding: 9px 13px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.05);
    color: var(--ink);
    border: 1px solid rgba(255, 255, 255, 0.09);
    font-size: 13px;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

.hero-statboard {
    display: grid;
    gap: 14px;
}

.hero-stat {
    position: relative;
    overflow: hidden;
    padding: 18px 18px 16px;
    border-radius: 20px;
    background: linear-gradient(180deg, rgba(14, 28, 54, 0.78) 0%, rgba(9, 18, 35, 0.88) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

.hero-stat::after {
    content: "";
    position: absolute;
    inset: auto -30px -40px auto;
    width: 120px;
    height: 120px;
    background: radial-gradient(circle, rgba(87, 216, 255, 0.16), transparent 70%);
    pointer-events: none;
}

.hero-stat-label {
    display: block;
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
}

.hero-stat-value {
    display: block;
    color: var(--ink);
    font-size: 26px;
    line-height: 1.05;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero-stat-copy {
    display: block;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.55;
}

.control-card,
.insight-card,
.plot-shell,
.summary-shell {
    position: relative;
    overflow: hidden;
    padding: 20px 20px 16px;
}

.panel-kicker {
    color: var(--accent);
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 8px;
}

.panel-title {
    margin: 0 0 8px;
    color: var(--ink);
    font-size: 24px;
    line-height: 1.15;
}

.panel-copy {
    margin: 0 0 14px;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.7;
}

.insight-stack {
    display: grid;
    gap: 12px;
}

.insight-item {
    display: grid;
    grid-template-columns: 44px 1fr;
    gap: 12px;
    align-items: start;
    padding: 12px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.insight-index {
    display: grid;
    place-items: center;
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(87, 216, 255, 0.2), rgba(141, 255, 178, 0.16));
    color: var(--ink);
    font-weight: 800;
    letter-spacing: 0.08em;
}

.insight-body strong {
    display: block;
    color: var(--ink);
    font-size: 14px;
    margin-bottom: 4px;
}

.insight-body p {
    margin: 0;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.55;
}

.comparison-band {
    display: flex;
    justify-content: space-between;
    gap: 20px;
    align-items: center;
    padding: 18px 22px;
    margin: 6px 0 16px;
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: linear-gradient(135deg, rgba(11, 20, 38, 0.94), rgba(20, 34, 64, 0.82));
    box-shadow: var(--shadow-soft);
}

.comparison-band.win {
    border-color: rgba(141, 255, 178, 0.22);
    box-shadow: 0 22px 60px rgba(0, 0, 0, 0.22), 0 0 0 1px rgba(141, 255, 178, 0.05);
}

.comparison-band.warn {
    border-color: rgba(255, 143, 124, 0.20);
}

.comparison-band.idle {
    border-color: rgba(87, 216, 255, 0.18);
}

.band-kicker {
    color: var(--accent);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 700;
    margin-bottom: 6px;
}

.band-title {
    color: var(--ink);
    font-size: 28px;
    line-height: 1.05;
    font-weight: 850;
    margin-bottom: 6px;
}

.band-copy {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.6;
    max-width: 760px;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin: 4px 0 8px;
}

.metric-card {
    position: relative;
    overflow: hidden;
    padding: 18px 18px 16px;
    min-height: 140px;
    background: linear-gradient(180deg, rgba(14, 24, 46, 0.9) 0%, rgba(9, 18, 35, 0.82) 100%);
}

.metric-card::after {
    content: "";
    position: absolute;
    inset: auto -40px -50px auto;
    width: 130px;
    height: 130px;
    background: radial-gradient(circle, rgba(87, 216, 255, 0.14), transparent 70%);
    pointer-events: none;
}

.metric-card.spotlight {
    border-color: rgba(87, 216, 255, 0.24);
    box-shadow: 0 22px 56px rgba(0, 0, 0, 0.24), 0 0 0 1px rgba(87, 216, 255, 0.05);
}

.metric-label {
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    line-height: 1.1;
    color: var(--ink);
}

.metric-value.win {
    color: var(--accent-2);
}

.metric-value.warn {
    color: #ffd27d;
}

.metric-subtext {
    margin-top: 8px;
    font-size: 13px;
    color: var(--muted);
    line-height: 1.5;
}

.summary-shell {
    padding-bottom: 12px;
}

.summary-title {
    margin: 0 0 8px;
    font-size: 24px;
    color: var(--ink);
}

.summary-copy {
    margin: 0 0 12px;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.7;
}

.score-pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 10px 14px;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(87, 216, 255, 0.18), rgba(141, 255, 178, 0.12));
    border: 1px solid rgba(255, 255, 255, 0.09);
    color: var(--ink);
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    white-space: nowrap;
}

.plot-title {
    margin: 0 0 10px;
    font-size: 22px;
    color: var(--ink);
}

.plot-copy {
    margin: 0 0 12px;
    font-size: 14px;
    color: var(--muted);
    line-height: 1.6;
}

.log-panel textarea {
    min-height: 330px !important;
    border-radius: 18px !important;
    border: 1px solid rgba(110, 163, 255, 0.18) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}

.gr-button-primary {
    background: linear-gradient(135deg, var(--button-a), var(--button-b)) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    box-shadow: 0 16px 34px rgba(47, 115, 255, 0.34) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

.gr-button-primary:hover {
    filter: brightness(1.05);
    transform: translateY(-1px);
    box-shadow: 0 18px 38px rgba(56, 209, 255, 0.34) !important;
}

.gr-slider input[type="range"] {
    accent-color: var(--accent);
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .gr-box,
.gradio-container .gr-group {
    border-color: var(--panel-border) !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container .wrap {
    background: rgba(7, 13, 28, 0.72) !important;
    color: var(--ink) !important;
}

.gradio-container .label-wrap span,
.gradio-container label,
.gradio-container .prose,
.gradio-container .prose p {
    color: var(--ink) !important;
}

.gradio-container .prose strong {
    color: var(--ink) !important;
}

@keyframes rise-in {
    from {
        opacity: 0;
        transform: translateY(16px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (max-width: 900px) {
    .hero-grid {
        grid-template-columns: 1fr;
    }

    .hero-title {
        font-size: 34px;
    }

    .metric-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .comparison-band {
        flex-direction: column;
        align-items: flex-start;
    }
}

@media (max-width: 640px) {
    .metric-grid {
        grid-template-columns: 1fr;
    }

    .hero-panel,
    .control-card,
    .insight-card,
    .summary-shell,
    .metric-card,
    .plot-shell,
    .comparison-band {
        border-radius: 16px;
    }

    .hero-panel {
        padding: 28px 24px 24px;
    }
}
"""

HERO_HTML = """
<div class="app-shell">
  <section class="hero-panel">
    <div class="hero-grid">
      <div class="hero-copy-block">
        <div class="hero-brow">Reinforcement learning command deck</div>
        <h1 class="hero-title">Supply Chain RL <span class="hero-accent">Control Room</span></h1>
        <p class="hero-copy">
          Stress-test the trained control policy against a random decision-maker under the exact same shocks,
          demand spikes, and supplier delays. This dashboard is built to feel like an executive control surface,
          not just a model demo.
        </p>
        <div class="hero-chip-row">
          <span class="hero-chip">Matched seeded scenarios</span>
          <span class="hero-chip">Deterministic trained policy</span>
          <span class="hero-chip">Reward, fulfillment, and stockout tracking</span>
        </div>
      </div>
      <div class="hero-statboard">
        <div class="hero-stat">
          <span class="hero-stat-label">Evaluation mode</span>
          <span class="hero-stat-value">5 synchronized rollouts</span>
          <span class="hero-stat-copy">
            Both agents face the same turbulence so the result reflects decision quality rather than lucky randomness.
          </span>
        </div>
        <div class="hero-stat">
          <span class="hero-stat-label">Primary outcome</span>
          <span class="hero-stat-value">Service level with discipline</span>
          <span class="hero-stat-copy">
            The strongest policy keeps fulfillment high while containing stockouts and preserving total reward.
          </span>
        </div>
      </div>
    </div>
  </section>
</div>
"""

DEFAULT_HIGHLIGHTS = """
<div class="app-shell">
  <section class="comparison-band idle">
    <div>
      <div class="band-kicker">Awaiting simulation</div>
      <div class="band-title">Shape the network, then light up the deck.</div>
      <div class="band-copy">
        The result board will spotlight the winning policy, reward edge, service-level lift, stockout reduction, and checkpoint in play.
      </div>
    </div>
    <div class="score-pill">5 matched episodes / 100 steps each</div>
  </section>
</div>
"""

GUIDE_PANEL_HTML = """
<div class="insight-card">
  <div class="panel-kicker">Evaluation protocol</div>
  <h3 class="panel-title">One network. Two decision-makers. Same turbulence.</h3>
  <p class="panel-copy">
    The visual comparison is only useful if the simulation is fair. This deck keeps the matchup disciplined.
  </p>
  <div class="insight-stack">
    <div class="insight-item">
      <div class="insight-index">01</div>
      <div class="insight-body">
        <strong>Matched disruptions</strong>
        <p>Demand spikes and supplier delays are synchronized across both agents.</p>
      </div>
    </div>
    <div class="insight-item">
      <div class="insight-index">02</div>
      <div class="insight-body">
        <strong>Deterministic policy replay</strong>
        <p>The trained controller is evaluated without action noise so the signal stays clean.</p>
      </div>
    </div>
    <div class="insight-item">
      <div class="insight-index">03</div>
      <div class="insight-body">
        <strong>Averaged scoreboards</strong>
        <p>Charts summarize five full 100-step episodes to reduce single-rollout luck.</p>
      </div>
    </div>
  </div>
</div>
"""


def get_model_candidates(n_warehouses):
    model_map = {
        2: "agent_w2_r3",
        3: "agent_w3_r5",
        4: "agent_w4_r7",
        5: "agent_w5_r8",
        6: "agent_w6_r10",
    }
    base_name = model_map.get(int(n_warehouses), "supply_chain_agent")
    return [
        (f"{base_name}_a2c", A2C, "A2C"),
        (base_name, PPO, "PPO"),
        ("supply_chain_agent_a2c", A2C, "A2C"),
        ("supply_chain_agent", PPO, "PPO"),
    ]


def load_agent(n_warehouses):
    errors = []

    for model_path, model_cls, algorithm_name in get_model_candidates(n_warehouses):
        if not os.path.exists(model_path + ".zip"):
            continue
        try:
            return model_cls.load(model_path), model_path, algorithm_name
        except Exception as e:
            errors.append(f"{model_path}.zip ({algorithm_name}): {e}")

    if errors:
        print("Model load errors:")
        for error in errors:
            print(f"  - {error}")

    fallback_path, _, fallback_algorithm = get_model_candidates(n_warehouses)[0]
    return None, fallback_path, fallback_algorithm


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


def run_episode(n_warehouses, n_retailers, seed, agent=None):
    env = SupplyChainEnv(
        n_warehouses=int(n_warehouses),
        n_retailers=int(n_retailers)
    )

    obs, _ = env.reset(seed=seed)
    rewards, fulfillments, stockouts = [], [], []
    total_reward = 0

    for step in range(100):
        if agent is not None:
            action = predict_agent_action(agent, obs, env)
        else:
            action = env.action_space.sample()

        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        rewards.append(reward)
        fulfillments.append(info['fulfillment_rate'] * 100)
        stockouts.append(info.get('stockout_rate', 0) * 100)
        if done:
            break

    summary = env.get_episode_summary()
    return rewards, fulfillments, stockouts, summary, total_reward


def aggregate_runs(n_warehouses, n_retailers, seeds, agent=None):
    rewards_runs, fulfill_runs, stockout_runs = [], [], []
    summaries, total_rewards = [], []

    for seed in seeds:
        rewards, fulfillments, stockouts, summary, total_reward = run_episode(
            n_warehouses, n_retailers, seed=seed, agent=agent
        )
        rewards_runs.append(np.array(rewards, dtype=np.float32))
        fulfill_runs.append(np.array(fulfillments, dtype=np.float32))
        stockout_runs.append(np.array(stockouts, dtype=np.float32))
        summaries.append(summary)
        total_rewards.append(total_reward)

    mean_summary = {
        "total_steps": int(round(np.mean([s.get("total_steps", 0) for s in summaries]))),
        "avg_reward": round(float(np.mean([s.get("avg_reward", 0) for s in summaries])), 3),
        "fulfillment_rate_%": round(float(np.mean([s.get("fulfillment_rate_%", 0) for s in summaries])), 1),
        "stockout_rate_%": round(float(np.mean([s.get("stockout_rate_%", 0) for s in summaries])), 1),
        "total_demand": round(float(np.mean([s.get("total_demand", 0) for s in summaries])), 1),
        "total_fulfilled": round(float(np.mean([s.get("total_fulfilled", 0) for s in summaries])), 1),
    }

    return {
        "rewards": np.mean(np.stack(rewards_runs), axis=0),
        "fulfillments": np.mean(np.stack(fulfill_runs), axis=0),
        "stockouts": np.mean(np.stack(stockout_runs), axis=0),
        "summary": mean_summary,
        "total_reward": float(np.mean(total_rewards)),
    }


def metric_delta(trained_value, random_value, suffix="", invert=False):
    delta = trained_value - random_value
    if invert:
        delta = random_value - trained_value
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{abs(delta):.1f}{suffix}"


def style_plot_axis(ax):
    ax.set_facecolor('#0b152b')
    ax.grid(axis='y', color='#28405d', alpha=0.34, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors='#d9e8ff', labelsize=8.5)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#375474')
        ax.spines[spine].set_linewidth(1.0)


def annotate_terminal(ax, x_value, y_value, text, color):
    ax.scatter([x_value], [y_value], color=color, s=34, zorder=5)
    ax.annotate(
        text,
        (x_value, y_value),
        xytext=(8, 0),
        textcoords='offset points',
        color='white',
        fontsize=8,
        va='center',
        bbox=dict(boxstyle='round,pad=0.25', fc=color, ec='none', alpha=0.9),
    )


def label_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        offset = 1.2 if height >= 0 else -3.0
        va = 'bottom' if height >= 0 else 'top'
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            f'{height:.1f}',
            ha='center',
            va=va,
            color='white',
            fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.22', fc='#091321', ec='none', alpha=0.72),
        )


def build_highlights_html(
    n_warehouses,
    n_retailers,
    model_name,
    algorithm_name,
    trained_agent_loaded,
    t_summary,
    r_summary,
    t_total,
    r_total,
    improvement,
    num_seeds,
):
    result_label = "Trained policy is in command" if t_total >= r_total else "Random baseline steals the lead"
    result_tone = "win" if t_total >= r_total else "warn"
    model_note = (
        f"Loaded {algorithm_name} checkpoint {model_name}.zip"
        if trained_agent_loaded
        else f"No trained {algorithm_name} checkpoint was available, so the trained lane fell back to random actions."
    )
    scenario_note = (
        f"{n_warehouses} warehouses / {n_retailers} retailers / "
        f"{num_seeds} matched episodes / 100 steps each"
    )

    cards = [
        (
            "Reward edge",
            f"{improvement:+.1f}%",
            f"Trained {t_total:.1f} total reward vs random {r_total:.1f}.",
            result_tone,
            "spotlight",
        ),
        (
            "Fulfillment lift",
            metric_delta(
                t_summary.get("fulfillment_rate_%", 0),
                r_summary.get("fulfillment_rate_%", 0),
                suffix=" pts",
            ),
            f"Trained {t_summary.get('fulfillment_rate_%', 0):.1f}% vs random {r_summary.get('fulfillment_rate_%', 0):.1f}%.",
            "win",
            "",
        ),
        (
            "Stockout reduction",
            metric_delta(
                t_summary.get("stockout_rate_%", 0),
                r_summary.get("stockout_rate_%", 0),
                suffix=" pts",
                invert=True,
            ),
            f"Lower is better. Trained {t_summary.get('stockout_rate_%', 0):.1f}% vs random {r_summary.get('stockout_rate_%', 0):.1f}%.",
            "win",
            "",
        ),
        (
            "Active checkpoint",
            model_name,
            f"{model_note}",
            "win" if trained_agent_loaded else "warn",
            "",
        ),
    ]

    cards_html = "".join(
        f"""
        <div class="metric-card {extra_class}">
          <div class="metric-label">{label}</div>
          <div class="metric-value {tone}">{value}</div>
          <div class="metric-subtext">{subtext}</div>
        </div>
        """
        for label, value, subtext, tone, extra_class in cards
    )

    return f"""
    <div class="app-shell">
      <section class="comparison-band {result_tone}">
        <div>
          <div class="band-kicker">Live verdict</div>
          <div class="band-title">{result_label}</div>
          <div class="band-copy">{scenario_note}</div>
        </div>
        <div class="score-pill">{algorithm_name} / {model_name}.zip</div>
      </section>
      <div class="metric-grid">
        {cards_html}
      </div>
    </div>
    """


def run_demo(n_warehouses, n_retailers):
    n_warehouses = int(n_warehouses)
    n_retailers  = int(n_retailers)
    seeds = EVAL_SEEDS
    trained_agent, model_name, algorithm_name = load_agent(n_warehouses)

    trained_results = aggregate_runs(
        n_warehouses, n_retailers, seeds=seeds, agent=trained_agent
    )
    random_results = aggregate_runs(
        n_warehouses, n_retailers, seeds=seeds, agent=None
    )

    t_rewards = trained_results["rewards"]
    t_fulfill = trained_results["fulfillments"]
    t_summary = trained_results["summary"]
    t_total = trained_results["total_reward"]

    r_rewards = random_results["rewards"]
    r_fulfill = random_results["fulfillments"]
    r_summary = random_results["summary"]
    r_total = random_results["total_reward"]

    steps = list(range(1, len(t_rewards) + 1))
    trained_color = '#57d8ff'
    trained_fill = '#1588b8'
    random_color = '#ff8f7c'
    random_fill = '#c65f49'
    fulfillment_color = '#8dffb2'

    fig1, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), dpi=120)
    fig1.patch.set_facecolor('#081220')
    for ax in axes:
        style_plot_axis(ax)

    axes[0].plot(
        steps, t_rewards, color=trained_color, linewidth=2.7, label='Trained agent',
        solid_capstyle='round'
    )
    axes[0].plot(
        steps, r_rewards, color=random_color, linewidth=2.2, label='Random agent',
        alpha=0.82, solid_capstyle='round'
    )
    axes[0].fill_between(steps, t_rewards, 0, color=trained_fill, alpha=0.16)
    axes[0].fill_between(steps, r_rewards, 0, color=random_fill, alpha=0.10)
    axes[0].axhline(0, color='#d9e8ff', linestyle='--', alpha=0.22, linewidth=1)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_title('Reward trajectory', color='white', fontsize=14, fontweight='bold', loc='left')
    axes[0].text(
        0.02, 0.92, 'Higher is better',
        transform=axes[0].transAxes, color='#90a7c8', fontsize=8.5
    )
    axes[0].set_xlabel('Step', color='white')
    axes[0].set_ylabel('Reward signal', color='white')
    axes[0].legend(
        facecolor='#091321',
        edgecolor='#263f5c',
        labelcolor='white',
        framealpha=0.92,
        fontsize=8.5,
        loc='lower left',
    )
    annotate_terminal(axes[0], steps[-1], t_rewards[-1], f"{t_rewards[-1]:.2f}", trained_color)
    annotate_terminal(axes[0], steps[-1], r_rewards[-1], f"{r_rewards[-1]:.2f}", random_color)

    axes[1].plot(
        steps, t_fulfill, color=fulfillment_color, linewidth=2.7, label='Trained agent',
        solid_capstyle='round'
    )
    axes[1].plot(
        steps, r_fulfill, color=random_color, linewidth=2.2, label='Random agent',
        alpha=0.82, solid_capstyle='round'
    )
    axes[1].fill_between(steps, t_fulfill, color='#3d9f73', alpha=0.15)
    axes[1].fill_between(steps, r_fulfill, color=random_fill, alpha=0.10)
    axes[1].set_ylim(0, 105)
    axes[1].set_title('Service level curve', color='white', fontsize=14, fontweight='bold', loc='left')
    axes[1].text(
        0.02, 0.92, 'Average fulfillment across matched episodes',
        transform=axes[1].transAxes, color='#90a7c8', fontsize=8.5
    )
    axes[1].set_xlabel('Step', color='white')
    axes[1].set_ylabel('Fulfillment %', color='white')
    axes[1].legend(
        facecolor='#091321',
        edgecolor='#263f5c',
        labelcolor='white',
        framealpha=0.92,
        fontsize=8.5,
        loc='lower left',
    )
    annotate_terminal(axes[1], steps[-1], t_fulfill[-1], f"{t_fulfill[-1]:.1f}%", fulfillment_color)
    annotate_terminal(axes[1], steps[-1], r_fulfill[-1], f"{r_fulfill[-1]:.1f}%", random_color)

    fig1.tight_layout(pad=2.6)

    fig2, ax2 = plt.subplots(figsize=(10.4, 5.2), dpi=120)
    fig2.patch.set_facecolor('#081220')
    style_plot_axis(ax2)

    metrics      = ['Fulfillment %', 'Stockout %', 'Avg Reward x10', 'Total Reward']
    trained_vals = [
        t_summary.get('fulfillment_rate_%', 0),
        t_summary.get('stockout_rate_%', 0),
        t_summary.get('avg_reward', 0) * 10,
        t_total
    ]
    random_vals = [
        r_summary.get('fulfillment_rate_%', 0),
        r_summary.get('stockout_rate_%', 0),
        r_summary.get('avg_reward', 0) * 10,
        r_total
    ]

    x     = np.arange(len(metrics))
    width = 0.35
    bars1 = ax2.bar(x - width/2, trained_vals, width,
                    label='Trained agent', color=trained_color, alpha=0.9,
                    edgecolor='#7ce6ff', linewidth=1.1)
    bars2 = ax2.bar(x + width/2, random_vals,  width,
                    label='Random agent',  color=random_color, alpha=0.86,
                    edgecolor='#ffb2a5', linewidth=1.1)

    all_values = trained_vals + random_vals
    min_value = min(all_values)
    max_value = max(all_values)
    pad = max(8, (max_value - min_value) * 0.20)
    ax2.set_ylim(min_value - pad, max_value + pad)
    ax2.axhline(0, color='#d9e8ff', alpha=0.22, linewidth=1)
    ax2.set_title('Matched evaluation scoreboard', color='white',
                  fontsize=14, fontweight='bold', loc='left')
    ax2.text(
        0.01, 0.95,
        f'{n_warehouses}W / {n_retailers}R / {len(seeds)} synchronized rollouts',
        transform=ax2.transAxes, color='#90a7c8', fontsize=8.5
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, color='white', fontsize=9)
    ax2.set_ylabel('Score snapshot', color='white')
    ax2.legend(
        facecolor='#091321',
        edgecolor='#263f5c',
        labelcolor='white',
        framealpha=0.92,
        fontsize=8.5,
        loc='upper left',
    )
    label_bars(ax2, bars1)
    label_bars(ax2, bars2)

    fig2.tight_layout(pad=2.4)

    def winner(t, r, higher_better=True):
        if higher_better:
            return "Trained wins" if t > r else "Random wins"
        return "Trained wins" if t < r else "Random wins"

    t_fulfill_rate = t_summary.get('fulfillment_rate_%', 0)
    r_fulfill_rate = r_summary.get('fulfillment_rate_%', 0)
    t_stockout     = t_summary.get('stockout_rate_%', 0)
    r_stockout     = r_summary.get('stockout_rate_%', 0)
    t_avg          = t_summary.get('avg_reward', 0)
    r_avg          = r_summary.get('avg_reward', 0)

    improvement = ((t_total - r_total) / (abs(r_total) + 1e-9)) * 100
    highlights_html = build_highlights_html(
        n_warehouses=n_warehouses,
        n_retailers=n_retailers,
        model_name=model_name,
        algorithm_name=algorithm_name,
        trained_agent_loaded=trained_agent is not None,
        t_summary=t_summary,
        r_summary=r_summary,
        t_total=t_total,
        r_total=r_total,
        improvement=improvement,
        num_seeds=len(seeds),
    )

    summary_text = f"""
{'='*55}
  COMPARISON SUMMARY
  Config: {n_warehouses} Warehouses x {n_retailers} Retailers
  Average over {len(seeds)} matched evaluation episodes
{'='*55}

METRIC                TRAINED      RANDOM        RESULT
{'-'*55}
Fulfillment Rate   {t_fulfill_rate:>8.1f}%   {r_fulfill_rate:>8.1f}%    {winner(t_fulfill_rate, r_fulfill_rate)}
Stockout Rate      {t_stockout:>8.1f}%   {r_stockout:>8.1f}%    {winner(t_stockout, r_stockout, False)}
Avg Reward         {t_avg:>8.3f}    {r_avg:>8.3f}    {winner(t_avg, r_avg)}
Total Reward       {t_total:>8.3f}    {r_total:>8.3f}    {winner(t_total, r_total)}
Total Steps        {t_summary.get('total_steps', 0):>8}    {r_summary.get('total_steps', 0):>8}
{'-'*55}
Total Demand       {t_summary.get('total_demand', 0):>8.1f}    {r_summary.get('total_demand', 0):>8.1f}
Total Fulfilled    {t_summary.get('total_fulfilled', 0):>8.1f}    {r_summary.get('total_fulfilled', 0):>8.1f}
{'='*55}
Improvement: {improvement:+.1f}% better total reward vs random
Model file: {model_name}.zip ({algorithm_name})
{'='*55}
"""
    return highlights_html, fig1, fig2, summary_text


# ── Gradio UI ──
with gr.Blocks(title="Supply Chain RL", fill_width=True) as demo:
    gr.HTML(HERO_HTML)

    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            with gr.Group(elem_classes="control-card"):
                gr.HTML(
                    """
                    <div class="panel-kicker">Scenario composer</div>
                    <h3 class="panel-title">Re-shape the network and replay the same shocks.</h3>
                    <p class="panel-copy">
                      Drag the sliders to change the supply network footprint, then launch the synchronized evaluation to see how the trained policy behaves under pressure.
                    </p>
                    """
                )
                warehouses = gr.Slider(2, 6, value=3, step=1, label="Warehouses in the network")
                retailers  = gr.Slider(3, 10, value=5, step=1, label="Retailers served")
                btn = gr.Button("Launch matched simulation", variant="primary", size="lg", elem_id="run-deck-button")
        with gr.Column(scale=4):
            gr.HTML(GUIDE_PANEL_HTML)

    highlights = gr.HTML(value=DEFAULT_HIGHLIGHTS)

    with gr.Row(equal_height=True):
        with gr.Column(scale=6):
            with gr.Group(elem_classes="plot-shell"):
                gr.HTML(
                    """
                    <div class="panel-kicker">Trajectory view</div>
                    <h3 class="plot-title">How each policy behaves over time</h3>
                    <p class="plot-copy">
                      Watch reward and fulfillment move through the episode horizon to see whether performance is consistent or fragile.
                    </p>
                    """
                )
                chart1 = gr.Plot(label="Average step-by-step performance")
        with gr.Column(scale=5):
            with gr.Group(elem_classes="plot-shell"):
                gr.HTML(
                    """
                    <div class="panel-kicker">Scoreboard</div>
                    <h3 class="plot-title">At-a-glance policy comparison</h3>
                    <p class="plot-copy">
                      This panel compresses the matchup into the four business-facing numbers that matter most.
                    </p>
                    """
                )
                chart2 = gr.Plot(label="Matched evaluation comparison")

    with gr.Group(elem_classes="summary-shell"):
        gr.HTML(
            """
            <div class="panel-kicker">Detailed transcript</div>
            <h3 class="summary-title">Comparison log</h3>
            <p class="summary-copy">
              Use the detailed readout when you want the exact averaged totals behind the headline verdict.
            </p>
            """
        )
        summary = gr.Textbox(
            label="Detailed evaluation log",
            lines=20,
            value="Run a simulation to populate the evaluation summary.",
            show_label=False,
            elem_classes="log-panel",
        )

    btn.click(
        run_demo,
        inputs=[warehouses, retailers],
        outputs=[highlights, chart1, chart2, summary]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(), css=APP_CSS)
