# -*- coding: utf-8 -*-
"""
OpenEnv Inference Server
========================
Flask server exposing the Supply Chain RL environment via HTTP endpoints.
Built for: Meta PyTorch OpenEnv Hackathon x Scaler School of Technology
"""
import sys
import os

# Ensure the supply-chain-rl subdirectory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "supply-chain-rl"))

from flask import Flask, request, jsonify
from supply_chain_env import SupplyChainEnv, Action

app = Flask(__name__)
env = SupplyChainEnv()


@app.route("/reset", methods=["POST"])
def reset():
    """Reset the environment and return the initial observation."""
    data = request.json or {}
    seed = data.get("seed", None)
    task = data.get("task", None)

    # If a task config is provided, recreate the env with that task
    if task:
        global env
        env = SupplyChainEnv(task=task)

    obs = env.reset(seed=seed)
    return jsonify({
        "observation": obs.model_dump(),
        "info": {}
    })


@app.route("/step", methods=["POST"])
def step():
    """Take one step in the environment with the given action."""
    data = request.json
    action_data = data.get("action", {})

    # Support both {"restock_quantities": [...]} and raw list [...]
    if isinstance(action_data, list):
        action = Action(restock_quantities=action_data)
    elif isinstance(action_data, dict):
        action = Action(**action_data)
    else:
        return jsonify({"error": "Invalid action format"}), 400

    obs, reward, done, info = env.step(action)
    return jsonify({
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "terminated": done,
        "truncated": False,
        "info": info.model_dump()
    })


@app.route("/grade", methods=["GET", "POST"])
def grade():
    """Return the current episode grade (0.0 - 1.0)."""
    return jsonify({
        "grade": env.grade()
    })


@app.route("/state", methods=["GET", "POST"])
def state():
    """Return the current environment state."""
    obs = env.state()
    return jsonify({
        "observation": obs.model_dump()
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
