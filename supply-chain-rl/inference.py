from flask import Flask, request, jsonify
import your_env  # your supply chain env

app = Flask(__name__)
env = your_env.SupplyChainEnv()

@app.route("/reset", methods=["POST"])
def reset():
    obs, info = env.reset()
    return jsonify({
        "observation": obs.tolist() if hasattr(obs, 'tolist') else obs,
        "info": info
    })

@app.route("/step", methods=["POST"])
def step():
    data = request.json
    action = data["action"]
    obs, reward, terminated, truncated, info = env.step(action)
    return jsonify({
        "observation": obs.tolist() if hasattr(obs, 'tolist') else obs,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)