---
title: Supply Chain Rl
emoji: 🚀
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 4.31.5
app_file: supply-chain-rl/app.py
pinned: false
---
# Supply Chain RL
**Meta PyTorch OpenEnv Hackathon Submission**

## Environment Overview & Motivation
This repository provides a custom OpenEnv environment simulating real-world Supply Chain Management. The agent is tasked with making daily restocking decisions across multiple warehouses to fulfill stochastic retail demand. It aims to bridge the gap between academic RL puzzles and complex, noisy environments humans interact with daily. The agent is rewarded for maintaining high fulfillment rates while minimizing extreme overstock and handling random demand spikes or supplier delays.

## Task Configurations

The environment operates under an OpenEnv schema, testing agents across three difficulty tasks:

* **easy**: Stable supply chain with normal baseline demand (100% normal days).
* **medium**: Occasional demand shocks and supplier delays (70% normal, 20% spike, 10% delay).
* **hard**: Frequent severe shocks preventing restocking and doubling demand natively (40% normal, 35% spike, 25% delay).

## Spaces Definition (Pydantic OpenEnv Models)

* **Observation Model**:
  * `inventory` (List[float]): Current stock at each warehouse.
  * `demand` (List[float]): Current outstanding order demands at each retailer.
  * `timestep` (int): Number of steps elapsed.
  * `max_steps` (int): Maximum episode length.

* **Action Model**:
  * `restock_quantities` (List[float]): Quantity to order for each warehouse (0-10 units bounded).

* **Reward Model**:
  * `value` (float): The actual scalar reward between -1.0 and 1.0 driving RL loop optimization.
  * `incremental_fulfilled` (float): Extra context regarding how many actual unit bounds were served this timestep.

* **Info Model**: Contains deterministic metrics like `fulfillment_rate` and `stockout_rate` used for grading.

## Setup and Usage Instructions

### Local Execution (Python)
1. Install requirements:
   ```bash
   pip install pydantic numpy gym stable-baselines3 openai gradio shimmy
   ```
2. Validate compliance:
   ```bash
   openenv validate
   ```
3. Run Local UI Dashboard:
   ```bash
   python app.py
   ```
   Navigate to `http://localhost:7860` in your browser.

### Containerized Execution (Docker)
This repository is pre-configured for Docker and Hugging Face space deployments:
```bash
docker build -t openenv .
docker run -p 7860:7860 openenv
```

## Baseline Performance Scores

A baseline `inference.py` is included utilizing the strict OpenAI API (`gpt-4o-mini`) via API key prompt-structuring.
* Easy Task Score: ~0.94
* Medium Task Score: ~0.83
* Hard Task Score: ~0.65 

(*Generated scores based on 0.0 - 1.0 grader API bounds dynamically populated in `env.grade()`*)
