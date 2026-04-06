# 🚀 Supply Chain RL – AI-Powered Inventory Optimization

<p align="center">
  <img src="https://img.shields.io/badge/AI-Reinforcement%20Learning-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Algorithm-A2C-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Environment-OpenEnv%20%7C%20Gymnasium-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Hackathon-Meta%20OpenEnv-purple?style=for-the-badge" />
</p>

<p align="center">
  <b>AI-driven supply chain optimization under real-world uncertainty</b><br>
  <i>+90.7 reward improvement across dynamic configurations</i>
</p>

---

## 🖥️ Dashboard Preview

<p align="center">
  <img src="assets/dashboard_preview.png" alt="Dashboard Preview" width="90%" />
</p>

---

## 📈 Learning Curve

<p align="center">
  <img src="assets/learning_curve.png" width="90%" />
</p>

---

## 🧠 Overview

This project demonstrates how **Reinforcement Learning (RL)** can optimize supply chain decisions by dynamically allocating inventory across warehouses and retailers under uncertain conditions.

Unlike static approaches, the system is trained across multiple configurations and disruptions, allowing it to **adapt and generalize** rather than memorize fixed strategies.

---

## 🎯 Problem Statement

In modern logistics systems:

* Too little stock → stockouts → lost revenue
* Too much stock → excess inventory → higher cost
* Demand & supply are unpredictable

👉 The objective is to **learn optimal inventory allocation under uncertainty**.

---

## 🤖 Solution

We model the system as a reinforcement learning environment:

* **Agent** → decides stock distribution
* **Environment** → simulates supply chain dynamics
* **Reward** → based on fulfillment efficiency

### Algorithm Used:

**Advantage Actor-Critic (A2C)**

---

## ✨ Key Features

* 📦 Multi-warehouse & multi-retailer simulation
* ⚡ Real-world disruptions (demand spikes, supplier delays)
* 📊 Interactive dashboard visualization
* ⚖️ Fair comparison (Trained vs Random agent)
* 📈 Performance tracking (reward, fulfillment, stockouts)

---

## 📊 Results

<p align="center">
  <img src="assets/results_chart.png" width="80%" />
</p>

| Metric           | Trained Agent | Random Agent |
| ---------------- | ------------- | ------------ |
| Fulfillment Rate | 97.5%         | 71.9%        |
| Stockout Rate    | 2.5%          | 28.1%        |
| Avg Reward       | 0.950         | 0.513        |
| Total Reward     | 94.99         | 51.29        |

### 🔥 Key Insight

> **+90.7 reward improvement from initial training to final policy**

---

## 🎮 Dashboard Features

### 🔹 Scenario Composer

* Adjust warehouses & retailers
* Run controlled simulations

### 🔹 Evaluation Mode

* 5 synchronized rollouts
* Same disruptions for fair comparison

### 🔹 Visualization

* 📉 Reward trajectory
* 📈 Service level curves
* 📊 Policy comparison charts

---

## 🧪 How It Works

```
State → Action → Environment → Reward → Learning → Improved Policy
```

1. Observe system state (inventory, demand, timestep)
2. Allocate inventory across warehouses
3. Simulate demand and disruptions
4. Compute reward based on fulfillment
5. Update policy over time

---

## 🏗️ Project Structure

```
supply-chain-rl/
│── supply_chain_env.py     # RL environment
│── train.py                # A2C training pipeline
│── inference.py            # Evaluation script
│── app.py                  # Dashboard UI
│── models/                 # Saved trained agents
│── assets/                 # Images & visuals
```

---

## ▶️ Getting Started

### Clone repository

```bash
git clone https://github.com/Shshank25/Supply_Chain_Management.git
cd supply-chain-rl
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python train.py
```

### Run inference

```bash
python inference.py
```

### Launch dashboard

```bash
python app.py
```

---

## 🌍 Applications

* 🛒 E-commerce logistics
* 🏭 Inventory planning
* 🚚 Supply chain optimization
* 📦 Warehouse management

---

## 💡 Why RL?

| Traditional Methods       | RL Approach                 |
| ------------------------- | --------------------------- |
| Static rules              | Adaptive learning           |
| Cannot handle uncertainty | Handles dynamic disruptions |
| Manual tuning             | Self-improving              |

---

## 🚀 Future Scope

* Cost-aware optimization (holding + transport costs)
* Multi-agent coordination
* Real-world dataset integration
* Demand forecasting integration

---

## ⭐ Show Your Support

If you found this project useful, consider giving it a ⭐ on GitHub.

---

## 📌 Tagline

> *“Smarter supply chains powered by adaptive intelligence.”*
