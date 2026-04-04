def run_demo(mode, n_warehouses, n_retailers):
    """Run one episode and return detailed metrics log."""
    env = SupplyChainEnv(
        n_warehouses=int(n_warehouses),
        n_retailers=int(n_retailers)
    )

    # Load model fresh each time
    try:
        agent = PPO.load("supply_chain_agent", env=env)
    except Exception as e:
        return f"Error loading model: {e}"

    obs, _ = env.reset()
    total_reward = 0
    log = []

    log.append("=" * 45)
    log.append(f"  🏭 Supply Chain Simulation")
    log.append(f"  Mode: {mode}")
    log.append(f"  Warehouses: {int(n_warehouses)} | "
               f"Retailers: {int(n_retailers)}")
    log.append("=" * 45)

    for step in range(100):
        if mode == "🤖 Trained Agent":
            action, _ = agent.predict(obs)
        else:
            action = env.action_space.sample()

        obs, reward, done, _, info = env.step(action)
        total_reward += reward

        if (step + 1) % 10 == 0:
            log.append(
                f"Step {step+1:03d} | "
                f"Reward: {reward:+.3f} | "
                f"Fulfilled: "
                f"{info['fulfillment_rate']*100:.1f}% | "
                f"Efficiency: {info['efficiency_score']:.1f}"
            )
        if done:
            break

    summary = env.get_episode_summary()
    log.append("\n" + "=" * 45)
    log.append("  📊 EPISODE SUMMARY")
    log.append("=" * 45)
    log.append(f"  ✅ Fulfillment Rate : "
               f"{summary.get('fulfillment_rate_%', 0):.1f}%")
    log.append(f"  ❌ Stockout Rate    : "
               f"{summary.get('stockout_rate_%', 0):.1f}%")
    log.append(f"  📦 Total Demand     : "
               f"{summary.get('total_demand', 0):.1f}")
    log.append(f"  ✔️  Total Fulfilled  : "
               f"{summary.get('total_fulfilled', 0):.1f}")
    log.append(f"  🏆 Avg Reward       : "
               f"{summary.get('avg_reward', 0):.3f}")
    log.append(f"  ⏱️  Total Steps      : "
               f"{summary.get('total_steps', 0)}")
    log.append("=" * 45)

    return "\n".join(log)