from supply_chain_env import SupplyChainEnv

env = SupplyChainEnv()
obs, _ = env.reset()
print("Initial observation:", obs)

for step in range(3):
    action = env.action_space.sample()
    obs, reward, done, truncated, _ = env.step(action)
    print(f"Step {step + 1}: reward={reward:.2f}, done={done}, truncated={truncated}")
