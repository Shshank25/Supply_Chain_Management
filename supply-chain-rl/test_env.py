"""
Automated Grader for Supply Chain RL Environment
=================================================
Tests all aspects of the environment for the
Meta PyTorch OpenEnv Hackathon submission.
"""

from supply_chain_env import SupplyChainEnv
import numpy as np

print("=" * 55)
print("  Supply Chain RL Environment — Automated Grader")
print("=" * 55)

env = SupplyChainEnv()
passed = 0
failed = 0

# ─────────────────────────────────────────
# TEST 1 — reset() works
# ─────────────────────────────────────────
try:
    obs, info = env.reset()
    assert obs is not None,        "obs is None"
    assert isinstance(obs, np.ndarray), "obs must be numpy array"
    assert obs.shape == (9,),      f"wrong shape: {obs.shape}"
    print("✅ Test 1 Passed : reset() returns valid observation")
    passed += 1
except Exception as e:
    print(f"❌ Test 1 Failed : reset() — {e}")
    failed += 1

# ─────────────────────────────────────────
# TEST 2 — step() works
# ─────────────────────────────────────────
try:
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    assert obs is not None,             "obs is None"
    assert isinstance(reward, float),   "reward must be float"
    assert isinstance(done, bool),      "done must be bool"
    print("✅ Test 2 Passed : step() returns valid outputs")
    passed += 1
except Exception as e:
    print(f"❌ Test 2 Failed : step() — {e}")
    failed += 1

# ─────────────────────────────────────────
# TEST 3 — Reward is within valid range
# ─────────────────────────────────────────
try:
    obs, _ = env.reset()
    rewards = []
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        rewards.append(reward)
        if done:
            break
    assert not all(r == 0 for r in rewards), "reward is always 0"
    assert min(rewards) >= -1.1,  f"reward too low: {min(rewards)}"
    assert max(rewards) <= 1.1,   f"reward too high: {max(rewards)}"
    print(f"✅ Test 3 Passed : reward range [{min(rewards):.2f}, {max(rewards):.2f}]")
    passed += 1
except Exception as e:
    print(f"❌ Test 3 Failed : reward range — {e}")
    failed += 1

# ─────────────────────────────────────────
# TEST 4 — Episode ends at 100 steps
# ─────────────────────────────────────────
try:
    obs, _ = env.reset()
    steps = 0
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        steps += 1
        if done:
            break
    assert steps <= 100, f"episode went {steps} steps, expected 100"
    print(f"✅ Test 4 Passed : episode ends at {steps} steps")
    passed += 1
except Exception as e:
    print(f"❌ Test 4 Failed : episode length — {e}")
    failed += 1

# ─────────────────────────────────────────
# TEST 5 — Zero action (order nothing)
# ─────────────────────────────────────────
try:
    obs, _ = env.reset()
    action = np.zeros(3, dtype=np.float32)
    obs, reward, done, _, _ = env.step(action)
    assert isinstance(reward, float), "reward must be float"
    print(f"✅ Test 5 Passed : zero action reward = {reward:.3f}")
    passed += 1
except Exception as e:
    print(f"❌ Test 5 Failed : zero action — {e}")
    failed += 1

# ─────────────────────────────────────────
# TEST 6 — Max action (order maximum)
# ─────────────────────────────────────────
try:
    obs, _ = env.reset()
    action = np.ones(3, dtype=np.float32) * 10
    obs, reward, done, _, _ = env.step(action)
    assert isinstance(reward, float), "reward must be float"
    print(f"✅ Test 6 Passed : max action reward = {reward:.3f}")
    passed += 1
except Exception as e:
    print(f"❌ Test 6 Failed : max action — {e}")
    failed += 1

# ─────────────────────────────────────────
# TEST 7 — reset() fully resets state
# ─────────────────────────────────────────
try:
    obs1, _ = env.reset()
    for _ in range(20):
        env.step(env.action_space.sample())
    obs2, _ = env.reset()
    assert env.timestep == 0, "timestep not reset to 0"
    assert env.inventory is not None, "inventory is None after reset"
    print("✅ Test 7 Passed : reset() fully restores initial state")
    passed += 1
except Exception as e:
    print(f"❌ Test 7 Failed : reset state — {e}")
    failed += 1

# ─────────────────────────────────────────
# TEST 8 — Disruptions work
# ─────────────────────────────────────────
try:
    disruptions_seen = set()
    obs, _ = env.reset()
    for _ in range(200):
        result = env._inject_disruption()
        can_restock, multiplier = result
        if not can_restock:
            disruptions_seen.add("supplier_delay")
        if multiplier > 1:
            disruptions_seen.add("demand_spike")
        if len(disruptions_seen) == 2:
            break
    assert len(disruptions_seen) > 0, "no disruptions occurred"
    print(f"✅ Test 8 Passed : disruptions working {disruptions_seen}")
    passed += 1
except Exception as e:
    print(f"❌ Test 8 Failed : disruptions — {e}")
    failed += 1

# ─────────────────────────────────────────
# TEST 9 — Observation space is valid
# ─────────────────────────────────────────
try:
    obs, _ = env.reset()
    assert env.observation_space.contains(obs), \
        "obs not in observation_space"
    print("✅ Test 9 Passed : observation is within defined space")
    passed += 1
except Exception as e:
    print(f"❌ Test 9 Failed : observation space — {e}")
    failed += 1

# ─────────────────────────────────────────
# TEST 10 — Full episode runs completely
# ─────────────────────────────────────────
try:
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    while True:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break
    print(f"✅ Test 10 Passed: full episode — {steps} steps, "
          f"total reward = {total_reward:.2f}")
    passed += 1
except Exception as e:
    print(f"❌ Test 10 Failed: full episode — {e}")
    failed += 1

# ─────────────────────────────────────────
# FINAL SCORE
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print(f"  RESULTS: {passed}/10 tests passed")
if failed == 0:
    print("  🎉 ALL TESTS PASSED — Ready for submission!")
else:
    print(f"  ⚠️  {failed} test(s) failed — fix before submitting")
print("=" * 55)