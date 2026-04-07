"""
Automated Grader for Supply Chain RL Environment
=================================================
Tests all aspects of the OpenEnv compliant environment for the
Meta PyTorch OpenEnv Hackathon submission.
"""

from supply_chain_env import SupplyChainEnv, Observation, Action, Reward, Info
import numpy as np

print("=" * 55)
print("  Supply Chain RL Environment — Automated Grader")
print("=" * 55)

passed = 0
failed = 0
env = SupplyChainEnv(task="medium")

# ─────────────────────────────────────────
# TEST 1 — reset() works and returns Pydantic Model
# ─────────────────────────────────────────
try:
    obs = env.reset()
    assert obs is not None, "obs is None"
    assert isinstance(obs, Observation), f"obs must be Pydantic Observation, got {type(obs)}"
    assert hasattr(obs, 'inventory'), "Observation missing inventory field"
    print("✅ Test 1 Passed : reset() returns valid Pydantic Observation")
    passed += 1
except Exception as e:
    print(f"❌ Test 1 Failed : reset() — {e}")
    failed += 1

# ─────────────────────────────────────────
# TEST 2 — state() method exists and works
# ─────────────────────────────────────────
try:
    state_obs = env.state()
    assert isinstance(state_obs, Observation), "state() must return Observation"
    print("✅ Test 2 Passed : state() returns current state observation")
    passed += 1
except Exception as e:
    print(f"❌ Test 2 Failed : state() — {e}")
    failed += 1

# ─────────────────────────────────────────
# TEST 3 — step() accepts Action and returns 4 values with Pydantic Models
# ─────────────────────────────────────────
try:
    action = Action(restock_quantities=[2.0, 2.0, 2.0])
    obs, reward, done, info = env.step(action)
    assert isinstance(obs, Observation), "returned obs must be Observation model"
    assert isinstance(reward, Reward), "returned reward must be Reward model"
    assert isinstance(done, bool), "done must be bool"
    assert isinstance(info, Info), "info must be Info model"
    print("✅ Test 3 Passed : step() returns (Observation, Reward, bool, Info)")
    passed += 1
except Exception as e:
    print(f"❌ Test 3 Failed : step() — {e}")
    failed += 1

# ─────────────────────────────────────────
# TEST 4 — Programmatic grade() check
# ─────────────────────────────────────────
try:
    grade = env.grade()
    assert isinstance(grade, float), "grade() must return float"
    assert 0.0 <= grade <= 1.0, "grade() must be between 0.0 and 1.0"
    print("✅ Test 4 Passed : grade() returns valid score")
    passed += 1
except Exception as e:
    print(f"❌ Test 4 Failed : grade() — {e}")
    failed += 1

# ─────────────────────────────────────────
# TEST 5 — Multi-task configurations
# ─────────────────────────────────────────
try:
    env_e = SupplyChainEnv(task="easy")
    env_h = SupplyChainEnv(task="hard")
    assert env_e.task == "easy"
    assert env_h.task == "hard"
    print("✅ Test 5 Passed : Tasks (easy, hard) load properly")
    passed += 1
except Exception as e:
    print(f"❌ Test 5 Failed : tasks — {e}")
    failed += 1

# ─────────────────────────────────────────
# FINAL SCORE
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print(f"  RESULTS: {passed}/5 tests passed")
if failed == 0:
    print("  🎉 ALL TESTS PASSED — OpenEnv Spec Compliant!")
else:
    print(f"  ⚠️  {failed} test(s) failed — fix before submitting")
print("=" * 55)