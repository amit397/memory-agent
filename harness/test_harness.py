"""
Phase 0 self-check (stdlib asserts, no framework).  Run: python harness/test_harness.py

Proves the four things Phase 0 must guarantee:
  1. determinism      -- same seed reproduces an identical run
  2. replay           -- recorded proposals replay byte-identically (the real-LLM mechanism)
  3. gate discriminates -- a small budget binds (PASS), a huge budget does not (FAIL)
  4. positive control -- a driver with real signal beats a null driver, so the
                         apparatus can detect a true effect before we trust any number
"""

import statistics
from loop import (run_episode, binding_gate, LoopConfig,
                  SyntheticDriver, ReplayDriver, MockBackend, LogEverything)


def _run(driver, seed, turns=30, budget=8000):
    return run_episode(driver, MockBackend(seed=seed), LogEverything(), LoopConfig(turns, budget, seed))


def test_determinism():
    a = _run(SyntheticDriver(mode="gradient"), 1)
    b = _run(SyntheticDriver(mode="gradient"), 1)
    assert a["turns"] == b["turns"] and a["best_bpb"] == b["best_bpb"]


def test_replay():
    a = _run(SyntheticDriver(mode="gradient"), 2)
    b = run_episode(ReplayDriver(a["proposals"]), MockBackend(seed=2), LogEverything(), LoopConfig(30, 8000, 2))
    assert a["turns"] == b["turns"], "replay from recorded proposals must match the original run"


def test_gate_passes_when_binding():
    rep = binding_gate()                       # defaults: budget 8000
    assert rep["passed"], rep


def test_gate_fails_when_not_binding():
    rep = binding_gate(config=LoopConfig(budget=1_000_000))
    assert not rep["passed"], "a huge budget must not bind"


def test_positive_control():
    grad = [_run(SyntheticDriver(mode="gradient"), s) for s in range(5)]
    rand = [_run(SyntheticDriver(mode="random"), s) for s in range(5)]
    g_bpb = statistics.mean(r["best_bpb"] for r in grad)
    r_bpb = statistics.mean(r["best_bpb"] for r in rand)
    g_acc = statistics.mean(r["accept_rate"] for r in grad)
    r_acc = statistics.mean(r["accept_rate"] for r in rand)
    print(f"  positive control: gradient bpb={g_bpb:.4f} acc={g_acc:.2f} | "
          f"random bpb={r_bpb:.4f} acc={r_acc:.2f}")
    assert g_bpb < r_bpb - 0.03, (g_bpb, r_bpb)   # real signal lowers the score
    assert g_acc > r_acc + 0.10, (g_acc, r_acc)   # and raises acceptance


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok:", name)
    print("all checks passed")
