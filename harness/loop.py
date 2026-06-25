"""
Phase 0 -- autoresearch memory harness: instrumentation + binding-regime gate.

Smallest programmatic version of the autoresearch loop needed to:
  - run experiments under a swappable CompactionPolicy (only B3 / log-everything here),
  - log full per-turn memory state to JSONL, deterministically replayable,
  - measure whether a small context budget actually BINDS (the Phase-0 gate, README sec 3).

Mock-only by design (runs on any machine, no GPU): a synthetic driver proposes
hyperparameter changes and a mock backend scores them through a KNOWN response
surface, so proposal quality genuinely moves the score. That doubles as the
positive control (test_harness.py) proving the apparatus can detect a real effect.

Real LLM driver and real (down-tuned, SDPA) train.py backend are stubbed for Phase 1.
References: README.md sec 3 (binding regime), sec 6 (B3 baseline), sec 10 (Phase 0).
"""

import json
import os
import random
import statistics
from dataclasses import dataclass, asdict


# --- token counting (pluggable) --------------------------------------------------
# ponytail: chars/4 proxy, no tokenizer dependency so the harness runs anywhere.
# Swap for the real driver model's tokenizer when the real LLM is wired (Phase 1).
def count_tokens(text):
    return max(1, len(text) // 4)


# --- experiment space ------------------------------------------------------------
# Normalized [0,1] stand-ins for real train.py knobs. The point is a known surface
# with a single optimum, not fidelity to the real hyperparameters (Phase 1).
PARAMS = ("depth", "matrix_lr", "embedding_lr", "weight_decay", "batch")
BASELINE = {k: 0.2 for k in PARAMS}
OPTIMUM = {k: 0.7 for k in PARAMS}


@dataclass
class Proposal:
    changes: dict     # param -> new normalized value
    reasoning: str    # prose; the bulk of the memory that overflows the budget


@dataclass
class EvalResult:
    val_bpb: float
    peak_vram_gb: float
    status: str       # "ok" | "crash"


@dataclass
class TurnRecord:
    """One past experiment, as it sits in the agent's memory."""
    turn: int
    changes: dict
    reasoning: str
    val_bpb: float
    status: str
    accepted: bool


def render_record(r):
    head = (f"[turn {r.turn}] changes={r.changes} -> val_bpb={r.val_bpb:.4f} "
            f"status={r.status} accepted={r.accepted}\n")
    return head + r.reasoning + "\n"


def serialize(records):
    return "".join(render_record(r) for r in records)


# --- compaction policy seam ------------------------------------------------------
@dataclass
class CompactionResult:
    kept: list
    dropped: list
    events: list
    retrievals: list
    post_tokens: int


class CompactionPolicy:
    """The seam Phase 1's B1/B2/Cx plug into. Phase 0 ships only B3."""
    name = "base"

    def compact(self, memory, budget, tok):
        raise NotImplementedError


class LogEverything(CompactionPolicy):
    """B3: keep everything; when it exceeds budget, drop oldest (naive recency truncation)."""
    name = "B3_log_everything"

    def compact(self, memory, budget, tok):
        if tok(serialize(memory)) <= budget:
            return CompactionResult(list(memory), [], [], [], tok(serialize(memory)))
        # ponytail: O(n^2) re-serialize per drop; fine at tens of turns, prefix-sum if it grows.
        kept, dropped = list(memory), []
        while kept and tok(serialize(kept)) > budget:
            dropped.append(kept.pop(0))
        events = [{"type": "recency_truncate", "dropped": len(dropped)}] if dropped else []
        return CompactionResult(kept, dropped, events, [], tok(serialize(kept)))


# --- training backend seam -------------------------------------------------------
class Backend:
    def evaluate(self, params, turn):
        raise NotImplementedError


@dataclass
class MockBackend(Backend):
    """Known response surface: val_bpb = base + sum w*(p-opt)^2 + tiny noise.
    Lower is better; proposals toward OPTIMUM genuinely improve the score -- this
    is the positive control. Params outside [0,1] model a VRAM-blowup crash."""
    seed: int = 0
    base: float = 1.0
    w: float = 0.4
    noise_std: float = 0.003

    def evaluate(self, params, turn):
        if any(not (0.0 <= v <= 1.0) for v in params.values()):
            return EvalResult(float("inf"), 99.0, "crash")
        bowl = sum(self.w * (params[k] - OPTIMUM[k]) ** 2 for k in PARAMS)
        noise = random.Random(self.seed * 1_000_003 + turn).gauss(0, self.noise_std)  # positional int seed -> deterministic, replay-safe
        vram = 4.0 + 6.0 * params["depth"] + 4.0 * params["batch"]
        return EvalResult(self.base + bowl + noise, vram, "ok")


class RealBackend(Backend):
    """Phase 1: edit train.py, run down-tuned (SDPA, not FA3) on the 8GB GPU, parse val_bpb."""
    def evaluate(self, params, turn):
        raise NotImplementedError("Real down-tuned training backend lands in Phase 1.")


# --- driver seam -----------------------------------------------------------------
class Driver:
    def propose(self, context, current, turn, rng):
        raise NotImplementedError


def _filler(target_tokens, turn):
    # ponytail: templated filler sized to a token target (chars/4); this is the memory
    # bloat the budget must contend with. Real LLM reasoning replaces it in Phase 1.
    unit = (f"Considering turn {turn}: adjusting hyperparameters to reduce val_bpb; "
            f"weighing model capacity against the fixed time budget and VRAM ceiling. ")
    s = ""
    while len(s) < target_tokens * 4:
        s += unit
    return s


@dataclass
class SyntheticDriver(Driver):
    mode: str = "gradient"        # "gradient" = real signal toward OPTIMUM; "random" = null
    reasoning_tokens: int = 600
    lr: float = 0.12              # steady improver: stays in the descending regime most of the run
    explore: float = 0.02         # so BOTH score and acceptance-rate channels separate vs the null driver

    def propose(self, context, current, turn, rng):
        changes = {}
        for k in PARAMS:
            if self.mode == "gradient":
                step = self.lr * (OPTIMUM[k] - current[k]) + rng.gauss(0, self.explore)
            else:
                step = rng.gauss(0, 0.1)
            changes[k] = current[k] + step
        return Proposal(changes, _filler(self.reasoning_tokens, turn))


@dataclass
class ReplayDriver(Driver):
    """Replays recorded proposals -- the mechanism the real (stochastic) LLM needs."""
    recorded: list

    def propose(self, context, current, turn, rng):
        assert turn < len(self.recorded), f"replay exhausted at turn {turn} (recorded {len(self.recorded)})"
        return self.recorded[turn]


class RealLLMDriver(Driver):
    """Phase 1: call the real driver model (provider+model TBD), temp 0, record outputs."""
    def propose(self, context, current, turn, rng):
        raise NotImplementedError("Real LLM driver lands in Phase 1 (needs provider+model+key).")


# --- the loop --------------------------------------------------------------------
@dataclass
class LoopConfig:
    num_turns: int = 30
    budget: int = 8000        # pre-registered small context budget (README sec 3); adjustable
    seed: int = 0


def run_episode(driver, backend, policy, config, log_path=None, tok=count_tokens):
    assert config.num_turns >= 1, "num_turns must be >= 1"
    rng = random.Random(config.seed)
    memory, current, best, turns = [], dict(BASELINE), None, []
    for t in range(config.num_turns):
        required = tok(serialize(memory))          # demand if we logged everything
        cr = policy.compact(memory, config.budget, tok)
        prop = driver.propose(serialize(cr.kept), current, t, rng)
        cand = {**current, **prop.changes}
        res = backend.evaluate(cand, t)
        accepted = res.status == "ok" and (best is None or res.val_bpb < best)
        if accepted:
            best, current = res.val_bpb, cand
        memory.append(TurnRecord(t, prop.changes, prop.reasoning, res.val_bpb, res.status, accepted))
        turns.append({
            "turn": t, "policy": policy.name, "budget": config.budget,
            "required_tokens": required, "post_tokens": cr.post_tokens,
            "overflow": required > config.budget,
            "margin": required / config.budget if config.budget else float("inf"),
            "compaction_events": cr.events, "dropped": len(cr.dropped), "retrievals": cr.retrievals,
            "proposal": {"changes": prop.changes, "reasoning_tokens": tok(prop.reasoning)},
            "result": {"val_bpb": res.val_bpb, "peak_vram_gb": res.peak_vram_gb, "status": res.status},
            "accepted": accepted, "best_bpb": best,
        })
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            for row in turns:
                f.write(json.dumps(row) + "\n")
    return {
        "seed": config.seed, "policy": policy.name, "budget": config.budget, "best_bpb": best,
        "accept_rate": sum(x["accepted"] for x in turns) / config.num_turns,
        "turns": turns,
        "proposals": [Proposal(r.changes, r.reasoning) for r in memory],  # for replay
    }


# --- Phase-0 binding-regime gate -------------------------------------------------
@dataclass
class GateThresholds:
    turn_overflow_frac: float = 0.5   # a run binds if >= this fraction of its turns overflow
    run_binding_frac: float = 0.8     # the budget binds if >= this fraction of runs bind
    median_margin: float = 1.5        # ...and the median peak (required/budget) is >= this


def binding_gate(num_runs=10, config=None, thresholds=None, base_seed=0):
    config = config or LoopConfig()
    thresholds = thresholds or GateThresholds()
    per_run = []
    for r in range(num_runs):
        c = LoopConfig(config.num_turns, config.budget, base_seed + r)
        log = run_episode(SyntheticDriver(mode="gradient"), MockBackend(seed=c.seed), LogEverything(), c)
        overflow_frac = sum(t["overflow"] for t in log["turns"]) / c.num_turns
        peak_margin = max(t["margin"] for t in log["turns"])
        per_run.append({"seed": c.seed, "overflow_frac": overflow_frac,
                        "peak_margin": peak_margin, "binds": overflow_frac >= thresholds.turn_overflow_frac})
    frac_runs_binding = sum(x["binds"] for x in per_run) / num_runs
    med_margin = statistics.median(x["peak_margin"] for x in per_run)
    passed = frac_runs_binding >= thresholds.run_binding_frac and med_margin >= thresholds.median_margin
    return {"passed": passed, "frac_runs_binding": frac_runs_binding, "median_peak_margin": med_margin,
            "thresholds": asdict(thresholds), "budget": config.budget, "per_run": per_run}


if __name__ == "__main__":
    rep = binding_gate()
    run_episode(SyntheticDriver(mode="gradient"), MockBackend(seed=0), LogEverything(),
                LoopConfig(), log_path=os.path.join(os.path.dirname(__file__), "logs", "example.jsonl"))
    th = rep["thresholds"]
    print("=== Phase 0 binding-regime gate ===")
    print(f"budget = {rep['budget']} tokens")
    print(f"thresholds: >={th['turn_overflow_frac']:.0%} turns overflow, "
          f"in >={th['run_binding_frac']:.0%} of runs, median peak margin >={th['median_margin']}x")
    print(f"runs binding:       {rep['frac_runs_binding']:.0%}")
    print(f"median peak margin: {rep['median_peak_margin']:.2f}x")
    print("RESULT:", "PASS -- scarcity is real, proceed" if rep["passed"] else "FAIL -- shrink budget")
    for r in rep["per_run"]:
        print(f"  seed {r['seed']}: {r['overflow_frac']:.0%} turns overflow, "
              f"peak {r['peak_margin']:.2f}x, binds={r['binds']}")
