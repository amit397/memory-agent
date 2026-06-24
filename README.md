# Memory Management Agent — Proving Necessity Before Building

**Status:** Planning — ready for implementation scoping
**Predecessor:** `context-window-research-agent` — tested memory *ordering/curation* on a self-improving autoresearch agent. Verified result: with history that **fit in context**, neutral strategies (baseline, recency, ranked-by-delta) tied, but aggressive curation (`quality_filter`) significantly *hurt* (acceptance rate halved, p ≤ 0.0004). It never reached the overflow regime; this project is the long-horizon/compression study the predecessor named as its next step.
**Reference paper:** Meta-Harness (Lee et al., 2026) — argues for exposing *full, uncompressed* history via a filesystem and retrieving on demand. We treat that as the lossless bar to approach, not a thesis to endorse.
**Last updated:** 2026-06-23

---

## 0. What This Project Is For

I want to build a product: a **memory-management agent** — a pluggable policy that decides what an LLM agent keeps verbatim, what it summarizes, what it evicts, and what it defers to retrieval, so the agent stays correct on long loops without paying to carry its entire history every turn.

I am not building it yet. I am proving it is *necessary* first. The discipline here is the one the predecessor established: a design that can only win, or that can't lose because the regime is trivial, is not an experiment. So this README is structured as two gates the product must clear before any build:

- **Part 1 — Does memory compression work at all?** On a controlled, *binding* regime (the self-improving autoresearch loop from the predecessor, run under a small fixed context budget so history genuinely overflows), does any compression policy beat a log-everything sliding window — improve the agent's delivered score at lower token cost?
- **Part 2 — Does it transfer to coding agents?** Does the Part 1 winner improve a real coding agent (SWE-bench), proving the effect holds for the LLM-with-harness systems the product targets?

If Part 1 produces a null, the product is unjustified and we write up why. If Part 1 wins but Part 2 doesn't transfer, we've learned the effect is domain-specific and the product scope narrows. Either outcome is a real result.

### Why the autoresearch loop is a fair test, not a stand-in

A coding agent is an LLM with a harness; so is the autoresearch agent. In both, the lever under test is identical: **memory content → the agent's decision quality.** In the autoresearch loop, compressed memory either preserves enough about past experiments for the agent to propose good ones (token-prediction score improves) or it doesn't. In a coding loop, compressed memory either preserves enough about errors and file state for the agent to fix the bug or it doesn't. Same mechanism, different readout. Part 1 is therefore a genuine instance of the phenomenon, not a proxy for it.

The seam that remains — and the reason Part 2 exists — is that *what counts as critical memory* differs by domain: experiment results and failures in autoresearch, versus error strings, file paths, and constraints in coding. A policy can win Part 1 by preserving the right thing for that domain and still need Part 2 to confirm it preserves the right thing for coding. One consequence to plan for: token-prediction score is a dense, continuous signal; SWE-bench pass/fail is sparse and binary, so an effect that is obvious in Part 1 can be statistically invisible in Part 2 from signal sensitivity alone. Part 2 must be powered with enough tasks to detect the Part 1 effect, or we will mistake "underpowered" for "didn't transfer."

---

## 1. The Question

On a long agentic loop, the cheapest way to be correct is to keep every observation and retrieve from it on demand — the Meta-Harness design. But full history does not stay cheap: as horizon grows, "store everything and retrieve adaptively" pushes cost into a growing store, repeated retrieval round-trips, and a context window that creeps toward the cap every turn. On a long loop with a cheap model, the binding constraint is **tokens-per-task and round-trips-per-task** — assuming model quality is held fixed, which Part 1 does by construction.

> **Central question.** Is there a form of *memory compression* that replicates the task performance of full-history adaptive retrieval at materially lower token cost — first on a controlled self-improving loop (Part 1), then on a coding agent (Part 2)?

This is framed *against* the paper's implicit thesis that compression is the wrong lever, not in agreement with it. The honest version of the hypothesis is not "compression beats retrieval" — it is "a good compression policy can sit on the success/cost Pareto frontier close to full retrieval, and retrieval is itself a *deferred* form of compression we can partially precompute."

---

## 2. What the Predecessor Found (and Why This Continues It)

The predecessor is not an empty null. Its results were verified against the repo (`report.md`, `results_claude/summary.md`, `cross_rung_summary.md`). Three facts drive this project's design:

**1. The regime was non-binding — confirmed in writing.** The predecessor's README: *"Every run was short enough that the full history fit in the context window."* At a 10-step horizon the recency arm "never truncates," so it tested *ordering*, not deletion. By construction it could not measure what compression does under scarcity. **This is the single flaw the new project must not reproduce — and reusing the same loop makes that the central risk (see §3).**

**2. Neutral strategies tied; aggressive curation hurt — significantly.** Across 10 seeds on a clean gemma rung (temp 0), baseline, recency-10, and ranked-by-delta were statistically indistinguishable. But `quality_filter` — showing the agent only its *successful* past experiments — delivered ~1/5 the improvement of the others (p ≤ 0.0004, Cohen's d ≈ −1.9 to −2.3). The mechanism was a **halved proposal acceptance rate** (17% vs ~52%): dropping the failures starved the agent of the contrast it needed to get edits accepted.

**3. The predecessor named this project as its next step:** *"a long-horizon / compression study — push the agent past the point where its history overflows the context window, and measure how hard the history can be compressed before delivered quality drops."*

The transferable lesson is a prior we design against: **lossy compression that discards negative/failure examples degrades a cheap agent — even though it looks like sensible compression.** It converges with Meta-Harness's own caution against compressing diagnostic signal. Compression here is also a *strictly stronger* lever than the ordering the predecessor tested: reordering changes the arrangement of information; compression changes which information exists at all. That power is why it can save real tokens and why it can silently destroy task success.

---

## 3. The Binding Regime (Part 1's Load-Bearing Step)

The predecessor's loop, rerun as-is, would reproduce the predecessor's flaw — history would again fit, and compression could not bite. So **Part 1's entire validity rests on forcing scarcity.** The lever is a **small fixed context budget**: cap the agent's working context well below what a full run's history needs, so the store overflows within a few iterations rather than thousands. This is cheaper than extending the horizon at full budget (which might never overflow before the self-improvement curve plateaus) and it makes the binding regime reproducible by config.

**Two ways to bind, and the artificiality caveat.** Scarcity can come from a *small absolute budget* (cap a large-window model) or from a *long horizon at the model's native window*. These are not physically equivalent. An artificial cap on a large-window model tests pure information loss; a genuine long-horizon run also exposes the model's own long-context degradation — position bias, lost-in-the-middle, attention dilution — which a policy must fight in production but which simply does not exist in a cramped window. A policy tuned only against eviction in a dense small window may do nothing against attention dilution at a true 100k+ horizon, so an 8k ranking need not predict a 100k ranking. Two guards: **(a)** where possible, set the budget to a base model whose *native* window genuinely equals it (a small model, as the predecessor used), so the cap is not artificial and the attention physics are real; and **(b)** treat scale-invariance as a gate, not an assumption — at least one Part 1 arm runs at native window with a genuinely long horizon, and a policy's ranking must hold as the horizon is swept from small-budget toward native-window-long-horizon (H5, promoted to a validity gate in Phase 3) before it earns a Part 2 run.

**Phase 0 gate (hard, non-negotiable).** Before any compression policy is compared, confirm the budget actually binds: a frozen configuration where the log-everything baseline overflows the budget on ≥ a pre-set fraction of turns, for ≥ a pre-set fraction of runs, with a median overflow margin (peak required tokens ÷ budget) above a pre-set threshold (e.g. ≥ 1.5×). If the budget does not bind, every downstream number is noise — exactly how the predecessor's main lever failed. **Because Part 1 is now the primary result, this gate is the most load-bearing step in the project, not a warm-up:** if the small budget does not produce real scarcity with a measurable effect, there is no primary result.

---

## 4. The Memory Model (Layered Store)

Two layers, because the cost lives in different places and they interact.

- **L1 — Working context (intra-episode).** The live message/tool-call history of a single rollout. The current shipping policy in the coding harness keeps the last N turns verbatim, LLM-summarizes older turns, triggers background summarization near a budget threshold, and emergency-truncates large tool results — this is baseline **B2** below.
- **L2 — Persistent memory (cross-episode).** Knowledge carried across turns and tasks: scored, embedded memory entries with confidence/trust, injected through a per-turn `search → verify → inject → maintain` pipeline.
- **The seam.** What gets *promoted* L1→L2 and *retrieved* L2→L1 is the policy surface we optimize. Compression is not one knob; it is a policy over *evict, summarize, embed-and-defer, or keep-verbatim*, applied at each layer and at the boundary.

In Part 1 the object being compressed is the autoresearch agent's record of past experiments and their outcomes. In Part 2 it is the coding agent's tool outputs, file reads, and error traces. The policies are the same; the content differs — which is the seam §0 names.

---

## 5. Hypotheses (falsifiable, pre-registered)

- **H1 (causal).** Holding model, tasks, and seeds fixed, changing only the compression policy changes delivered score. If false, compression is irrelevant — stop.
- **H2 (frontier).** There exists a compression policy that Pareto-dominates the log-everything sliding window: equal-or-higher score at strictly lower **all-in** tokens-per-task (see §7 on counting the compressor's own cost).
- **H3 (parity).** The best compression policy reaches within a margin Δ of full-history adaptive retrieval's score, at a fraction φ of its tokens-per-task. **Δ and φ are left general for now and pre-committed once the binding regime is characterized in Phase 0** — we will fix them before the comparison runs, not after.
- **H4 (recoverability).** Lossy compression with a *recoverable pointer* (evicted detail stays on disk and is retrievable) outperforms equally-aggressive lossy compression without recovery, isolating how much of compression's damage is reversible.
- **H5 (horizon scaling + scale-invariance gate).** Compression's token-cost advantage over full retrieval grows with loop length while the score gap stays flat or grows — *and* the ranking of policies is stable as the horizon is swept from small-budget toward native-window-long-horizon. The first clause is the product's scaling argument; the second is a validity gate (Phase 3): if rankings invert across the sweep, the small-budget Part 1 result does not generalize and must not be carried to Part 2.
- **H6 (transfer).** The Part 1 winner produces a detectable, same-sign improvement on the Part 2 coding benchmark. If it does not, the effect is domain-specific and the product scope narrows.

---

## 6. Baselines

The comparison set is the contribution. Each policy runs on identical instances and seeds.

| ID | Policy | Role |
|----|--------|------|
| **B1 — reference** | Adaptive full-history retrieval (Meta-Harness style): full logs on disk, retrieved on demand; nothing precompressed, nothing deleted | **Performant, lossless reference.** Its role is to be a strong full-information bar that is *comparable to the in-house memory methods (B2/B3)* on the same tasks and seeds — not to reproduce the paper's published number. All compression candidates are reported as a (score Δ, token Δ) relative to B1. B1's **retrieval utilization** (does the cheap model invoke the tool at all? recall@k of needed on-disk facts) is logged, so a non-retrieving B1 is visible rather than silently depressing the bar. |
| **B2** | Recency-keep + LLM summary (the coding harness's current default compaction) | The shipping incumbent a compression policy must beat to justify replacing it |
| **B3** | Log-everything sliding window / naive recency truncation, no summary | The cheap lossy floor and the *primary thing a method must beat in Part 1* |
| **B0** *(required where feasible)* | Full history in-context, no compression | Lossless ceiling **and anti-strawman anchor** — run wherever it fits, by default. If a compression policy ≈ B1 but both ≪ B0, B1 is failing to retrieve (not a real bar) and the apparent win is hollow. Infeasible in the deepest binding regime by design — that infeasibility is itself a result. |
| **Cx** | Candidate compression policies (§7) | The contribution; lossy, each measured as (score Δ, token Δ) vs B1 |

We deliberately keep the *magnitude* of the B1/B2/B3 comparison general until the regime is characterized; the research is not complete, and pre-committing exact margins before Phase 0 would be fabricated precision.

---

## 7. Compression Methods to Test

Ordered from structural/cheap to semantic/expensive. Each is a drop-in policy.

1. **Recency window** (B3) — keep last N, drop the rest. Control / floor.
2. **Structured/extractive compression** — keep tool calls and results in a compact schema (verb, target, status, key fields); drop conversational prose. Near-zero extra model cost.
3. **Observation compression with pointers** — truncate large outputs to head/tail plus a retrievable pointer to the full artifact on disk. The hybrid H4 predicts wins.
4. **Running hierarchical summary** — recursively summarize older turns into a rolling summary. Test granularity and refresh cadence.
5. **Salience-based eviction** — score each item by predicted future relevance (heuristic first: recency × reference-count × is-error × is-constraint; learned later) and evict the lowest.
6. **Anchored summaries (needle + failure preservation)** — summarize narrative but preserve *verbatim* the tokens that are unrecoverable if dropped: file paths, identifiers, error strings, explicit constraints, and **failed/rejected attempts**. Encodes two convergent warnings — Meta-Harness's (don't compress diagnostic signal) and the predecessor's (dropping failures halves acceptance). Predicted to beat naive summarization; tested directly. *Implementation note: distinguishing prose from a structural constraint does not need a full AST — a cheap lexer/regex pass (paths, identifiers, error patterns, diff hunks, code fences) plus a "tool outputs and code blocks are verbatim by default; only natural-language turns get summarized" rule covers most of it; over/under-inclusion is measured by the stratified needle probe (§10, Phase 1).*
7. **Retrieval-as-compression** (= B1, reframed) — full-history retrieval as the limit case where compression ratio is 1.0 but injection is on-demand. Lets us place retrieval on the same frontier plot.
8. **Hybrid: structural compress + on-demand recovery** — cheap structural compression every turn (#2/#3) backed by B1's filesystem so evicted detail is recoverable. The expected product configuration.

**Counting the compressor's own cost (required).** Methods #4 and #6 spend extra model calls to produce summaries; that I/O is real token cost. Every "tokens-per-task" figure must be **all-in**, including the summarizer's input and output. A semantic method that summarizes every turn can erase its own savings, and only #2 is genuinely near-free — so H2's "strictly lower tokens" claim is evaluated net of compression overhead, not on the size of the shrunken context alone.

---

## 8. Metrics

**Primary (pre-registered).**
- *Part 1:* the autoresearch loop's delivered improvement (token-prediction score, e.g. bits-per-byte) and proposal **acceptance rate** — the predecessor's two main signals.
- *Part 2:* task pass rate on the coding benchmark.

**Cost.** All-in tokens (input+output, including compressor) per task; peak context size; retrieval round-trips per task; USD; and **critical-path latency** — wall-clock added *on the blocking path* per turn, which separates synchronous compressors (#4/#5, paid before every turn) from background ones (B2 summarizes async, off the path). Reported as the **score-vs-tokens Pareto frontier**, with latency a secondary frontier axis at the product gate (Phase 4). *Latency is near-irrelevant to Part 1's offline loop and carries a placeholder threshold until the product gate — the primary detail here lives in the autoresearch score and token axes.*

**Diagnostic.**
- **Recoverability rate, stratified by needle type** — when a fact compressed out of context is later needed, does the agent recover it (via retrieval) or fail? Reported separately for *statistical-aggregate* facts (a trend across past results) and *exact-token* facts (a file path, identifier, or error string), because the autoresearch metric rewards the former while coding survival depends on the latter (§10, Phase 1). Exact-token retention is the cross-domain predictor.
- **B1 retrieval utilization** — fraction of needed on-disk facts the model actually retrieves, and whether it triggers the tool at all; guards against scoring against a B1 the cheap model cannot drive.
- **Compression-induced failure taxonomy** — classify every failure a policy causes that B1 does not: lost constraint, lost error detail, lost prior-state, hallucinated-from-summary, redundant re-derivation, and — carried from the predecessor — *acceptance collapse from dropped failures*. A deliverable in its own right.
- **Information density** — fraction of injected tokens referenced (attended/cited/acted-on) in the next k turns. Separates "compressed well" from "compressed lucky."

---

## 9. Evaluation Protocol — Emulating Meta-Harness

We adopt Meta-Harness's evaluation and selection protocol wholesale, because it is precisely *how they established their design was best*, and reusing it makes our comparison legible against theirs. Six commitments, each taken from the paper:

1. **Pareto-dominance over (score, cost), no scalar fixed in advance.** They "evaluate candidates under Pareto dominance and report the resulting frontier" rather than "committing to a single scalar objective in advance," then read the operating point off the curve. We do the same: every policy is a point on a score-vs-all-in-tokens frontier (§8), and "best" means Pareto-dominant, not top of one number. This is the formal version of H2/H3.
2. **Strict search-set / held-out test-set split.** Their proposer "never sees test-set results; its only feedback comes from the search set," and test sets are held out "until the final evaluation." We freeze a search set for tuning any policy knobs and a disjoint test set touched exactly once, at the end. No policy is selected on test data.
3. **Equal evaluation budget per method.** Because "evaluation is the main computational bottleneck," they "give each method the same budget of proposal harness evaluations." We compute-match: every baseline (B1/B2/B3) and candidate (Cx) gets the same runs/seeds, so a win is never bought with extra compute.
4. **Final test-set evaluation on the frontier.** They run for a fixed budget, then perform one final test-set evaluation on the Pareto frontier. We mirror it: select frontier policies on the search set, then report a single locked test-set number with 95% CIs.
5. **Domain-appropriate baselines, standard metric.** They compare against the state-of-the-art hand-designed strategies in each domain on that domain's standard metric. Ours are B1 (full retrieval), B2 (recency+summary incumbent), and B3 (log-everything floor), scored on the standard metric — token-prediction score in Part 1, SWE-bench pass rate in Part 2.
6. **Generalization via held-out models / OOD.** They confirm the discovered system on out-of-distribution datasets and across five held-out base models, not just the search distribution. Our Part 2 (coding transfer) *is* this step; where cheap, we add a held-out-model check in Part 1 so the winner isn't tuned to a single base model.

**One honest scope note.** Meta-Harness *searches code space* for the best harness; we compare a *frozen, pre-registered set* of memory policies, not searching for new ones. So what we emulate is their **evaluation and selection protocol** (1–6) — the part that establishes which design is best — not their proposer/search loop. The autoresearch loop in Part 1 is the system the policies ride on, not a search over the policies themselves.

---

## 10. Experimental Plan

### Phase 0 — Instrumentation + binding-regime gate (cheap; the fix for the prior project's core flaw)
Wrap the autoresearch loop so every run logs full per-turn memory state (pre/post-compaction messages, compaction events/stats, retrievals) to JSONL, deterministically replayable (fixed seed, temp 0). Then enforce §3: tune the **small fixed context budget** until the log-everything baseline overflows by the pre-set margin for the pre-set fraction of turns/runs. Do not proceed until the gate holds.

### Phase 1 — Part 1: the compression comparison (primary result, H1–H4)
On the binding autoresearch loop, run every baseline (B1/B2/B3, B0 where feasible) and every candidate (Cx) on identical seeds. Paired comparisons; report mean ± 95% CI and the Pareto frontier. This is where the **power analysis** lives and where the central claim — does any compression policy beat log-everything at lower all-in token cost — is settled. A controlled needle probe (plant a fact early that is provably required late, padded with realistic-but-irrelevant traffic) is run *within* this phase to isolate information loss; it is treated as a diagnostic for recoverability, **not** as the headline result, since by construction it favors anchoring methods (#6).

The probe is **stratified by needle type** — *statistical-aggregate* facts vs. *exact-token* facts (paths, identifiers, error strings) — and **exact-token retention is a hard gate for advancing a policy to Part 2**, scored separately from token-prediction improvement. This is the antidote to Part 1's central false-positive risk: a semantic-summarization policy can smooth the bits-per-byte curve while shredding the exact strings SWE-bench depends on, topping the Part 1 score yet being catastrophic on code. Such a policy is barred from Part 2 by the exact-token gate regardless of its score. Separately, **at least one Part 1 arm runs at the base model's native window with a genuinely long horizon**, so compression is exercised against long-context attention degradation, not only against eviction in a cramped window.

### Phase 2 — Part 2: transfer to a coding agent (external validity, H6)
Take the Part 1 winner(s) only and run them on a coding agent over a **frozen, long-horizon slice of SWE-bench Verified** (and/or TerminalBench-2), sized by the Phase-1 power analysis so the Part 1 effect is detectable on a sparse binary metric. Paired across policies on identical instances. This confirms — or refutes — that the effect holds for the systems the product targets.

### Phase 3 — Ablations + scale-invariance gate (mechanism, H5)
**Horizon sweep as a validity gate, not just an ablation:** sweep loop length from the small-budget regime toward native-window-long-horizon and confirm the Part 1 policy ranking is stable (does the cost advantage grow with length, and does the ranking hold? — H5). A policy whose ranking inverts across the sweep does not advance to Part 2 — this is how the small-budget result earns the right to generalize, and the direct answer to the "artificial window" critique. Also: component ablation of the winning hybrid (structural compression vs pointer recovery vs needle anchoring); phase-dependence (compress harder early vs late).

### Phase 4 — Decision gate (product trigger)
Build the memory-management agent **iff** a policy (a) Pareto-dominates B2 and beats B3 in Part 1, (b) meets the pre-committed (Δ, φ) parity bar against B1, (c) passes the exact-token-retention gate (Phase 1) and the scale-invariance gate (Phase 3), (d) transfers in Part 2 with no catastrophic failure class in the taxonomy, and (e) stays under the pre-registered **critical-path latency cap** *(placeholder — to be set by the product owner; e.g. a method that cuts tokens 40% but doubles per-turn blocking latency fails this gate even if it wins the token frontier)*. Otherwise, write up the null/failure-mode result and stop. The gate is defined *now*, not after seeing results.

---

## 11. Statistical Rigor

Non-negotiable, because the predecessor's weakness was inconclusiveness, not wrong direction.

- **Pre-registration.** Primary metrics, the Phase-0 gate thresholds, and the Phase-4 gate are fixed before runs; (Δ, φ) are fixed immediately after Phase 0 characterizes the regime, before the comparison.
- **Paired design.** Same instances and seeds across policies → paired tests (Wilcoxon / paired bootstrap), which detect smaller effects at lower N.
- **Variance always.** Every "A beats B" carries a 95% CI. A single run is an anecdote.
- **Power analysis** in Phase 1 sets the seed/task count for Phase 2.
- **Honest null.** If no policy beats B3/B2, that is the result, with a mechanism hypothesis.

---

## 12. Risks and Mitigations

- **The small budget doesn't actually bind.** The most likely silent failure, and now fatal because Part 1 is primary. → Phase 0 gate; reselect budget/tasks until overflow is confirmed.
- **Effect swamped by noise.** → Paired design + power analysis + more seeds on the cheap Part 1 loop before committing Part 2 spend.
- **Compressor cost eats the savings.** → All-in token accounting (§7); a method that can't beat B3 net of its own summarization overhead does not count.
- **Part 1 winner doesn't transfer.** → That is H6, a finding, not a bug; report the domain-specificity and narrow product scope.
- **Small-budget regime mis-simulates production (attention physics).** An artificial cap tests information loss but not the long-context degradation (lost-in-the-middle, attention dilution) of a real native-window horizon, so an 8k ranking may not hold at 100k. → Match the budget to a real small *native* window where possible; run ≥1 native-window long-horizon Part 1 arm; gate on scale-invariance (Phase 3) before any Part 2 spend.
- **Part 1 false positive on a code-catastrophic method.** The token-prediction metric can reward a summarizer that destroys exact paths/identifiers/errors — a Part 1 winner that breaks SWE-bench. → Stratified needle probe with **exact-token retention as a hard advancement gate** (Phase 1).
- **B1 reference is weak on a cheap model (strawman risk).** A cheap model may lack the metacognition to know when to query the filesystem, cratering B1 and handing compression a hollow win. → Keep B1 an *internal* comparable bar (vs B2/B3); **log B1 retrieval utilization**; and run **B0 as the anti-strawman anchor** wherever it fits — if compression ≈ B1 but both ≪ B0, the "win" is B1's retrieval failure, not compression quality.
- **Latency ignored (product axis).** Token parity ≠ time parity; synchronous compressors add blocking wall-clock a token frontier hides. → Critical-path latency is a Phase-4 gate with a placeholder cap, blocking methods penalized vs background. Irrelevant to Part 1's offline loop by design.
- **Policy leaks task-specific tuning.** → Freeze the policy across all tasks; no per-task knobs; report on held-out tasks.

---

## 13. If Both Gates Pass — Building the Product (brief)

The deliverable is a pluggable **compaction/memory policy** behind the coding harness's existing seams, not a fork. At a high level: refactor the hardcoded keep-recent-N + summarize logic into a swappable `CompactionPolicy` so B2, B3, and the winning hybrid are interchangeable by config; extend the L2 retrieval/eviction scoring with the salience signals from method #5; and back lossy compression with on-disk full logs exposed as a retrieval tool (the Meta-Harness "recover the detail you dropped" path — the single most important capability to port). Ship behind a flag, default off, until production telemetry reproduces the offline frontier. The detailed integration spec is deferred until Part 1 and Part 2 actually clear their gates — we are proving necessity before building.

---

## 14. What This Project Is Not

- Not a leaderboard chase. The goal is the success/cost *frontier*, not a single SOTA number.
- Not prompt engineering by vibes. Policies are frozen and compared under controlled, paired conditions.
- Not a re-test of memory *ordering* — the predecessor settled that (ordering null; aggressive curation negative) in a regime where everything fit. This tests memory *content under a binding cost budget*.
- Not an endorsement of the paper's thesis. It is a stress test of that thesis in the cheap, long-horizon, cost-bound regime the paper did not target.
- Not a build. It is the necessity proof that gates the build.

---

### Open decisions for the implementation agent
- The exact small-budget value and overflow-margin thresholds for the Phase-0 gate.
- The size of the Part 2 SWE-bench/TB2 slice (set by the Phase-1 power analysis).
- Concrete numbers for the pre-committed parity bar (Δ score points, φ token fraction), fixed after Phase 0.
- Whether salience scoring (method #5) starts heuristic or learned.
- Cold-start handling: policy behavior in the first few turns when there is little to compress.
