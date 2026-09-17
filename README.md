# Grounded Support Agent

*A self-verifying RAG system with a measured abstention policy*

A multi-agent customer-support pipeline built around one constraint: **every autonomous answer must be verified against a retrievable source, or the system hands off to a human instead of guessing.** The interesting question this project answers isn't "how accurate is it?" — it's **"what percentage of traffic do you have to give up to guarantee every autonomous answer is grounded and auditable?"**

> **Origin-story note, for transparency:** the verify-or-abstain architecture (Critic + retry + Escalation) was the original Phase 1 design, decided before any evaluation or fine-tuning work began. The "this project measures the cost of that guarantee" framing came later, as a synthesis reached after the eval and fine-tuning work — presented here as insight gained along the way, not the original day-one hypothesis.

---

## Architecture

```
User Query → Router → Retriever → Generator → Critic → Response
                                       ↑                    |
                                       |___ retry (max 2) ___|
                                                              |
                                                        Escalation (human handoff)
```

| Component | Role | Notes |
|---|---|---|
| **Router** | Classifies intent | `contact_human_agent` / `contact_customer_service` intents bypass the bot entirely and go straight to Escalation — a deliberate product decision, not a failure path |
| **Retriever** | FAISS + `sentence-transformers/all-MiniLM-L6-v2`, row-based chunking (~26.9K chunks, top-k=4) | Deterministic — same query always returns the same chunks |
| **Generator** | Produces the answer from query + retrieved context | Qwen2.5-7B-Instruct, 4-bit quantized |
| **Critic** | Checks groundedness: is the answer supported by the *retrieved* context, and does it actually address the question? Outputs structured JSON | Parse failure defaults to `grounded=false` |
| **Escalation** | Deterministic handoff to a human, no LLM call | Triggered by retry-exhaustion or router bypass |

**Why abstention instead of always answering:** in customer support, a confidently wrong answer (a bad refund-policy claim, a wrong cancellation step) is more expensive than a deferred one — it creates a support ticket *and* a trust problem. The Critic + retry + escalation loop exists to make that trade-off explicit and measurable, not to hide it.

**Stack:** FAISS · sentence-transformers · Bitext Customer Support LLM dataset (~27K rows) · Qwen2.5-7B-Instruct (local, 4-bit) · QLoRA/PEFT · LangGraph · built via Google Antigravity, run on free-tier Colab/Kaggle T4 GPUs.

---

## Two findings this project is actually about

### Finding 1 — Two metrics moved that architecturally *couldn't* move, and that was the real discovery

Comparing a baseline eval against a fine-tuned model, `context_precision` and `context_recall` both improved by ~9–10 points. That's impossible — the retriever (FAISS index, embeddings) is deterministic and untouched by a Generator-only fine-tune; those numbers cannot move from that intervention.

**Root cause:** escalated queries were being dropped from the metric averages, and the escalation rate differed between the "before" and "after" runs (11.6% vs 22.8%). The two averages were being computed over different, non-comparable subsets of queries — the hardest queries were leaving the harder run's denominator, mechanically inflating its precision/recall. This is an architectural argument, not just a suspicion: retrieval is deterministic and untouched by a Generator-only fine-tune, so precision/recall *cannot* move from this intervention — full stop. A later, separate fine-tuning run (routing bug fixed, discussed below) showed exactly this: precision/recall flat, as the architecture predicts, when the same survivorship-bias confound wasn't present.

This also raised a real suspicion about the headline completeness gain (+0.43) reported at the time — that it may have been partly the same artifact rather than a clean win. That's flagged as a suspicion, not confirmed for that specific run; see the fine-tuning section below for what the one clean, routing-fixed measurement actually shows.

### Finding 2 — The fine-tune regressed groundedness, and the cause was a mismatch between what I trained it on and what graded it

Fine-tuning targets were `(query + context → gold reference answer)` pairs from the dataset. The Critic grades the model's actual output against the *retrieved chunk* at inference time — not against the gold answer. Those two aren't the same text. When they diverge, the model produces something that *sounds* like an ideal answer instead of sticking to what was actually retrieved, and the Critic correctly flags it as ungrounded.

Plain version: **RAG controls what the model can see. Fine-tuning controls what it does with what it sees.** The eval showed retrieval was working (context precision 0.76, recall 0.758) — the model just wasn't committing to the specifics it had access to. That's a behavior problem, and I tried to fix a behavior problem with weight updates trained on the wrong target.

---

## Getting the evaluation trustworthy (before trusting any of the above)

Three separate measurement bugs, each initially hidden behind numbers that looked fine or good on the surface:

- **Chunking fragmentation.** Character-based chunking (500 chars/50 overlap) split Q&A pairs mid-sentence. Switched to row-based chunking (1 pair = 1 chunk). Faithfulness moved 0.687 → 0.754.
- **Eval-set leakage.** A 100-sample eval came back with grounded_rate = 1.0 — suspiciously perfect. Eval queries were sitting inside the FAISS index, so the retriever was returning each query's own source row verbatim. Fixed by excluding eval rows from the index before building it. Corrected grounded_rate: 0.86. *Perfect metrics are the biggest red flag in an eval, not the best result.*
- **Train/test leakage in the fine-tuning data.** The same 250 held-out eval rows were initially also being drawn from for training examples. Fixed by splitting into a training-only pool and a fine-tuning-blind test pool.

---

## The fine-tuning experiment: collapse, diagnosis, and a real recovery

**Why QLoRA at all:** the diagnosis (Finding 2, in plain terms) pointed at behavior, not knowledge — a weight-update problem, not a retrieval problem. QLoRA was the compute-feasible way to test that hypothesis on a free-tier T4 (full fine-tuning of a 7B model needs far more VRAM than is available; QLoRA fits in ~5–6GB by quantizing the base model to 4-bit and training only small adapter matrices on top).

**Training data (~1,350 examples):** a 5%/95% split by fixed seed, stratified across intent categories, so the training sample proportionally represents every category rather than over-weighting failure cases. Training context retrieved through the actual retriever, so training-time context matches inference-time context.

**First eval: 100% escalation — zero grounded answers.** Unrelated quality metrics (correctness, tone) stayed flat or even ticked up. That contradiction — how can "correctness" look fine if nothing was ever accepted? — was the signal that something had broken structurally, not gradually degraded.

Two separate causes, both confirmed directly rather than inferred:

1. **Gold-answer template contamination.** The Bitext dataset's gold answers contain unresolved `{{placeholder}}` tokens (e.g. `{{Order Number}}`) as a known property of the raw data. The model learned to reproduce this template syntax *verbatim* instead of substituting real values — confirmed by reading raw generator output directly, not by inference.
2. **One adapter, reused across every pipeline role.** The same QLoRA adapter, trained only on Generator-style free-text answers, was being reused for the Router, Critic, and eval Judge — all of which require structured JSON output the adapter was never trained to produce. Confirmed with a controlled test: toggling the adapter off for the Critic's calls only restored valid JSON and correct grading (9/10 on a spot check); with the adapter on, the Critic produced plain prose with no JSON structure, so every call failed to parse and silently defaulted to `grounded=false` — enough on its own to explain 100% escalation, independent of generator quality. The eval Judge, running under the same corrupted adapter, gave a correctness score of 4/5 to a literal escalation boilerplate message — proof the Judge itself was hallucinating, which voids that first run's quality metrics entirely.

**Fix:** per-role adapter routing — `enable_adapter_layers()` / `disable_adapter_layers()` toggled so only the Generator runs through the QLoRA weights; Router, Critic, and Judge run on the base model. Zero extra VRAM, same model, one switch.

**Re-measured the same adapter, routing fixed, same 250 held-out queries:**

| Metric | Pre-fine-tune baseline | Post-fix (routing corrected) | Δ vs. baseline |
|---|---|---|---|
| answer_relevancy | 0.953 | 0.891 | −0.062 |
| context_precision | 0.76 | 0.765 | +0.005 (flat, as expected — retrieval is untouched by a Generator-only change) |
| context_recall | 0.758 | 0.769 | +0.011 (flat) |
| correctness | 3.82 | 3.64 | −0.18 |
| completeness | 3.45 | 3.28 | −0.17 |
| tone | 3.92 | 3.9 | −0.02 |
| **grounded_rate** | **0.884** | **0.648** | **−0.236** |
| **escalation_rate** | **0.116** | **0.352** | **+0.236** |

**Two things are both true here:** fixing the routing bug recovered the pipeline from total collapse (escalation 100% → 35.2%) — a massive, real recovery. But it's still a genuine regression against doing nothing at all (baseline escalation was 11.6%). The fine-tune made groundedness worse, not better, even once every measurement bug was removed. The Critic caught it correctly both times — degraded answers were routed to human handoff rather than served.

**Lesson:** you cannot fine-tune one role in a multi-agent pipeline and assume it's safe to reuse the resulting weights everywhere else in the pipeline. Each role has a different output contract, and a single adapter optimized for one of them can silently break the others.

**A note on an earlier, smaller run:** an initial ~370-example fine-tuning attempt was also run and reported (grounded_rate 0.884→0.772, escalation 0.116→0.228) before the routing bug was discovered. That run's Critic and Judge were *also* running through its adapter, so those numbers carry the same contamination risk as the impossible deltas below — they were never independently re-verified with routing fixed. Treated here as unverified, not as a second clean data point.

---

## Escalation audit — what the regression actually looked like

*Scope note: this audit is drawn from the earlier ~370-sample run's escalated cases (the run whose aggregate numbers were never independently re-verified with routing fixed — see the fine-tuning section above). The content analysis below is a read of actual generated text, independent of that run's aggregate-metric reliability, but it is not from the 1,350-sample run discussed above it.*

Auditing the true Critic-triggered escalations (excluding by-design human-handoff bypasses) against the hypothesis "maybe the model just got more honest about refusing hard queries":

- **Rejected.** Reference answers show the escalated queries were answerable from the retrieved context. This isn't honest refusal.
- 68% of Critic rejections were flagged as the answer "not addressing the question" — not fabrication.
- The dominant failure pattern was generic, templated customer-service hedging (apologizing, asking a clarifying question) instead of using the specific procedural details already sitting in the retrieved context.

**Representative case:**
> Query: *"need help with canceling order {{Order Number}}"*
> Reference answer: a specific 5-step cancellation walkthrough
> Generated answer (flagged ungrounded): acknowledges the request, apologizes, never states the steps

Verdict: the fine-tuned model got less direct on action-oriented queries — failing to *commit* to available specifics, defaulting to safe-sounding filler instead of either the right answer or a genuine refusal.

---

## Hypotheses tested (and mostly rejected)

| Hypothesis | Test | Verdict |
|---|---|---|
| Escalation increase = model becoming more honest | Audit of true escalations vs. reference answers | **Rejected** — hedging, not honest refusal |
| More balanced (stratified) training data would reduce escalation | Second QLoRA run, ~1,350 examples | **Rejected, badly** — escalation went to 100%, driven by a separate adapter-routing bug |
| Training-data category skew was the primary cause of escalation | Composition analysis + the stratified re-run | **Partially rejected** — skew contributed, but the training-target/grading-target mismatch and the adapter-routing bug were more fundamental |
| Context precision/recall genuinely improved from QLoRA | Architectural analysis — retriever is deterministic and untouched by a Generator-only change | **Rejected** — confirmed as survivorship bias |
| The completeness gain (+0.43) was entirely real | The +0.43 figure is from the earlier 370-sample run, never independently re-verified with routing fixed | **Unconfirmed for that specific run — but the one clean, routing-fixed measurement available (a different, later run) shows completeness dropping, not rising (3.45→3.28), which is at minimum inconsistent with the reported gain** |

---

## What I'd do differently

1. **Try prompt engineering before QLoRA.** A structured format constraint ("your first sentence must state a specific action step") might have gotten partial improvement for free, without touching weights. This project jumped to fine-tuning without that baseline.
2. **Align training targets with retrieved chunks, not corpus gold answers**, so what the model is trained to produce matches what the Critic actually grades against.
3. **Isolate the QLoRA adapter to the Generator from day one** — never let a role-specific adapter run through roles it wasn't trained for.

## Known open items (documented, not swept away)

- Escalation rate as reported still mixes by-design bypass escalations with genuine Critic-triggered ones — not yet separated.
- A logging bug mislabels a small number of retry-exhausted rows as escalations that never reached the escalation node — inflates the reported escalation rate by roughly a percentage point; not yet fixed.
- One escalation-audit report's stated case count doesn't reconcile with the escalation rate on the same sample — flagged, not resolved.
- A proposed test (counting hedging-style n-grams in the training targets vs. pre-fine-tune generations, to check whether the corpus's own answer style taught part of the hedging behavior) was designed but not run.
- Deployment (FastAPI + Gradio, cost/latency logging) is scoped but not built.

---

## Quick reference

**Did the fine-tuning help?**
No — it regressed groundedness. The value wasn't in the result, it was in the diagnosis: a training-target/grading-target mismatch, a dataset-hygiene issue in the gold answers themselves, and a pipeline bug where one adapter was silently running roles it was never trained for. All three are specific, traceable, and fixable.

**What was the most important bug you found?**
The eval-set leakage bug. Eval queries were indexed in FAISS, so the retriever found the exact query as its own top match — perfect retrieval, meaningless evaluation, grounded_rate showing a false 1.0. Every downstream conclusion would have been wrong if this hadn't been caught first.
