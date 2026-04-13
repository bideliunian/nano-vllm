# autorefine: Throughput Auto-Optimization Pipeline for nano-vllm

This is an autonomous experimentation loop that iteratively improves
nano-vllm's inference throughput — modelled directly on `autoresearch/`.

---

## Setup

To start a new experiment session, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr13`). The branch `autorefine/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autorefine/<tag>` from current main.
3. **Read the in-scope files** for full context:
   - `autorefine/evaluator.py` — fixed benchmark, defines the metric. **Do not modify.**
   - `autorefine/candidate.py` — the file you tune. Config parameters exposed here.
   - `nanovllm/config.py` — all valid Config fields and their constraints.
   - `nanovllm/engine/model_runner.py` — CUDA graph logic (lines 191, 228).
   - `nanovllm/engine/scheduler.py` — scheduling logic (lines 29–58).
4. **Verify a model exists**: confirm `$MODEL_PATH` or `~/huggingface/Qwen3-0.6B/` is present.
5. **Initialize results.tsv**: create it with just the header row.
6. **Confirm and go**.

---

## The Metric

**decode_throughput (tokens/sec) — HIGHER IS BETTER.**

Measured as `total_output_tokens / elapsed_seconds` across 256 concurrent
requests with fixed random prompts (seed 42, input 100–512 tokens, output
100–256 tokens). Each benchmark run takes roughly 1–3 minutes depending on
GPU and configuration.

The first run establishes the **baseline** — run the evaluator unchanged.

---

## Output Format

After a run completes, the log contains a summary block:

```
---
decode_throughput:    1234.56
elapsed_seconds:      87.32
peak_memory_gb:       18.3
total_output_tokens:  40576
num_seqs:             256
```

Extract the key metric with:

```bash
grep "^decode_throughput:" run.log
grep "^peak_memory_gb:" run.log
```

---

## Logging Results

Log every experiment to `results.tsv` (tab-separated, NOT comma-separated).
**Do not commit results.tsv to git** — leave it untracked.

Header and format (5 columns):

```
commit	decode_throughput	memory_gb	status	description
```

1. `commit` — short git hash (7 chars): `git rev-parse --short HEAD`
2. `decode_throughput` — value from log (e.g. `1234.56`); use `0.00` for crashes
3. `memory_gb` — value from log, rounded to `.1f` (e.g. `18.3`); use `0.0` for crashes
4. `status` — `keep`, `discard`, or `crash`
5. `description` — short text of what this experiment tried

Example:

```
commit	decode_throughput	memory_gb	status	description
a1b2c3d	1021.34	18.3	keep	baseline
b2c3d4e	1087.55	18.5	keep	max_num_seqs=768
c3d4e5f	998.12	18.3	discard	max_num_seqs=1024 (preemption overhead)
d4e5f6g	0.00	0.0	crash	kvcache_block_size=64 (violates %256 constraint)
```

---

## The Experiment Loop

LOOP FOREVER:

1. Look at git state: current branch and commit.
2. Propose one experimental change.
3. Modify `autorefine/candidate.py` (config tuning) and/or engine files (structural changes).
4. `git commit -am "description of experiment"`
5. Run: `python autorefine/evaluator.py $MODEL_PATH > run.log 2>&1`
6. Extract: `grep "^decode_throughput:\|^peak_memory_gb:" run.log`
7. If grep is empty → crash. Run `tail -n 50 run.log` to see the traceback.
8. Record in `results.tsv`.
9. If `decode_throughput` **improved (higher)** → keep the commit, advance the branch.
10. If equal or worse → `git reset --hard HEAD~1` (discard), record as `discard`.

---

## What You CAN Modify

**Primary (start here — zero structural risk):**
- `autorefine/candidate.py` — modify the values in `get_engine_kwargs()`.

**Secondary (structural changes, higher risk):**
- `nanovllm/engine/model_runner.py` — CUDA graph batch-size ladder and eager fallback.
- `nanovllm/engine/scheduler.py` — scheduling strategies.

---

## What You CANNOT Modify

- `autorefine/evaluator.py` — the ground-truth benchmark. Modifying it invalidates all comparisons.
- `nanovllm/engine/llm_engine.py` — core engine loop.
- `nanovllm/models/` — model architecture files.
- `nanovllm/layers/` — layer implementations (attention, linear, norm, sampler).
- `nanovllm/config.py` — Config definition (your changes go in candidate.py instead).

---

## Experiment Ideas (ordered by risk)

### Tier 1 — Config tuning only (`candidate.py`)

These never risk breaking the engine. Start here.

| Parameter | Default | Ideas to try |
|---|---|---|
| `max_num_seqs` | 512 | 128, 256, 384, 768, 1024 — more seqs = better decode GPU utilisation |
| `max_num_batched_tokens` | 16384 | 8192, 32768, 65536 — larger = fewer prefill scheduler steps |
| `kvcache_block_size` | 256 | 512, 1024 — larger blocks = less overhead, but coarser prefix caching |
| `gpu_memory_utilization` | 0.9 | 0.85, 0.95 — more = more KV blocks, fewer preemptions |
| `enforce_eager` | False | True — measures pure CUDA-graph overhead |

### Tier 2 — CUDA graph ladder (`model_runner.py`)

The graph batch-size list is at **line 228**:
```python
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
```
The eager fallback threshold is at **line 191**: `input_ids.size(0) > 512`.

Experiment ideas:
- Finer small-batch steps: `[1,2,3,4,5,6,7,8,10,12,14,16, ...]` — reduces padding waste for odd-sized decode batches.
- Mixed granularity: `[1,2,4,8] + range(16,64,8) + range(64, max_bs+1, 32)` — fewer large graphs, smaller memory footprint.
- Raise eager cutoff to 1024 to allow graph replay for larger batches.

### Tier 3 — Scheduler strategies (`scheduler.py`)

The prefill and decode scheduling loops are at **lines 25–58**.

Experiment ideas:
- **Chunked prefill**: cap tokens added per step to e.g. 2048, so prefill steps interleave with decode steps. Reduces decode stall when a large prefill batch monopolises the GPU.
- **Decode-first**: flush all running sequences before admitting new prefill, reducing preemption churn on large decode batches.
- **Lookahead allocation**: reserve blocks for sequences that are nearly out of KV budget before they actually preempt.

---

## Constraints

- `kvcache_block_size` **must be a multiple of 256** (enforced in `Config.__post_init__`).
- `max_num_batched_tokens >= max_model_len` (default 4096) — otherwise Config raises.
- **VRAM**: some increase is acceptable if throughput gain is meaningful. Avoid OOM.
- **Simplicity**: a small gain with added complexity is not worth keeping; a simplification with neutral throughput is worth keeping.
- **Timeout**: if a run exceeds 10 minutes, kill it (`Ctrl-C`) and treat as a crash.

---

## Crashes

- If the crash is trivial (typo, missing import) → fix and re-run.
- If the idea is fundamentally broken → log as `crash`, discard, move on.
- Common crash causes: `kvcache_block_size` not multiple of 256; `max_num_batched_tokens` < `max_model_len`; OOM during KV cache allocation.

---

## Never Stop

Once the loop begins, **do not pause to ask the human whether to continue**. Keep iterating until manually interrupted. If ideas run dry, combine previous near-misses, sweep parameters more finely, or try Tier 3 scheduler experiments. The loop runs until the human stops it.
