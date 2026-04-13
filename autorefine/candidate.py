# autorefine/candidate.py — AGENT MODIFIES THIS FILE
"""
Primary tuning surface for throughput experiments.

The agent modifies the values in get_engine_kwargs() to experiment with
different Config parameters. All keys must be valid fields of Config
(nanovllm/config.py). Constraints:
  - kvcache_block_size must be a multiple of 256
  - max_num_batched_tokens >= max_model_len (default max_model_len=4096)
  - enforce_eager=True disables CUDA graphs (useful for comparison runs)

For structural changes (CUDA graph ladder, scheduler logic), the agent
may also edit nanovllm/engine/model_runner.py or nanovllm/engine/scheduler.py
directly and commit those alongside any changes here.
"""


def get_engine_kwargs() -> dict:
    return dict(
        max_num_batched_tokens=16384,  # prefill token budget per scheduler step
        max_num_seqs=512,              # max concurrent sequences in the scheduler
        gpu_memory_utilization=0.9,    # fraction of GPU memory reserved for KV cache
        kvcache_block_size=256,        # tokens per KV cache block (must be multiple of 256)
        enforce_eager=False,           # set True to disable CUDA graphs for comparison
    )
