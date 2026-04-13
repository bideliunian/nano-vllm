"""
autorefine/evaluator.py — NEVER MODIFIED BY AGENT

Fixed throughput benchmark for nano-vllm. This is the ground-truth evaluator;
modifying it invalidates all cross-experiment comparisons.

Usage:
    python autorefine/evaluator.py <model_path>
    MODEL_PATH=~/huggingface/Qwen3-0.6B python autorefine/evaluator.py

Output (grep-parseable, printed to stdout):
    ---
    decode_throughput:    1234.56
    elapsed_seconds:      87.32
    peak_memory_gb:       18.3
    total_output_tokens:  40576
    num_seqs:             256

The primary metric is decode_throughput (tokens/sec). HIGHER IS BETTER.
"""

import os
import sys
import time
import torch
from random import seed, randint

# ── FIXED BENCHMARK CONSTANTS — never change these ────────────────────────────
NUM_SEQS = 256          # concurrent requests in the benchmark batch
MIN_INPUT_LEN = 100     # minimum prompt length (tokens)
MAX_INPUT_LEN = 512     # maximum prompt length (tokens)
MIN_OUTPUT_LEN = 100    # minimum generation length (tokens)
MAX_OUTPUT_LEN = 256    # maximum generation length (tokens)
BENCH_SEED = 42         # RNG seed for reproducible prompt/length sampling
WARMUP_SEQS = 4         # sequences for warmup pass (results discarded)
WARMUP_OUTPUT_LEN = 32  # output tokens per warmup sequence
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # Resolve model path from CLI arg or environment variable
    if len(sys.argv) >= 2:
        model_path = os.path.expanduser(sys.argv[1])
    elif "MODEL_PATH" in os.environ:
        model_path = os.path.expanduser(os.environ["MODEL_PATH"])
    else:
        print("Usage: python autorefine/evaluator.py <model_path>", file=sys.stderr)
        print("       or set MODEL_PATH environment variable", file=sys.stderr)
        sys.exit(1)

    # Import here so import errors surface clearly in run.log
    from nanovllm import LLM, SamplingParams
    from autorefine.candidate import get_engine_kwargs

    # Build deterministic prompts and sampling params
    seed(BENCH_SEED)
    prompt_token_ids = [
        [randint(1, 50000) for _ in range(randint(MIN_INPUT_LEN, MAX_INPUT_LEN))]
        for _ in range(NUM_SEQS)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.8,
            ignore_eos=True,
            max_tokens=randint(MIN_OUTPUT_LEN, MAX_OUTPUT_LEN),
        )
        for _ in range(NUM_SEQS)
    ]

    # Warmup sequences (short, fixed)
    warmup_prompts = [[1] * 64] * WARMUP_SEQS
    warmup_params = [SamplingParams(temperature=0.8, max_tokens=WARMUP_OUTPUT_LEN)] * WARMUP_SEQS

    # Initialise engine with config from candidate.py
    kwargs = get_engine_kwargs()
    llm = LLM(model_path, **kwargs)

    # Warmup pass — discard results and reset peak memory counter
    llm.generate(warmup_prompts, warmup_params, use_tqdm=False)
    torch.cuda.reset_peak_memory_stats()

    # Benchmark pass
    t_start = time.perf_counter()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - t_start

    total_output_tokens = sum(sp.max_tokens for sp in sampling_params)
    decode_throughput = total_output_tokens / elapsed
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

    print("---")
    print(f"decode_throughput:    {decode_throughput:.2f}")
    print(f"elapsed_seconds:      {elapsed:.2f}")
    print(f"peak_memory_gb:       {peak_memory_gb:.1f}")
    print(f"total_output_tokens:  {total_output_tokens}")
    print(f"num_seqs:             {NUM_SEQS}")


if __name__ == "__main__":
    main()
