"""
Benchmark: nano-vllm vs vllm vs HuggingFace baseline (Qwen3-0.6B)
Measures throughput (tok/s), latency, GPU memory, and CPU memory.
"""

import os
import gc
import json
import time
import psutil
import torch
from random import randint, seed


MODEL_PATH = os.path.expanduser("models/Qwen3-0.6B")
NUM_SEQS = 64
MAX_INPUT_LEN = 512
MAX_OUTPUT_LEN = 512
MAX_MODEL_LEN = 2048
WARMUP_PROMPTS = ["Warmup: hello world"]


def get_gpu_memory_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2, torch.cuda.max_memory_allocated() / 1024**2


def get_cpu_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def reset_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def generate_inputs():
    seed(42)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, MAX_INPUT_LEN))]
        for _ in range(NUM_SEQS)
    ]
    max_tokens_list = [randint(100, MAX_OUTPUT_LEN) for _ in range(NUM_SEQS)]
    return prompt_token_ids, max_tokens_list


def bench_nanovllm(prompt_token_ids, max_tokens_list):
    from nanovllm import LLM, SamplingParams

    reset_gpu()
    cpu_before = get_cpu_memory_mb()

    llm = LLM(MODEL_PATH, enforce_eager=False, max_model_len=MAX_MODEL_LEN)
    gpu_after_load, _ = get_gpu_memory_mb()
    cpu_after_load = get_cpu_memory_mb()

    # warmup
    llm.generate(WARMUP_PROMPTS, SamplingParams(max_tokens=8))
    torch.cuda.reset_peak_memory_stats()

    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=mt)
        for mt in max_tokens_list
    ]
    total_tokens = sum(max_tokens_list)

    t = time.perf_counter()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - t

    _, gpu_peak = get_gpu_memory_mb()
    cpu_after = get_cpu_memory_mb()

    return {
        "engine": "nano-vllm",
        "total_output_tokens": total_tokens,
        "elapsed_s": round(elapsed, 2),
        "throughput_tok_s": round(total_tokens / elapsed, 2),
        "gpu_model_mb": round(gpu_after_load, 1),
        "gpu_peak_mb": round(gpu_peak, 1),
        "cpu_rss_before_mb": round(cpu_before, 1),
        "cpu_rss_after_mb": round(cpu_after, 1),
    }


def bench_vllm(prompt_token_ids, max_tokens_list):
    from vllm import LLM, SamplingParams

    reset_gpu()
    cpu_before = get_cpu_memory_mb()

    llm = LLM(MODEL_PATH, enforce_eager=False, max_model_len=MAX_MODEL_LEN)
    gpu_after_load, _ = get_gpu_memory_mb()
    cpu_after_load = get_cpu_memory_mb()

    # warmup
    llm.generate(WARMUP_PROMPTS, SamplingParams(max_tokens=8))
    torch.cuda.reset_peak_memory_stats()

    prompts = [dict(prompt_token_ids=p) for p in prompt_token_ids]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=mt)
        for mt in max_tokens_list
    ]
    total_tokens = sum(max_tokens_list)

    t = time.perf_counter()
    llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - t

    _, gpu_peak = get_gpu_memory_mb()
    cpu_after = get_cpu_memory_mb()

    return {
        "engine": "vllm",
        "total_output_tokens": total_tokens,
        "elapsed_s": round(elapsed, 2),
        "throughput_tok_s": round(total_tokens / elapsed, 2),
        "gpu_model_mb": round(gpu_after_load, 1),
        "gpu_peak_mb": round(gpu_peak, 1),
        "cpu_rss_before_mb": round(cpu_before, 1),
        "cpu_rss_after_mb": round(cpu_after, 1),
    }


def bench_baseline(prompt_token_ids, max_tokens_list):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    reset_gpu()
    cpu_before = get_cpu_memory_mb()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()
    gpu_after_load, _ = get_gpu_memory_mb()
    cpu_after_load = get_cpu_memory_mb()

    # warmup
    warmup_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    with torch.no_grad():
        model.generate(warmup_ids, max_new_tokens=8)
    torch.cuda.reset_peak_memory_stats()

    total_tokens = sum(max_tokens_list)
    generated_tokens = 0

    t = time.perf_counter()
    # baseline processes one prompt at a time (no batched inference engine)
    for token_ids, max_tok in zip(prompt_token_ids, max_tokens_list):
        input_ids = torch.tensor([token_ids], device="cuda")
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_tok,
                temperature=0.6,
                do_sample=True,
            )
        generated_tokens += out.shape[1] - input_ids.shape[1]
    elapsed = time.perf_counter() - t

    _, gpu_peak = get_gpu_memory_mb()
    cpu_after = get_cpu_memory_mb()

    return {
        "engine": "baseline (HF transformers)",
        "total_output_tokens": generated_tokens,
        "elapsed_s": round(elapsed, 2),
        "throughput_tok_s": round(generated_tokens / elapsed, 2),
        "gpu_model_mb": round(gpu_after_load, 1),
        "gpu_peak_mb": round(gpu_peak, 1),
        "cpu_rss_before_mb": round(cpu_before, 1),
        "cpu_rss_after_mb": round(cpu_after, 1),
    }


def print_results(results):
    print("\n" + "=" * 80)
    print(f"{'Metric':<30} ", end="")
    for r in results:
        print(f"{r['engine']:>20}", end="")
    print()
    print("-" * 80)

    metrics = [
        ("Output Tokens", "total_output_tokens", ""),
        ("Elapsed Time", "elapsed_s", "s"),
        ("Throughput", "throughput_tok_s", "tok/s"),
        ("GPU Model Load", "gpu_model_mb", "MB"),
        ("GPU Peak (inference)", "gpu_peak_mb", "MB"),
        ("CPU RSS Before", "cpu_rss_before_mb", "MB"),
        ("CPU RSS After", "cpu_rss_after_mb", "MB"),
    ]
    for label, key, unit in metrics:
        suffix = f" {unit}" if unit else ""
        print(f"{label:<30} ", end="")
        for r in results:
            print(f"{r[key]:>18}{suffix:>2}", end="")
        print()
    print("=" * 80)


def main():
    import sys
    prompt_token_ids, max_tokens_list = generate_inputs()
    total = sum(max_tokens_list)
    print(f"Config: {NUM_SEQS} seqs, max_input={MAX_INPUT_LEN}, max_output={MAX_OUTPUT_LEN}")
    print(f"Total output tokens to generate: {total}")
    print(f"Model: {MODEL_PATH}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = []

    # Each engine runs in a subprocess to get clean GPU state
    if len(sys.argv) > 1:
        engine = sys.argv[1]
        if engine == "nanovllm":
            r = bench_nanovllm(prompt_token_ids, max_tokens_list)
        elif engine == "vllm":
            r = bench_vllm(prompt_token_ids, max_tokens_list)
        elif engine == "baseline":
            r = bench_baseline(prompt_token_ids, max_tokens_list)
        print(json.dumps(r))
        return

    # Run all three in subprocesses for clean GPU state
    for engine in ["nanovllm", "vllm", "baseline"]:
        print(f"\n>>> Running {engine}...")
        import subprocess
        proc = subprocess.run(
            [sys.executable, __file__, engine],
            capture_output=True, text=True, timeout=600,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if proc.returncode != 0:
            print(f"  FAILED (exit {proc.returncode})")
            print(proc.stderr[-2000:] if proc.stderr else "(no stderr)")
            continue
        # find the JSON line in output
        for line in proc.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                r = json.loads(line)
                results.append(r)
                print(f"  {r['throughput_tok_s']} tok/s, GPU peak {r['gpu_peak_mb']} MB")
                break
        else:
            print(f"  No JSON output found")
            print(proc.stdout[-1000:])

    if results:
        print_results(results)


if __name__ == "__main__":
    main()
