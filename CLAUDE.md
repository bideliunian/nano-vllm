# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run inference example
python example.py

# Benchmark nano-vllm
python bench.py

# Benchmark nano-vllm vs vLLM
python benchmark_all.py

# Download a model for testing
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ --local-dir-use-symlinks False
```

There are no tests, linting configs, or CI pipelines in this project.

## Architecture

Nano-vLLM is a ~1,200-line reimplementation of vLLM supporting Qwen3 models. It has four layers:

### Public API (`nanovllm/`)
- `LLM` (`llm.py`) — thin wrapper over `LLMEngine` exposing `generate()`
- `SamplingParams` (`sampling_params.py`) — temperature, max_tokens, ignore_eos. Temperature must be > 1e-10 (no greedy sampling; uses Gumbel-max trick instead)
- `Config` (`config.py`) — model path, KV cache settings, tensor parallelism degree, memory utilization, max batch size

### Engine (`nanovllm/engine/`)
- **`LLMEngine`** (`llm_engine.py`) — top-level orchestrator; owns the tokenizer, `Scheduler`, and `ModelRunner`; drives the `step()` loop
- **`Scheduler`** (`scheduler.py`) — manages waiting/running queues of `Sequence` objects; returns prefill or decode batches; preempts sequences under memory pressure
- **`BlockManager`** (`block_manager.py`) — divides GPU KV cache into fixed 256-token blocks with hash-based prefix caching for cross-request block reuse; tracks reference counts
- **`Sequence`** (`sequence.py`) — one request; holds token_ids, block table, status (WAITING → RUNNING → FINISHED)
- **`ModelRunner`** (`model_runner.py`) — allocates KV cache, prepares CUDA inputs, captures CUDA graphs for decode (batch sizes 1/2/4/8 and multiples of 16), runs model, calls sampler; tensor parallelism via `torch.multiprocessing.spawn` (rank 0 drives, ranks 1+ listen over shared memory)

### Model (`nanovllm/models/`, `nanovllm/layers/`)
- `models/qwen3.py` — `Qwen3ForCausalLM`; loads safetensors weights with packed weight remapping (gate+up projections merged, q+k+v merged)
- `layers/attention.py` — prefill uses `flash_attn_varlen_func`; decode uses `flash_attn_with_kvcache` with block tables; Triton kernel for KV slot writes
- `layers/linear.py` — `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`, `MergedColumnParallelLinear` for tensor parallelism
- `layers/layernorm.py` — RMSNorm with fused add+norm residual (`add_rms_forward`, `@torch.compile`)
- `layers/sampler.py` — temperature-scaled Gumbel-max sampling (`@torch.compile`)
- `layers/rotary_embedding.py` — precomputed RoPE cache (`@torch.compile`)

### Utilities (`nanovllm/utils/`)
- `context.py` — thread-local attention context (prefill/decode mode, sequence metadata) consumed by attention layers
- `loader.py` — safetensors loading with packed weight remapping

## Key Design Decisions

- **Prefix caching**: KV blocks are content-addressed by hash; identical token prefixes across requests share blocks at zero cost
- **CUDA graphs**: decode steps are graph-captured per batch size; prefill always runs eagerly
- **Slot mapping**: each token's KV position is a flat integer index into the block-linear KV cache, computed from block table + block offset
- **Residual fusion**: `RMSNorm.add_rms_forward` fuses the residual add into the norm to reduce memory bandwidth
- **Tensor parallelism**: column-parallel layers shard outputs across ranks; row-parallel layers shard inputs and all_reduce at the end

## Submodule

`autoresearch/` (karpathy/autoresearch) is an unrelated autonomous ML research agent that iteratively modifies and trains a GPT model within a 5-minute budget. It is not part of the inference stack.
