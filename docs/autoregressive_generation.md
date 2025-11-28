# Autoregressive Generation

**Autoregressive generation** means generating tokens one at a time, where each new token depends on all previous tokens.

## The Process

1. Feed prompt to model, get probability distribution for next token
2. Sample/select next token
3. Feed that token back into the model
4. Repeat until done

```
Input:  "The sky is"
Step 1: Model predicts "blue"  -> "The sky is blue"
Step 2: Model predicts "and"   -> "The sky is blue and"
Step 3: Model predicts "clear" -> "The sky is blue and clear"
...
```

## Why It's Slow

Each token requires a full forward pass through the model. You can't parallelize because token N+1 depends on token N.

## Prefill vs Decode

- **Prefill**: Process all prompt tokens in parallel (batch operation, very fast)
- **Decode**: Generate output tokens one-by-one (autoregressive, slower)

This is why benchmarks show dramatically different speeds for prefill vs decode. For example:

| Phase   | Speed      | Notes                          |
|---------|------------|--------------------------------|
| Prefill | ~65k tok/s | Parallel batch processing      |
| Decode  | ~40 tok/s  | Sequential, autoregressive     |

The autoregressive constraint is the fundamental bottleneck in LLM inference.

## Implications

- **Time to First Token (TTFT)**: Dominated by prefill time. Longer prompts = longer TTFT.
- **Generation Speed**: Limited by decode speed, regardless of hardware.
- **Optimization Strategies**: KV caching, speculative decoding, and batching help mitigate the bottleneck.
