# semantic08 · preserving successful single-turn calls

semantic08 branches from v2. Only `blk.27.ffn_down.weight` changed relative to that parent. The byte-level audit found 3,115,246 changed F16 values within the 1024 × 3072 matrix. The stored update's Frobenius norm was 11.19% of the parent matrix norm.

1. Inspect the native responses behind v2's official scores. In two BFCL irrelevance categories, exclude credited responses containing an attempted or malformed tool call. This left 2278 single-turn successes to preserve, from 2310 official credits. Freeze the development and guard memberships. [Development and guard IDs](splits-semantic08.json)
2. Cache v2 activations on fixed prefixes. Fit a supervised correction with F16 rounding in the forward pass and a straight-through gradient. Protect token margins across whole successful calls, including later arguments. Recheck genuine no-call decisions using free generation.
3. Start the final repair from semantic03, retaining v2 call and A6 teachers. Protect 794 successful no-call responses from semantic03 and two v2 responses for regressions. The final cache contained 116,303 positions. Use a learning rate of 0.000001, a relative norm cap of 15%, refresh difficult guard rows every 25 steps and evaluate every 400 steps. Select step 800, export to F16, then check actual completions in llama.cpp.
4. Compare paired task IDs across the full evaluation. After screening malformed calls, semantic08 gained 308 single-turn cases and lost zero relative to v2. Against the original it gained 248 and lost 107. The broader preservation check retained 3272/3453 required v2 credits. The 181 losses comprised eight multi-turn, 14 memory and 159 format-variant cases.

BFCL outcomes influenced the design throughout this series. Excluding the 224 fit IDs alone would not produce an independent test. The external API-Bank difference relative to v2 was +0.54 percentage points (paired 95% interval −0.83 to +1.98). IFEval strict prompt fell by 3.14 points (interval −6.28 to −0.18).

[Weights and usage](https://huggingface.co/rexarrior/Qwen3-0.6B-Tool-Repair-semantic08-GGUF) · [Evaluation across checkpoints](README.md)
