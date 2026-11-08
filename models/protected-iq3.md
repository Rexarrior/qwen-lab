# Qwen3-0.6B · Protected IQ3_S

We quantized the official checkpoint directly via F32. Most projections (140) use IQ3_S. All 28 attention value and 28 MLP down projections receive extra bits: the first two value projections and the first down projection use Q5_K, and the other 53 use Q4_K. The tied embedding / output head uses Q5_K; 113 normalization tensors stay in F32. No additional fine-tuning. The calibration input is the importance matrix distributed by [Unsloth](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF); we did not reconstruct its original calibration text.

**Run:** Use the GGUF files from the matching release directory.

```sh
llama-server -m Qwen3-0.6B-OpenResearch-Protected-IQ3_S.gguf -c 2048
```

[Compression code](../compression/README.md) · [Scores and run identifiers](protected-iq3.json).

## In-house evaluation

Counts are passes out of all tasks. Fresh, A6 and A8 score complete trajectories; A6 and A8 are known task sets. The language perplexity uses the same 2,040 scored tokens from *Alice in Wonderland* on CPU; lower is better.

| Check | Result |
|---|---:|
| Fresh tool-use trajectories | 4/48 |
| Known A6 trajectories | 4/48 |
| Known A8 trajectories | 5/48 |
| Next action from supplied history | 8/24 |
| Full continuation from supplied history | 8/24 |
| Strict English | 22/96 |
| Alice perplexity ↓ | 28.1534 |

## External evaluations

BFCL v4: 21 categories without web search. Family accuracies are shown separately; these are not a full leaderboard overall. API-Bank uses 744 offline-evaluable calls; IFEval scores prompts and individual instructions separately.

| Check | Result |
|---|---:|
| BFCL · non-live | 41.71% |
| BFCL · live | 24.13% |
| BFCL · irrelevance | 77.25% |
| BFCL · live relevance | 43.75% |
| BFCL · multi turn | 0.00% |
| BFCL · memory | 1.51% |
| BFCL · format sensitivity | 10.19% |
| API-Bank · correct calls | 51/744 |
| IFEval · strict prompt | 283/541 |
| IFEval · strict instruction | 515/834 |
| IFEval · loose prompt | 303/541 |
| IFEval · loose instruction | 546/834 |

These results describe this artifact under the listed protocols. The small tool-use and English sets are research diagnostics.
