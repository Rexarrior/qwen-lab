# Fine-tuning and weight editing

Three standalone F16 GGUF checkpoints trace different approaches to tool use in Qwen3-0.6B. WSP38 combines manual edits with supervised adaptation. v3-C7 explores targeted repair with explicit preservation constraints. semantic08 takes a separate branch from v2 and adapts one matrix to preserve successful single-turn behavior.

| Checkpoint and method | HF weights | A6 /48 | BFCL single-turn /3641 | Multi-turn /800 | Memory /465 | Format variants /5200 | API-Bank /744 | IFEval strict /541 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Original Qwen3-0.6B | [Upstream](https://huggingface.co/Qwen/Qwen3-0.6B) | 24 | 2463 | 24 | 17 | 1082 | 133 | 315 |
| [WSP38](wsp38.md) | [GGUF](https://huggingface.co/rexarrior/Qwen3-0.6B-WSP38-GGUF) | 38 | 2137 | 17 | 20 | 1129 | 135 | 277 |
| v2 parent | — | 38 | 2310 | 18 | 24 | 1133 | 135 | 308 |
| [v3-C7](v3-c7.md) | [GGUF](https://huggingface.co/rexarrior/Qwen3-0.6B-Tool-Repair-v3-C7-GGUF) | 40 | 2281 | 4 | 13 | 1000 | 127 | 287 |
| [semantic08](semantic08.md) | [GGUF](https://huggingface.co/rexarrior/Qwen3-0.6B-Tool-Repair-semantic08-GGUF) | 40 | 2602 | 14 | 17 | 1082 | 139 | 291 |

A6 was used for development. Its small local agent harness allowed four turns, a 2048-token context and 96 output tokens per turn. For semantic08 the A6 result comes from the archived local evaluation. Broad BFCL results were available during later model development. The table uses official per-family counts on 21 categories with web search excluded. Format sensitivity repeats 200 base cases in 26 formats. Adding these columns does not yield an official leaderboard score. BFCL used greedy decoding, thinking disabled, seed 42, 40,960 context tokens and a 4096-token output limit. [BFCL evaluator revision](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard)

On API-Bank, each model received the reference dialogue history for its next API call. The offline executable subset contained 744 of the 753 generated calls. IFEval strict prompt requires the response to satisfy all checked instructions for that prompt. API-Bank and IFEval were external transfer checks. The package includes file sizes and SHA-256 hashes for verification. [Release manifest](checkpoints.json)
