# WSP38 · changing tool-use decisions

WSP38 investigates how much of a small model's tool use can be repaired by changing its weights. It combines three hand-selected token scales, a supervised least-squares correction and a merged LoRA. On the 48 A6 tasks used during selection, the original model solved 24, the three scales alone solved 32, and WSP38 solved 38.

1. Start from the F16 Qwen3-0.6B GGUF. Multiply three complete rows of the tied token embedding by 1.5 for `<tool_call>` (151657), 1.2 for `<|im_end|>` (151645) and 1.3 for `lookup` (21020). Compute in float32 and store in F16. These scales affect both token input representations and output logits. The intended effects are to encourage a call, help terminate a turn, and improve tool selection. The study did not isolate a causal effect for every row.
2. Fit the output projection of the last MLP to labeled examples in two least-squares stages. Apply the first correction with a factor of two. The adaptation pool included 96 training tasks, 224 agent turns and 20 language examples.
3. Fit rank-16 LoRA updates on the attention and MLP projections of blocks 16–27 with AdamW, learning rate 0.0005, for two epochs (140 updates). Merge half of the learned update into the weights. The released GGUF contains the merged weights.
4. Rebuild and compare the file hash. In the historical replication the export matched the measured file byte for byte.

On the frozen transfer set of new phrasings and values from the same task families, scores were 30/48 for WSP38 and 20/48 for the original. On the JSON regression set they were 7/24 and 10/24 respectively. The manual-rows-only ablation and the merged WSP38 checkpoint are different experimental conditions.

[Weights and usage](https://huggingface.co/rexarrior/Qwen3-0.6B-WSP38-GGUF) · [Evaluation across checkpoints](README.md)
