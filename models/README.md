# Evaluation

Fresh, A6 and A8 each contain 48 tool-use trajectories. A case passes only if the model completes the task; A6 and A8 are known task sets. Oracle checks supply the correct history, and score the immediate next action and the entire remaining trajectory separately. Strict English is a separate 96-case test. Alice perplexity is measured on CPU on the same 2,040 scored tokens. The native codebook runtime was used for the P9 and P10 entries.

External testing uses 21 non-web BFCL v4 categories, with format sensitivity reported separately, 744 offline-evaluable API-Bank examples and IFEval (541 prompts / 834 instructions). Family accuracies do not constitute a full BFCL leaderboard score. A displayed zero is a measured zero; absent results are not zeros.

- [q4-q6-q8](q4-q6-q8.md) · [JSON](q4-q6-q8.json)
- [protected-iq3](protected-iq3.md) · [JSON](protected-iq3.json)
- [f5](f5.md) · [JSON](f5.json)
- [p9](p9.md) · [JSON](p9.json)
- [p10](p10.md) · [JSON](p10.json)
