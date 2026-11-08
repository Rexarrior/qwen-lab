# Qwen3 compression lab

Five experiments on the same [Qwen3-0.6B checkpoint](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca). How much of a small chat model survives when we spend fewer bits on each weight? We compared two fixed precision maps, a learned rank-eight correction, and two vector-codebook models.

| Model | Fresh tool use | A6 | A8 | Strict English | Alice PPL ↓ |
|---|---:|---:|---:|---:|---:|
| [q4-q6-q8](models/q4-q6-q8.md) | 24/48 | 16/48 | 8/48 | 26/96 | 23.99 |
| [protected-iq3](models/protected-iq3.md) | 4/48 | 4/48 | 5/48 | 22/96 | 28.15 |
| [f5](models/f5.md) | 5/48 | 12/48 | 9/48 | 20/96 | 38.75 |
| [p9](models/p9.md) | 0/48 | 0/48 | 7/48 | 13/96 | 35.47 |
| [p10](models/p10.md) | 16/48 | 21/48 | 32/48 | 21/96 | 54.69 |

Each model page describes what changed, how to load the result, and what passed. The P9 and P10 weights come from two independent training runs. P9 better predicts the text sample, while P10 completes more tool-use tasks. The tables distinguish this small diagnostic suite from independent evaluations.

- [Compression code and commands](compression/README.md): mixed-precision GGUF, F5 factor synthesis, weighted codebook fitting, recovery and native inference.
- [Evaluation protocol and result files](models/README.md): task definitions, denominators, scope and limitations.
- [Smoke check](compression/smoke.py): builds a codebook from a small matrix and checks the two recovery gradient paths against autograd.

This repository contains code, precision maps, training settings and measured results. Run instructions that take a model file require the corresponding weights or portable kit. Historical full-model training used separate calibration data, teacher caches and starting kits; the per-projection example can be run without those inputs.

Model derivatives follow the upstream Apache 2.0 license. The llama.cpp patch retains its [MIT license](compression/LICENSE-llama.cpp).
