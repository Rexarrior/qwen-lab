# Compression recipes

Run commands from the repository root. The examples use Python 3.11 with NumPy 2.2.6, PyTorch 2.8.0 and safetensors 0.5.3 on macOS. The numerical smoke check has no model or network dependency:

```sh
python -m pip install numpy==2.2.6 torch==2.8.0 safetensors==0.5.3
python compression/smoke.py
```

## Mixed precision: Q4/Q6 + Q8 and Protected IQ3_S

Start from the official Qwen3-0.6B, converted to F32 GGUF with a single tied token embedding / output head. Use llama-quantize from [llama.cpp 8bb3cf0d](https://github.com/ggml-org/llama.cpp/tree/8bb3cf0d216a5ae48360179f2931daf284be4d01). The original experiments used [Unsloth’s pinned importance matrix](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/50968a4468ef4233ed78cd7c3de230dd1d61a56b/imatrix_unsloth.dat); its SHA-256 is checked before quantization. The original calibration text is not needed to use this file. A different converter may produce different GGUF metadata.

```sh
python compression/quantize.py q4-q6-q8 original-f32.gguf imatrix_unsloth.dat q4-q6-q8.gguf
python compression/quantize.py protected-iq3 original-f32.gguf imatrix_unsloth.dat protected-iq3.gguf
```

Each .types file assigns an explicit GGML type to every matrix; norms stay F32. See the [Q4/Q6](../models/q4-q6-q8.md) and [Protected](../models/protected-iq3.md) results.

## F5: learn a correction to a frozen quantized base

[f5_training.py](f5_training.py) contains the shared decoder, straight-through rounding of local INT4 codes, rank-eight linear update and terminal-weighted loss from the F5 run. The conditioning vector identifies the tensor family, factor role and layer. During training, codes, decoder and FiLM parameters are updated jointly; the original quantized base stays fixed. See [training settings](f5-training.json). This is the training kernel; replaying all 112 historical updates additionally requires the parent kit, interleaved task rows and teacher distributions.

To recover the 196 factor pairs from an F5 compact correction kit (directory containing factor-codec.json):

```sh
PYTHONPATH=compression python -c "from factor_decoder import decode_kit; print(len(decode_kit('path/to/f5-kit')[0]))"
```

For inference load the exported GGUF adapter with its matching base, as shown on the [F5 page](../models/f5.md).

## P9 and P10: fit and train vector codebooks

[fit_projection.py](fit_projection.py) accepts a float weight matrix and a matrix of calibration activations, both in .npy format. Inputs must have the same width, divisible by 128. It estimates the input Gram matrix, fits a 256-entry book by weighted EM and chooses codes with block error feedback. The output archive stores the book, group scales and indices. This example runs one projection; it does not train the whole model.

```sh
python compression/fit_projection.py weights.npy activations.npy packed.npz --d 4
```

[vq_recovery.py](vq_recovery.py) implements both update rules. P9 updates only books, P10 also adjusts scales within ±10%; indices and the token embedding stay fixed. The [P9](p9-training.json) and [P10](p10-training.json) configs record the differences in learning rate, regularization and loss. See the individual model pages for their scores.

**Portable inference:** unzip one model’s portable-vq-kit.zip into a directory containing recipe.json, the books and indices, tokenizer and model config. Decode it without original weights:

```sh
PRACTICAL_DENIED_PATHS='["/unavailable-original-checkpoint"]' python compression/vq_decode.py kit kit/model.safetensors
```

**Native inference on Apple Silicon:** the native GGUF refers to codebooks compiled into the kernels. The repository contains the kernel changes but no learned tables. Supply both unpacked kits, in P9/P10 order, to recreate the exact patch. In a separate llama.cpp checkout, starting from the repository root:

```sh
python compression/native_patch.py p9-kit p10-kit native.patch
git clone https://github.com/ggml-org/llama.cpp.git
git -C llama.cpp checkout 8bb3cf0d216a5ae48360179f2931daf284be4d01
git -C llama.cpp apply ../native.patch
cmake -S llama.cpp -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build --config Release --target llama-server -j 4
```

This build reuses GGML type identifiers and must be reserved for the two matching native GGUFs. For portable deployment, the dense decoded kit needs no patched runtime.
