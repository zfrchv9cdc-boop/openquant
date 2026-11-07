# OpenQuant

Experimental AI weight quantization and GPU benchmarking toolkit.

## Features
- Symmetric INT8 and packed INT4 quantization
- Groupwise INT4 compression
- Fused Triton INT4 GPU matrix multiplication
- GPU benchmarking and kernel configuration testing

## Requirements
- Python 3.11 or newer
- NumPy
- NVIDIA CUDA GPU and compatible PyTorch/Triton for GPU features

## Commands
- openquant --version
- openquant cpu
- openquant gpu
- openquant tune

## Limitations
This is an experimental prototype, not a complete LLM inference
engine. Numerical quality and performance depend on the workload.
GPU benchmarks exclude initial weight quantization and transfers.


## Sentence encoding (experimental v0.7)

Install the optional inference dependencies and prepare the
compressed MiniLM checkpoint before using this command.

Encode one sentence:

```text
openquant encode --text "Hello world" --output reports/embedding.json
```

Encode a UTF-8 file containing one sentence per line:

```text
openquant encode --input sentences.txt --output reports/batch.json
```

Use `--graphs` to enable CUDA Graphs for repeated input shapes.
One-shot encoding runs eagerly by default to avoid unnecessary
capture overhead. Outputs contain text and embeddings, so handle
the JSON files according to your data privacy requirements.
