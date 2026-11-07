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



## Portable INT8 PyTorch checkpoints (experimental)

OpenQuant can convert standard PyTorch `nn.Linear` layers to
symmetric, weight-only INT8 and save the compressed model state
using SafeTensors.

Supported:
- Nested standard linear layers
- Inference on CPU or CUDA
- Per-output-channel INT8 scales
- Preserved biases
- Mixed compressed/uncompressed modules
- Checkpoint SHA-256 integrity validation
- Restoration into a compatible architecture

Important limitations:
- A compatible PyTorch model architecture is required.
- Arbitrary model Python code is not serialized.
- Training and optimizer state are not supported.
- Other layer types are preserved but not automatically quantized.
- Complex parameter sharing between distinct modules is not
  guaranteed to survive conversion.
- The checkpoint checksum detects corruption, not malicious
  replacement of both the file and its manifest.

Create and validate a portable checkpoint locally:

    python demo_checkpoint_roundtrip_v07.py

Run the checkpoint tests:

    python -m pytest -q tests/test_int8_checkpoint_v07.py

The saved checkpoint contains INT8 weights for converted layers
and retains ordinary tensors for unconverted modules.

