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
