
from dataclasses import dataclass
import numpy as np


@dataclass
class QuantizedWeights:
    data: np.ndarray
    scales: np.ndarray
    bits: int
    shape: tuple


def quantize(weights: np.ndarray, bits: int = 8):
    if bits not in (4, 8):
        raise ValueError("Only INT4 and INT8 supported")

    weights = np.asarray(weights, dtype=np.float32)

    if weights.ndim != 2:
        raise ValueError("Expected a 2D weight matrix")

    qmax = (1 << (bits - 1)) - 1

    maximum = np.max(
        np.abs(weights), axis=1, keepdims=True
    )

    scales = np.where(
        maximum == 0, 1.0, maximum / qmax
    ).astype(np.float32)

    q = np.clip(
        np.rint(weights / scales),
        -qmax,
        qmax
    ).astype(np.int8)

    if bits == 8:
        data = q
    else:
        # Store two signed 4-bit values per byte.
        flat = q.flatten()
        if len(flat) % 2:
            flat = np.pad(flat, (0, 1))

        unsigned = flat.astype(np.uint8) & 0x0F

        data = (
            unsigned[0::2]
            | (unsigned[1::2] << 4)
        ).astype(np.uint8)

    return QuantizedWeights(
        data=data,
        scales=scales,
        bits=bits,
        shape=weights.shape,
    )


def dequantize(result: QuantizedWeights):
    if result.bits == 8:
        q = result.data.reshape(result.shape)
    elif result.bits == 4:
        packed = result.data

        low = (packed & 0x0F).astype(np.int8)
        high = (packed >> 4).astype(np.int8)

        low = np.where(low >= 8, low - 16, low)
        high = np.where(high >= 8, high - 16, high)

        q = np.empty(len(packed) * 2, dtype=np.int8)
        q[0::2] = low
        q[1::2] = high

        q = q[:np.prod(result.shape)].reshape(result.shape)
    else:
        raise ValueError("Unsupported quantization format")

    return q.astype(np.float32) * result.scales
