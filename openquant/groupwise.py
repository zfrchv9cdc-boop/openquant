from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Groupwise4:
    data: np.ndarray
    scales: np.ndarray
    shape: tuple[int, int]
    group_size: int


def quantize_int4_groupwise(weights, group_size=32):
    a = np.asarray(weights, dtype=np.float32)

    if a.ndim != 2 or min(a.shape) == 0:
        raise ValueError("Expected nonempty 2D weights")

    if group_size < 2 or group_size % 2 != 0:
        raise ValueError("Group size must be even and >= 2")

    if not np.isfinite(a).all():
        raise ValueError("Weights must be finite")

    rows, cols = a.shape

    padded_cols = (
        (cols + group_size - 1) // group_size
    ) * group_size

    padded = np.pad(
        a, ((0, 0), (0, padded_cols - cols))
    )

    blocks = padded.reshape(rows, -1, group_size)

    scales3 = (
        np.max(np.abs(blocks), axis=2, keepdims=True)
        / 7.0
    )

    scales3 = np.where(
        scales3 == 0, 1.0, scales3
    ).astype(np.float32)

    integers = np.clip(
        np.rint(blocks / scales3), -7, 7
    ).astype(np.int8).reshape(-1)

    unsigned = integers.astype(np.uint8) & 15

    data = (
        unsigned[0::2] | (unsigned[1::2] << 4)
    ).astype(np.uint8)

    return Groupwise4(
        data=data,
        scales=scales3.squeeze(-1),
        shape=(rows, cols),
        group_size=group_size
    )


def dequantize_int4_groupwise(q):
    rows, cols = q.shape

    low = (q.data & 15).astype(np.int16)
    high = (q.data >> 4).astype(np.int16)

    unpacked = np.empty(
        q.data.size * 2, dtype=np.int16
    )

    unpacked[0::2] = low
    unpacked[1::2] = high

    signed = np.where(
        unpacked >= 8, unpacked - 16, unpacked
    ).astype(np.float32)

    blocks = signed.reshape(
        rows, -1, q.group_size
    )

    restored = (
        blocks * q.scales[:, :, None]
    ).reshape(rows, -1)

    return restored[:, :cols]
