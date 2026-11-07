
"""OpenQuant v0.7 - reusable compressed sentence encoder.

Features:
- Plain text to normalized sentence embeddings
- Packed INT4 model execution
- Fixed-shape CUDA Graph caching
- Configurable graph cache capacity
- Eager fallback
- Serialized thread-safe inference
- CPU or GPU result tensors

Experimental CUDA-only implementation.
"""

import gc
import threading
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from openquant.inference_v07 import (
    MODEL_ID,
    encode_sentences,
    load_packed_model,
)


DEFAULT_ARCHIVE = (
    "models/minilm/minilm_int4_g32.safetensors"
)

BUCKETS = (16, 32, 64, 128, 256)


class FixedShapeGraph:
    """Capture a fixed-shape encoder computation."""

    def __init__(self, model, example_tokens):
        self.model = model

        self.keys = tuple(sorted(example_tokens))

        self.inputs = {
            key: example_tokens[key].clone()
            for key in self.keys
        }

        current = torch.cuda.current_stream()
        warmup = torch.cuda.Stream()

        warmup.wait_stream(current)

        with torch.cuda.stream(warmup):
            with torch.inference_mode():
                for _ in range(4):
                    encode_sentences(
                        self.model,
                        self.inputs,
                    )

        current.wait_stream(warmup)

        self.graph = torch.cuda.CUDAGraph()

        with torch.inference_mode():
            with torch.cuda.graph(self.graph):
                self.output = encode_sentences(
                    self.model,
                    self.inputs,
                )

        torch.cuda.synchronize()

    def __call__(self, tokens):
        if tuple(sorted(tokens)) != self.keys:
            raise ValueError("CUDA Graph input keys changed")

        for key in self.keys:
            src = tokens[key]
            dst = self.inputs[key]

            if (
                src.shape != dst.shape
                or src.dtype != dst.dtype
                or src.device != dst.device
            ):
                raise ValueError(
                    "CUDA Graph input metadata changed"
                )

        with torch.inference_mode():
            for key in self.keys:
                self.inputs[key].copy_(tokens[key])

            self.graph.replay()

            # Graph outputs are overwritten during replay.
            # Always return an independent tensor.
            return self.output.clone()


class OpenQuantEncoder:
    """GPU sentence encoder backed by OpenQuant packed weights."""

    def __init__(
        self,
        archive=DEFAULT_ARCHIVE,
        *,
        model_id=MODEL_ID,
        batch_size=12,
        max_length=128,
        enable_graphs=True,
        max_graphs=2,
        graph_max_tokens=2048,
        model=None,
        tokenizer=None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "OpenQuant v0.7 inference requires CUDA"
            )

        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        if not 4 <= max_length <= 512:
            raise ValueError(
                "max_length must be between 4 and 512"
            )

        if max_graphs < 0 or graph_max_tokens < 1:
            raise ValueError("Invalid graph cache settings")

        self.batch_size = batch_size
        self.max_length = max_length
        self.enable_graphs = bool(enable_graphs)
        self.max_graphs = max_graphs
        self.graph_max_tokens = graph_max_tokens

        self.device = torch.device("cuda")
        self.lock = threading.RLock()

        self.graphs = {}
        self.graph_errors = {}
        self.stats = Counter()

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
            )

        self.tokenizer = tokenizer

        if model is None:
            archive = Path(archive)

            if not archive.is_file():
                raise FileNotFoundError(archive)

            model, self.compressed_modules = (
                load_packed_model(archive)
            )
        else:
            self.compressed_modules = None

        self.model = model.to(self.device).eval()

    def _prepare(self, texts):
        """Tokenize, bucket lengths, and transfer to CUDA."""

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        if (
            "input_ids" not in encoded
            or "attention_mask" not in encoded
        ):
            raise ValueError(
                "Tokenizer must return IDs and an attention mask"
            )

        original_length = encoded["input_ids"].shape[1]

        target_length = next(
            (
                bucket
                for bucket in BUCKETS
                if bucket >= original_length
            ),
            original_length,
        )

        if target_length > self.max_length:
            target_length = original_length

        padded = {}

        for key, tensor in encoded.items():
            if tensor.ndim != 2:
                raise ValueError(
                    f"Unsupported token tensor: {key}"
                )

            difference = target_length - tensor.shape[1]

            if difference > 0:
                pad_value = 0

                if key == "input_ids":
                    pad_value = (
                        self.tokenizer.pad_token_id or 0
                    )

                tensor = F.pad(
                    tensor,
                    (0, difference),
                    value=pad_value,
                )

            padded[key] = (
                tensor.to(self.device).contiguous()
            )

        return padded

    @staticmethod
    def _key(tokens):
        ids = tokens["input_ids"]

        return (
            int(ids.shape[0]),
            int(ids.shape[1]),
            tuple(sorted(tokens)),
        )

    def _forward(self, tokens, use_graph):
        """Choose a cached graph or the existing eager path."""

        key = self._key(tokens)

        batch, sequence, _ = key

        eligible = (
            use_graph
            and self.enable_graphs
            and self.max_graphs > 0
            and batch * sequence <= self.graph_max_tokens
        )

        if eligible and key in self.graphs:
            self.stats["graph_hits"] += 1
            return self.graphs[key](tokens)

        if eligible and len(self.graphs) < self.max_graphs:
            # Capture a graph and validate it against eager inference.
            with torch.inference_mode():
                reference = encode_sentences(
                    self.model, tokens
                ).detach().clone()

            try:
                captured = FixedShapeGraph(
                    self.model, tokens
                )

            except RuntimeError as exc:
                self.graph_errors[str(key)] = repr(exc)
                self.stats["capture_failures"] += 1

                # The eager path remains available.
                self.stats["eager_calls"] += 1

                with torch.inference_mode():
                    return encode_sentences(
                        self.model, tokens
                    )

            actual = captured(tokens)

            torch.cuda.synchronize()

            # Do not cache an incorrect graph.
            torch.testing.assert_close(
                actual.float(),
                reference.float(),
                rtol=0.001,
                atol=0.001,
            )

            self.graphs[key] = captured
            self.stats["graph_captures"] += 1

            return actual

        self.stats["eager_calls"] += 1

        with torch.inference_mode():
            return encode_sentences(
                self.model, tokens
            )

    def encode(
        self,
        texts,
        *,
        use_graph=True,
        return_cpu=True,
    ):
        """Encode one string or a list into shape [N, hidden_size]."""

        if isinstance(texts, str):
            texts = [texts]

        if (
            not isinstance(texts, (list, tuple))
            or not texts
            or not all(isinstance(x, str) for x in texts)
        ):
            raise ValueError(
                "Provide a string or a nonempty list of strings"
            )

        # A single model and its graph buffers cannot be
        # replayed simultaneously by multiple callers.
        with self.lock:
            results = []

            for start in range(
                0, len(texts), self.batch_size
            ):
                chunk = texts[
                    start:start + self.batch_size
                ]

                tokens = self._prepare(chunk)

                output = self._forward(
                    tokens,
                    use_graph=use_graph,
                )

                if return_cpu:
                    output = output.cpu()

                results.append(output)

            return torch.cat(results, dim=0)

    def cache_info(self):
        """Return cache statistics and current GPU allocator usage."""

        with self.lock:
            return {
                "cached_graphs": len(self.graphs),
                "max_graphs": self.max_graphs,
                "shapes": [
                    [key[0], key[1]]
                    for key in self.graphs
                ],
                "stats": dict(self.stats),
                "capture_errors": dict(self.graph_errors),
                "memory_allocated_bytes": (
                    torch.cuda.memory_allocated()
                ),
                "memory_reserved_bytes": (
                    torch.cuda.memory_reserved()
                ),
            }

    def clear_graph_cache(self):
        """Release graph references and request unused cache cleanup."""

        with self.lock:
            self.graphs.clear()
            self.graph_errors.clear()

            gc.collect()
            torch.cuda.empty_cache()
