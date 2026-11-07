
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
import math
import threading
from collections import Counter, OrderedDict
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

    def __init__(
        self, model, example_tokens, *, warmup_stream=None
    ):
        self.model = model

        self.keys = tuple(sorted(example_tokens))

        self.inputs = {
            key: example_tokens[key].clone()
            for key in self.keys
        }

        current = torch.cuda.current_stream()
        warmup = (
            warmup_stream
            if warmup_stream is not None
            else torch.cuda.Stream()
        )

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
        max_graph_cache_mb=128,
        min_capture_free_mb=256,
        graph_capture_min_calls=1,
        graph_capture_min_budget_mb=0,
        graph_capture_max_attempts=0,
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

        if max_graph_cache_mb < 0:
            raise ValueError('Invalid graph memory budget')
        if min_capture_free_mb < 0:
            raise ValueError('Invalid capture headroom')

        if (
            isinstance(graph_capture_min_budget_mb, bool)
            or not isinstance(graph_capture_min_budget_mb, (int, float))
            or not math.isfinite(graph_capture_min_budget_mb)
            or graph_capture_min_budget_mb < 0
        ):
            raise ValueError(
                'graph_capture_min_budget_mb must be finite and nonnegative'
            )

        if (
            isinstance(graph_capture_max_attempts, bool)
            or not isinstance(graph_capture_max_attempts, int)
            or graph_capture_max_attempts < 0
        ):
            raise ValueError(
                'graph_capture_max_attempts must be a nonnegative integer'
            )

        self.max_graph_cache_bytes = int(
            max_graph_cache_mb * 1024 * 1024
        )
        self.min_capture_free_bytes = int(
            min_capture_free_mb * 1024 * 1024
        )
        self.graph_capture_min_budget_bytes = int(
            graph_capture_min_budget_mb * 1024 * 1024
        )

        self.graph_capture_max_attempts = (
            graph_capture_max_attempts
        )
        self.graph_capture_attempts = 0

        self.batch_size = batch_size
        self.max_length = max_length
        self.enable_graphs = bool(enable_graphs)
        self.max_graphs = max_graphs
        self.graph_max_tokens = graph_max_tokens

        self.device = torch.device("cuda")
        self.lock = threading.RLock()
        self._warmup_stream = None

        self.graphs = OrderedDict()
        self.graph_costs = {}
        self.graph_cache_bytes = 0
        self.graph_errors = {}
        self.stats = Counter()

        if (
            not isinstance(graph_capture_min_calls, int)
            or graph_capture_min_calls < 1
        ):
            raise ValueError(
                'graph_capture_min_calls must be positive'
            )

        self.graph_capture_min_calls = (
            graph_capture_min_calls
        )
        self.pending_graph_shapes = OrderedDict()
        self.pending_shape_limit = 64

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


    def _eager(self, tokens):
        self.stats["eager_calls"] += 1

        with torch.inference_mode():
            return encode_sentences(self.model, tokens)

    def _evict_lru(self):
        """Release the least recently used cached graph."""

        key, graph = self.graphs.popitem(last=False)

        cost = self.graph_costs.pop(key)
        self.graph_cache_bytes -= cost

        # Explicitly destroy graph execution before eviction.
        torch.cuda.synchronize()
        graph.graph.reset()
        del graph
        gc.collect()
        torch.cuda.empty_cache()

        self.stats["graph_evictions"] += 1

    def _forward(self, tokens, use_graph):
        """Execute using an admitted graph or eager fallback."""

        key = self._key(tokens)
        batch, sequence, _ = key

        eligible = (
            use_graph
            and self.enable_graphs
            and self.max_graphs > 0
            and self.max_graph_cache_bytes > 0
            and batch * sequence <= self.graph_max_tokens
        )

        if not eligible:
            return self._eager(tokens)

        if key in self.graphs:
            self.graphs.move_to_end(key)
            self.stats["graph_hits"] += 1
            return self.graphs[key](tokens)

        # Optional model-specific preflight threshold.
        # Not a physical VRAM guarantee.
        if (
            self.graph_capture_min_budget_bytes > 0
            and self.max_graph_cache_bytes
            < self.graph_capture_min_budget_bytes
        ):
            self.stats["preflight_budget_fallbacks"] += 1
            return self._eager(tokens)

        # Prevent repeated attempts for shapes already rejected.
        if str(key) in self.graph_errors:
            self.stats["blocked_shape_fallbacks"] += 1
            return self._eager(tokens)

        if self.graph_capture_min_calls > 1:
            calls = self.pending_graph_shapes.get(key, 0) + 1
            self.pending_graph_shapes[key] = calls
            self.pending_graph_shapes.move_to_end(key)

            if len(self.pending_graph_shapes) > self.pending_shape_limit:
                self.pending_graph_shapes.popitem(last=False)

            if calls < self.graph_capture_min_calls:
                self.stats["cold_shape_fallbacks"] += 1
                return self._eager(tokens)

        if (
            self.graph_capture_max_attempts > 0
            and self.graph_capture_attempts
            >= self.graph_capture_max_attempts
        ):
            self.stats["capture_limit_fallbacks"] += 1
            return self._eager(tokens)

        free_bytes, _ = torch.cuda.mem_get_info()

        if free_bytes < self.min_capture_free_bytes:
            self.stats["low_memory_fallbacks"] += 1
            return self._eager(tokens)

        # Preserve existing graphs until the candidate passes
        # capture, correctness, and budget admission checks.
        with torch.inference_mode():
            reference = encode_sentences(
                self.model, tokens
            ).detach().clone()

        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()

        self.graph_capture_attempts += 1

        try:
            if self._warmup_stream is None:
                self._warmup_stream = torch.cuda.Stream(
                    device=self.device
                )

            candidate = FixedShapeGraph(
                self.model, tokens,
                warmup_stream=self._warmup_stream,
            )

            result = candidate(tokens)
            torch.cuda.synchronize()

        except RuntimeError as exc:
            self.graph_errors[str(key)] = repr(exc)
            self.stats["capture_failures"] += 1

            del reference
            gc.collect()
            torch.cuda.empty_cache()

            return self._eager(tokens)

        # Incorrect graph output is an actual correctness
        # failure, not something to silently conceal.
        torch.testing.assert_close(
            result.float(),
            reference.float(),
            rtol=0.001,
            atol=0.001,
        )

        # Subtract the returned clone; it is not retained
        # by the graph cache.
        result_bytes = (
            result.numel() * result.element_size()
        )

        measured_cost = max(
            0,
            torch.cuda.memory_allocated()
            - allocated_before
            - result_bytes,
        )

        # Include permanent input and output buffers as
        # a minimum, even when allocations are reused.
        static_bytes = (
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in candidate.inputs.values()
            )
            + candidate.output.numel()
            * candidate.output.element_size()
        )

        cost = int(max(measured_cost, static_bytes))

        del reference

        if cost > self.max_graph_cache_bytes:
            self.graph_errors[str(key)] = (
                f"Graph allocation {cost} exceeds "
                f"cache budget {self.max_graph_cache_bytes}"
            )

            self.stats["budget_rejections"] += 1

            # Release execution resources of rejected graph.
            torch.cuda.synchronize()
            candidate.graph.reset()
            del result, candidate
            gc.collect()
            torch.cuda.empty_cache()

            return self._eager(tokens)

        # Admit the new graph by evicting older graphs
        # until the retained-allocation budget fits.
        while (
            self.graphs
            and (
                len(self.graphs) >= self.max_graphs
                or self.graph_cache_bytes + cost
                > self.max_graph_cache_bytes
            )
        ):
            self._evict_lru()

        self.graphs[key] = candidate
        self.graph_costs[key] = cost
        self.pending_graph_shapes.pop(key, None)
        self.graph_cache_bytes += cost

        self.stats["graph_captures"] += 1

        return result

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
                "graph_cache_bytes": self.graph_cache_bytes,
                "graph_budget_bytes": self.max_graph_cache_bytes,
                "graph_capture_min_calls": self.graph_capture_min_calls,
                "graph_capture_max_attempts": self.graph_capture_max_attempts,
                "graph_capture_attempts": self.graph_capture_attempts,
                "graph_capture_min_budget_bytes": self.graph_capture_min_budget_bytes,
                "pending_shapes": [
                    [key[0], key[1], count]
                    for key, count in self.pending_graph_shapes.items()
                ],
                "graph_costs_bytes": {
                    str(key): value
                    for key, value in self.graph_costs.items()
                },
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
        """Discard cached graphs and reset memory accounting."""

        with self.lock:
            torch.cuda.synchronize()

            # Explicitly release all cached graph executions.
            for cached_graph in self.graphs.values():
                cached_graph.graph.reset()

            self.graphs.clear()
            self.graph_costs.clear()
            self.graph_errors.clear()
            self.pending_graph_shapes.clear()
            self.graph_cache_bytes = 0

            gc.collect()
            torch.cuda.empty_cache()
