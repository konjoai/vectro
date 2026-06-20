"""Pipeline telemetry & observability for vectro — v5.3.0.

Provides structured, per-stage metrics (throughput, cosine fidelity,
compression ratio, latency) via pluggable callback hooks.

Public API
----------
TelemetryEvent        — immutable dataclass emitted after every pipeline stage.
TelemetryHook         — callable protocol: (TelemetryEvent) -> None.
TelemetryCollector    — manages a set of hooks; call emit() to fan-out.
InMemoryTelemetryCollector — subclass that stores every event in a list.
attach_telemetry()    — convenience helper: wraps a CompressionPipeline to
                        emit events automatically on every run().
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# TelemetryEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TelemetryEvent:
    """Immutable record emitted after each pipeline stage (or full run).

    Attributes
    ----------
    stage_name : str
        Quantization mode label, e.g. ``"int8"``.
    stage_index : int
        Zero-based position of this stage in the pipeline.
    latency_ms : float
        Wall-clock time taken by this stage in milliseconds.
    input_shape : Tuple[int, ...]
        Shape of the array *fed into* this stage.
    output_shape : Tuple[int, ...]
        Shape of the array *produced by* this stage.
    input_dtype : str
        NumPy dtype string of the input array, e.g. ``"float32"``.
    output_dtype : str
        NumPy dtype string of the output array.
    compression_ratio : float
        ``input_bytes / output_bytes`` for this stage alone.
    throughput_vecs_per_sec : float
        Vectors processed per second (``input_shape[0] / latency_s``).
        Returns ``0.0`` when ``latency_ms`` is zero.
    cosine_fidelity : Optional[float]
        Mean cosine similarity between the float32 input and the
        float32-decoded output of this stage.  ``None`` when the output
        cannot be decoded back to float32 (e.g. packed binary).
    extra : Dict[str, Any]
        Arbitrary key-value metadata for application-level context.
    """

    stage_name: str
    stage_index: int
    latency_ms: float
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    input_dtype: str
    output_dtype: str
    compression_ratio: float
    throughput_vecs_per_sec: float
    cosine_fidelity: Optional[float]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of all fields."""
        d = asdict(self)
        # tuples are not JSON-native; coerce to lists
        d["input_shape"] = list(self.input_shape)
        d["output_shape"] = list(self.output_shape)
        return d

    def to_json(self) -> str:
        """Serialise this event to a compact JSON string."""
        return json.dumps(self.to_dict(), separators=(",", ":"))


# ---------------------------------------------------------------------------
# Hook protocol & collector
# ---------------------------------------------------------------------------

# A hook is any callable that accepts a single TelemetryEvent.
TelemetryHook = Callable[[TelemetryEvent], None]


class TelemetryCollector:
    """Manages a set of :data:`TelemetryHook` callbacks.

    Attach hooks with :meth:`attach`, detach with :meth:`detach`, and
    fan-out to all registered hooks by calling :meth:`emit`.

    Usage example::

        collector = TelemetryCollector()
        collector.attach(lambda ev: print(ev.stage_name, ev.latency_ms))
        # … run a pipeline …
        collector.emit(event)
    """

    def __init__(self) -> None:
        self._hooks: List[TelemetryHook] = []

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def attach(self, hook: TelemetryHook) -> None:
        """Register *hook* to receive future events."""
        if hook not in self._hooks:
            self._hooks.append(hook)

    def detach(self, hook: TelemetryHook) -> None:
        """Unregister *hook*.  No-op if the hook was not registered."""
        try:
            self._hooks.remove(hook)
        except ValueError:
            pass

    def clear(self) -> None:
        """Remove all registered hooks."""
        self._hooks.clear()

    @property
    def hook_count(self) -> int:
        """Number of currently registered hooks."""
        return len(self._hooks)

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def emit(self, event: TelemetryEvent) -> None:
        """Call every registered hook with *event* in registration order."""
        for hook in list(self._hooks):  # snapshot to tolerate mid-loop detach
            hook(event)


class InMemoryTelemetryCollector(TelemetryCollector):
    """A :class:`TelemetryCollector` that stores every emitted event.

    Useful in tests and interactive exploration::

        collector = InMemoryTelemetryCollector()
        attach_telemetry(pipeline, collector)
        pipeline.run(vectors)
        print(collector.events[0].latency_ms)
    """

    def __init__(self) -> None:
        super().__init__()
        self._events: List[TelemetryEvent] = []
        # Auto-register the internal store hook
        self.attach(self._store)

    def _store(self, event: TelemetryEvent) -> None:
        self._events.append(event)

    @property
    def events(self) -> List[TelemetryEvent]:
        """Read-only view of all collected events."""
        return list(self._events)

    def clear_events(self) -> None:
        """Discard all stored events (hooks are preserved)."""
        self._events.clear()

    def export_json(self) -> str:
        """Export all stored events as a JSON array string."""
        return json.dumps([e.to_dict() for e in self._events], separators=(",", ":"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_fidelity(original: np.ndarray, reconstructed: np.ndarray) -> Optional[float]:
    """Return mean cosine similarity between *original* and *reconstructed*.

    Both arrays must be 2-D (n_vectors, dim) float32.  Returns ``None`` when
    either array is degenerate (zero-norm rows), to avoid division-by-zero.
    """
    orig_f32 = original.astype(np.float32, copy=False)
    recon_f32 = reconstructed.astype(np.float32, copy=False)
    if orig_f32.ndim == 1:
        orig_f32 = orig_f32[np.newaxis, :]
    if recon_f32.ndim == 1:
        recon_f32 = recon_f32[np.newaxis, :]
    if orig_f32.shape != recon_f32.shape:
        return None
    # Per-row L2 norms — accumulate in FP32
    orig_norms = np.linalg.norm(orig_f32, axis=1)  # (n,)
    recon_norms = np.linalg.norm(recon_f32, axis=1)  # (n,)
    valid = (orig_norms > 0) & (recon_norms > 0)
    if not valid.any():
        return None
    dots = np.einsum("ij,ij->i", orig_f32[valid], recon_f32[valid])  # FP32 accumulation
    cosines = dots / (orig_norms[valid] * recon_norms[valid])
    return float(np.mean(np.clip(cosines, -1.0, 1.0)))


def _decode_stage_output(stage_output: Any, vectro: Any) -> np.ndarray:
    """Decode a compress() result back to a float32 ndarray.

    Vectro.compress() returns BatchQuantizationResult or QuantizationResult
    rather than a plain numpy array.  This helper reconstructs the float32
    vectors so they can be used for cosine-fidelity comparison and as input
    to the next pipeline stage.
    """
    # BatchQuantizationResult — has reconstruct_batch()
    if hasattr(stage_output, "reconstruct_batch") and callable(stage_output.reconstruct_batch):
        try:
            decoded = stage_output.reconstruct_batch()
            return np.asarray(decoded, dtype=np.float32)
        except Exception:
            pass

    # QuantizationResult — use vectro.decompress()
    if (
        hasattr(stage_output, "quantized")
        and hasattr(stage_output, "scales")
        and hasattr(vectro, "decompress")
    ):
        try:
            decoded = vectro.decompress(stage_output)
            arr = np.asarray(decoded, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            return arr
        except Exception:
            pass

    # Plain numpy array — just ensure float32
    if hasattr(stage_output, "astype"):
        try:
            return stage_output.astype(np.float32)
        except (ValueError, TypeError):
            pass

    return stage_output


# ---------------------------------------------------------------------------
# attach_telemetry()
# ---------------------------------------------------------------------------


def attach_telemetry(
    pipeline: Any,  # CompressionPipeline — avoid circular import
    collector: Optional[TelemetryCollector] = None,
    *,
    measure_cosine_fidelity: bool = True,
) -> TelemetryCollector:
    """Wrap *pipeline*'s ``run()`` to emit :class:`TelemetryEvent` objects.

    The original ``run()`` is replaced with a thin wrapper that calls the
    original, measures per-stage metrics, and fans them out through
    *collector*.  The wrapper is transparent: it returns the same
    ``(PipelineResult, output_array)`` tuple as the original.

    Parameters
    ----------
    pipeline :
        A :class:`~python.async_pipeline.CompressionPipeline` instance.
    collector :
        The collector to emit events to.  When ``None``, a fresh
        :class:`InMemoryTelemetryCollector` is created and returned.
    measure_cosine_fidelity :
        When ``True`` (default), the wrapper dequantizes each stage's
        output back to float32 and computes cosine fidelity against the
        stage's float32 input.  Set to ``False`` to skip this and reduce
        overhead.

    Returns
    -------
    TelemetryCollector
        The *collector* that will receive events (newly created when the
        caller passed ``None``).
    """
    if collector is None:
        collector = InMemoryTelemetryCollector()

    # Preserve the original run() so we can call it
    _original_run = pipeline.run

    def _instrumented_run(vectors: np.ndarray):  # type: ignore[override]
        """Instrumented replacement for CompressionPipeline.run()."""
        import time as _time

        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")

        vectro = pipeline._get_vectro()
        current = vectors.astype(np.float32)
        input_bytes_total = current.nbytes
        stage_latencies: List[float] = []
        stage_names: List[str] = []

        for idx, stage in enumerate(pipeline.stages):
            stage_input = current.copy()  # snapshot for fidelity comparison
            input_shape = current.shape
            input_bytes = current.nbytes

            kwargs: dict = {}
            if stage.profile is not None:
                kwargs["profile"] = stage.profile
            if stage.group_size is not None:
                kwargs["group_size"] = stage.group_size

            t0 = _time.perf_counter()
            raw_result = vectro.compress(current, precision_mode=stage.mode, **kwargs)
            elapsed_ms = (_time.perf_counter() - t0) * 1000.0

            stage_latencies.append(elapsed_ms)
            stage_names.append(stage.mode)

            # Unwrap (result, metadata) tuples produced by some modes
            if isinstance(raw_result, tuple):
                stage_output = raw_result[0]
            else:
                stage_output = raw_result

            # Decode the quantization result back to a float32 ndarray so we
            # can compute cosine fidelity and feed the next stage correctly.
            decoded = _decode_stage_output(stage_output, vectro)

            output_shape = decoded.shape if hasattr(decoded, "shape") else (0,)
            output_bytes = decoded.nbytes if hasattr(decoded, "nbytes") else input_bytes
            output_dtype = str(decoded.dtype) if hasattr(decoded, "dtype") else "unknown"

            # Cosine fidelity
            fidelity: Optional[float] = None
            if (
                measure_cosine_fidelity
                and hasattr(decoded, "shape")
                and decoded.shape == stage_input.shape
            ):
                fidelity = _cosine_fidelity(stage_input, decoded)

            # Throughput
            n_vecs = input_shape[0] if input_shape else 1
            latency_s = elapsed_ms / 1000.0
            throughput = float(n_vecs) / latency_s if latency_s > 0 else 0.0

            event = TelemetryEvent(
                stage_name=stage.mode,
                stage_index=idx,
                latency_ms=elapsed_ms,
                input_shape=tuple(input_shape),
                output_shape=tuple(output_shape),
                input_dtype=str(stage_input.dtype),
                output_dtype=output_dtype,
                compression_ratio=float(input_bytes) / float(max(output_bytes, 1)),
                throughput_vecs_per_sec=throughput,
                cosine_fidelity=fidelity,
            )
            collector.emit(event)
            current = decoded

        # Build PipelineResult to stay compatible with the original return type
        from .async_pipeline import PipelineResult

        total_output_bytes = current.nbytes if hasattr(current, "nbytes") else input_bytes_total
        pr = PipelineResult(
            stages=stage_names,
            input_shape=vectors.shape,
            output_shape=current.shape if hasattr(current, "shape") else (0,),
            input_dtype=str(vectors.dtype),
            output_dtype=str(current.dtype) if hasattr(current, "dtype") else "unknown",
            total_latency_ms=sum(stage_latencies),
            stage_latencies_ms=stage_latencies,
            compression_ratio=float(input_bytes_total) / float(max(total_output_bytes, 1)),
        )
        return pr, current

    # Monkey-patch the pipeline instance (not the class)
    pipeline.run = _instrumented_run
    return collector
