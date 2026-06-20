"""Tests for telemetry — python/telemetry.py (v5.3.0).

Covers:
- TelemetryEvent construction and serialisation
- TelemetryCollector attach / detach / clear / emit
- InMemoryTelemetryCollector stores events and exports JSON
- attach_telemetry() integrates with CompressionPipeline
- cosine fidelity gate (>= 0.9999 on unit-vector inputs)
- throughput positivity
- multi-stage event ordering
- JSON round-trip
"""

from __future__ import annotations

import json
import unittest

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.telemetry import (
    TelemetryEvent,
    TelemetryCollector,
    InMemoryTelemetryCollector,
    attach_telemetry,
)
from python.async_pipeline import CompressionPipeline, PipelineStage
from python.vectro import Vectro


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_event(**overrides) -> TelemetryEvent:
    defaults = dict(
        stage_name="int8",
        stage_index=0,
        latency_ms=1.5,
        input_shape=(10, 64),
        output_shape=(10, 64),
        input_dtype="float32",
        output_dtype="float32",
        compression_ratio=4.0,
        throughput_vecs_per_sec=6666.0,
        cosine_fidelity=0.9999,
    )
    defaults.update(overrides)
    return TelemetryEvent(**defaults)


# ---------------------------------------------------------------------------
# TelemetryEvent
# ---------------------------------------------------------------------------


class TestTelemetryEvent(unittest.TestCase):
    def test_construction(self):
        ev = _make_event()
        self.assertEqual(ev.stage_name, "int8")
        self.assertEqual(ev.stage_index, 0)
        self.assertAlmostEqual(ev.latency_ms, 1.5)

    def test_to_dict_keys(self):
        ev = _make_event()
        d = ev.to_dict()
        expected = {
            "stage_name",
            "stage_index",
            "latency_ms",
            "input_shape",
            "output_shape",
            "input_dtype",
            "output_dtype",
            "compression_ratio",
            "throughput_vecs_per_sec",
            "cosine_fidelity",
            "extra",
        }
        self.assertEqual(set(d.keys()), expected)

    def test_to_dict_shapes_are_lists(self):
        ev = _make_event()
        d = ev.to_dict()
        self.assertIsInstance(d["input_shape"], list)
        self.assertIsInstance(d["output_shape"], list)

    def test_to_json_valid(self):
        ev = _make_event()
        parsed = json.loads(ev.to_json())
        self.assertEqual(parsed["stage_name"], "int8")

    def test_json_round_trip(self):
        ev = _make_event(cosine_fidelity=None)
        parsed = json.loads(ev.to_json())
        self.assertIsNone(parsed["cosine_fidelity"])

    def test_immutable(self):
        ev = _make_event()
        with self.assertRaises(Exception):
            ev.stage_name = "nf4"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TelemetryCollector
# ---------------------------------------------------------------------------


class TestTelemetryCollector(unittest.TestCase):
    def test_attach_and_emit(self):
        received = []
        collector = TelemetryCollector()
        collector.attach(received.append)
        collector.emit(_make_event())
        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], TelemetryEvent)

    def test_detach(self):
        received = []
        collector = TelemetryCollector()
        collector.attach(received.append)
        collector.detach(received.append)
        collector.emit(_make_event())
        self.assertEqual(len(received), 0)

    def test_detach_unknown_is_noop(self):
        collector = TelemetryCollector()
        collector.detach(lambda e: None)  # should not raise

    def test_clear(self):
        received = []
        collector = TelemetryCollector()
        collector.attach(received.append)
        collector.clear()
        collector.emit(_make_event())
        self.assertEqual(len(received), 0)

    def test_hook_count(self):
        collector = TelemetryCollector()
        self.assertEqual(collector.hook_count, 0)
        collector.attach(lambda e: None)
        self.assertEqual(collector.hook_count, 1)

    def test_duplicate_attach_ignored(self):
        hook = lambda e: None
        collector = TelemetryCollector()
        collector.attach(hook)
        collector.attach(hook)
        self.assertEqual(collector.hook_count, 1)


# ---------------------------------------------------------------------------
# InMemoryTelemetryCollector
# ---------------------------------------------------------------------------


class TestInMemoryCollector(unittest.TestCase):
    def test_stores_events(self):
        collector = InMemoryTelemetryCollector()
        collector.emit(_make_event(stage_name="int8"))
        collector.emit(_make_event(stage_name="nf4"))
        self.assertEqual(len(collector.events), 2)

    def test_events_order(self):
        collector = InMemoryTelemetryCollector()
        collector.emit(_make_event(stage_index=0))
        collector.emit(_make_event(stage_index=1))
        self.assertEqual(collector.events[0].stage_index, 0)
        self.assertEqual(collector.events[1].stage_index, 1)

    def test_export_json_is_array(self):
        collector = InMemoryTelemetryCollector()
        collector.emit(_make_event())
        parsed = json.loads(collector.export_json())
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 1)

    def test_clear_events(self):
        collector = InMemoryTelemetryCollector()
        collector.emit(_make_event())
        collector.clear_events()
        self.assertEqual(len(collector.events), 0)


# ---------------------------------------------------------------------------
# attach_telemetry() + pipeline integration
# ---------------------------------------------------------------------------


class TestAttachTelemetry(unittest.TestCase):
    def _make_pipeline(self, modes=None):
        modes = modes or ["int8"]
        stages = [PipelineStage(mode=m) for m in modes]
        return CompressionPipeline(stages, vectro=Vectro())

    def test_returns_collector(self):
        pipeline = self._make_pipeline()
        collector = attach_telemetry(pipeline)
        self.assertIsInstance(collector, InMemoryTelemetryCollector)

    def test_accepts_custom_collector(self):
        pipeline = self._make_pipeline()
        custom = InMemoryTelemetryCollector()
        returned = attach_telemetry(pipeline, custom)
        self.assertIs(returned, custom)

    def test_event_emitted_on_run(self):
        pipeline = self._make_pipeline()
        collector = attach_telemetry(pipeline)
        vectors = RNG.standard_normal((10, 64)).astype(np.float32)
        pipeline.run(vectors)
        self.assertEqual(len(collector.events), 1)

    def test_event_stage_name(self):
        pipeline = self._make_pipeline(["int8"])
        collector = attach_telemetry(pipeline)
        vectors = RNG.standard_normal((10, 64)).astype(np.float32)
        pipeline.run(vectors)
        self.assertEqual(collector.events[0].stage_name, "int8")

    def test_throughput_positive(self):
        pipeline = self._make_pipeline()
        collector = attach_telemetry(pipeline)
        vectors = RNG.standard_normal((50, 64)).astype(np.float32)
        pipeline.run(vectors)
        self.assertGreater(collector.events[0].throughput_vecs_per_sec, 0.0)

    def test_compression_ratio_positive(self):
        pipeline = self._make_pipeline()
        collector = attach_telemetry(pipeline)
        vectors = RNG.standard_normal((10, 64)).astype(np.float32)
        pipeline.run(vectors)
        self.assertGreater(collector.events[0].compression_ratio, 0.0)

    def test_cosine_fidelity_int8_unit_vectors(self):
        """Cosine fidelity must be >= 0.9999 on unit-vector inputs."""
        pipeline = self._make_pipeline(["int8"])
        collector = attach_telemetry(pipeline, measure_cosine_fidelity=True)
        # Unit vectors: each row normalised to L2=1
        raw = RNG.standard_normal((100, 128)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        unit_vecs = raw / norms
        pipeline.run(unit_vecs)
        fidelity = collector.events[0].cosine_fidelity
        self.assertIsNotNone(fidelity)
        self.assertGreaterEqual(
            fidelity,
            0.9999,
            f"INT8 cosine fidelity {fidelity:.6f} < 0.9999 on unit vectors",
        )

    def test_multi_stage_event_count(self):
        pipeline = self._make_pipeline(["int8", "int8"])
        collector = attach_telemetry(pipeline)
        vectors = RNG.standard_normal((10, 64)).astype(np.float32)
        pipeline.run(vectors)
        self.assertEqual(len(collector.events), 2)

    def test_multi_stage_indices(self):
        pipeline = self._make_pipeline(["int8", "int8"])
        collector = attach_telemetry(pipeline)
        vectors = RNG.standard_normal((10, 64)).astype(np.float32)
        pipeline.run(vectors)
        self.assertEqual(collector.events[0].stage_index, 0)
        self.assertEqual(collector.events[1].stage_index, 1)

    def test_latency_nonneg(self):
        pipeline = self._make_pipeline()
        collector = attach_telemetry(pipeline)
        pipeline.run(RNG.standard_normal((10, 64)).astype(np.float32))
        self.assertGreaterEqual(collector.events[0].latency_ms, 0.0)

    def test_run_returns_pipeline_result(self):
        from python.async_pipeline import PipelineResult

        pipeline = self._make_pipeline()
        attach_telemetry(pipeline)
        pr, out = pipeline.run(RNG.standard_normal((10, 64)).astype(np.float32))
        self.assertIsInstance(pr, PipelineResult)


if __name__ == "__main__":
    unittest.main()
