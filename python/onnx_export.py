"""ONNX export for the INT8 dequantization graph (opset 17).

Produces a portable ONNX model that reproduces the NumPy dequantization path
from :func:`~python.interface.reconstruct_embeddings`:

.. code-block:: text

    Cast(quantized INT8 → FLOAT)           → quantized_f32     [N, D]
    Unsqueeze(scales FLOAT, axes=[1])       → scales_2d         [N, 1]
    Mul(quantized_f32, scales_2d)           → reconstructed     [N, D]

The graph uses opset 17 and keeps dimensions dynamic (N is a free axis).

Usage::

    result = Vectro().compress(embeddings, profile="balanced")

    # Build in-memory model
    model = to_onnx_model(result)

    # Write to disk
    export_onnx(result, "dequant.onnx")

Runtime inference example (requires onnxruntime)::

    import onnxruntime as ort
    sess = ort.InferenceSession("dequant.onnx")
    restored = sess.run(
        None,
        {
            "quantized": result.quantized,
            "scales":    result.scales,
        },
    )[0]
"""

from __future__ import annotations


# Lazy import so users without onnx installed can still import vectro.
try:
    import onnx  # type: ignore[import]
    import onnx.helper as _oh  # type: ignore[import]

    _HAVE_ONNX = True
except ImportError:
    _HAVE_ONNX = False

_OPSET_VERSION = 17
_OPSET_DOMAIN = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_onnx_model(result: object) -> "onnx.ModelProto":  # type: ignore[name-defined]
    """Build an ONNX ModelProto for the INT8 dequantization graph.

    Parameters
    ----------
    result:
        A ``QuantizationResult`` or compatible object (must have ``.quantized``,
        ``.scales``, and optionally ``.precision_mode``).  Only ``int8`` results
        are supported; ``int4`` raises :class:`ValueError`.

    Returns
    -------
    onnx.ModelProto
        In-memory ONNX model.  Call ``model.SerializeToString()`` to get bytes,
        or pass the model directly to :func:`export_onnx`.

    Raises
    ------
    RuntimeError
        If the ``onnx`` package is not installed.
    ValueError
        If *result* has ``precision_mode == "int4"``.
    """
    if not _HAVE_ONNX:
        raise RuntimeError(
            "onnx is required for ONNX export. Install with: pip install 'onnx>=1.14'"
        )

    precision_mode = getattr(result, "precision_mode", "int8")
    if precision_mode == "int4":
        raise ValueError(
            "INT4 results are not supported by the ONNX exporter. Pass an int8 result instead."
        )

    # ------------------------------------------------------------------
    # Graph inputs
    # ------------------------------------------------------------------
    # quantized: INT8   [N, D]
    # scales:    FLOAT  [N]
    quantized_input = _oh.make_tensor_value_info("quantized", onnx.TensorProto.INT8, ["N", "D"])
    scales_input = _oh.make_tensor_value_info("scales", onnx.TensorProto.FLOAT, ["N"])

    # ------------------------------------------------------------------
    # Graph output
    # ------------------------------------------------------------------
    # reconstructed: FLOAT [N, D]
    reconstructed_output = _oh.make_tensor_value_info(
        "reconstructed", onnx.TensorProto.FLOAT, ["N", "D"]
    )

    # ------------------------------------------------------------------
    # axes initialiser — Unsqueeze in opset 13+ requires axes as a tensor
    # rather than an attribute.
    # ------------------------------------------------------------------
    axes_initializer = _oh.make_tensor(
        name="axes_1",
        data_type=onnx.TensorProto.INT64,
        dims=[1],
        vals=[1],
    )

    # ------------------------------------------------------------------
    # Nodes:  Cast → Unsqueeze → Mul
    # ------------------------------------------------------------------
    cast_node = _oh.make_node(
        "Cast",
        inputs=["quantized"],
        outputs=["quantized_f32"],
        to=onnx.TensorProto.FLOAT,
    )

    unsqueeze_node = _oh.make_node(
        "Unsqueeze",
        inputs=["scales", "axes_1"],
        outputs=["scales_2d"],
    )

    mul_node = _oh.make_node(
        "Mul",
        inputs=["quantized_f32", "scales_2d"],
        outputs=["reconstructed"],
    )

    # ------------------------------------------------------------------
    # Assemble graph → model
    # ------------------------------------------------------------------
    graph = _oh.make_graph(
        nodes=[cast_node, unsqueeze_node, mul_node],
        name="vectro_int8_dequant",
        inputs=[quantized_input, scales_input],
        outputs=[reconstructed_output],
        initializer=[axes_initializer],
    )

    model = _oh.make_model(
        graph,
        opset_imports=[_oh.make_opsetid(_OPSET_DOMAIN, _OPSET_VERSION)],
        ir_version=8,
    )
    model.doc_string = (
        "Vectro INT8 dequantization — opset 17. "
        "Inputs: quantized (INT8 [N,D]), scales (FLOAT [N]). "
        "Output: reconstructed (FLOAT [N,D])."
    )

    onnx.checker.check_model(model)
    return model


def export_onnx(result: object, path: str) -> None:
    """Serialize the ONNX dequantization graph to *path*.

    Parameters
    ----------
    result:
        A ``QuantizationResult`` (or compatible) from Vectro.
    path:
        Destination file path, conventionally ending in ``.onnx``.

    Raises
    ------
    RuntimeError
        If the ``onnx`` package is not installed.
    ValueError
        If *result* has ``precision_mode == "int4"``.
    """
    model = to_onnx_model(result)
    with open(path, "wb") as fh:
        fh.write(model.SerializeToString())
