"""Shared helpers for the analyze.py / plot.py CLI dispatchers."""
import json
import math
from pathlib import Path

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy arrays and scalars to plain Python.

    Non-finite floats (nan/inf/-inf) are converted to JSON ``null`` instead
    of the bare ``NaN``/``Infinity`` tokens the stdlib encoder would
    otherwise emit for them -- those tokens are not valid JSON per RFC 8259
    and are rejected by strict parsers (e.g. ``jq``, JS ``JSON.parse``).
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                # Object-dtype arrays hold arbitrary Python objects (.tolist()
                # does no conversion here); sanitize each element so a stray
                # numpy scalar or unserializable object doesn't leave behind
                # invalid JSON or abort the dump.
                return [_sanitize(v) for v in obj.tolist()]
            return _sanitize(obj.tolist())
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return _finite_or_none(float(obj))
        if isinstance(obj, np.complexfloating):
            # JSON has no native complex type; represent as a real/imag pair.
            return {
                "real": _finite_or_none(float(obj.real)),
                "imag": _finite_or_none(float(obj.imag)),
            }
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _finite_or_none(value):
    """Map non-finite floats (nan/inf/-inf) to None; finite values pass through."""
    return value if math.isfinite(value) else None


def _sanitize_key(key):
    """Convert a dict key to something JSON can use as an object key.

    JSON object keys must be str/int/float/bool/None; the stdlib encoder
    stringifies int/float/bool/None keys for us, but raises TypeError on a
    numpy-typed key (e.g. np.int64) and ValueError on a non-finite float key
    once ``allow_nan=False``. Convert numpy scalars to their native Python
    type first, then fall back to ``str()`` for anything that still isn't a
    valid key type (non-finite floats, complex, ...).
    """
    if isinstance(key, np.generic):
        key = key.item()
    if key is None or isinstance(key, (str, int, bool)):
        return key
    if isinstance(key, float):
        return key if math.isfinite(key) else str(key)
    return str(key)


def _sanitize(obj):
    """Recursively replace non-finite floats (and stray numpy/unserializable
    values) ahead of json.dump.

    This exists because ``np.float64`` -- and plain Python ``float`` -- are
    native JSON types as far as the stdlib encoder is concerned, so
    ``_NumpyEncoder.default()`` is never called for them: the encoder would
    otherwise serialize nan/inf directly as the invalid ``NaN``/``Infinity``
    tokens, with or without ``allow_nan``. Walking the structure ourselves
    replaces those values with ``None`` up front, before ``write_json``'s
    ``allow_nan=False`` would ever get a chance to raise on them.
    """
    if isinstance(obj, float):
        return _finite_or_none(obj)
    if isinstance(obj, complex):
        # Covers plain Python complex (e.g. what .tolist() yields for a
        # complex-dtype ndarray) as well as np.complex128, which subclasses
        # complex -- keep this in sync with the {real, imag} form
        # _NumpyEncoder.default() uses for other numpy complex scalars, so
        # complex arrays and complex scalars serialize identically.
        return {
            "real": _finite_or_none(float(obj.real)),
            "imag": _finite_or_none(float(obj.imag)),
        }
    if isinstance(obj, dict):
        return {_sanitize_key(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        # Any remaining numpy scalar (np.float32, np.complex128, ...): reuse
        # _NumpyEncoder's conversions, then sanitize the result the same way.
        try:
            return _sanitize(_NumpyEncoder().default(obj))
        except TypeError:
            return str(obj)
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def write_json(data, path: str) -> None:
    """Serialize *data* to *path*, creating parent directories as needed."""
    out = Path(path)
    if out.suffix.lower() != ".json":
        raise ValueError(f"Output file must end in .json; got: {path}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            _sanitize(data), f, cls=_NumpyEncoder, indent=2, allow_nan=False
        )


def read_json(path: str) -> dict:
    """Load a JSON file produced by analyze.py."""
    with open(path) as f:
        return json.load(f)


def validate_checkpoint(path: str) -> Path:
    """Return resolved Path or raise FileNotFoundError."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return p


def validate_figure_output(path: str) -> Path:
    """Verify extension is .png / .svg / .pdf; create parent dirs."""
    p = Path(path)
    if p.suffix.lower() not in {".png", ".svg", ".pdf"}:
        raise ValueError(
            f"Output figure must end in .png, .svg, or .pdf; got: {path}"
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
