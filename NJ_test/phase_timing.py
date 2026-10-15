"""NJ_test/phase_timing.py

Non-invasive per-phase timers for OCSMesh's HfunCollector.

Wraps selected private methods with perf_counter timers (no OCSMesh
changes needed) so summary.json can split ``meshdata_time_s`` into
detection vs. application vs. dispatch.

Times are INCLUSIVE and can nest, so do not sum them:

    apply_features                      everything below
      apply_contours / apply_channels   detection + dispatch + integration
        contour_detect / channel_detect shape extraction (rank 0, serial)
        dispatch_<kind>                 distributed apply (Pool / MPI)
      apply_flow_limiters, apply_const_val, apply_linefeatures,
      apply_patch, apply_constraints

Usage: call ``install()`` once after ``import ocsmesh``, then
``snapshot()`` to get ``{name: {"seconds": s, "calls": n}}``.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Dict

_logger = logging.getLogger("nj_mesh.phase_timing")
_TIMES: Dict[str, Dict[str, float]] = {}
_INSTALLED = False

# (class name in ocsmesh.hfun.collector, attribute, label)
_TARGETS = [
    ("HfunCollector", "_apply_features",      "apply_features"),
    ("HfunCollector", "_apply_features_fast", "apply_features_fast"),
    ("HfunCollector", "_apply_contours",      "apply_contours"),
    ("HfunCollector", "_apply_flow_limiters", "apply_flow_limiters"),
    ("HfunCollector", "_apply_const_val",     "apply_const_val"),
    ("HfunCollector", "_apply_linefeatures",  "apply_linefeatures"),
    ("HfunCollector", "_apply_patch",         "apply_patch"),
    ("HfunCollector", "_apply_channels",      "apply_channels"),
    ("HfunCollector", "_apply_constraints",   "apply_constraints"),
    ("HfunCollector", "_dispatch_refinement_tasks", "dispatch_refinement"),
    ("_RefinementContourCollector", "calculate", "contour_detect"),
    ("_ChannelRefineCollector",     "calculate", "channel_detect"),
]


def _wrap(fn, label):
    @functools.wraps(fn)
    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            rec = _TIMES.setdefault(label, {"seconds": 0.0, "calls": 0})
            rec["seconds"] += time.perf_counter() - t0
            rec["calls"] += 1
    return timed


def _wrap_dispatch(fn):
    """dispatch_refinement receives ``kind`` as 2nd positional arg."""
    @functools.wraps(fn)
    def timed(self, kind, *args, **kwargs):
        label = f"dispatch_{kind}"
        t0 = time.perf_counter()
        try:
            return fn(self, kind, *args, **kwargs)
        finally:
            rec = _TIMES.setdefault(label, {"seconds": 0.0, "calls": 0})
            rec["seconds"] += time.perf_counter() - t0
            rec["calls"] += 1
    return timed


def install() -> None:
    """Patch the collector classes (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return
    from ocsmesh.hfun import collector as mod

    for cls_name, attr, label in _TARGETS:
        cls = getattr(mod, cls_name, None)
        fn = getattr(cls, attr, None) if cls is not None else None
        if fn is None:
            _logger.warning(f"phase timing: {cls_name}.{attr} not found")
            continue
        wrapped = (_wrap_dispatch(fn) if label == "dispatch_refinement"
                   else _wrap(fn, label))
        setattr(cls, attr, wrapped)
    _INSTALLED = True


def snapshot() -> Dict[str, Dict[str, float]]:
    return {
        k: {"seconds": round(v["seconds"], 2), "calls": int(v["calls"])}
        for k, v in sorted(_TIMES.items())
    }
