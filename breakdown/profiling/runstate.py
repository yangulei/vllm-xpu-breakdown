# SPDX-License-Identifier: Apache-2.0
"""What the server remembers about profiling: the run, and the model configs.

The run state is a :class:`breakdown.runs.RunState`, persisted to
``output/profile/<run_id>/state.json``, because everything downstream -- the
shape matrix, the benchmark, the optimizer -- is derived from a profile, and
losing it to a server restart meant profiling again: minutes on a real model.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from breakdown import runs
from breakdown.trace_common import _detect_device_via_torch

logger = logging.getLogger("vllm_xpu_breakdown")


#: The accelerator this host has. Cached at import: it cannot change while the
#: process runs.
DEVICE = _detect_device_via_torch() or "xpu"

#: The stage name the profile's runs are stored under (see :mod:`breakdown.runs`).
STAGE = "profile"

# ---- Config Cache ----
# Persists successfully loaded model configs to disk so they appear as suggestions.

_CONFIG_CACHE_DIR = Path(__file__).parent / "output" / "config_cache"


_config_cache_lock = threading.Lock()


def _cache_key(model_id: str) -> str:
    """Convert model_id to a safe filename."""
    return model_id.replace("/", "__")


def _save_config_cache(model_id: str, config: dict[str, Any]) -> None:
    """Persist config.json to disk cache."""
    key = _cache_key(model_id)
    path = _CONFIG_CACHE_DIR / f"{key}.json"
    with _config_cache_lock:
        path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")


def _load_cached_model_ids() -> list[str]:
    """Return list of model IDs that have been successfully cached."""
    ids: list[str] = []
    with _config_cache_lock:
        for p in sorted(_CONFIG_CACHE_DIR.glob("*.json")):
            ids.append(p.stem.replace("__", "/"))
    return ids


def _load_cached_config(model_id: str) -> dict[str, Any] | None:
    """Load a cached config from disk, or None if not cached."""
    key = _cache_key(model_id)
    path = _CONFIG_CACHE_DIR / f"{key}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


#: The current (or last) profiling run. A profile is the input to every later
#: stage, so it is also **persisted** to ``output/profile/<run_id>/state.json``
#: (:func:`save_state`): a server restart, or a second browser tab, used to lose
#: the run everything downstream is derived from, and the only way back was to
#: profile again - minutes on a real model.
_profile_state = runs.RunState(result=None, model_id=None, settings=None)

#: Kept as a name because the routes take it around the state they mutate.
_profile_lock = _profile_state.lock


def state() -> dict[str, Any]:
    """The current profiling state (restored from disk on first use)."""
    with _profile_lock:
        if _profile_state["status"] == "idle":
            _restore_latest()
        return _profile_state


def begin(model_id: str, settings: dict[str, Any]) -> str:
    """Mark a run as started and return its id."""
    return _profile_state.begin(runs.new_run_id(model_id.split("/")[-1]),
                                model_id=model_id, settings=settings)


def save_state() -> None:
    """Persist the current state, so it outlives this process."""
    run_id = _profile_state.get("run_id")
    if not run_id:
        return
    try:
        runs.write_state(STAGE, run_id, _profile_state)
    except OSError:
        logger.warning("could not persist the profile run", exc_info=True)


def _restore_latest() -> None:
    """Adopt the newest completed run on disk, if there is one."""
    found = runs.latest_state(STAGE)
    if not found:
        return
    run_id, saved = found
    if saved.get("status") != "done" or not saved.get("result"):
        return
    _profile_state.update(saved)
    _profile_state["run_id"] = run_id
    logger.info("restored profile run %s (%s)", run_id, saved.get("model_id"))

def _norm_quant(q: object) -> str | None:
    """Normalize a quantization selection: "", "auto", "none" → None."""
    if not q or str(q).lower() in ("auto", "none"):
        return None
    return str(q).lower()


def _profile_template_for(model_id: str, quantization: object = None
                          ) -> tuple[dict, dict | None, str | None]:
    """The latest completed profile graph, validated against a request.

    Returns ``(template, profile_settings, error)``; ``error`` is a
    user-facing message when the state cannot serve this model/quantization.
    Shared by the Shape Matrix export and the ``/api/perf/*`` pipeline.
    """
    with _profile_lock:
        state_status = _profile_state["status"]
        state_model = _profile_state.get("model_id")
        state_result = _profile_state.get("result")
        profile_settings = _profile_state.get("settings")
    if state_status != "done" or not state_result:
        return {}, None, (
            "The Shape Matrix is derived from a profiling run, but no "
            "completed run is available. Run a profile first.")
    template = state_result.get("graph")
    if not template or not (template.get("prefill") or template.get("decode")):
        return {}, None, ("The latest profile has no reconstructed graph to "
                          "derive shapes from.")
    if state_model and state_model != model_id:
        return {}, None, (f"Latest profile is for '{state_model}', not "
                          f"'{model_id}'. Profile that model or switch the "
                          "model ID.")

    # The derived shapes/dtypes/memory are only valid for the quantization the
    # run actually used, so the requested quantization must match the profiled
    # one.
    requested_quant = _norm_quant(quantization)
    profiled_quant = _norm_quant(
        (profile_settings or {}).get("quantization")
        if profile_settings else
        template.get("config", {}).get("quantization")
    )
    if requested_quant != profiled_quant:
        return {}, None, (
            f"Latest profile used quantization '{profiled_quant or 'none'}', "
            f"not '{requested_quant or 'none'}'. Re-profile with the requested "
            "quantization or change the selection.")
    return template, profile_settings, None


def is_running() -> bool:
    with _profile_lock:
        return _profile_state["status"] == "running"
