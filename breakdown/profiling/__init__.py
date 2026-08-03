# SPDX-License-Identifier: Apache-2.0
"""Profiling: run vLLM once, and turn the traces into a reconstructed graph.

This was one 1200-line module. It is four, along the seams the work already
had:

===============  =======================================================
:mod:`.runstate`    what the server remembers -- the run, and model configs
:mod:`.launch`   running vLLM under the profiler
:mod:`.traces`   trace files -> one reconstructed result
:mod:`.uploads`  the same, from files a browser sent
===============  =======================================================

The public surface is re-exported here, so ``breakdown.profiling.<name>``
means what it always did for the routes, the CLI and the tools.
"""
from __future__ import annotations

from breakdown.profiling import runstate  # noqa: F401
from breakdown.profiling.runstate import (  # noqa: F401
    DEVICE,
    STAGE,
    _CONFIG_CACHE_DIR,
    _load_cached_config,
    _load_cached_model_ids,
    _norm_quant,
    _profile_template_for,
    _restore_latest,
    _save_config_cache,
    begin,
    is_running,
    save_state,
    state,
)
from breakdown.profiling.traces import (  # noqa: F401
    _build_result_from_traces,
    _merge_two_pass_result,
    _parse_trace_filename,
    _rank0_first,
    _trace_rank,
    trace_download_name,
    trace_path_for,
)
from breakdown.profiling.launch import (  # noqa: F401
    _build_layer_override,
    _get_block_size,
    _get_vocab_size,
    _make_token_ids,
    _run_profile,
    _scheduler_pin,
    _set_num_hidden_layers,
    fit_max_model_len,
)
from breakdown.profiling.uploads import (  # noqa: F401
    build_from_uploads,
    save_uploads,
)


#: The mutable run state is *not* bound here.
#:
#: ``breakdown.profiling._profile_state`` must always be whatever
#: :mod:`.runstate` currently holds. Re-exporting it by value would bind the
#: object that existed at import time, so a caller that swaps the state --
#: which the tests do, to stage a finished profile without running one --
#: would be swapping something the rest of the package no longer reads. PEP
#: 562 module ``__getattr__`` forwards the lookup instead, which is only
#: consulted when the normal one fails, so these two names must stay unbound
#: above.
_FORWARDED = ("_profile_state", "_profile_lock")


def __getattr__(name: str):
    if name in _FORWARDED:
        return getattr(runstate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
