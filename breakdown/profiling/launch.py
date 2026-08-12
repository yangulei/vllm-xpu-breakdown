# SPDX-License-Identifier: Apache-2.0
"""Running vLLM once, under the profiler.

Everything between "the user pressed Profile" and "there are trace files":
building the engine for a reduced-layer, prefix-cached, exact-length run,
pinning the scheduler so a batch is not chunked into partial waves,
installing the capture-time hooks, and driving the warmup and profiled passes.

It lives here rather than in ``app.py`` because none of it is about HTTP: the
CLI, the tests and the fixture capture tool all need the same run, and the web
route is one caller among several.
"""
from __future__ import annotations

import functools
import importlib
import logging
import os
import time
import traceback

from breakdown.model_info import (
    fetch_model_config, min_profile_layers, summarize_config)
from breakdown.profiling import runstate
from breakdown.profiling.runstate import save_state
from breakdown.profiling.traces import (
    _build_result_from_traces, _merge_two_pass_result)

logger = logging.getLogger("vllm_xpu_breakdown")

# ---- Profile API ----


def _set_num_hidden_layers(hf_config, n: int):
    """Set the decoder layer count where it actually lives in the HF config.

    Module-level (picklable) so it can be passed as a ``hf_overrides`` callable
    to vLLM, which pickles the config when spawning the EngineCore subprocess.
    Some multimodal models (e.g. MiniMax-M3) nest ``num_hidden_layers`` under
    ``text_config``; a top-level override is ignored there.
    """
    text_cfg = getattr(hf_config, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "num_hidden_layers"):
        text_cfg.num_hidden_layers = n
    else:
        hf_config.num_hidden_layers = n
    return hf_config


def _build_layer_override(
    profiled_layers: int,
    quantization: str | None,
    layers_under_text_config: bool,
):
    """Build the ``hf_overrides`` value for reduced-layer profiling.

    Normally we return a callable that sets ``num_hidden_layers`` where it
    actually lives (top level, or nested under ``text_config`` for multimodal
    models like MiniMax-M3). vLLM applies callables in place, preserving the
    rest of the config.

    When ``quantization`` is requested, vLLM's ``get_quant_config`` rejects a
    callable ("hf_overrides must be a dict ...") because it reads the quant
    config out of ``hf_overrides``. In that case we return a **dict** override
    targeting the right key instead — vLLM applies nested ``text_config`` dicts
    recursively.
    """
    if quantization:
        if layers_under_text_config:
            return {"text_config": {"num_hidden_layers": profiled_layers}}
        return {"num_hidden_layers": profiled_layers}
    # Module-level partial so it pickles for the spawned EngineCore subprocess.
    return functools.partial(_set_num_hidden_layers, n=profiled_layers)


def _make_token_ids(n: int, vocab_size: int, seed: int) -> list[int]:
    """Deterministically build ``n`` valid, non-special token ids.

    Ids are drawn from a safe interior range of the vocabulary (avoiding the
    low ids that are typically special/control tokens) using a cheap hash of the
    position and ``seed``. Two calls with the same ``(n, vocab_size, seed)``
    yield the identical sequence, which is what lets the prefix-cache warm pass
    and the profiled pass share an exact-match context prefix.
    """
    if n <= 0:
        return []
    lo = 256
    hi = max(lo + 1, vocab_size - 256)
    span = hi - lo
    return [lo + ((i * 2654435761 + seed * 40503 + 12345) % span) for i in range(n)]


def _get_block_size(llm, default: int = 16) -> int:
    """Read the KV-cache block size from a constructed vLLM engine (robust)."""
    engine = getattr(llm, "llm_engine", None)
    for attr in ("vllm_config", "engine_config", "model_config"):
        cfg = getattr(engine, attr, None)
        cc = getattr(cfg, "cache_config", None)
        bs = getattr(cc, "block_size", None)
        if bs:
            return int(bs)
    cc = getattr(engine, "cache_config", None)
    bs = getattr(cc, "block_size", None)
    if bs:
        return int(bs)
    return default


def _get_vocab_size(llm, summary: dict, default: int = 32000) -> int:
    """Best-effort tokenizer vocabulary size for synthetic-prompt generation."""
    try:
        tok = llm.get_tokenizer()
        vs = getattr(tok, "vocab_size", None) or len(tok)
        if vs and vs > 512:
            return int(vs)
    except Exception:
        pass
    vs = summary.get("vocab_size")
    return int(vs) if vs and vs > 512 else default


# Descriptive trace filename produced by the download endpoint, e.g.
#   vllm_trace_MiniMax-M3_XPU_eager_decode_ctx2048_in1536_out8_bs32_tp4_6layers.json.gz
# The stable ``_ctx…_in…_out…_bs…_tp…[_quant]_…layers`` tail (plus the optional
# ``_prefill``/``_decode`` pass tag and the ``_device_mode`` before it) lets the
# upload endpoint recover the full profiled configuration, making a
# download -> upload reconstruction a lossless round-trip.

def fit_max_model_len(max_model_len: int, query_len: int | None,
                      context_len: int | None, max_tokens: int) -> int:
    """``max_model_len`` grown to cover the decode budget, if it must be.

    +16 of slack for the chat template's own tokens.
    """
    if not query_len:
        return int(max_model_len)
    needed = int(query_len) + int(context_len or 0) + int(max_tokens) + 16
    return max(int(max_model_len), needed)

def _scheduler_pin(prefill_batch: int, decode_batch: int,
                   query_len: int) -> dict[str, int]:
    """Engine settings that make every step run the *full* requested batch.

    Left to its defaults, vLLM's continuous-batching scheduler caps
    per-iteration concurrency (by ``max_num_seqs``, and by how many sequences'
    KV fits in cache) and runs an oversized batch in *partial-batch waves* - a
    batch of 32 dispatched as 29 + 3. Each wave has a different row count, so
    its ops neither symbolize to ``B`` nor merge with the full-batch ops, and
    the reconstructed decode graph grows duplicated ``29``/``3`` nodes.

    ``max_num_seqs`` admits the whole batch in one iteration;
    ``max_num_batched_tokens`` is sized to also admit a whole batch's prefill
    tokens in a single step (prefill pass: ``prefill_batch x query_len``;
    decode pass: ``decode_batch`` single-token prefills) so a full-shape step
    is never chunked. If the batch's KV does not fit device memory, raise
    ``gpu_memory_utilization`` or lower Context/Batch rather than letting the
    run silently split.
    """
    max_batch = max(int(prefill_batch), int(decode_batch))
    prefill_step_tokens = int(prefill_batch) * max(int(query_len), 1)
    return {
        "max_num_seqs": max_batch,
        "max_num_batched_tokens": max(prefill_step_tokens, max_batch, 2048),
    }


def _configure_text_only_profile(engine_kwargs: dict) -> None:
    """Disable multimodal processing for the app's text-only prompts."""
    engine_kwargs["language_model_only"] = True


def _validate_profile_batch(summary: dict, decode_batch: int) -> str | None:
    """Return an actionable error for unsafe recurrent-state batches."""
    if summary.get("linear_attention") and decode_batch > 1:
        return (
            "This model uses recurrent linear attention with a large state per "
            "sequence. Decode Batch greater than 1 can exhaust XPU memory during "
            "vLLM startup warmup. Set Decode Batch to 1 and retry."
        )
    return None


def _profile_gpu_memory_utilization(
    summary: dict, requested: float | None
) -> float | None:
    """Leave headroom for recurrent-state gathers during XPU warmup."""
    if not summary.get("linear_attention"):
        return requested
    return min(float(requested), 0.5) if requested is not None else 0.5


def _enable_trusted_apply_model_serialization() -> str | None:
    """Allow this app's trusted hook functions through vLLM V1 RPC."""
    previous = os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION")
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    return previous


def _restore_trusted_apply_model_serialization(previous: str | None) -> None:
    """Restore the serialization policy that preceded a profiling run."""
    if previous is None:
        os.environ.pop("VLLM_ALLOW_INSECURE_SERIALIZATION", None)
    else:
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = previous

def _run_profile(model_id: str, mode: str, max_model_len: int,
                 batch_size: int, max_tokens: int, prompt: str,
                 num_profile_layers: int | None = None,
                 tp_size: int = 1,
                 quantization: str | None = None,
                 gpu_memory_utilization: float | None = None,
                 query_len: int | None = None,
                 context_len: int | None = None,
                 prefill_batch_size: int | None = None,
                 decode_batch_size: int | None = None):
    """Run profiling in a background thread using vLLM's native profiler.

    On XPU hardware, vLLM automatically selects XPUWorker which uses
    the correct profiler activities (["CPU", "XPU"]).

    Args:
        num_profile_layers: If set (e.g. 1), override the model to load only
            this many layers. Timing is then scaled by actual_layers/profiled
            for the full model estimate. Enables profiling models too large
            for the GPU.
        tp_size: tensor parallel size (default 1). With TP>1, vLLM creates
            one trace file per rank; we parse all and aggregate timing.
        quantization: quantization method (e.g. "fp8", "gptq", "awq").
            Passed as --quantization to vLLM.
        gpu_memory_utilization: fraction of device memory vLLM may use. Lower
            it (e.g. 0.8) when vLLM's init footprint leaves too little headroom
            for the default (0.92) on small-VRAM cards. None keeps vLLM default.
        query_len: number of *new* prompt tokens the profiled prefill computes
            (the "Query Len"). Drives the prefill token dimension ``S``.
        context_len: number of prior context tokens the query attends to (the
            "Context Len"). When >0, those tokens are pre-computed in an
            un-profiled warm pass and served from the prefix cache (APC) during
            the profiled run, so the profiled prefill computes only ``query_len``
            new tokens while attention still reads the full ``context_len+query_len``
            KV. Rounded down to a KV block boundary so the whole context caches.
        prefill_batch_size: number of concurrent sequences for the **prefill**
            phase (typically 1 in real serving). When it differs from
            ``decode_batch_size`` the run is profiled in two passes — a prefill
            pass at this batch and a decode pass at ``decode_batch_size`` — and
            the two phase graphs are merged. ``None`` falls back to
            ``batch_size`` (single pass, legacy behaviour).
        decode_batch_size: number of concurrent sequences for the **decode**
            phase (often 32/64/128). See ``prefill_batch_size``. ``None`` falls
            back to ``batch_size``.
    """
    # The engine must fit the longest sequence it will ever see: cached context
    # + new query tokens + the decode tokens we generate. The caller sizes
    # max_model_len from query+context, which leaves no decode headroom, so the
    # bump belongs here -- it used to live in the HTTP route, which meant the
    # headless path never got it and a CLI profile could fail engine start-up
    # with a request longer than max_model_len.
    max_model_len = fit_max_model_len(max_model_len, query_len, context_len,
                                      max_tokens)

    serialization_policy = _enable_trusted_apply_model_serialization()
    try:
        from vllm import LLM, SamplingParams, TokensPrompt

        # Fetch model config for analysis
        try:
            config = fetch_model_config(model_id)
            summary = summarize_config(config)
        except Exception:
            config = {}
            summary = {}

        actual_layers = summary.get("num_layers") or 1
        if num_profile_layers == "min":
            # Auto-calculate minimum layers needed
            profiled_layers = min_profile_layers(summary)
        elif num_profile_layers:
            profiled_layers = int(num_profile_layers)
        else:
            profiled_layers = actual_layers
        layer_scale = actual_layers / profiled_layers

        trace_dir = os.path.abspath("output/traces")
        os.makedirs(trace_dir, exist_ok=True)

        engine_kwargs: dict = {
            "model": model_id,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tp_size,
            "profiler_config": {
                "profiler": "torch",
                "torch_profiler_dir": trace_dir,
                "torch_profiler_record_shapes": True,
                "torch_profiler_with_stack": True,
                "torch_profiler_with_flops": True,
                "torch_profiler_use_gzip": True,
            },
            "trust_remote_code": True,
        }

        # Always use dummy weights for profiling — timing doesn't depend on
        # weight values, and dummy avoids KeyError when layers are reduced.
        engine_kwargs["load_format"] = "dummy"

        # Optionally cap device memory usage (leaves headroom for vLLM's init
        # footprint on small-VRAM cards; None keeps vLLM's default).
        effective_memory_utilization = _profile_gpu_memory_utilization(
            summary, gpu_memory_utilization
        )
        if effective_memory_utilization is not None:
            engine_kwargs["gpu_memory_utilization"] = effective_memory_utilization

        # Every generated prompt is text-only. This explicit mode also handles
        # local text-only copies of multimodal checkpoints that omit processor
        # assets and no longer expose enough config to detect the vision tower.
        _configure_text_only_profile(engine_kwargs)

        # Sparse-attention models (e.g. MiniMax-M3) select fixed-size KV blocks
        # via the lightning indexer, so the KV-cache block size must match the
        # sparse block size; otherwise vLLM cannot reconcile a common kernel
        # block size across the sparse/full attention backends.
        sparse_block = summary.get("sparse_block_size")
        if sparse_block:
            engine_kwargs["block_size"] = int(sparse_block)

        # Normalize the query/context sizing knobs. ``query_len`` sets the
        # number of new prompt tokens the profiled prefill computes; when
        # ``context_len`` > 0 we serve that many prior tokens from the prefix
        # cache so the profiled prefill sees ``S = query_len`` new tokens
        # attending to a ``context_len``-token KV context.
        query_len = int(query_len) if query_len else 0
        context_len = int(context_len) if context_len else 0
        use_token_prompts = query_len > 0
        # The context length actually served from the prefix cache, floored to a
        # whole number of KV blocks (set below once the block size is known). The
        # graph reconstruction symbolizes this value as ``C`` so attention KV
        # dims read ``C`` / ``S+C`` instead of a bare number.
        profiled_context_len = 0

        # Enable Automatic Prefix Caching so the context prefix computed in the
        # warm pass is reused (not recomputed) during the profiled run.
        if context_len > 0:
            engine_kwargs["enable_prefix_caching"] = True

        # Quantization method
        if quantization:
            engine_kwargs["quantization"] = quantization

        # Override layer count for reduced-layer profiling.
        if profiled_layers < actual_layers:
            # Some multimodal models (e.g. MiniMax-M3) nest the decoder layer
            # count under ``text_config``. A top-level ``num_hidden_layers``
            # override is silently ignored there, so the full model is built
            # and exhausts device memory (UR_RESULT_ERROR_DEVICE_LOST). The
            # helper returns a callable normally, but a dict when quantization
            # is set (vLLM's ``get_quant_config`` requires a dict override).
            engine_kwargs["hf_overrides"] = _build_layer_override(
                profiled_layers,
                quantization,
                bool(summary.get("layers_under_text_config")),
            )

        # Set compile / eager mode
        if mode == "compile":
            os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "3"
            engine_kwargs["enforce_eager"] = False
        else:
            os.environ.pop("VLLM_TORCH_COMPILE_LEVEL", None)
            engine_kwargs["enforce_eager"] = True

        # Resolve the per-phase batch sizes. When prefill and decode batches
        # differ we profile two passes (real serving prefills ~1 sequence while
        # decoding many); otherwise a single pass reproduces legacy behaviour.
        pf_batch = int(prefill_batch_size) if prefill_batch_size else int(batch_size)
        dc_batch = int(decode_batch_size) if decode_batch_size else int(batch_size)
        two_pass = pf_batch != dc_batch


        engine_kwargs.update(
            _scheduler_pin(pf_batch, dc_batch, int(query_len or 1)))

        llm = LLM(**engine_kwargs)

        if use_token_prompts:
            block_size = _get_block_size(llm)
            vocab_size = _get_vocab_size(llm, summary)
            # Round the context down to a whole number of KV blocks so the entire
            # context prefix is cacheable (a trailing partial block would be
            # recomputed and shift the profiled prefill token count).
            ctx_aligned = (context_len // block_size) * block_size
            profiled_context_len = ctx_aligned
            ctx_ids = _make_token_ids(ctx_aligned, vocab_size, seed=0)
        else:
            ctx_aligned = 0
            ctx_ids = []

        def _list_trace_files() -> set:
            return {os.path.join(trace_dir, f) for f in os.listdir(trace_dir)
                    if f.endswith(".json") or f.endswith(".json.gz")}

        def _install_span_hooks() -> bool:
            """Install capture-time module-name span hooks in the worker(s).

            Registers forward hooks that emit ``record_function(
            "module::<qname>::<Cls>")`` spans around every module's forward, so
            the trace carries real attribute names (``q_norm``/``k_norm``,
            ``self_attn``, ...) and ``build_graph_from_trace`` reconstructs the
            tree with exact names. This is the *only* source of real module
            names, so a failure is logged loudly: the run still produces a
            graph, but every module falls back to a class heuristic.
            """
            try:
                from breakdown.module_hooks import install_module_span_hooks_on
                counts = llm.apply_model(install_module_span_hooks_on)
                total = sum(c for c in (counts or []) if isinstance(c, int))
                if not total:
                    logger.error(
                        "module span hooks: apply_model returned %r — no hooks "
                        "installed, module names will be class-only", counts)
                    return False
                logger.info("module span hooks: installed %d hooks across %d "
                            "worker(s)", total, len(counts or []))
            except Exception:
                logger.error("module span hooks: install FAILED; the trace will "
                             "carry no module:: spans and module names will fall "
                             "back to the name overlay", exc_info=True)
                return False
            # Kernel-launch spans: the operands of kernels launched straight
            # from Python (Triton, pybind11 extensions), which leave no cpu_op
            # and so no recorded shapes. Not fatal - without them such ops stay
            # shape-less and the replay benchmark reports them as uncovered.
            try:
                from breakdown.kernel_hooks import install_kernel_span_hooks_on
                kcounts = llm.apply_model(install_kernel_span_hooks_on)
                ktotal = sum(c for c in (kcounts or []) if isinstance(c, int))
                logger.info("kernel span hooks: installed %d hooks across %d "
                            "worker(s)", ktotal, len(kcounts or []))
            except Exception:
                logger.warning("kernel span hooks: install failed; "
                               "Python-launched kernels will have no recorded "
                               "operands", exc_info=True)
            return True

        def _remove_span_hooks(installed: bool) -> None:
            if not installed:
                return
            for mod_name, fn_name in (("breakdown.module_hooks",
                                       "remove_module_span_hooks_on"),
                                      ("breakdown.kernel_hooks",
                                       "remove_kernel_span_hooks_on")):
                try:
                    mod = importlib.import_module(mod_name)
                    llm.apply_model(getattr(mod, fn_name))
                except Exception:
                    logger.warning("%s: remove failed", mod_name, exc_info=True)

        def _profiled_pass(pass_batch: int, pass_query_len: int,
                           pass_max_tokens: int):
            """Warm + run one profiled generate at the given batch/query size.

            ``pass_max_tokens`` is the number of tokens to generate this pass:
            the decode pass uses the full decode budget (so decode steps are
            captured), while the prefill pass uses **1** — it only needs the
            single prefill step (``S`` = ``query_len``), and generating extra
            decode tokens would only bloat the trace and slow the run.

            ``ignore_eos`` keeps every sequence alive for the full budget so the
            profiled decode step reflects the requested batch (a sequence hitting
            EOS early would shrink the observed decode concurrency ``B``).

            Returns ``(rank_files, cache_hit_note)``: the trace file(s) newly
            written by this pass (newest first, capped to ``tp_size``) and an
            optional prefix-cache-miss note.
            """
            note = None
            pass_sampling = SamplingParams(max_tokens=pass_max_tokens,
                                           ignore_eos=True)
            if use_token_prompts:
                def _full_prompt(query_seed: int) -> "TokensPrompt":
                    q = _make_token_ids(pass_query_len, vocab_size, seed=query_seed)
                    return TokensPrompt(prompt_token_ids=ctx_ids + q)

                # Distinct query per batch item (shared context prefix) so every
                # sequence genuinely prefills its own tokens instead of
                # cache-hitting a sibling; profiled seeds differ from warmup
                # seeds so the profiled queries are never served from cache.
                def _batch(base_seed: int) -> list["TokensPrompt"]:
                    return [_full_prompt(base_seed + b) for b in range(pass_batch)]

                # Warm the shared prefix cache once (un-profiled) so the profiled
                # run reads the context from cache instead of recomputing it.
                if ctx_ids:
                    llm.generate(
                        [TokensPrompt(prompt_token_ids=ctx_ids)],
                        SamplingParams(max_tokens=1), use_tqdm=False,
                    )
                for w in range(2):
                    llm.generate(_batch(1000 * (w + 1)), pass_sampling,
                                 use_tqdm=False)
                profiled_prompts = _batch(900000)

                before = _list_trace_files()
                _spans = _install_span_hooks()
                llm.start_profile()
                outputs = llm.generate(profiled_prompts, pass_sampling,
                                       use_tqdm=False)
                llm.stop_profile()
                _remove_span_hooks(_spans)

                # Verify the context was served from cache. A miss means the
                # profiled prefill recomputed the whole context (S = context +
                # query), so record a note rather than silently misreporting.
                if context_len > 0 and outputs:
                    cached = getattr(outputs[0], "num_cached_tokens", None)
                    if cached is not None and cached < ctx_aligned:
                        note = (
                            f"Prefix cache hit only {cached}/{ctx_aligned} "
                            "context tokens; profiled prefill may include "
                            "context recompute."
                        )
            else:
                # --- Legacy text-prompt path (Query/Context Len unspecified) ---
                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
                conversations = [conversation] * pass_batch
                prompts = [prompt] * pass_batch

                # First warmup pass also detects chat-template support.
                try:
                    llm.chat(conversations, pass_sampling, use_tqdm=False)
                    use_chat = True
                except Exception:
                    llm.generate(prompts, pass_sampling, use_tqdm=False)
                    use_chat = False

                def _run_inference():
                    if use_chat:
                        llm.chat(conversations, pass_sampling, use_tqdm=False)
                    else:
                        llm.generate(prompts, pass_sampling, use_tqdm=False)

                # Warmup primes Triton JIT / autotuning so the profiled trace
                # reflects steady-state timing. Detection pass above is warmup
                # #1; run 2 more for 3 total.
                for _ in range(2):
                    _run_inference()

                before = _list_trace_files()
                _spans = _install_span_hooks()
                llm.start_profile()
                _run_inference()
                llm.stop_profile()
                _remove_span_hooks(_spans)

            # torch's profiler writes the trace on stop_profile; wait briefly for
            # the new file(s) to appear, then return them (newest first).
            new_files: list[str] = []
            for _ in range(20):
                new_files = sorted(_list_trace_files() - before,
                                   key=os.path.getmtime, reverse=True)
                if len(new_files) >= tp_size:
                    break
                time.sleep(0.5)
            if not new_files:
                new_files = sorted(_list_trace_files(),
                                   key=os.path.getmtime, reverse=True)
            return new_files[:tp_size], note

        # --- Run the profiled pass(es) ---
        if two_pass:
            # Prefill pass: batch = prefill_batch, real query_len, generate only
            # 1 token so the trace holds exactly the prefill step (S=query_len)
            # — we keep only its prefill phase. Decode pass: batch = decode_batch,
            # query_len forced to 1 so decode is 1 new token/seq (matches real
            # decode and avoids OOM from prefilling decode_batch x query_len
            # tokens), generating the full decode budget — we keep its decode
            # phase.
            pre_files, note_pre = _profiled_pass(pf_batch, query_len,
                                                 pass_max_tokens=1)
            dec_query = 1 if use_token_prompts else 0
            dec_files, note_dec = _profiled_pass(dc_batch, dec_query,
                                                 pass_max_tokens=max_tokens)
            cache_hit_note = note_pre or note_dec
        else:
            # Single pass yields both phases from one run, so it needs the full
            # decode budget.
            single_files, cache_hit_note = _profiled_pass(
                dc_batch, query_len, pass_max_tokens=max_tokens)

        # --- Parse trace files & build the result ---
        # With TP>1, vLLM produces one trace file per rank; each pass returns all
        # of its rank files (mtime order) and ``_build_result_from_traces`` picks
        # the rank-0 file via ``_rank0_first``.
        def _build(files: list[str], bsz: int, qlen: int | None) -> dict:
            if not files:
                raise RuntimeError(
                    f"No trace files found in {trace_dir}. "
                    "Profiling may have failed in the worker process."
                )
            return _build_result_from_traces(
                files,
                model_id=model_id,
                summary=summary,
                tp_size=tp_size,
                batch_size=bsz,
                mode=mode,
                max_model_len=max_model_len,
                max_tokens=max_tokens,
                quantization=quantization,
                profiled_layers=profiled_layers,
                actual_layers=actual_layers,
                layer_scale=layer_scale,
                        query_len=qlen,
                context_len=profiled_context_len or None,
            )

        if two_pass:
            res_pre = _build(pre_files, pf_batch, query_len or None)
            res_dec = _build(dec_files, dc_batch,
                             1 if use_token_prompts else None)
            profile_result = _merge_two_pass_result(
                res_pre, res_dec, pf_batch, dc_batch)
        else:
            profile_result = _build(single_files, dc_batch, query_len or None)

        profile_result["query_len"] = query_len or None
        profile_result["context_len"] = context_len or None
        profile_result["context_len_aligned"] = profiled_context_len or None
        if cache_hit_note:
            profile_result["cache_hit_note"] = cache_hit_note

        with runstate._profile_lock:
            runstate._profile_state["status"] = "done"
            runstate._profile_state["result"] = profile_result
            runstate._profile_state["error"] = None
        save_state()

    except Exception:
        with runstate._profile_lock:
            runstate._profile_state["status"] = "error"
            runstate._profile_state["error"] = traceback.format_exc()
        save_state()
    finally:
        _restore_trusted_apply_model_serialization(serialization_policy)
        os.environ.pop("VLLM_TORCH_COMPILE_LEVEL", None)
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
