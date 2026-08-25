# Training notes

Running log of the QLoRA fine-tune run on Kaggle (`notebooks/kaggle_train.ipynb`) --
what broke, why, and what fixed it. Read this before touching the training notebook
again; it exists so the debugging already done doesn't get repeated. Companion to
`LABELLING-NOTES.md` (that one covers the labelling/eval side; this one covers
training/Kaggle execution).

Last updated: 2026-08-25.

## Current status

**v10 SUCCEEDED -- first genuinely complete, valid QLoRA adapter.** Kernel
`mattarmario/alternance-extractor-train`, version 10, COMPLETE in ~2h (7171s). Printed
`Adapter saved to /kaggle/working/qlora-adapter (36,981,856 bytes, verified non-empty)`
-- passed the hard assertion, and 37MB matches the expected size for 18.4M trainable
LoRA params almost exactly. This is 1 epoch on `train.jsonl` (541 rows), bf16,
batch=2/accumulation=8, `loss_type="nll"`.

**Next up**: hand this adapter off to `kaggle_benchmark.ipynb` (still needs the
train->benchmark chaining solved, see "Chaining train -> benchmark" section below),
run the benchmark, score against the Groq baseline. Then decide on the 3-epoch "real"
run. User has stepped away and asked me to proceed autonomously through the rest of the
pipeline without further check-ins, only stopping for a genuine blocker.

**Full version history**: v1 P100/PyTorch incompatibility -> v2 trl API break -> v3
chunked_nll bug -> v4 OOM (batch=8 too big) -> v5 **trained a full epoch (2h11m)**, then
OOM'd on eval (eval batch size never set) -> v6 fp16 attempt, dtype crash -> v7 fp16
attempt #2, same crash, abandoned fp16 -> v8 bf16 revert, first fully COMPLETE run, but
silently empty adapter -> v9 device_map={"": 0} attempt, OOM'd immediately, reverted ->
**v10 model.save_pretrained() fix -- SUCCESS, valid adapter saved.**

**Kaggle account**: `mattarmario`, API token stored at `C:\Users\mmmar\.kaggle\access_token`.
GPU quota: 30h/week, ~0.15h used so far across all attempts -- quota is not a constraint.

**Next step once this finishes successfully**: run `notebooks/kaggle_benchmark.ipynb`
(needs the trained adapter as input -- see "Chaining train -> benchmark" below for how
that hasn't been solved yet).

## Bug history (all four hit in sequence, each only surfaced by actually running it)

1. **P100/PyTorch incompatibility.** Kaggle's default GPU assignment gave a Tesla P100
   (compute capability sm_60). The pinned PyTorch build in Kaggle's image only supports
   sm_70+. bitsandbytes crashed: `Error named symbol not found ... ops.cu`. **Fix**: set
   `machine_shape: "NvidiaTeslaT4"` in `kernel-metadata.json` (valid values are
   `NvidiaTeslaT4`, `NvidiaTeslaP100`, `Tpu1VmV38` -- found in
   `kagglesdk/kernels/types/kernels_api_service.py`'s docstring, not documented anywhere
   user-facing).

2. **trl 1.10.0 API break.** The notebook was drafted against an older trl where
   `SFTTrainer(max_seq_length=..., packing=...)` worked directly. Current trl (pulled via
   `!pip install -U trl`, always latest) moved both onto a separate `SFTConfig` object
   (which subclasses `TrainingArguments`). Error: `TypeError: SFTTrainer.__init__() got
   an unexpected keyword argument 'max_seq_length'`. **Fix**: `TrainingArguments` ->
   `SFTConfig`, moved `max_length` (renamed from `max_seq_length`) and `packing` onto it.

3. **trl 1.10.0 chunked_nll bug.** `SFTConfig`'s default `loss_type` is `"chunked_nll"`
   (a memory-saving loss variant). Its internal patching
   (`_patch_chunked_ce_lm_head` in `trl/trainer/sft_trainer.py`) assumes `model.forward`
   is always a bound method with `__func__`. For this LoRA+4-bit model it's a
   `functools.partial` instead (likely from accelerate's device dispatch hooks on a
   4-bit-quantized model). Error: `AttributeError: 'functools.partial' object has no
   attribute '__func__'`. **Fix**: `loss_type="nll"` (standard, mathematically
   equivalent, skips the buggy patch). Real trl bug, not fixable from our side --
   worth checking if a later trl release patches this, so this workaround can be dropped.

4. **OOM from losing chunked_nll's memory savings.** `"nll"` (unlike `"chunked_nll"`)
   materializes the full `[batch * seq_len, vocab_size]` logits tensor in one shot.
   Qwen's vocab is ~152k tokens; at batch=8 (raised earlier for speed -- see below) with
   sequences up to 3584 tokens, that tensor alone tried to allocate 14GB and OOM'd
   against the T4's ~14.56GB usable capacity (`OutOfMemoryError: CUDA out of memory.
   Tried to allocate 14.08 GiB`). **Fix**: batch 8->2, grad accumulation 2->8 (same
   effective batch size 16, just smaller per-step memory footprint).

5. **Eval-batch OOM.** `per_device_eval_batch_size` was never set in `SFTConfig`, so it
   silently defaulted to Hugging Face's default of 8 -- while training correctly used
   batch=2. Training itself ran fine for a full epoch (257 steps, ~2h11m); the crash hit
   during the epoch-end eval pass, same full-vocab-logits OOM as bug #4 but triggered by
   eval instead of train (`Tried to allocate 10.89 GiB`). **Fix**: pin
   `per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE`.

6. **bf16/fp16 dtype mismatch -- fp16 abandoned, reverted to bf16.** Switching to
   `fp16=True` (bug #5's fix) crashed during gradient-norm clipping:
   `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not
   implemented for 'BFloat16'`. First hypothesis: `bnb_4bit_compute_dtype=torch.float16`
   only affects the quantized base weights' compute, so pinned
   `torch_dtype=torch.float16` on `from_pretrained` too, to make the unquantized parts
   (embeddings, LoRA-injected layers) agree. **Same crash persisted (v7).** Traced
   further: peft's LoRA-on-bnb-Linear4bit path reads `base_layer.compute_dtype` to
   decide the new adapter's dtype
   (`peft/tuners/tuners_utils.py::_get_base_layer_device_and_dtype`), and transformers
   passes `quantization_config.bnb_4bit_compute_dtype` straight through to
   `bnb.nn.Linear4bit`'s constructor (`transformers/integrations/bitsandbytes.py`) --
   both looked correct on paper, so the real mismatch point was never confirmed (no
   local GPU to actually reproduce and verify; possible version drift between what's
   installed locally for inspection vs what Kaggle's `!pip install -U` grabs at push
   time). **Rather than burn a third attempt guessing blind, reverted to bf16
   entirely** (`bf16=True`, `bnb_4bit_compute_dtype=torch.bfloat16`,
   `torch_dtype=torch.bfloat16`). Key insight that justified reverting instead of
   continuing to chase fp16: **bf16 mixed-precision training doesn't use `GradScaler`
   at all** -- bf16's exponent range matches fp32, so it never needs loss scaling the
   way fp16 does. That's *why* bf16 completed a full epoch (v5) while fp16 kept
   crashing regardless of which specific parameter was the actual culprit -- reverting
   sidesteps the whole bug class rather than patching around one instance of it.
   **If revisiting fp16 for speed later, do it with real GPU access to verify locally
   rather than reasoning from static source inspection alone.**

7. **Silently-empty adapter save.** v8 (bf16, 1 epoch) completed with `COMPLETE` status,
   no error, and printed "Adapter saved to /kaggle/working/qlora-adapter" -- **the first
   fully successful run, ~2h14m total.** But `adapter_model.safetensors` was genuinely
   0 bytes and no tokenizer files were written at all. Confirmed with two independent
   fresh downloads (ruled out a transfer glitch -- other large files in the same output,
   e.g. `train.jsonl` at 1.8MB, downloaded correctly; Kaggle's `kernels_list_files` API
   size field is also unreliable/broken -- reported near-identical tiny numbers for
   every file regardless of real size, don't trust it). `trainer.save_model()` /
   `tokenizer.save_pretrained()` silently wrote nothing usable, no exception raised.
   Leading suspect (not yet confirmed with a real run): `device_map="auto"` invokes
   accelerate's hook-based `dispatch_model()` even on a single GPU -- the same mechanism
   already confirmed to turn `model.forward` into a `functools.partial` for bug #3
   (chunked_nll). **Fix**: `device_map={"": 0}` (pin to the single T4 explicitly,
   there's no real multi-device need here) instead of `"auto"`. **Also added a hard
   assertion right after saving** (adapter file >1MB, tokenizer.json exists) so a repeat
   of this fails immediately and loudly instead of silently succeeding and only being
   caught after a ~2h14m run plus a fresh download in a separate notebook.

   **v9 correction: `device_map={"": 0}` OOM'd at the very first training step**
   (`Tried to allocate 7.04 GiB`) -- `"auto"` apparently has a smaller memory footprint
   than pinning explicitly (mechanism unconfirmed), so that change was reverted without
   ever confirming it fixed the save bug in the first place. **Real fix (v10)**: call
   `model.save_pretrained(OUTPUT_DIR)` directly on the `PeftModel` object instead of
   `trainer.save_model(OUTPUT_DIR)`. The latter goes through `accelerator.unwrap_model()`
   before saving; something in that indirection (likely interacting with `"auto"`'s
   hook-based dispatch) silently produced an empty save with no exception. Saving
   directly from the `model` object we built via `get_peft_model()` is unambiguous about
   whose weights get written, and is the more common QLoRA-tutorial pattern for exactly
   this reason. Kept `device_map="auto"` (proven memory footprint) and the hard
   save-verification assertion from the first fix attempt.

## Speed tuning attempted

Original notebook draft: batch=4, accumulation=4. I raised this to batch=8/accum=2 as a
"free" speedup (the 4-bit base model + 18M LoRA params seemed to leave plenty of T4
headroom) -- this is what triggered bug #4 above. **Net lesson: the real memory
constraint here is the loss computation's full-vocab logits tensor (batch x seq_len x
vocab_size), not the model/optimizer footprint** -- so batch-size headroom estimates
based on model size alone are wrong for this loss_type. Current batch=2/accum=8 is the
safe, working setting. Have NOT tried re-raising batch size now that `loss_type="nll"`
specifically (as opposed to model size) is the known constraint -- if a future session
wants to retune for speed, the lever to pull is batch size against the logits-tensor
math (`batch * seq_len * vocab_size * bytes_per_element`), not general "does the model
fit" intuition.

**Real data point**: batch=2, bf16, unchunked `"nll"` loss measured at ~30s/step on a T4
(257 steps = ~2h11m for one epoch). That's slow for a 1.5B model -- switched to fp16 in
version 6 as the most likely fix (T4/Turing has no native bf16 tensor-core support).
Haven't yet measured whether fp16 actually speeds this up; check version 6's logs once
it finishes for the real per-step time before assuming this worked.

**Second parallel account (`jadbadawi09`) was tried as a speed hedge** -- pushed a
batch=4 variant there to test in parallel while the safe batch=2 run continued on
`mattarmario`. **Failed for an unrelated reason**: that account's session had no
internet access at all (`Temporary failure in name resolution` for both PyPI and
GitHub), despite `enable_internet: true` in kernel-metadata.json. Almost certainly
because Kaggle requires phone verification before granting real internet access to
notebooks, regardless of the setting -- a Kaggle-account-side fix, not something doable
via the API. **Second token stored at `C:\Users\mmmar\.kaggle2_jad_token`** if that
account gets phone-verified later and someone wants to retry the parallel-hedge idea.

## Kaggle API limitations discovered (useful for future sessions driving this via API)

- `kaggle kernels logs <ref>` returns **nothing while a kernel is still RUNNING** --
  logs only populate once the run reaches a terminal state (COMPLETE/ERROR/CANCELLED).
  There is no live log streaming through this CLI. Don't waste time polling logs mid-run.
- `kaggle kernels status <ref>` (and the underlying API response object) exposes only
  `status` and `failureMessage` -- **no timestamps, no elapsed time, no step/epoch
  progress**. There's no way to distinguish "still queued" from "running slowly" from
  the API alone once status flips to RUNNING. If a tighter ETA matters, the only real
  option is checking the kernel's page in a browser (not tried this session) or just
  waiting for it to finish.
- The polling `until`-loop pattern can misfire on transient network errors (a
  `RemoteDisconnected` exception string doesn't match `RUNNING|QUEUED`, so a naive loop
  treats it as "done"). Always check for actual terminal-state strings
  (`KernelWorkerStatus.(COMPLETE|ERROR|CANCELLED)`) rather than "not still
  running/queued", or a network blip ends the poll early with a false read.
- `kaggle kernels logs` output needs `PYTHONIOENCODING=utf-8` (or redirect to a file
  and read with a tool that handles UTF-8) -- raw console printing crashes with
  `'charmap' codec can't encode characters` on Windows due to non-ASCII content in logs.
- Auth: this CLI version (2.2.4) uses a single bearer token (`KAGGLE_API_TOKEN` env var
  or `~/.kaggle/access_token` file), not the older `kaggle.json` username+key format.
  `KAGGLE_CONFIG_DIR` does NOT relocate the token file for this CLI version (tested,
  silently fell back to the default account) -- use `KAGGLE_API_TOKEN=<token>` inline
  per-command to reliably target a specific account when juggling more than one.
- `machine_shape` valid values (`NvidiaTeslaT4`, `NvidiaTeslaP100`, `Tpu1VmV38`) are
  undocumented in any CLI help text -- only found by reading
  `kagglesdk/kernels/types/kernels_api_service.py`'s docstring directly.

## Chaining train -> benchmark (not yet solved)

`kaggle_benchmark.ipynb` expects the trained adapter at
`/kaggle/input/alternance-extractor-adapter` (a Kaggle Dataset input, hardcoded path).
Once `kaggle_train.ipynb` finishes successfully, the adapter needs to get from that run's
`/kaggle/working/qlora-adapter` output into a Kaggle Dataset named
`alternance-extractor-adapter` before the benchmark notebook can use it as-is. Options,
neither tried yet:
- `kaggle kernels output mattarmario/alternance-extractor-train -p <local dir>` to pull
  the adapter files locally, then `kaggle datasets create` to publish them as a new
  Dataset with that exact name.
- Or edit `kaggle_benchmark.ipynb`'s `ADAPTER_DIR` to instead reference the training
  kernel's output directly via `kernel_sources` in its `kernel-metadata.json` (Kaggle
  auto-mounts a referenced kernel's output, though the exact mount path wasn't verified
  this session).

## Open items

- Confirm training actually completes (still running as of this note) and check the
  final train/eval loss curve makes sense (decreasing, not NaN/diverging).
- Decide train -> benchmark adapter handoff (see above) once there's an adapter to hand off.
- Consider whether to re-attempt a larger batch size now that the real constraint
  (logits tensor size under `loss_type="nll"`) is understood, if this run's wall-clock
  time turns out to be inconveniently long.
- If `jadbadawi09` gets phone-verified, the parallel-hedge setup is ready to reuse
  (token already saved, push script pattern established in this session).
