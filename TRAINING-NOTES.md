# Training notes

Running log of the QLoRA fine-tune run on Kaggle (`notebooks/kaggle_train.ipynb`) --
what broke, why, and what fixed it. Read this before touching the training notebook
again; it exists so the debugging already done doesn't get repeated. Companion to
`LABELLING-NOTES.md` (that one covers the labelling/eval side; this one covers
training/Kaggle execution).

Last updated: 2026-08-26.

## THE BIG ONE: loss was computed over the whole sequence, not the answer

Found while diagnosing why the 1-epoch model never abstains (see `RESULTS.md`). **This is
the highest-leverage fix in the project so far and it invalidates every training run
before v13.**

trl decides which tokens the loss covers from the *dataset shape*:

| dataset shape | trl treats it as | loss covers |
|---|---|---|
| `{"messages": [...]}` | conversational **language modeling** | **the entire sequence** |
| `{"prompt": ..., "completion": ...}` | **prompt-completion** | the completion only |

The notebook emitted `{"messages": [...]}`. trl also *explicitly refuses*
`completion_only_loss` for that shape (`"completion_only_loss argument is not supported
for language modeling datasets"`), so there was no way to scope it without changing shape.

Measured with the real Qwen tokenizer over all 541 rows:

```
SYSTEM_PROMPT   1566 tokens, BYTE-IDENTICAL in all 541 examples
full sequence   median 2174 tokens
JSON answer     median  129 tokens  = 6.0% of the sequence (min 4.0%, max 10.7%)
```

So **~94% of the gradient was spent teaching the model to reproduce a constant system
prompt and the input posting** -- text it never generates at inference. This also explains
the deceptively healthy `mean_token_accuracy` of 0.90: most of what it was scored on was a
system prompt identical in every example, i.e. trivially memorised.

It is the most plausible cause of the 1-epoch model's defining failure: non-null
`required_skills` on 99/100 postings vs gold's 49/100 (fp=85, tn=1). **Not a label problem**
-- those labels are non-null only 44% of the time; the model simply had almost no gradient
teaching it what a correct answer looks like, let alone when to abstain.

**Fix (v13)**: emit `{"prompt": [system, user], "completion": [assistant]}`, plus explicit
`completion_only_loss=True` and an `assert` on dataset shape so a silent regression cannot
recur. ~16x more effective gradient on the actual task for identical compute.

**`assistant_only_loss=True` is NOT a usable alternative here -- do not try it.** It
requires the chat template to mark assistant spans with `{% generation %}`. Qwen2.5-Instruct's
template has none: verified directly that
`apply_chat_template(..., return_assistant_tokens_mask=True)` returns a mask selecting
**zero** tokens (transformers even warns `return_assistant_tokens_mask==True but chat
template does not contain 'generation' keyword`). Setting it would have trained on nothing
at all, silently.

**Caveat when comparing runs**: v13's `eval_loss` is computed over the completion only, so
its absolute value is NOT comparable to v10's full-sequence `eval_loss` (0.4439). Compare
benchmark scores, not losses, across that boundary.

## Label-quality audit (2026-08-26) -- 87 further fixes to train.jsonl

Compared per-field non-null rates in `train.jsonl` (raw Groq labels, only
required_skills/nice_to_have_skills previously cleaned) against the hand-corrected test
gold. **The point: any field where the two distributions disagree is a field where the
model is being taught a different convention than it will be graded on.**

`company` was the outlier -- **93% non-null in train vs 76% in gold (+17pp)**: the model
was learning to always name an employer when the right answer is null ~24% of the time.
Three mechanically-verifiable defect classes found and nulled:

| field | rows fixed | what |
|---|---|---|
| `company` | 28 | 16 bare department codes / `"NN - City"` fragments, 12 broker-not-employer (11x ISCOD, 1x LE CABRH) |
| `salary_range` | 51 | digit-free boilerplate -- `eval/score.py` already scores these as null, so training on them taught the model to emit text worth nothing |
| `education_level` | 8 | all `"Bac+3"`, token absent from `raw_text` under a whitespace-insensitive check -- Groq's documented signature hallucination |

Gaps vs gold after: `company` +17pp -> **+11pp**, `education_level` +7pp -> **+5pp**,
`salary_range` +5pp -> **-4pp**. All 541 rows re-validate against `JobPosting`.

**Residual `company` gap (+11pp, ~60 rows) is NOT pattern-matchable** -- it's subtler
hallucination (plausible-looking company names with no textual support). Fixing it needs
per-posting reading, same as the required_skills pass. That is the highest-value remaining
label-quality work if the benchmark shows `company` underperforming.

List-field conventions were checked for train/gold drift and are broadly aligned
(`required_skills` non-null 44% train vs 49% gold; mean items 3.02 vs 3.78 -- train is
slightly *terser*, so expect a small recall penalty on that field, not a precision one).

## What the 1-epoch run's metrics actually showed

From `checkpoint-33/trainer_state.json` (worth reading after every run -- it is the only
place the loss curve survives, the Kaggle log stream does not carry it):

```
step 10  loss 1.0329  mean_token_accuracy 0.7784  lr 1e-4   grad_norm 0.359
step 20  loss 0.5494  mean_token_accuracy 0.8805  lr 1e-4   grad_norm 0.124
step 30  loss 0.4885  mean_token_accuracy 0.8946  lr 0.0    grad_norm 0.096
eval@33  eval_loss 0.4439  eval_mean_token_accuracy 0.9043
```

Two conclusions, both acted on:
1. **`eval_loss` (0.4439) is BELOW final train loss (0.4885)**, both still falling steeply
   -> the model is **underfit, not overfit**. 3 epochs is justified; there was no evidence
   for stopping at 1.
2. **`learning_rate` was already 0.0 by step 30.** 514 train rows / effective batch 16 =
   only 33 optimizer steps, and the default linear scheduler annealed the LR to zero across
   them, so most of the run trained at a near-dead LR. 3 epochs (~99 steps) spreads the
   decay usefully; added `warmup_ratio=0.05` so the first steps don't hit 2e-4 cold.

## Current status

**v13 (3 epochs, completion-only loss, cleaned labels) COMPLETED** -- started ~02:17,
finished ~12:15, so **~10h wall-clock**, well over the ~6h estimated from v10's per-step
pace. Adapter saved and asserted non-empty (36,981,856 bytes). Log confirms
`WARNING: this trl build rejects ['warmup_ratio'] -- proceeding without them`: the
signature-filter added after v11 worked exactly as intended, degrading gracefully and
visibly instead of killing a 10h run over a non-essential hyperparameter. **So v13 ran
without warmup** -- if warmup is wanted later, use `warmup_steps` (an int) instead, which
older SFTConfig builds do accept.

Re-benchmark of v13 in progress.

**Benchmark of the 1-epoch model is DONE** -- see `RESULTS.md` for the full table.
Headline: Groq macro_f1 **0.891** / exact-match 34.0% vs fine-tuned **0.510** / 0.0%,
both 100% JSON-valid. That model is the v10 adapter, trained *before* both the label
cleanup and the loss-scoping fix, so it is a floor, not the headline number.

**To re-benchmark when v13 lands** (all verified, just needs running):
1. `kaggle kernels output mattarmario/alternance-extractor-train -p <dir>` (~15 min; pull
   `qlora-adapter/`).
2. Copy `adapter_config.json`, `adapter_model.safetensors`, `tokenizer.json`,
   `tokenizer_config.json`, `chat_template.jinja` into a clean folder alongside the
   existing `dataset-metadata.json` (id `mattarmario/alternance-extractor-adapter`).
   Skip `checkpoint-*/` -- `optimizer.pt` alone is 74MB and is not needed for inference.
3. `kaggle datasets version -p <folder> -m "3-epoch completion-only-loss adapter"`.
4. Re-push the benchmark kernel: `kaggle kernels push -p <benchmark folder>` (its
   `dataset_sources` already points at that dataset slug and picks up the newest version).
5. Compare against the Run 1 table in `RESULTS.md`.

## Bug history (each only surfaced by actually running it)

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

## Chaining train -> benchmark -- SOLVED

Downloaded v10's adapter locally (`kaggle kernels output`), assembled a clean folder with
just `adapter_config.json`, `adapter_model.safetensors`, `tokenizer.json`,
`tokenizer_config.json`, `chat_template.jinja` (skipped the bulky training-only files --
`optimizer.pt`, `rng_state.pth`, etc. -- from the `checkpoint-33/` subdirectory, which
also has a valid copy of everything from the epoch-end `save_strategy="epoch"` checkpoint,
confirming `save_strategy="epoch"` was never actually broken, only the final
`trainer.save_model()` call was). Published as a private Kaggle Dataset via
`kaggle datasets create -p <folder>` with a `dataset-metadata.json` id
`mattarmario/alternance-extractor-adapter` -- matches exactly what
`kaggle_benchmark.ipynb`'s `ADAPTER_DIR = "/kaggle/input/alternance-extractor-adapter"`
expects once attached as a `dataset_sources` input.

Pushed `kaggle_benchmark.ipynb` as kernel `mattarmario/alternance-extractor-benchmark`,
`kernel-metadata.json` with `"dataset_sources": ["mattarmario/alternance-extractor-adapter"]`
and `machine_shape: "NvidiaTeslaT4"` (avoid the P100 issue from training).

**Mount path gotcha, cost 2 failed runs to find**: the notebook's `ADAPTER_DIR =
"/kaggle/input/alternance-extractor-adapter"` followed the classic, universally-documented
Kaggle dataset mount convention -- and failed both times with a confusing
`HFValidationError`-wrapped-in-`ValueError` (peft's `PeftModel.from_pretrained` couldn't
find `adapter_config.json` locally, fell back to treating the path as a HF Hub repo ID,
which then failed repo-id validation). Added a throwaway diagnostic cell that walked
`/kaggle/input` -- the dataset files were genuinely there the whole time, just mounted at
`/kaggle/input/datasets/mattarmario/alternance-extractor-adapter/` instead. **This Kaggle
API version (kagglesdk / OAuth-token CLI) uses a different mount path than the classic
one every tutorial documents** -- fixed `ADAPTER_DIR` accordingly, removed the diagnostic
cell. Worth remembering for any future dataset-as-kernel-input wiring on this account.

## Open items

- Confirm training actually completes (still running as of this note) and check the
  final train/eval loss curve makes sense (decreasing, not NaN/diverging).
- Decide train -> benchmark adapter handoff (see above) once there's an adapter to hand off.
- Consider whether to re-attempt a larger batch size now that the real constraint
  (logits tensor size under `loss_type="nll"`) is understood, if this run's wall-clock
  time turns out to be inconveniently long.
- If `jadbadawi09` gets phone-verified, the parallel-hedge setup is ready to reuse
  (token already saved, push script pattern established in this session).
