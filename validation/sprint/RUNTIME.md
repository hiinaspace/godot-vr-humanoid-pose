# Runtime path (Round 4a): trained checkpoint -> ONNX -> Godot live capture

Property this round delivers: **the trained model runs causally frame by frame from Godot's live-capture
path.** Evidence: ONNX equals the torch full-sequence prediction to 1.7e-6 m (gate 1e-4) on 5 val clips;
a headless Godot test replays a tracker stream through the running sidecar and gets the torch outputs
back to 7e-7 m; per-frame latency is measured below. Everything is parametrised by checkpoint
(arch, window, input mode, contact head are read from the checkpoint's `cfg`).

## 1. What the model is, as a runtime

Read this first, it changes the latency picture. `train.py` trains a **window model**, not a free-running
recurrent stream: output frame `t` is a pure function of the newest `W+1` tracker frames (`W=40`, 60 Hz,
so 0.67 s). The window is canonicalised at its *last* frame (head yaw and x/z origin, `make_feats`), the
LSTMs start from zero state at the window start, and only the newest output is read. Consequences:

* There is no carried recurrent state. The runtime state is a **ring buffer of raw tracker frames**
  (`runtime.TimedWindow`); each frame re-runs the 40-step LSTMs and spatial transformers over the window.
  (Carrying the LSTM state instead would be a different model: the canonicalisation frame changes
  every frame, so the cached LSTM inputs of the previous frame are wrong.)
* Model lookahead is **0 frames** (window ends at the newest sample; `tests/test_runtime_r4a.py::test_zero_lookahead`
  perturbs future frames and checks that earlier outputs are bit-identical, torch and ONNX).
* Cost per frame is ~0.83 GMAC (1.7 GFLOP): 12 LSTMs x 40 steps = 252 M, first spatial block 566 M,
  second spatial block 14 M (only the newest step is evaluated; exact trim), embeddings/head < 1 M.
* The model's time base is **60 Hz**; the headset runs at 72-120 Hz. `TimedWindow` samples the window on
  a 60 Hz grid anchored at the newest sample (linear position, nlerp rotation, interpolating only
  between past samples), so no delay is added and on an exact 60 Hz stream the window is bit-identical
  to training's. Godot timestamps are `Time.get_ticks_msec()/1000` (1 ms quantisation): that is
  up to ~1 mm of position jitter at 1 m/s inside the interpolation, not checked further.

## 2. Files

| file | role |
|---|---|
| `validation/sprint/export_onnx.py` | `--ckpt X --out Y.onnx`; single graph: raw window + static -> world rot matrices, hips, (contact). Feature construction (heading canonicalisation, velocities, head-relative hands), decode and canonical->world are *in the graph*. Writes `Y.json` and ONNX `metadata_props` (`vrhp_meta`). |
| `validation/sprint/runtime.py` | torch-free runtime: `TimedWindow` (rate conversion), `LearnedRuntime` (ORT + numpy FK), numpy twins. |
| `validation/vr_pose_validation/learned_estimator.py` | the sidecar `learned` estimator (`create_estimator("learned")`). |
| `validation/scripts/run_live_capture_recorder.py` | gains `--estimator learned --learned-model/-threads/-frame/-skeleton/-height-m`, `--no-raw-log`. |
| `validation/sprint/parity.py` | `gate` (ONNX vs torch full-sequence on 5 val clips) and `fixture` (Godot test inputs/expected). |
| `validation/sprint/bench_runtime.py` | per-stage latency percentiles. |
| `validation/sprint/run_model_parity.py` | one command: export, fixture, sidecar, headless Godot, teardown. |
| `godot/tests/run_model_parity.gd` | the Godot test. |
| `tests/test_runtime_r4a.py` | unit tests (numpy twins, window, zero lookahead, ONNX == torch for hmdposer and avatarposer on random-init nets). |

## 3. Commands (sayu, CPU only)

```sh
export UV_PROJECT_ENVIRONMENT=$HOME/.cache/venvs/godot-vr-humanoid-pose-r4a CUDA_VISIBLE_DEVICES=
uv sync --group ml --group runtime --group dev
CK=/mnt/s/code/godot-vr-humanoid-pose/.cache/sprint/runs/e1/best.pt          # any train.py checkpoint
uv run python -m validation.sprint.export_onnx --ckpt $CK --out .cache/r4a/e1.onnx
uv run python -m validation.sprint.parity gate --ckpt $CK --onnx .cache/r4a/e1.onnx   # exit 1 if > 1e-4 m
uv run python -m validation.sprint.bench_runtime --onnx .cache/r4a/e1.onnx --frames 10000 --threads 1 2
uv run python -m validation.sprint.run_model_parity --ckpt $CK --onnx .cache/r4a/e1.onnx --threads 2
# protocol-only floor (trivial estimator, no model):  ... run_model_parity ... --protocol-floor [--paced-hz 60]
```

`run_model_parity` picks a free port, starts the sidecar as a child, runs
`godot --headless --path godot --script res://tests/run_model_parity.gd`, and always stops the sidecar
(SIGTERM, then SIGKILL). Exit code = Godot's. Godot 4.7.2 (sayu) runs the project's 4.6 scripts without
changes; the only noise is the missing `godot-openvr-overlay` autoload (an untracked dependency).

## 4. Measured (E1 `hmdposer`, 10 000 steps; AMD Ryzen 7 9800X3D, onnxruntime 1.31 CPU, fp32)

### Parity (ONNX frame by frame vs `ModelPredictor.predict_clip_meta`, FK with GT bone lengths)

| clip (1000 frames) | max abs position | max abs hips | max rotation |
|---|---|---|---|
| cmu/01/01_01 | 1.4e-6 m | 6e-7 m | 0.0001 deg |
| cmu/01/01_14 | 9e-7 m | 7e-7 m | 0.0001 deg |
| xsens/S09/train_S09_A02 | 1.7e-6 m | 1.7e-6 m | 0.0001 deg |
| xsens/S09/train_S09_A02_mirror | 1.5e-6 m | 7e-7 m | 0.0001 deg |
| xsens/S09/train_S09_A03 | 6e-7 m | 5e-7 m | 0.0001 deg |

Gate 1e-4 m: pass with 60x margin. The Godot test (through the socket, JSON, GDScript) gets 6.6e-7 m / 0.0001 deg on the 60 Hz
replay and 7.2e-7 m on the 90 Hz resampled replay; resetting (explicit message and timestamp restart)
reproduces outputs exactly (max diff 0). These clips are "val" of `splits_v1`; the E1 weights saw only train clips.

### Per-frame compute (10 000 frames, one thread = 1 ORT intra-op thread)

Host: sayu, 16 hardware threads, load average 36-40 from other jobs during the runs, so p99/max are pessimistic
(p50 barely moves: a quiet-host spot check gave 15.7 ms at 1 thread). Real-world player CPUs are slower than
this Zen 5; treat the numbers as a floor. ms, p50 / p99 / max over 10 000 steps of a real tracker stream:

| stage | 1 thread | 2 threads |
|---|---|---|
| window build (rate conversion, numpy) | 0.31 / 0.63 / 2.4 | 0.23 / 0.46 / 3.1 |
| ORT run: feature construction + network + decode (the whole graph) | 15.6 / 25.5 / 34.6 | 8.2 / 14.3 / 29.0 |
| FK (14 joints, numpy) | 0.11 / 0.25 / 0.6 | 0.08 / 0.18 / 1.0 |
| **step** = window + ORT + FK | **16.0 / 26.2 / 35.5** | **8.6 / 14.8 / 31.1** |
| whole sidecar request (JSON parse, sample unpack, step, response JSON dump) | 16.4 / 25.5 / 42.4 | 10.0 / 19.8 / 1308 (one OS stall) |

Feature construction is inside the ORT graph (a few hundred tiny ops, well under 1 ms of the 15.6); the
separate numpy cost is the window (0.3 ms) and FK (0.1 ms). More threads do not help much: 4 threads gave
9.0 ms (12 LSTMs of 40 sequential 256x1024 matvecs bound the critical path), 8 threads got worse under contention.

### End-to-end latency ledger (frames at 90 Hz = 11.11 ms)

One frame is 11.11 ms. "Age" is how stale the pose on screen is, measured from the tracker sample.

| item | 1 ORT thread | 2 ORT threads | notes |
|---|---|---|---|
| model lookahead | **0 frames** | **0** | window ends at the newest sample (test_zero_lookahead) |
| rate conversion to the 60 Hz grid | 0 | 0 | newest sample enters exactly; older grid points are interpolated between past samples |
| filter lag | 0 | 0 | no smoothing in the path; Round 3's One-Euro / foot lock are not in the runtime and would add their own lag |
| compute, window + ORT + FK (p50 / p99) | 1.44 / 2.36 | 0.77 / 1.33 | 16.0 / 26.2 ms and 8.6 / 14.8 ms |
| sidecar JSON + dataclass + TCP | ~0.05 | ~0.1 | protocol floor measured with the trivial estimator: 0.26 ms p50 round trip incl. Godot stringify/parse |
| waiting for the render loop to pick the response up | 0.5 avg, up to 1 | 0.5 avg, up to 1 | the overlay polls once per rendered frame |
| **age at render, measured (Godot, requests at 60 Hz, stop-and-wait)** | p50 3.0, p99 4.5 (33 / 50 ms) | p50 1.5, p99 3.0 (16.7 / 33 ms) | host load ~45; see round trip below |

Reading: compute is the whole ledger. A 90 Hz render loop with fp32 and 2 threads shows a pose that is ~1.5-2
frames old (hands are pinned to the real trackers by IK, so this is the torso/legs/elbows); one thread is ~3 frames
and cannot sustain 60 Hz. The model itself adds no latency beyond compute.

### Round trip through the existing NDJSON protocol (Godot `StreamClient` <-> sidecar, localhost)

Godot headless test (`run_model_parity.gd`) against the real sidecar, one request in flight. "unpaced" sends
the next frame as soon as the response arrives; "paced" is a real-time tick (`stop-and-wait like live_overlay_debug.gd`:
a tick is skipped if the previous request is still in flight). RTT in ms, p50 / p99 / max:

| run | RTT | achieved rate | skipped ticks | age at render p50 / p99 |
|---|---|---|---|---|
| 2 threads, unpaced 60 Hz stream (900 frames) | 11.0 / 20.4 / 29.7 | back-to-back | n/a | n/a |
| 2 threads, paced 60 Hz, 6 s (two runs) | 10.8 / 19.5 / 26.2 and 12.5 / 24.5 / 35.8 | 58 and 53 Hz | 12 and 44 of 359 | 16.7 / 33.3 ms |
| 2 threads, paced 90 Hz, 6 s | 12.3 / 18.4 / 48.1 | **53 Hz** | 220 of 540 | 22.2 / 22.2 ms |
| 1 thread, unpaced 60 Hz stream | 19.8 / 31.7 / 37.4 | back-to-back | n/a | n/a |
| 1 thread, paced 60 Hz, 6 s | 20.7 / 40.4 / 42.9 | **31 Hz** | 173 of 359 | 33.3 / 50.0 ms |
| protocol floor (trivial estimator), paced 90 Hz | 0.26 / 1.06 / 2.56 | 90 Hz | 0 of 540 | 11.1 / 11.1 ms |

(host load average 35-45 for all rows; the Godot test asserts p99 <= `--max-p99-ms`, default 30, which the 1-thread paced run fails
by design, so run it with `--threads 2`.) Verdict: **90 Hz stop-and-wait is not achievable with this fp32 model
on this machine** (compute 8.6 ms p50 + 1 ms of protocol + tick quantisation gives 53 Hz); 60 Hz is nearly, with 2
ORT threads (53-58 Hz achieved, load dependent). Suggested fixes, in order of effort: (1) request at the model's 60 Hz and let
the renderer hold/extrapolate (hands are IK-pinned anyway); (2) keep up to 2 requests in flight (request ids
already exist, the sidecar answers in order), which removes the tick-skipping without a protocol change;
(3) a streaming-friendly retrain (section 8) or int8 (section 7) to cut compute 3-20x.

## 5. Running it in the headset (owner steps)

Machine: the Windows PC with SteamVR. The r4a worktree is on the shared drive as
`S:\code\godot-vr-humanoid-pose-r4a`, and the E1 export is already at `.cache\r4a\e1.onnx` in it (re-export
on sayu for another checkpoint). Branch `sprint/2026-10-r4a-runtime`.

```powershell
cd S:\code\godot-vr-humanoid-pose-r4a
uv sync --group runtime                       # onnxruntime + onnx; no torch needed at runtime
# once: the untracked OpenVR overlay addon (see godot/README.md)
New-Item -ItemType Junction -Path godot\addons\godot-openvr-overlay -Target S:\code\godot-openvr-overlay\project\addons\godot-openvr-overlay
# terminal 1: sidecar (the model sees your standing head-joint height: eye height minus ~10 cm)
uv run python -m validation.scripts.run_live_capture_recorder --estimator learned --learned-model .cache\r4a\e1.onnx --learned-threads 2 --learned-height-m 1.70 --no-raw-log
# terminal 2: the overlay scene (res://scenes/live_overlay_debug.tscn is the main scene)
& 'S:\lib\godot\bin\godot.windows.editor.x86_64.console.exe' --path godot --scene res://scenes/live_overlay_debug.tscn -- --estimate-hz 60
```

* `--estimate-hz 60` is the model's native rate (stop-and-wait stays inside the 16.7 ms budget with 2
  threads); `--estimate-hz 0` sends a request on every rendered frame and lets the round trip set the
  rate; `90` is the headset rate but see the ledger: fp32 compute does not fit 11 ms.
* The sidecar prints `[learned] 300 frames @ N Hz | compute p50 .. p99 .. ms` every 300 requests. Please
  paste that line with the verdict.
* Smoke test without a headset first: the same scene runs on synthetic 3-point input in a window
  (`-- --no-openvr-overlay --estimate-hz 60`); in the repo's headless check the overlay got 152
  responses in 400 frames with zero `estimate_error`.
* Calibration: the overlay's flow (floor target, countdown, T-pose) is unchanged and does not feed the
  model: height comes from `--learned-height-m` (the Godot side always sends 1.7), arm length and
  shoulder width from population ratios. Per-joint bone lengths follow from the height.
* What to judge (short): (1) left/right: raise the left hand, the skeleton's left arm follows (not
  swapped, not mirrored); (2) elbows bend the right way and do not flip while you rotate your wrists
  (hand-rotation inputs use an unverified Godot-grip -> training-hand-frame mapping, see below); (3)
  turn in place 360 degrees: the body should follow the head yaw without drifting; (4) walk a few steps and
  stop: feet should not slide away; (5) latency: wave the hand quickly and compare the mirrored hand
  with the controller. Report which of these fail and the sidecar stats line.

Open input-convention risks (none verifiable without the headset): hand tracker orientation frame
(`learned_estimator.HAND_M`, +-90 degree yaw: grip -Z to the forearm axis; the palm-roll is a guess);
`--learned-frame model` bypasses it. If wrists look wrong while position-driven parts are fine, the
Round 3 `--inputs pos|posrot` ablation checkpoints remove the dependency on hand rotations and are the
better runtime candidates. The estimator emits the 14 canonical joints with their world rotations in the
rest.py canonical rest pose (not avatar bones; a retarget step is Round 5), plus `neck` for the debug
overlay. Foot contact (if the checkpoint has the head) is exposed as `LearnedEstimator.last_contact`
but not yet sent over the protocol, and no foot lock / One-Euro filter is applied.

## 6. Protocol findings (stop-and-wait)

* Protocol overhead is negligible: with the trivial estimator, a 90 Hz paced Godot client gets
  round trips of 0.26 ms p50 / 1.06 ms p99 (JSON build + TCP + parse in both directions), zero skipped ticks.
* The cost is model compute. Stop-and-wait with one request in flight cannot hold 90 Hz when compute
  is ~10 ms: a response that arrives just after a tick makes the client skip that tick (measured
  achieved rates are in the table). Request ids already allow pipelining (depth 2, the sidecar answers in
  order); a client change that keeps one request in flight *per model frame* (not per render frame) at
  the model's 60 Hz and extrapolates/holds between responses removes the skipped-tick quantisation
  without a protocol change. Not implemented (client-side policy, Round 5).
* `LiveCaptureParquetWriter.append_raw_message` opens and appends a file for every message; with
  `--no-raw-log` it is skipped (the live overlay does not need it; recording mode does).

## 7. Experiment: int8 dynamic quantisation (not adopted)

`onnxruntime.quantization.quantize_dynamic` (QInt8) on the same graph: 1 thread step 5.9 ms p50 (2.7x faster; 3.9 ms with 2 threads), joints move by 2.6 cm on average vs the fp32 output (max 9.6 cm in the parity gate), while MPJPE vs GT on 3 clips x 500 frames barely changed (7.02 -> 7.07 cm). Not adopted: it
fails the 1e-4 m parity gate by design and the effect on jitter/foot sliding was not measured.

## 8. Path to a shippable addon

Ops observed in the exported graph (opset 17, 1393 nodes before ORT folding, 546 after; 11.48 M
parameters, 46 MB fp32):

* network: `LSTM` x12 (hidden 256, seq 40, batch 1), `Gemm`/`MatMul` (embeddings, QKV, out-proj, FFN 256-1024-256,
  head 1536-256-165, attention scores/values over 6 tokens), `LayerNormalization` x18, `Softmax` x6,
  `Relu` x6 (FFN), `LeakyRelu` x22;
* feature/decode glue: `Einsum` x2, `Atan`, `Sin`, `Cos`, `Sign`, `ReduceL2`, `Clip`, `Where`, `Sqrt`, `Div`,
  `Gather`, `Slice`, `Concat`, `Reshape`, `Transpose`, `Mod` (cross products), `ScatterND`, `Cast`.
  No dynamic shapes, no control flow, no custom ops, fixed `W`.

| | GDExtension + ONNX Runtime (CPU) | hand-written C++ inference |
|---|---|---|
| Effort | ~200 lines (godot-cpp, MIT; ORT, MIT); the graph and this directory's fixtures/tests carry over unchanged | LSTM, Linear, LayerNorm, 6-token attention, ReLU/LeakyReLU + ~100 lines of feature glue; per-op parity tests against the ONNX graph |
| Speed | ORT/MLAS as measured above (SIMD, threads) | a naive scalar loop is several times slower than MLAS (not measured here); needs a real GEMM (Eigen, or an AVX2/AVX-512/NEON kernel) to match |
| Footprint | `onnxruntime.dll` (roughly 10+ MB, not checked) + the 46 MB (23 MB if fp16-stored) weights | tens of KB of code + weights |
| Platforms | desktop Win/Linux first-class; Quest/Android needs the ORT mobile build | any target with a C++ compiler |
| Risk | GDExtension build/compat per Godot version (4.6 vs 4.7 not tried here); new arch ops need an ORT-supported opset (they are, today) | silent numeric drift on every retrain unless the per-op parity test stays in CI |

Recommendation: wrap ORT first (same graph, bit-for-bit the parity tested here), run inference on a worker
thread with a double-buffered "latest pose" so the render thread never blocks on a 10-16 ms call, and
re-export on every retrain. Hand-write only if a standalone/mobile target or binary size matters and the op
list stays this short. The bigger shipping lever is not the runtime but the **window recompute**: ~1.7 GFLOP
per frame is desktop-only. A streaming variant (canonicalise with a causal running heading rather than the
window-end heading, carry LSTM state) would cost ~35 M MAC per frame (about 24x less) and gives the true
stateful single-step graph; that is a training-recipe change for Round 3/5, not an export change.

## 9. Compatibility with Round 3 checkpoints

`export_onnx.load_checkpoint` reads `cfg.model` (`hmdposer` | `avatarposer`), `cfg.window` (sets `train.L`),
`cfg.inputs` (feature mask baked into the graph, local copy of `feat_mask`) and infers the head width from the
state dict (contact head: +2 logits -> a sigmoid `contact` output). Checked:

* real E1 (`hmdposer`, round 2): parity above;
* random-init `avatarposer` and `hmdposer`: unit tests (ONNX == torch predictor, zero lookahead, reset);
* real Round 3 `r3_contact` (contact head, 89 outputs) and `r3_jit` (jit-loss), exported and gated against **Round 3's own
  `train.py` at 17e46ff** (a `git archive` snapshot in `.cache/r4a/r3snap`, since this worktree is based on `184e846`):
  max |dpos| 9.4e-7 m / 1.0e-6 m, and the contact probabilities match to 2.4e-6.

Not covered: Round 3's inference-time post-processes in `ModelPredictor` (`postfix`, `footlock`, One-Euro `smooth`,
`honest_fk`) are numpy steps outside the network and are **not** in this runtime; after Round 3 lands, port the ones
that make the final recipe (they are causal per-frame / scan filters, so they fit `LearnedEstimator.estimate`).
