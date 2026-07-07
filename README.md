# godot-vr-humanoid-pose

Full-body humanoid pose estimation for PC-VR avatars in [Godot](https://godotengine.org/),
driven from sparse mainstream VR tracking — HMD plus two controllers, with
optional hip, chest, elbow, knee, and foot trackers.

> **Status: early work in progress — not yet usable.** This repository is
> currently research and validation scaffolding: a motion-capture validation
> harness, a procedural baseline solver, synthetic tracker-noise modeling, a
> learned upper-body experiment, and a live VR capture pipeline. There is no
> packaged, drop-in Godot addon yet. It is public to share the approach and to
> build in the open on top of prior sparse-tracker research.

## What it targets

The goal is convincing social-VR-style avatar pose from sparse VR tracking,
kept open source and suitable for commercial games. "IK" here means driving a
full humanoid skeleton from the trackers a mainstream VR user actually has — not
a general-purpose inverse-kinematics library.

- **3-point is the floor.** A convincing HMD + two-controller mode with a
  procedural lower body and foot grounding, for users without extra trackers.
- **6-point+ FBT is a co-equal target**, not deferred polish: tracker
  override/blending, calibration UX, and per-tracker dropout handling are
  first-class.
- **Tracking loss and transitions are first-class** — holstered controllers,
  hands out of view, stale trackers, and seated↔standing transitions are design
  requirements, not edge cases.
- **Latency budget: ≤75–80 ms** movement→avatar pose, tracked per estimator
  configuration.

The quality yardstick is a *parity floor* against an in-repo procedural baseline
(never regress plausibility/stability) plus a *fidelity stretch* spent where it
is perceptually visible: upper body, torso facing, and transitions.

## Approach

- **Procedural "Basis" solver** — calibration, virtual spine/hips, two-bone arms
  with swivel strategies, and a sticky-feet gait planner. The always-available
  baseline and correctness floor. Named after and modeled on the procedural
  avatar posing in [Basis](https://github.com/BasisVR/Basis) (MIT), an
  open-source social-VR platform.
- **Learned upper-body residual** — a small causal model (GRU/TCN/MLP variants)
  over sparse tracker windows, decoded through a differentiable, MANIKIN-style
  runtime layer (torso heading/lean, shoulder offsets, elbow swivel) on top of
  the procedural base pose, with FK rotation targets and runtime-matched temporal
  losses.
- **Tracker-noise modeling** — a synthetic corruption model (dropout run-length,
  hold-last vs invalid-confidence, reacquisition excursion, jitter, latency,
  calibration bias, prediction overshoot) with per-role profiles that can be
  *fitted from real device logs* rather than invented.
- **Validation harness** — CMU ASF/AMC intake into a canonical humanoid
  wide-Parquet cache, regional MPJPE/MPJVE, contact/slip/grounding, gait
  plausibility, frequency-band diagnostics, worst-clip tables, and plots.
- **Live VR capture** — a Godot overlay scene plus a Python sidecar over a
  localhost JSON socket, with T-pose calibration and per-session Parquet output.

## Relationship to prior work

This project builds directly on the sparse-tracker VR pose literature, above all
**[SparsePoser](https://upc-virvig.github.io/SparsePoser/)** (Ponton et al.,
ACM TOG 2023) and its publicly released, CC BY 4.0 Xsens motion corpus and Unity
live-calibration flow, whose authors kindly confirmed the license terms for this
use. It also draws ideas from AvatarPoser, EgoPoser, MANIKIN (ETH Zürich SIPLab),
[DragPoser](https://upc-virvig.github.io/DragPoser/) (UPC-ViRVIG), LGSIK-Poser,
SSR-HMR, and VRArmIK. See [`validation/RESEARCH_NOTES.md`](validation/RESEARCH_NOTES.md)
for the implementation-reference notes and a cross-method read of how each handles
tracker noise, spine/shoulder inference, and torso-heading ambiguity.

The procedural baseline solver is named after and modeled on
[Basis](https://github.com/BasisVR/Basis) (MIT), an open-source social-VR
platform, used here as a ToS-safe open proxy for social-VR avatar behavior.

## Data and licensing stance

Training and evaluation deliberately stay on commercially-usable sources — CMU
raw mocap, our own VR capture, and the CC BY 4.0 SparsePoser Xsens corpus.
Non-commercial and SMPL-restricted assets (AMASS-trained checkpoints, BOXRR-23,
Who-Is-Alyx, etc.) are treated as eval-only or off-limits per their licenses.
Details in [`PLAN.md`](PLAN.md) (Data Policy) and
[`validation/DATA_STRATEGY.md`](validation/DATA_STRATEGY.md).

## Repository layout

| Path | Contents |
| --- | --- |
| [`PLAN.md`](PLAN.md) | Authoritative goals, architecture direction, data policy, milestones, open questions. |
| [`godot/`](godot/) | Godot-side runtime, live overlay/debug harness, and tests. |
| [`validation/`](validation/) | Python validation harness, estimators, synthetic trackers, tracker-noise fitting, and research notes. |
| [`tests/`](tests/) | Python test suite. |

## Development

Python tooling uses [uv](https://docs.astral.sh/uv/); Godot targets 4.6+
(developed against a 4.7 build).

```sh
uv sync
uv run python -m pytest         # Python tests
```

Formatting/lint/typecheck (ruff + ty) run via pre-commit; see
[`validation/README.md`](validation/README.md) for the validation and
live-capture command reference.

## License

Licensed under the MIT License — see [LICENSE](LICENSE).

Maintained by Hiina (<hiina@hiina.space>).
