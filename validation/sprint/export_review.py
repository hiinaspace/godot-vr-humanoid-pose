# ruff: noqa: E501
"""Owner review packet exporter (Round 4b): pick clips, run methods, inline one HTML viewer.

    uv run python -m validation.sprint.export_review --ckpt .cache/sprint/runs/e1/best.pt \
        --methods gt,basis,learned

Methods: ``gt``, ``basis``, ``learned`` (= the first ``--ckpt``) and, for extra checkpoints given as
``--ckpt label=path``, the label. Everything downstream (viewer colours, A/B pairing) is driven by
the method list in the exported JSON, so a Round-3 model is one ``--ckpt`` away.

Pipeline: (1) rank candidate clips (CMU test minus content-duplicate subject 84, Xsens val subject
S09 without mirror twins) by learned MPJPE-full, evaluating the model every ``--rank-stride`` frames
and Basis on every frame; (2) select 6+6 worst, 4 learned-beats-Basis, 4 sit/crouch, plus the
2026-07-06 live capture (no GT); (3) cut each clip to its worst 8 s window, run every method at 60 Hz
there; (4) compute oracle-v2 columns on the window; (5) write quantised int16-mm joint arrays, GT
foot-contact flags, error curves and jump lists as one JSON blob into ``viewer/index.html``
(``viewer/template.html`` with the blob substituted for ``PLACEHOLDER``).

All metric / contact definitions come from ``oracle_v2`` (floor = 2nd percentile of GT lowest foot
joint, contact = GT foot < floor + 5 cm and GT horizontal speed < 10 cm/s); the window floor is
used for both the columns and the markers so the two never disagree.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from validation.sprint import conventions as C
from validation.sprint import oracle as O
from validation.sprint import oracle_v2 as O2
from validation.sprint import rest as R
from validation.sprint import splits as S

HERE = Path(__file__).resolve().parent
VIEWER = HERE / "viewer"
DEFAULT_DATA_ROOT = Path("/mnt/s/code/godot-vr-humanoid-pose")
LIVE_SESSION = "live_capture/session=20260706T215223-dcc7f918"
LIVE_RUNS = {2: "range_of_motion", 3: "sit_stand"}
EXCLUDE_SUBJECTS = {"84", "141"}  # content duplicates of train subjects (round-4b brief)
WINDOW_S, AB_S = 8.0, 4.0
FPS = int(C.FPS)
MIN_FRAMES = 240
PLACEHOLDER = "/*__REVIEW_DATA__*/null"


def patch_paths(data_root: Path) -> None:
    """The worktree has no .cache; read the data caches of the main checkout."""
    S.CMU_DIR = data_root / ".cache/cmu/parquet/canonical_full"
    S.XSENS_DIR = data_root / ".cache/sparseposer/parquet/canonical"
    O.FITTED_NOISE_PATH = data_root / "validation/reports/fitted_tracker_noise.json"


# --- method runners -------------------------------------------------------------------------


def _worker(args):
    """Pool worker: calibration, inputs and the stateful Basis run for one clip (full rate)."""
    data_root, clip_id, variant = args
    patch_paths(Path(data_root))
    from validation.sprint import baselines

    clip = C.load_clip(S.clip_path(clip_id))
    calib, inputs = O.make_calib(clip), O.make_inputs(clip, variant)
    pos = O2.predicted_positions(baselines.create("basis").predict_clip(inputs, calib), clip, calib)
    return clip_id, dataclasses.astuple(calib), inputs.astype(np.float32), pos.astype(np.float32)


def basis_from_inputs(inputs: np.ndarray, calib: O.Calib) -> np.ndarray:
    from validation.sprint import baselines

    p = baselines.create("basis").predict_clip(inputs.astype(np.float64), calib)
    assert p.positions is not None
    return np.asarray(p.positions).astype(np.float32)


class Learned:
    """A trained checkpoint evaluated at arbitrary frame indices (causal 40-frame windows)."""

    def __init__(self, label: str, path: Path):
        import torch

        from validation.sprint import train as T

        self.torch, self.T, self.label, self.path = torch, T, label, path
        self.pred = T.ModelPredictor.load(path, dev="cpu")

    @property
    def name(self) -> str:
        return self.pred.name

    def positions_at(
        self,
        inputs: np.ndarray,
        calib: O.Calib,
        dataset: str,
        idx: np.ndarray,
        lengths: np.ndarray,
        bs: int = 256,
    ) -> np.ndarray:
        """Same computation as ``ModelPredictor.predict_clip_meta`` restricted to frames ``idx``."""
        torch, T, p = self.torch, self.T, self.pred
        with torch.no_grad():
            x = torch.tensor(inputs, dtype=torch.float32)
            x = torch.cat([x[:1].expand(T.L, -1), x])
            cal = torch.tensor([[calib.height_m, calib.arm_length_m, calib.shoulder_width_m]])
            ds = torch.tensor([0 if dataset == "cmu" else 1])
            st = T.static_feats(cal, ds, p.cfg.get("skel", True))
            rots, roots = [], []
            for i in range(0, len(idx), bs):
                t = torch.as_tensor(idx[i : i + bs])
                raw = x[t[:, None] + torch.arange(T.L + 1)[None]]
                feats, Q, origin, head_pc = T.make_feats(raw)
                rot6, hr = T.decode(p.model(feats, st.expand(len(t), -1))[:, -1:].float())
                m = T.sixd2mat(rot6[:, 0])
                m = torch.cat([(Q.transpose(-1, -2) @ m[:, 0])[:, None], m[:, 1:]], 1)
                hips = (Q.transpose(-1, -2) @ (head_pc[:, -1] + hr[:, 0])[..., None]).squeeze(
                    -1
                ) + origin
                rots.append(T.mat2quat(m).double().numpy())
                roots.append(hips.double().numpy())
        return R.fk(np.concatenate(rots), np.concatenate(roots), lengths).astype(np.float32)


# --- live capture ---------------------------------------------------------------------------


def _nlerp_track(t_src, q_src, t_dst):
    """Per-frame nlerp with sign continuity (frames are ~34 ms apart, so nlerp ~ slerp)."""
    q = q_src.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    out = np.stack([np.interp(t_dst, t_src, q[:, k]) for k in range(4)], axis=-1)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def load_live(data_root: Path, clip_index: int):
    """(inputs[T,24] at 60 Hz, calib) from the live capture's measured joints.

    The live recorder logged the three measured joints (head, left_hand, right_hand; ``estimated``
    False) in the Godot world frame at ~29 Hz. They are resampled to 60 Hz and pushed through the same
    synthetic-tracker generator as the oracle (``clean`` variant: fixed mounting offset only), so the
    live stream looks to the models like training inputs. No controller-grip calibration.
    """
    import polars as pl

    from validation.vr_pose_validation.geometry import Transform
    from validation.vr_pose_validation.synthetic_trackers import iter_three_point_samples

    j = pl.read_parquet(data_root / ".cache" / LIVE_SESSION / "joints.parquet").filter(
        (pl.col("clip_index") == clip_index) & ~pl.col("estimated")
    )
    names = ("head", "left_hand", "right_hand")
    tracks = {}
    for n in names:
        d = j.filter(pl.col("joint") == n).sort("frame_index")
        tracks[n] = (
            d["timestamp_seconds"].to_numpy(),
            d.select("joint_px", "joint_py", "joint_pz").to_numpy(),
            d.select("joint_qx", "joint_qy", "joint_qz", "joint_qw").to_numpy(),
        )
    t0, t1 = tracks["head"][0][0], tracks["head"][0][-1]
    td = np.arange(t0, t1, 1.0 / FPS)
    res = {}
    for n in names:
        ts, p, q = tracks[n]
        res[n] = (
            np.stack([np.interp(td, ts, p[:, k]) for k in range(3)], -1),
            _nlerp_track(ts, q, td),
        )
    cfg = O._tracker_config("clean", f"live/{clip_index}")
    frames = (
        {
            n: Transform(
                tuple(map(float, res[n][0][t])),  # ty: ignore[invalid-argument-type]
                tuple(map(float, res[n][1][t])),  # ty: ignore[invalid-argument-type]
            )
            for n in names
        }
        for t in range(len(td))
    )
    out = np.zeros((len(td), C.INPUT_DIM))
    for t, s in enumerate(iter_three_point_samples(frames, cfg)):
        for k, st in enumerate((s.hmd, s.left_hand, s.right_hand)):
            out[t, 7 * k : 7 * k + 3] = st.transform.position
            out[t, 7 * k + 3 : 7 * k + 7] = st.transform.rotation
            out[t, 21 + k] = 1.0 if st.confidence > 0.5 else 0.0
    h = 1.7  # calibration_json.human_height_m of the session; arm / shoulder from the xsens template
    tpl = C.TEMPLATES["xsens"]
    arm = h * (tpl["left_elbow"][1] + tpl["left_hand"][1])
    sw = h * (
        tpl["left_shoulder"][1] * tpl["left_shoulder"][0][0]
        - tpl["right_shoulder"][1] * tpl["right_shoulder"][0][0]
    )
    return out, O.Calib(h, arm, sw)


# --- analysis helpers -----------------------------------------------------------------------


def per_frame_err(pos: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pos - gt, axis=-1)[:, C.NONINPUT_IDX].mean(1)


def gt_contact(gt: np.ndarray) -> tuple[np.ndarray, float]:
    """[T,2] bool GT foot-contact (oracle-v2 rule, window floor) and the floor height."""
    gf = gt[:, O2.FEET]
    floor = float(np.percentile(gf[..., 1].min(1), 2))
    c = np.zeros((len(gt), 2), bool)
    c[1:-1] = (gf[1:-1, :, 1] < floor + O2.CONTACT_H) & (O2._hspeed(gf) < O2.CONTACT_V)
    c[0], c[-1] = c[1], c[-2]
    return c, floor


def pop_frames(gt: np.ndarray | None, pr: np.ndarray) -> list[int]:
    lp = R.swing_from_positions(pr)[:, O2.POP_JOINTS]
    dp = C.qgeodesic_deg(lp[1:], lp[:-1])
    ev = dp > O2.POP_PRED_DEG
    if gt is not None:
        lg = R.swing_from_positions(gt)[:, O2.POP_JOINTS]
        ev &= C.qgeodesic_deg(lg[1:], lg[:-1]) < O2.POP_GT_DEG
    start = ev.copy()
    start[1:] &= ~ev[:-1]
    return sorted({int(f) + 1 for f in np.where(start.any(1))[0]})


def separated_top(score: np.ndarray, k: int, gap: int) -> list[int]:
    out: list[int] = []
    for i in np.argsort(-score):
        if all(abs(int(i) - o) >= gap for o in out):
            out.append(int(i))
        if len(out) >= k:
            break
    return sorted(out)


def window_metrics(clip: C.ClipData, pos: np.ndarray) -> dict:
    m = O2.clip_metrics(clip, pos, O.Prediction(positions=pos))
    n = clip.n_frames
    sv, gv, ev = m["skate_vals"], m["ground_vals"], m["swivel_vals"]
    return {
        "full_cm": float(m["full"].mean() * 100),
        "upper_cm": float(m["upper"].mean() * 100),
        "lower_cm": float(m["lower"].mean() * 100),
        "hand_cm": float(m["hand"].mean() * 100),
        "jitter": float(m["jit_pred_sum"] / m["jit_n"]),
        "jitter_gt": float(m["jit_gt_sum"] / m["jit_n"]),
        "skate_cm_s": None if np.isnan(m["skate_clip"]) else float(m["skate_clip"]),
        "pen_p95_cm": float(np.percentile(np.maximum(0, -gv), 95)) if gv.size else None,
        "float_p95_cm": float(np.percentile(np.maximum(0, gv), 95)) if gv.size else None,
        "pops_per_min": float(m["pops"] / (n / C.FPS / 60.0)),
        "swivel_deg": float(ev.mean()) if ev.size else None,
        "swivel_p95_deg": float(np.percentile(ev, 95)) if ev.size else None,
        "contact_frames": int(sv.size),
    }


# --- ranking and selection ------------------------------------------------------------------


@dataclasses.dataclass
class Cand:
    clip_id: str
    dataset: str
    n: int
    score_learned: float = np.nan  # mean MPJPE-full (cm) over sampled frames
    score_basis: float = np.nan
    win: tuple[int, int] = (0, 0)
    crouch_frac: float = 0.0


def best_window(score: np.ndarray, idx: np.ndarray, n: int, w: int) -> tuple[int, int]:
    """Window of length w maximising the mean of ``score`` (sampled at frame indices ``idx``)."""
    if n <= w:
        return 0, n
    best, bs = -1e18, 0
    for s in range(0, n - w + 1, FPS // 2):
        sel = (idx >= s) & (idx < s + w)
        if sel.sum() < 4:
            continue
        v = float(score[sel].mean())
        if v > best:
            best, bs = v, s
    return bs, bs + w


def pool_ids() -> list[str]:
    ids = [c for c in S.clip_ids("cmu_test") if c.split("/")[1] not in EXCLUDE_SUBJECTS]
    ids += [c for c in S.clip_ids("xsens_val") if not c.endswith("_mirror")]
    return ids


def _npz(basis_dir: Path, cid: str) -> Path:
    return basis_dir / (cid.replace("/", "__") + ".npz")


def rank_pool(args, learned: Learned, log) -> tuple[list[Cand], dict]:
    cache = (
        args.cache
        / f"rank_{learned.label}_{learned.path.parent.name}_{args.variant}_s{args.rank_stride}.json"
    )
    ids = pool_ids()
    log(f"pool {len(ids)} clips; basis pass")
    basis_dir = args.cache / f"basis_{args.variant}"
    basis_dir.mkdir(exist_ok=True)
    todo = [i for i in ids if not _npz(basis_dir, i).exists()]
    if todo:
        with ProcessPoolExecutor(args.workers) as ex:
            for cid, calib, inputs, pos in ex.map(
                _worker, [(str(args.data_root), i, args.variant) for i in todo], chunksize=1
            ):
                np.savez(_npz(basis_dir, cid), inputs=inputs, pos=pos, calib=np.array(calib))
    log("basis done; learned rank pass")
    results = json.loads(cache.read_text()) if cache.exists() else {}
    cands: list[Cand] = []
    store: dict = {}
    for k, cid in enumerate(ids):
        clip = C.load_clip(S.clip_path(cid))
        if clip.n_frames < MIN_FRAMES:
            continue
        z = np.load(_npz(basis_dir, cid))
        gt = clip.positions
        idx = np.arange(0, clip.n_frames, args.rank_stride)
        if cid not in results:
            t0 = time.time()
            lp = learned.positions_at(
                z["inputs"],
                O.Calib(*map(float, z["calib"])),
                clip.dataset,
                idx,
                R.bone_lengths(gt),
                bs=512,
            )
            results[cid] = {"err_l": (per_frame_err(lp, gt[idx]) * 100).tolist()}
            cache.write_text(json.dumps(results))
            log(f"  ranked {k + 1}/{len(ids)} {cid} {len(idx)} fr {time.time() - t0:.0f}s")
        eb = per_frame_err(z["pos"], gt) * 100
        el = np.array(results[cid]["err_l"])
        c = Cand(cid, clip.dataset, clip.n_frames)
        c.score_learned, c.score_basis = float(el.mean()), float(eb[idx].mean())
        c.win = best_window(el, idx, clip.n_frames, int(WINDOW_S * FPS))
        c.crouch_frac = float((gt[:, 0, 1] < 0.65).mean())
        store[cid] = {"idx": idx, "el": el, "eb": eb[idx], "hips": gt[idx, 0, 1]}
        cands.append(c)
    return cands, store


def select(cands: list[Cand], store: dict) -> list[tuple[Cand, str]]:
    chosen: list[tuple[Cand, str]] = []
    used: set[str] = set()

    def take(pool: list[Cand], k: int, group: str, key):
        n = 0
        for c in sorted(pool, key=key):
            if c.clip_id in used:
                continue
            used.add(c.clip_id)
            chosen.append((c, group))
            n += 1
            if n == k:
                break

    take([c for c in cands if c.dataset == "cmu"], 6, "worst-cmu-test", lambda c: -c.score_learned)
    take(
        [c for c in cands if c.dataset == "xsens"], 6, "worst-xsens-s09", lambda c: -c.score_learned
    )
    take(cands, 4, "learned-beats-basis", lambda c: -(c.score_basis - c.score_learned))
    take(cands, 4, "sit-crouch", lambda c: -c.crouch_frac)
    w = int(WINDOW_S * FPS)
    for c, g in chosen:  # re-centre the window on the reason the clip was picked
        d = store[c.clip_id]
        if g == "learned-beats-basis":
            c.win = best_window(d["eb"] - d["el"], d["idx"], c.n, w)
        elif g == "sit-crouch":
            c.win = best_window(-d["hips"], d["idx"], c.n, w)
    return chosen


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def quant_pos(pos: np.ndarray, origin: np.ndarray) -> np.ndarray:
    q = np.round((pos - origin) * 1000.0)
    assert np.abs(q).max() < 32000, "int16 mm overflow"
    return q.astype("<i2")


# --- clip record ----------------------------------------------------------------------------


def build_clip_record(cid, group, win, method_pos, gt, clip, extra, clip_scores) -> dict:
    f0, f1 = win
    n = f1 - f0
    gap = FPS * 3 // 4
    ref = gt if gt is not None else next(v for k, v in method_pos.items() if k != "gt")
    origin = np.array([ref[0, 0, 0], 0.0, ref[0, 0, 2]], dtype=np.float32)
    rec: dict = {
        "id": cid,
        "group": group,
        "has_gt": gt is not None,
        "win": [int(f0), int(f1)],
        "n": int(n),
        "origin": [float(origin[0]), float(origin[2])],
        "pos": {m: b64(quant_pos(p, origin)) for m, p in method_pos.items()},
        "clip_scores": clip_scores,
        **extra,
    }
    metrics: dict = {}
    err: dict = {}
    jump: dict = {}
    if gt is not None and clip is not None:
        contact, floor = gt_contact(gt)
        rec["floor"] = floor
        rec["contact"] = b64(contact[:, 0].astype(np.uint8) | (contact[:, 1].astype(np.uint8) << 1))
        wclip = dataclasses.replace(
            clip,
            positions=np.ascontiguousarray(gt, dtype=np.float64),
            rotations=np.ascontiguousarray(clip.rotations[f0:f1]),
        )
        for m, p in method_pos.items():
            metrics[m] = window_metrics(wclip, p.astype(np.float64))
            e = per_frame_err(p, gt)
            err[m] = b64(np.round(e * 1000).astype("<u2"))
            if m == "gt":
                continue
            j = [
                {"f": f, "k": "err", "v": round(float(e[f] * 100), 1)}
                for f in separated_top(e, 5, gap)
            ]
            j += [{"f": f, "k": "pop"} for f in pop_frames(gt, p)[:6]]
            sk = np.zeros(n)
            sk[1:-1] = (O2._hspeed(p[:, O2.FEET]) * contact[1:-1]).max(1) * 100.0
            if sk.max() > 0:
                j += [
                    {"f": f, "k": "skate", "v": round(float(sk[f]), 1)}
                    for f in separated_top(sk, 3, gap)
                ]
            jump[m] = sorted(j, key=lambda r: r["f"])
    else:
        rec["floor"] = float(np.percentile(ref[:, O2.FEET][..., 1].min(1), 2))
        others = [m for m in method_pos if m != "gt"]
        base = method_pos[others[0]]
        for m in others:
            d = np.linalg.norm(method_pos[m] - base, axis=-1)[:, C.NONINPUT_IDX].mean(1)
            err[m] = b64(np.round(d * 1000).astype("<u2"))  # distance to the first method (no GT)
            pf = pop_frames(None, method_pos[m])
            j = (
                [
                    {"f": f, "k": "diff", "v": round(float(d[f] * 100), 1)}
                    for f in separated_top(d, 5, gap)
                ]
                if m != others[0]
                else []
            )
            j += [{"f": f, "k": "pop"} for f in pf[:6]]
            jump[m] = sorted(j, key=lambda r: r["f"])
            metrics[m] = {
                "pops_per_min": len(pf) / (n / C.FPS / 60.0),
                "jitter": float(O.jerk_norms(method_pos[m].astype(np.float64)).mean()),
            }
    rec["metrics"], rec["err"], rec["jump"] = metrics, err, jump
    # A/B segment: the 4 s part of the window where Basis and learned differ most
    if "basis" in method_pos and "learned" in method_pos:
        w = int(AB_S * FPS)
        if n <= w:
            seg = (0, n)
        else:
            if gt is not None:
                diff = np.abs(
                    per_frame_err(method_pos["basis"], gt)
                    - per_frame_err(method_pos["learned"], gt)
                )
            else:
                diff = np.linalg.norm(method_pos["basis"] - method_pos["learned"], axis=-1).mean(1)
            cs = np.concatenate([[0], np.cumsum(diff)])
            s = int(np.argmax(cs[w:] - cs[:-w]))
            seg = (s, s + w)
        rec["ab_seg"] = [int(seg[0]), int(seg[1])]
    return rec


# --- main -----------------------------------------------------------------------------------


def write_html(blob: dict, out: Path, log) -> None:
    tpl = (VIEWER / "template.html").read_text()
    assert PLACEHOLDER in tpl
    html = tpl.replace(PLACEHOLDER, json.dumps(blob, separators=(",", ":")))
    out.write_text(html)
    log(f"wrote {out} {len(html) / 1e6:.2f} MB, {len(blob['clips'])} clips")
    assert len(html) <= 8e6, "over the 8 MB budget"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", action="append", help="[label=]path; the first one is `learned`")
    ap.add_argument("--methods", default="gt,basis,learned")
    ap.add_argument("--variant", choices=O.VARIANTS, default="clean")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--rank-stride", type=int, default=6)
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-live", action="store_true")
    ap.add_argument("--cache", type=Path, default=S.REPO / ".cache/r4b")
    ap.add_argument("--out", type=Path, default=VIEWER / "index.html")
    ap.add_argument("--rebuild-html", action="store_true", help="re-inline cached data only")
    args = ap.parse_args(argv)
    args.cache.mkdir(parents=True, exist_ok=True)
    blob_path = args.cache / "review_data.json"

    def log(*a):
        print(time.strftime("%H:%M:%S"), *a, flush=True)

    if args.rebuild_html:
        write_html(json.loads(blob_path.read_text()), args.out, log)
        return 0
    if not args.ckpt:
        ap.error("--ckpt is required")

    patch_paths(args.data_root)
    import torch

    torch.set_num_threads(args.threads)
    ckpts: dict[str, Learned] = {}
    for i, spec in enumerate(args.ckpt):
        label, path = spec.split("=", 1) if "=" in spec else ("", spec)
        label = "learned" if i == 0 else (label or f"learned{i + 1}")
        ckpts[label] = Learned(label, Path(path))
    methods = [m for m in args.methods.split(",") if m]
    for m in methods:
        assert m in ("gt", "basis") or m in ckpts, f"unknown method {m}; ckpt labels {list(ckpts)}"
    primary = ckpts["learned"]

    cands, store = rank_pool(args, primary, log)
    chosen = select(cands, store)
    log("selected:")
    for c, g in chosen:
        log(
            f"  {g:20s} {c.clip_id:30s} n={c.n} win={c.win} learned={c.score_learned:.1f} basis={c.score_basis:.1f} crouch={c.crouch_frac:.2f}"
        )

    clips_out: list[dict] = []
    basis_dir = args.cache / f"basis_{args.variant}"
    for k, (c, g) in enumerate(chosen):
        t0 = time.time()
        clip = C.load_clip(S.clip_path(c.clip_id))
        z = np.load(_npz(basis_dir, c.clip_id))
        calib = O.Calib(*map(float, z["calib"]))
        f0, f1 = c.win
        gt = clip.positions[f0:f1].astype(np.float32)
        mp: dict[str, np.ndarray] = {}
        for m in methods:
            if m == "gt":
                mp[m] = gt
            elif m == "basis":
                mp[m] = z["pos"][f0:f1]
            else:
                mp[m] = ckpts[m].positions_at(
                    z["inputs"],
                    calib,
                    clip.dataset,
                    np.arange(f0, f1),
                    R.bone_lengths(clip.positions),
                )
        extra = {
            "dataset": clip.dataset,
            "activity": clip.activity_class,
            "n_clip": c.n,
            "variant": args.variant,
        }
        scores = {
            "basis_full_cm": round(c.score_basis, 2),
            "learned_full_cm": round(c.score_learned, 2),
        }
        clips_out.append(build_clip_record(c.clip_id, g, (f0, f1), mp, gt, clip, extra, scores))
        log(f"  [{k + 1}/{len(chosen)}] {c.clip_id} {time.time() - t0:.0f}s")

    if not args.no_live:
        for ci, name in LIVE_RUNS.items():
            inputs, calib = load_live(args.data_root, ci)
            n = len(inputs)
            basis = basis_from_inputs(inputs, calib)
            lengths = np.linalg.norm(
                C.template_offsets("xsens", calib.height_m, calib.arm_length_m), axis=-1
            )
            idx = np.arange(0, n, args.rank_stride)
            lp = primary.positions_at(inputs, calib, "xsens", idx, lengths, bs=512)
            diff = np.linalg.norm(basis[idx] - lp, axis=-1)[:, C.NONINPUT_IDX].mean(1)
            f0, f1 = best_window(diff, idx, n, int(WINDOW_S * FPS))
            mp = {}
            for m in methods:
                if m == "gt":
                    continue
                mp[m] = (
                    basis[f0:f1]
                    if m == "basis"
                    else ckpts[m].positions_at(inputs, calib, "xsens", np.arange(f0, f1), lengths)
                )
            extra = {
                "dataset": "live",
                "activity": name,
                "n_clip": n,
                "variant": "live (device poses)",
            }
            clips_out.append(
                build_clip_record(
                    f"live/20260706/{name}", "live-capture", (f0, f1), mp, None, None, extra, {}
                )
            )
            log(f"  live {name} window {f0}-{f1} of {n}")

    blob = {
        "version": 1,
        "fps": FPS,
        "oracle": O2.ORACLE_VERSION,
        "joints": list(C.JOINTS),
        "parents": [int(p) for p in C.PARENT],
        "methods": [{"id": m, "label": m} for m in methods],
        "ckpts": {m: {"path": str(c.path), "name": c.name} for m, c in ckpts.items()},
        "variant": args.variant,
        "excluded_subjects": sorted(EXCLUDE_SUBJECTS),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "clips": clips_out,
    }
    blob_path.write_text(json.dumps(blob, separators=(",", ":")))
    write_html(blob, args.out, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
