"""Phase-recognition metrics, computed per video and then averaged (the Cholec80 convention)."""
from __future__ import annotations

import numpy as np


def video_metrics(y: np.ndarray, pred: np.ndarray, n_phases: int = 7) -> dict:
    """Accuracy plus per-phase precision, recall and Jaccard for one video.
    A phase that is absent from both truth and prediction is NaN and skipped in averages."""
    out = {"accuracy": float(np.mean(y == pred)), "precision": [], "recall": [], "jaccard": []}
    for p in range(n_phases):
        t, q = y == p, pred == p
        inter, union = np.sum(t & q), np.sum(t | q)
        out["precision"].append(float(inter / q.sum()) if q.sum() else float("nan"))
        out["recall"].append(float(inter / t.sum()) if t.sum() else float("nan"))
        out["jaccard"].append(float(inter / union) if union else float("nan"))
    return out


def summarise(per_video: list[dict], n_boot: int = 5000, seed: int = 42) -> dict:
    """Mean over videos with a video-level bootstrap 95% CI; per-phase means over videos."""
    acc = np.array([v["accuracy"] for v in per_video])
    P, R, J = (np.array([v[k] for v in per_video], dtype=float) for k in ("precision", "recall", "jaccard"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(acc), (n_boot, len(acc)))

    def ci(values_per_video: np.ndarray) -> list[float]:
        boots = np.nanmean(values_per_video[idx], axis=1)
        return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    with np.errstate(invalid="ignore"):
        jv, pv, rv = np.nanmean(J, axis=1), np.nanmean(P, axis=1), np.nanmean(R, axis=1)
    return {"n_videos": int(len(acc)),
            "accuracy": float(acc.mean()), "accuracy_sd": float(acc.std(ddof=1)), "accuracy_ci95": ci(acc),
            "jaccard": float(np.nanmean(jv)), "jaccard_ci95": ci(jv),
            "precision": float(np.nanmean(pv)), "recall": float(np.nanmean(rv)),
            "per_phase": {"precision": np.nanmean(P, axis=0).tolist(), "recall": np.nanmean(R, axis=0).tolist(),
                          "jaccard": np.nanmean(J, axis=0).tolist()}}


def paired_permutation_p(a, b, n_perm: int = 20000, seed: int = 42) -> float:
    """Two-sided paired test on per-video differences."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    perm = np.abs((rng.choice([-1.0, 1.0], (n_perm, len(d))) * d).mean(axis=1))
    return float((np.sum(perm >= abs(d.mean()) - 1e-12) + 1) / (n_perm + 1))


def segments(seq: np.ndarray) -> list[tuple[int, int, int]]:
    """Run-length encode a label sequence as (label, start, end_exclusive)."""
    seq = np.asarray(seq)
    cuts = np.flatnonzero(np.diff(seq)) + 1
    starts, ends = np.r_[0, cuts], np.r_[cuts, len(seq)]
    return [(int(seq[s]), int(s), int(e)) for s, e in zip(starts, ends)]
