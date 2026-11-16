"""Cholec80 (Twinanda et al., IEEE TMI 2016): frame extraction and indexing.

80 laparoscopic cholecystectomy videos at 25 fps with a surgical-phase label for every frame.
Licence CC BY-NC-SA 4.0, https://camma.unistra.fr/datasets/ . This project used the 66 videos
that fitted a 50 GB download cap (see scripts/download_subset.py).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "cholec80"
FRAMES = ROOT / "data" / "frames"
PHASES = ["Preparation", "CalotTriangleDissection", "ClippingCutting", "GallbladderDissection",
          "GallbladderPackaging", "CleaningCoagulation", "GallbladderRetraction"]
NICE = ["Preparation", "Calot triangle dissection", "Clipping and cutting", "Gallbladder dissection",
        "Gallbladder packaging", "Cleaning and coagulation", "Gallbladder retraction"]
COLORS = ["#64748b", "#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2"]
FPS_VIDEO, FPS_OUT = 25, 1
TRAIN_MAX_ID = 40                       # standard protocol: videos 1-40 train, 41-80 test
VAL_IDS = (5, 13, 22, 30, 37)           # held out of training for model selection


def video_id(name: str) -> int:
    return int(re.search(r"video(\d+)", name).group(1))


def read_phase_labels(vid: int) -> np.ndarray:
    """Phase index for every original 25 fps frame."""
    df = pd.read_csv(RAW / "phase_annotations" / f"video{vid:02d}-phase.txt", sep="\t")
    return df["Phase"].map({p: i for i, p in enumerate(PHASES)}).to_numpy()


def extract_video(path: Path, out_dir: Path, height: int = 224, quality: int = 90) -> int:
    """Save one frame per second, resized so the short side is `height`. Returns frames written."""
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    step, i, n = FPS_VIDEO // FPS_OUT, 0, 0
    while True:
        if i % step == 0:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(round(w * height / h)), height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{i:07d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            n += 1
        elif not cap.grab():
            break
        i += 1
    cap.release()
    (out_dir / "DONE").write_text(str(n))
    return n


def extract_all(log=print) -> None:
    """Extract every fully downloaded video that has not been extracted yet."""
    for mp4 in sorted((RAW / "videos").glob("video*.mp4")):
        out = FRAMES / mp4.stem
        if (out / "DONE").exists():
            continue
        n = extract_video(mp4, out)
        log(f"extracted {mp4.stem}: {n} frames")


def index() -> pd.DataFrame:
    """One row per extracted frame: path, video, second, phase label, split."""
    rows = []
    for d in sorted(FRAMES.glob("video*")):
        if not (d / "DONE").exists():
            continue
        vid = video_id(d.name)
        labels = read_phase_labels(vid)
        for f in sorted(d.glob("*.jpg")):
            k = int(f.stem)
            if k < len(labels):
                rows.append((str(f), vid, k // FPS_VIDEO, int(labels[k])))
    df = pd.DataFrame(rows, columns=["path", "video", "second", "phase"])
    df["split"] = np.where(df.video > TRAIN_MAX_ID, "test", np.where(df.video.isin(VAL_IDS), "val", "train"))
    return df


if __name__ == "__main__":
    extract_all()
