"""Three-stage training and evaluation on Cholec80.

  1. fine-tune a ResNet-50 to classify the phase of single frames
  2. cache its 2048-d embedding for every frame (one per second of surgery)
  3. train a causal LSTM (and an offline BiLSTM) over each video's embedding sequence

Run:  python -m surgphase.train            (stages whose outputs exist are skipped)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from . import metrics as M
from .data import NICE, PHASES, ROOT, extract_all, index
from .models import TemporalLSTM, embed, frame_encoder

FEATS = ROOT / "data" / "features"
ENCODER = ROOT / "models" / "frame_encoder.pt"
LOG = ROOT / "train.log"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EPOCHS_CNN, EPOCHS_TEMPORAL, SEED = 5, 40, 42


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


class Frames(Dataset):
    def __init__(self, paths, labels, train: bool):
        self.paths, self.labels = list(paths), np.asarray(labels)
        aug = [T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)), T.RandomHorizontalFlip(),
               T.ColorJitter(0.2, 0.2, 0.2, 0.02)] if train else [T.CenterCrop(224)]
        self.tf = T.Compose(aug + [T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), int(self.labels[i])


def loader(df, train: bool, batch: int = 64) -> DataLoader:
    return DataLoader(Frames(df.path, df.phase, train), batch_size=batch, shuffle=train, drop_last=train,
                      num_workers=8, persistent_workers=False, pin_memory=True)


@torch.no_grad()
def predict_frames(model, df, dev) -> tuple[np.ndarray, np.ndarray]:
    """Class probabilities and 2048-d embeddings for every frame in df (in order)."""
    model.eval()
    probs, feats = [], []
    for x, _ in loader(df, False, 128):
        with torch.autocast(dev, enabled=dev == "cuda"):
            z = embed(model, x.to(dev, non_blocking=True))
            p = torch.softmax(model.fc(z).float(), 1)
        probs.append(p.cpu().numpy())
        feats.append(z.float().cpu().numpy().astype(np.float16))
    return np.concatenate(probs), np.concatenate(feats)


def stage1_frame_encoder(df, dev) -> None:
    tr, va = df[df.split == "train"], df[df.split == "val"]
    log(f"stage 1: fine-tuning ResNet-50 on {len(tr):,} frames from {tr.video.nunique()} videos; "
        f"validating on {len(va):,} frames from {va.video.nunique()} videos")
    torch.manual_seed(SEED)
    model = frame_encoder().to(dev)
    dl = loader(tr, True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    total = EPOCHS_CNN * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.05 * total)) * 0.5 * (1 + math.cos(math.pi * s / total)))
    scaler = torch.amp.GradScaler(dev, enabled=dev == "cuda")
    best = -1.0
    for ep in range(EPOCHS_CNN):
        model.train()
        losses = []
        for step, (x, y) in enumerate(dl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = F.cross_entropy(model(x).float(), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            if step % 200 == 0:
                log(f"  round {ep + 1} step {step}/{len(dl)} loss {np.mean(losses[-200:]):.3f}")
        probs, _ = predict_frames(model, va, dev)
        acc = float(np.mean(probs.argmax(1) == va.phase.to_numpy()))
        log(f"epoch {ep + 1}/{EPOCHS_CNN} loss {np.mean(losses):.3f} val acc {acc:.3f}")
        if acc > best:
            best = acc
            ENCODER.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), ENCODER)


def stage2_features(df, dev) -> None:
    model = frame_encoder(pretrained=False).to(dev)
    model.load_state_dict(torch.load(ENCODER, map_location=dev, weights_only=True))
    FEATS.mkdir(parents=True, exist_ok=True)
    todo = df[[not (FEATS / f"video{v:02d}.npz").exists() for v in df.video]].reset_index(drop=True)
    if len(todo):
        log(f"stage 2: embedding {len(todo):,} frames from {todo.video.nunique()} videos")
        probs, feats = predict_frames(model, todo, dev)          # one pass: worker start-up is slow
        for vid, rows in todo.groupby("video").indices.items():
            np.savez_compressed(FEATS / f"video{vid:02d}.npz", feats=feats[rows],
                                probs=probs[rows].astype(np.float16), y=todo.phase.to_numpy()[rows])
    log(f"stage 2: features cached for {df.video.nunique()} videos")


def load_video(vid: int):
    d = np.load(FEATS / f"video{vid:02d}.npz")
    return d["feats"].astype(np.float32), d["probs"].astype(np.float32), d["y"]


def stage3_temporal(videos: dict[str, list[int]], dev, bidirectional: bool) -> dict[int, np.ndarray]:
    """Train on train videos, pick the round with best validation accuracy, predict test videos."""
    name = "offline BiLSTM" if bidirectional else "causal LSTM"
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = TemporalLSTM(bidirectional=bidirectional).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    data = {v: load_video(v) for split in videos.values() for v in split}
    best, best_state = -1.0, None
    for ep in range(EPOCHS_TEMPORAL):
        model.train()
        losses = []
        for v in rng.permutation(videos["train"]):
            x, _, y = data[v]
            x = torch.tensor(x, device=dev)[None]
            x = x + 0.02 * torch.randn_like(x)                      # light feature noise
            loss = F.cross_entropy(model(x)[0], torch.tensor(y, device=dev, dtype=torch.long))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        acc = float(np.mean([np.mean(predict_seq(model, data[v][0], dev) == data[v][2]) for v in videos["val"]]))
        if acc > best:
            best, best_state = acc, {k: t.detach().clone() for k, t in model.state_dict().items()}
        if (ep + 1) % 5 == 0:
            log(f"  {name} round {ep + 1}/{EPOCHS_TEMPORAL}: loss {np.mean(losses):.3f}, validation accuracy {acc:.3f}")
    model.load_state_dict(best_state)
    torch.save(best_state, ROOT / "models" / ("bilstm.pt" if bidirectional else "lstm.pt"))
    log(f"stage 3: {name} best validation accuracy {best:.3f}")
    return {v: predict_seq(model, data[v][0], dev) for v in videos["test"]}


@torch.no_grad()
def predict_seq(model, feats: np.ndarray, dev) -> np.ndarray:
    model.eval()
    return model(torch.tensor(feats, device=dev)[None])[0].argmax(1).cpu().numpy()


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    extract_all(log)
    df = index()
    videos = {s: sorted(int(v) for v in df[df.split == s].video.unique()) for s in ("train", "val", "test")}
    log(f"videos: {len(videos['train'])} train, {len(videos['val'])} validation, {len(videos['test'])} test; "
        f"{len(df):,} frames at 1 fps")
    if not ENCODER.exists():
        stage1_frame_encoder(df, dev)
    stage2_features(df, dev)

    preds = {"frame_only": {v: load_video(v)[1].argmax(1) for v in videos["test"]},
             "causal_lstm": stage3_temporal(videos, dev, bidirectional=False),
             "offline_bilstm": stage3_temporal(videos, dev, bidirectional=True)}
    truth = {v: load_video(v)[2] for v in videos["test"]}
    labels = {"frame_only": "ResNet-50, single frames", "causal_lstm": "ResNet-50 + causal LSTM (live)",
              "offline_bilstm": "ResNet-50 + BiLSTM (whole video)"}
    out = {"videos": videos, "frames": int(len(df)), "phases": NICE, "systems": {}, "per_video": {}}
    for k, p in preds.items():
        per = [M.video_metrics(truth[v], p[v]) for v in videos["test"]]
        out["systems"][k] = {"label": labels[k], **M.summarise(per)}
        out["per_video"][k] = [r["accuracy"] for r in per]
        s = out["systems"][k]
        log(f"TEST {labels[k]}: accuracy {s['accuracy']:.3f} +/- {s['accuracy_sd']:.3f}, "
            f"precision {s['precision']:.3f}, recall {s['recall']:.3f}, Jaccard {s['jaccard']:.3f}")
    out["p_lstm_vs_frame"] = M.paired_permutation_p(out["per_video"]["causal_lstm"], out["per_video"]["frame_only"])
    out["p_bilstm_vs_lstm"] = M.paired_permutation_p(out["per_video"]["offline_bilstm"], out["per_video"]["causal_lstm"])
    out["phase_seconds_test"] = np.bincount(np.concatenate(list(truth.values())), minlength=len(PHASES)).tolist()
    cm = np.zeros((len(PHASES), len(PHASES)), int)
    for v in videos["test"]:
        np.add.at(cm, (truth[v], preds["causal_lstm"][v]), 1)
    out["confusion_causal_lstm"] = cm.tolist()
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(out, indent=2))
    (ROOT / "reports" / "predictions.json").write_text(json.dumps(
        {str(v): {"truth": M.segments(truth[v]), **{k: M.segments(preds[k][v]) for k in preds}}
         for v in videos["test"]}))
    from .report import write_report
    write_report()
    log("ALL DONE")


if __name__ == "__main__":
    main()
