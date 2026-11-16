"""Download a size-capped subset of Cholec80 without fetching the whole 75 GB archive.

The archive's zip directory is read with HTTP range requests and only the chosen members are
pulled, in 16 MB ranged chunks. A dropped connection costs at most one chunk: the transfer
resumes from the bytes already on disk, also across runs. Videos are chosen by dropping the
largest files until the total fits the budget.

Usage:  python scripts/download_subset.py --budget-gb 46.5
Dataset: Twinanda et al., EndoNet (IEEE TMI 2016). Licence CC BY-NC-SA 4.0. https://camma.unistra.fr/datasets/
"""
from __future__ import annotations

import argparse
import json
import struct
import time
import zlib
from pathlib import Path

import requests
from remotezip import RemoteZip

URL = "https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.zip"
DATA = Path(__file__).resolve().parents[1] / "data"
CHUNK = 16 * 1024 * 1024


def choose(infos: list[dict], budget_bytes: float) -> list[dict]:
    videos = sorted((i for i in infos if i["name"].lower().endswith(".mp4")), key=lambda i: -i["csize"])
    total = sum(v["csize"] for v in videos)
    while videos and total > budget_bytes:
        total -= videos.pop(0)["csize"]            # drop the largest remaining video
    return sorted(videos, key=lambda i: i["name"])


def get_range(session: requests.Session, start: int, end: int) -> bytes:
    """Bytes [start, end] of the archive, retried with back-off."""
    for attempt in range(12):
        try:
            r = session.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 120))
            if r.status_code == 206 and len(r.content) == end - start + 1:
                return r.content
        except (requests.RequestException, OSError):     # any network-level failure: wait, retry
            pass
        time.sleep(min(60, 5 * (attempt + 1)))
    raise SystemExit(f"network failed repeatedly at byte {start}; re-run the script to resume")


def fetch_member(session: requests.Session, info: dict, target: Path, progress) -> int:
    """Download one zip member with resume. Returns bytes transferred in this call."""
    head = get_range(session, info["offset"], info["offset"] + 29)
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    data_start = info["offset"] + 30 + name_len + extra_len
    part = target.with_suffix(target.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    moved = 0
    with part.open("ab") as f:
        while have < info["csize"]:
            n = min(CHUNK, info["csize"] - have)
            f.write(get_range(session, data_start + have, data_start + have + n - 1))
            have += n
            moved += n
            progress(n)
    if info["method"] == 0:                              # stored: bytes on disk are the file
        part.replace(target)
    else:                                                # deflated: inflate to the final file
        d = zlib.decompressobj(-15)
        with part.open("rb") as src, target.open("wb") as dst:
            while block := src.read(1 << 20):
                dst.write(d.decompress(block))
            dst.write(d.flush())
        part.unlink()
    return moved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-gb", type=float, default=46.5)
    args = ap.parse_args()
    out = DATA / "cholec80"
    out.mkdir(parents=True, exist_ok=True)
    with RemoteZip(URL) as z:
        infos = [{"name": i.filename, "size": i.file_size, "csize": i.compress_size,
                  "offset": i.header_offset, "method": i.compress_type}
                 for i in z.infolist() if not i.filename.endswith("/")]
    videos = choose(infos, args.budget_gb * 1e9)
    small = [i for i in infos if not i["name"].lower().endswith(".mp4")]
    keep = small + videos
    total = sum(i["csize"] for i in keep)
    (DATA / "subset.json").write_text(json.dumps(
        {"budget_gb": args.budget_gb, "videos": [v["name"] for v in videos],
         "video_gb": round(sum(v["csize"] for v in videos) / 1e9, 2), "total_bytes": total}, indent=1))
    status = DATA / "download_status.json"
    state = {"total": total, "done": 0, "session_bytes": 0, "t0": time.time(), "file": "", "finished": False}

    def progress(n: int) -> None:
        state["done"] += n
        state["session_bytes"] += n
        state["updated"] = time.time()
        status.write_text(json.dumps(state))
        rate = state["session_bytes"] / max(time.time() - state["t0"], 1)
        left = (total - state["done"]) / rate / 3600 if rate else 0
        print(f"\r{state['done'] / 1e9:6.2f} of {total / 1e9:.2f} GB | {rate / 1e6:4.1f} MB/s | "
              f"{left:4.1f} h left | {state['file']}      ", end="", flush=True)

    session = requests.Session()
    for n, i in enumerate(keep, 1):
        target = out / i["name"]
        if target.exists() and target.stat().st_size == i["size"]:
            state["done"] += i["csize"]
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(target.suffix + ".part")
        state["done"] += part.stat().st_size if part.exists() else 0
        state["file"] = i["name"]
        fetch_member(session, i, target, progress)
        rate = state["session_bytes"] / max(time.time() - state["t0"], 1) / 1e6
        print(f"[{n}/{len(keep)}] {i['name']} ok | {state['done'] / 1e9:.2f} of {total / 1e9:.2f} GB "
              f"| {rate:.1f} MB/s", flush=True)
    state["finished"] = True
    status.write_text(json.dumps(state))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
