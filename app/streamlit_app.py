"""SurgiPhase: which step of the operation is happening, second by second.

Run:  streamlit run app/streamlit_app.py
Research prototype on Cholec80 (CC BY-NC-SA 4.0, non-commercial research only). Not for clinical use.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from surgphase.data import COLORS, FPS_VIDEO, FRAMES, NICE

ROOT = Path(__file__).resolve().parents[1]
ROWS = [("truth", "Surgeon labels"), ("frame_only", "Single frames"),
        ("causal_lstm", "Causal LSTM (live)"), ("offline_bilstm", "BiLSTM (whole video)")]
st.set_page_config(page_title="SurgiPhase", page_icon="🔬", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(135deg,#083344 0%,#0e7490 55%,#22d3ee 100%);
       padding: 1.6rem 2rem; border-radius: 18px; color:#fff; margin-bottom: 1.2rem;}
.hero h1 {margin:0; font-size:2rem; color:#fff;} .hero p {margin:.3rem 0 0; opacity:.9;}
.chip {display:inline-block; padding:.2rem .7rem; border-radius:999px; color:#fff; font-size:.8rem;
       font-weight:600; margin:.15rem .25rem .15rem 0;}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🔬 SurgiPhase</h1><p>Recognises the seven phases of keyhole '
            'gallbladder surgery from the video, one prediction per second.</p></div>', unsafe_allow_html=True)

mpath, ppath = ROOT / "reports" / "metrics.json", ROOT / "reports" / "predictions.json"
if not (mpath.exists() and ppath.exists()):
    st.info("No results yet. Run `python -m surgphase.train` after downloading the videos.")
    st.stop()
m, pred = json.loads(mpath.read_text()), json.loads(ppath.read_text())
vids = [int(v) for v in m["videos"]["test"]]
acc = dict(zip(vids, m["per_video"]["causal_lstm"]))


def expand(segs) -> np.ndarray:
    return np.concatenate([np.full(e - s, lab) for lab, s, e in segs])


tab_video, tab_results, tab_about = st.tabs(["Explore an operation", "Results", "About"])

with tab_video:
    st.markdown("".join(f'<span class="chip" style="background:{c}">{n}</span>' for c, n in zip(COLORS, NICE)),
                unsafe_allow_html=True)
    order = sorted(vids, key=lambda v: -acc[v])
    vid = st.selectbox("Test operation (never seen in training)", order,
                       format_func=lambda v: f"Video {v}: {acc[v]:.1%} of seconds correct (causal LSTM)")
    p = pred[str(vid)]
    fig = go.Figure()
    for row, (key, label) in enumerate(ROWS):
        for lab, s, e in p[key]:
            fig.add_trace(go.Bar(x=[(e - s) / 60], base=[s / 60], y=[label], orientation="h",
                                 marker_color=COLORS[lab], marker_line_width=0, showlegend=False,
                                 hovertemplate=f"{NICE[lab]}<br>{s // 60}:{s % 60:02d} to {e // 60}:{e % 60:02d}<extra></extra>"))
    fig.update_layout(barmode="overlay", height=270, margin=dict(l=10, r=10, t=10, b=40),
                      xaxis_title="Minutes into the operation", yaxis=dict(autorange="reversed"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    truth = expand(p["truth"])
    c = st.columns(4)
    c[0].metric("Length", f"{len(truth) // 60} min")
    for col, (key, label) in zip(c[1:], ROWS[1:]):
        col.metric(label, f"{np.mean(expand(p[key]) == truth):.1%}")
    st.caption("Top row: what the surgeon-annotators marked. The causal model only sees the past, as it "
               "would live in theatre; the BiLSTM reviews the whole recording afterwards.")

    frames_dir = FRAMES / f"video{vid:02d}"
    if frames_dir.exists():
        t = st.slider("Jump to minute", 0.0, (len(truth) - 1) / 60, 0.0, 0.25)
        sec = min(len(truth) - 1, int(t * 60))
        f = frames_dir / f"{sec * FPS_VIDEO:07d}.jpg"
        a, b = st.columns([1.3, 1])
        if f.exists():
            a.image(str(f), use_container_width=True)
        live = int(expand(p["causal_lstm"])[sec])
        b.markdown(f"**Surgeon label**<br><span class='chip' style='background:{COLORS[int(truth[sec])]}'>"
                   f"{NICE[int(truth[sec])]}</span>", unsafe_allow_html=True)
        b.markdown(f"**Model (live)**<br><span class='chip' style='background:{COLORS[live]}'>{NICE[live]}</span>"
                   + (" ✅" if live == truth[sec] else " ❌"), unsafe_allow_html=True)
    else:
        st.caption("Video frames are not included in the repository (dataset licence). "
                   "Download the videos to browse frames here.")

with tab_results:
    s = m["systems"]
    c = st.columns(4)
    c[0].metric("Single frames", f"{s['frame_only']['accuracy']:.1%}")
    c[1].metric("+ causal LSTM", f"{s['causal_lstm']['accuracy']:.1%}",
                f"{s['causal_lstm']['accuracy'] - s['frame_only']['accuracy']:+.1%}")
    c[2].metric("+ BiLSTM (offline)", f"{s['offline_bilstm']['accuracy']:.1%}")
    c[3].metric("Test operations", s["causal_lstm"]["n_videos"])
    st.caption(f"Accuracy per second, averaged over operations. Trained on {len(m['videos']['train'])} "
               f"operations, tuned on {len(m['videos']['val'])}.")
    st.image(str(ROOT / "reports/figures/results.png"))
    st.image(str(ROOT / "reports/figures/ribbons.png"))
    st.dataframe(pd.DataFrame({NICE[i]: {"Precision": s["causal_lstm"]["per_phase"]["precision"][i],
                                         "Recall": s["causal_lstm"]["per_phase"]["recall"][i],
                                         "Jaccard": s["causal_lstm"]["per_phase"]["jaccard"][i]}
                               for i in range(len(NICE))}).T.round(3), use_container_width=True)

with tab_about:
    st.markdown("""
**Why phase recognition.** Knowing which step of an operation is under way lets software warn
of what comes next, alert staff, index recordings for teaching and estimate remaining time.

**How it works.** A ResNet-50 reads each frame; an LSTM follows the sequence, because one
frame is often ambiguous but operations follow an order.

**Data.** Cholec80 (University of Strasbourg), licence CC BY-NC-SA 4.0: research and
non-commercial use only. This project used 66 of the 80 operations (a 50 GB download limit),
so results are not directly comparable with papers that use all 80.

**Not for clinical use.**
""")
