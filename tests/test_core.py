import numpy as np
import pytest
import torch

from surgphase import metrics as M
from surgphase.data import PHASES, video_id
from surgphase.models import TemporalLSTM, embed, frame_encoder


def test_video_metrics_known_case():
    y = np.array([0, 0, 1, 1, 1, 2])
    p = np.array([0, 1, 1, 1, 2, 2])
    m = M.video_metrics(y, p, n_phases=4)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"][1] == pytest.approx(2 / 3) and m["recall"][1] == pytest.approx(2 / 3)
    assert m["jaccard"][1] == pytest.approx(2 / 4)
    assert np.isnan(m["jaccard"][3])                       # phase absent from truth and prediction


def test_summarise_averages_over_videos_and_skips_missing_phases():
    a = M.video_metrics(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), 3)     # perfect, phase 2 absent
    b = M.video_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 3)
    s = M.summarise([a, b], n_boot=200)
    assert s["accuracy"] == pytest.approx((1.0 + 0.75) / 2)
    assert s["n_videos"] == 2 and s["accuracy_ci95"][0] <= s["accuracy"] <= s["accuracy_ci95"][1]
    assert np.isnan(s["per_phase"]["jaccard"][2])
    assert s["jaccard"] == pytest.approx((1.0 + (0.5 + 2 / 3) / 2) / 2)


def test_segments_round_trip():
    seq = np.array([0, 0, 0, 1, 1, 3, 3, 3, 3, 1])
    segs = M.segments(seq)
    assert segs == [(0, 0, 3), (1, 3, 5), (3, 5, 9), (1, 9, 10)]
    rebuilt = np.concatenate([np.full(e - s, lab) for lab, s, e in segs])
    np.testing.assert_array_equal(rebuilt, seq)


def test_paired_permutation():
    rng = np.random.default_rng(0)
    base = rng.random(30)
    assert M.paired_permutation_p(base + 0.1, base) < 0.001
    assert M.paired_permutation_p(base, base) == 1.0


def test_video_id():
    assert video_id("video07.mp4") == 7 and video_id("D:/x/video41") == 41


def test_causal_lstm_cannot_see_the_future():
    torch.manual_seed(0)
    model = TemporalLSTM(dim=16, hidden=8, bidirectional=False, dropout=0.0).eval()
    x = torch.randn(1, 20, 16)
    y1 = model(x)
    x2 = x.clone()
    x2[0, 10:] = torch.randn(10, 16)                       # change only the future
    y2 = model(x2)
    assert torch.allclose(y1[0, :10], y2[0, :10], atol=1e-6)
    assert not torch.allclose(y1[0, 10:], y2[0, 10:])


def test_bilstm_does_use_the_future():
    torch.manual_seed(0)
    model = TemporalLSTM(dim=16, hidden=8, bidirectional=True, dropout=0.0).eval()
    x = torch.randn(1, 20, 16)
    x2 = x.clone()
    x2[0, 10:] = torch.randn(10, 16)
    assert not torch.allclose(model(x)[0, :10], model(x2)[0, :10])


def test_frame_encoder_shapes():
    model = frame_encoder(pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert model(x).shape == (2, len(PHASES))
        z = embed(model, x)
        assert z.shape == (2, 2048)
        assert torch.allclose(model.fc(z), model(x), atol=1e-4)


def test_temporal_model_learns_ordered_phases():
    """Synthetic 'operation': noisy one-hot features of phases in a fixed order."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    def video():
        lens = rng.integers(8, 20, 4)
        y = np.concatenate([np.full(n, i) for i, n in enumerate(lens)])
        x = 2.0 * np.eye(4, 16)[y] + rng.normal(0, 1.0, (len(y), 16))
        return torch.tensor(x, dtype=torch.float32)[None], torch.tensor(y)

    model = TemporalLSTM(dim=16, hidden=32, n_classes=4, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    for _ in range(300):
        x, y = video()
        loss = torch.nn.functional.cross_entropy(model(x)[0], y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    accs = []
    for _ in range(10):
        x, y = video()
        accs.append((model(x)[0].argmax(1) == y).float().mean().item())
    assert np.mean(accs) > 0.85          # single noisy frames alone would be far lower
