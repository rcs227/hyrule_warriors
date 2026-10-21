import torch

from dataset import N_CLASSES
from features import N_BINS
from model import HumTranscriber, compute_loss


def fake_batch(b=4, t=120):
    return {
        "features": torch.randn(b, N_BINS, t),
        "pitch": torch.randint(0, N_CLASSES, (b, t)),
        "onset": (torch.rand(b, t) > 0.9).float(),
        "onset_mask": torch.ones(b, t),
    }


def test_output_shapes():
    model = HumTranscriber().eval()
    pitch, onset = model(torch.randn(2, N_BINS, 77))
    assert pitch.shape == (2, 77, N_CLASSES) and onset.shape == (2, 77)


def test_one_training_step_lowers_the_loss():
    torch.manual_seed(0)
    model = HumTranscriber()
    batch = fake_batch()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(5):
        loss, _, _ = compute_loss(*model(batch["features"]), batch)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_masked_onsets_do_not_count():
    model = HumTranscriber().eval()
    batch = fake_batch()
    batch["onset_mask"] = torch.zeros_like(batch["onset_mask"])
    _, _, onset_loss = compute_loss(*model(batch["features"]), batch)
    assert onset_loss.item() == 0
