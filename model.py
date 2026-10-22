"""CNN + Bi-LSTM hum transcription model (see diagrams/training_first_model.png).

  CQT (180 bins x frames)
    -> CNN: learns local patterns such as a note's harmonics, and shrinks
       the frequency axis 180 -> 20
    -> Bi-LSTM: reads the whole sequence forwards and backwards, so each
       frame's prediction can use what comes before and after it
    -> two outputs per frame: which note is sounding (or silence), and how
       likely a new note starts here
"""
import torch
from torch import nn
import torch.nn.functional as F

from dataset import IGNORE, N_CLASSES
from features import N_BINS


def conv_block(c_in, c_out, pool_freq):
    layers = [
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
    ]
    if pool_freq > 1:
        layers.append(nn.MaxPool2d(kernel_size=(pool_freq, 1)))
    return nn.Sequential(*layers)


class HumTranscriber(nn.Module):
    def __init__(self, n_bins=N_BINS, n_classes=N_CLASSES, hidden=128, dropout=0.25):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(n_bins)
        self.cnn = nn.Sequential(
            conv_block(1, 16, 1),
            conv_block(16, 32, 3),     # 180 -> 60 frequency bins
            conv_block(32, 64, 3),     # 60 -> 20
            nn.Dropout(dropout),
        )
        self.project = nn.Linear(64 * (n_bins // 9), 256)
        self.lstm = nn.LSTM(256, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.pitch_head = nn.Linear(2 * hidden, n_classes)
        self.onset_head = nn.Linear(2 * hidden, 1)

    def forward(self, features):
        """features: (batch, bins, frames) -> pitch logits (batch, frames,
        classes) and onset logits (batch, frames)."""
        x = self.input_norm(features).unsqueeze(1)          # (B, 1, F, T)
        x = self.cnn(x)                                      # (B, C, F', T)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)       # (B, T, C*F')
        x = F.relu(self.project(x))
        x, _ = self.lstm(x)
        x = self.dropout(x)
        return self.pitch_head(x), self.onset_head(x).squeeze(-1)


def compute_loss(pitch_logits, onset_logits, batch, onset_weight=1.0, onset_pos_weight=5.0):
    """Pitch: cross-entropy per frame. Onset: weighted binary cross-entropy
    (note starts are rare, so positive frames count onset_pos_weight times)."""
    pitch_loss = F.cross_entropy(pitch_logits.transpose(1, 2), batch["pitch"],
                                 ignore_index=IGNORE)
    mask = batch["onset_mask"]
    onset_loss = F.binary_cross_entropy_with_logits(
        onset_logits, batch["onset"], reduction="none",
        pos_weight=torch.tensor(onset_pos_weight, device=onset_logits.device))
    onset_loss = (onset_loss * mask).sum() / mask.sum().clamp(min=1)
    return pitch_loss + onset_weight * onset_loss, pitch_loss, onset_loss
