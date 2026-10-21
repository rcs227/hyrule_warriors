"""PyTorch dataset: CQT windows paired with frame labels.

Each item is a fixed-length window (3 s by default) of one clip:
  features  (N_BINS, frames) float32 CQT
  pitch     (frames,) class index: 0 = silence, 1..N_PITCHES = MIDI 36..84,
            IGNORE where the label is unusable (padding, out-of-range note)
  onset     (frames,) 1.0 within one frame of a note start
  onset_mask(frames,) 0 where onset labels should not be trained on
            (padding, or datasets without reliable note starts)

Training windows are cut at random positions so every epoch sees different
parts of each clip. Evaluation uses whole clips (window=None).
"""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from features import load_cqt


PITCH_MIN, PITCH_MAX = 36, 84          # C2-C6
N_PITCHES = PITCH_MAX - PITCH_MIN + 1  # 49 notes
N_CLASSES = N_PITCHES + 1              # + silence
IGNORE = -100
ONSET_WIDTH = 1                        # frames either side of a note start


def midi_to_class(midi):
    """MIDI numbers (0 = silence) -> class indices."""
    midi = np.asarray(midi)
    classes = np.where(midi == 0, 0, midi - PITCH_MIN + 1)
    out_of_range = (midi != 0) & ((midi < PITCH_MIN) | (midi > PITCH_MAX))
    return np.where(out_of_range, IGNORE, classes).astype(np.int64)


def class_to_midi(classes):
    classes = np.asarray(classes)
    return np.where(classes == 0, 0, classes + PITCH_MIN - 1)


def widen_onsets(onset, width=ONSET_WIDTH):
    """Mark `width` frames either side of each note start as onset too."""
    wide = onset.astype(np.float32).copy()
    for shift in range(1, width + 1):
        wide[shift:] = np.maximum(wide[shift:], onset[:-shift])
        wide[:-shift] = np.maximum(wide[:-shift], onset[shift:])
    return wide


class HumDataset(Dataset):
    """Clips from one or more (features_dir, labels_dir) folders."""

    def __init__(self, clips, window=300, seed=None):
        """clips: list of (features_path, labels_path) pairs."""
        self.clips = list(clips)
        self.window = window
        self.rng = random.Random(seed)

    @staticmethod
    def from_dirs(features_dir, labels_dir, stems, **kwargs):
        features_dir, labels_dir = Path(features_dir), Path(labels_dir)
        pairs = [(features_dir / f"{s}.npz", labels_dir / f"{s}.npz") for s in stems]
        return HumDataset([p for p in pairs if p[0].exists() and p[1].exists()],
                          **kwargs)

    def __len__(self):
        return len(self.clips)

    def load(self, index):
        """Whole clip as numpy arrays (features, pitch, onset, onset_mask)."""
        features_path, labels_path = self.clips[index]
        cqt = load_cqt(features_path)
        with np.load(labels_path) as labels:
            pitch = midi_to_class(labels["pitch"])
            onset = widen_onsets(labels["onset"])
            has_onsets = bool(labels["has_onsets"]) if "has_onsets" in labels.files else True
        n = min(cqt.shape[1], len(pitch))
        onset_mask = np.full(n, 1.0 if has_onsets else 0.0, dtype=np.float32)
        return cqt[:, :n], pitch[:n], onset[:n], onset_mask

    def __getitem__(self, index):
        cqt, pitch, onset, onset_mask = self.load(index)
        if self.window is not None:
            cqt, pitch, onset, onset_mask = self._crop(cqt, pitch, onset, onset_mask)
        return {
            "features": torch.from_numpy(cqt),
            "pitch": torch.from_numpy(pitch),
            "onset": torch.from_numpy(onset),
            "onset_mask": torch.from_numpy(onset_mask),
        }

    def _crop(self, cqt, pitch, onset, onset_mask):
        n, w = len(pitch), self.window
        if n >= w:
            start = self.rng.randint(0, n - w)
            return (cqt[:, start:start + w], pitch[start:start + w],
                    onset[start:start + w], onset_mask[start:start + w])
        pad = w - n   # short clip: pad with silence the loss ignores
        return (np.pad(cqt, ((0, 0), (0, pad))),
                np.pad(pitch, (0, pad), constant_values=IGNORE),
                np.pad(onset, (0, pad)),
                np.pad(onset_mask, (0, pad)))
