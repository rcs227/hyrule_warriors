"""Audio features shared by every script.

All feature files use the same settings, so they are defined once here:
16 kHz audio, 100 frames per second, and a CQT with 3 bins per semitone
from C2 (MIDI 36) to just below C7 (MIDI 96). Three bins per semitone let
the model see hummed notes that fall between two piano keys.
"""
import json
from io import BytesIO

import librosa
import numpy as np


SAMPLE_RATE = 16_000
HOP_LENGTH = 160                  # 10 ms
FRAME_S = HOP_LENGTH / SAMPLE_RATE

CQT_FMIN_MIDI = 36                # C2
BINS_PER_SEMITONE = 3
N_OCTAVES = 5
N_BINS = 12 * BINS_PER_SEMITONE * N_OCTAVES   # 180

FEATURE_KEY = "cqt_log_magnitude"


def load_audio(source):
    """Load a path or raw WAV bytes as 16 kHz mono float audio."""
    if isinstance(source, (bytes, bytearray)):
        source = BytesIO(source)
    y, _ = librosa.load(source, sr=SAMPLE_RATE, mono=True)
    if y.size == 0:
        raise ValueError("audio contains no samples")
    return y


def compute_cqt(y):
    """Log-magnitude CQT, shape (N_BINS, n_frames), float16 to save space."""
    cqt = librosa.cqt(
        y=y,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        fmin=librosa.midi_to_hz(CQT_FMIN_MIDI),
        n_bins=N_BINS,
        bins_per_octave=12 * BINS_PER_SEMITONE,
    )
    return np.log1p(np.abs(cqt)).astype(np.float16)


def bin_to_midi(bins):
    return CQT_FMIN_MIDI + np.asarray(bins) / BINS_PER_SEMITONE


def midi_to_bin(midi):
    return np.round((np.asarray(midi) - CQT_FMIN_MIDI) * BINS_PER_SEMITONE).astype(int)


def features_to_npz_bytes(cqt, source):
    """Pack a CQT and its settings into compressed .npz bytes."""
    metadata = {
        "source": source,
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "cqt_fmin_midi": CQT_FMIN_MIDI,
        "bins_per_semitone": BINS_PER_SEMITONE,
        "n_bins": N_BINS,
    }
    buffer = BytesIO()
    np.savez_compressed(buffer, **{FEATURE_KEY: cqt},
                        metadata_json=np.array(json.dumps(metadata)))
    return buffer.getvalue()


def load_cqt(path):
    """Load a CQT saved by features_to_npz_bytes, checking its settings."""
    with np.load(path) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if (metadata.get("bins_per_semitone") != BINS_PER_SEMITONE
                or metadata.get("cqt_fmin_midi") != CQT_FMIN_MIDI):
            raise ValueError(f"{path} was made with old feature settings; "
                             "re-run the feature script")
        return data[FEATURE_KEY].astype(np.float32)
