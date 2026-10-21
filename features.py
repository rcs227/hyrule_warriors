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
PITCH_KEY = "pyin_midi"


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


def compute_pitch_track(y, n_frames):
    """pYIN pitch in MIDI numbers per frame (NaN when not hummed).

    Only used to align the labels (build_labels.py); the model never sees it.
    pYIN copes with a faint fundamental far better than reading the CQT,
    which matters for low male voices.
    """
    f0, voiced, _ = librosa.pyin(y, sr=SAMPLE_RATE, fmin=60, fmax=1000,
                                 hop_length=HOP_LENGTH)
    midi = np.where(voiced, librosa.hz_to_midi(f0), np.nan)
    midi = np.pad(midi[:n_frames], (0, max(0, n_frames - len(midi))),
                  constant_values=np.nan)
    return midi.astype(np.float16)


def bin_to_midi(bins):
    return CQT_FMIN_MIDI + np.asarray(bins) / BINS_PER_SEMITONE


def midi_to_bin(midi):
    return np.round((np.asarray(midi) - CQT_FMIN_MIDI) * BINS_PER_SEMITONE).astype(int)


def features_to_npz_bytes(cqt, source, pitch_track=None):
    """Pack a CQT (and optional pYIN track) into compressed .npz bytes."""
    metadata = {
        "source": source,
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "cqt_fmin_midi": CQT_FMIN_MIDI,
        "bins_per_semitone": BINS_PER_SEMITONE,
        "n_bins": N_BINS,
    }
    arrays = {FEATURE_KEY: cqt}
    if pitch_track is not None:
        arrays[PITCH_KEY] = pitch_track
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays,
                        metadata_json=np.array(json.dumps(metadata)))
    return buffer.getvalue()


def load_pitch_track(path):
    """The saved pYIN track as float32, or None if the file has none."""
    with np.load(path) as data:
        if PITCH_KEY not in data.files:
            return None
        return data[PITCH_KEY].astype(np.float32)


def load_cqt(path):
    """Load a CQT saved by features_to_npz_bytes, checking its settings."""
    with np.load(path) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if (metadata.get("bins_per_semitone") != BINS_PER_SEMITONE
                or metadata.get("cqt_fmin_midi") != CQT_FMIN_MIDI):
            raise ValueError(f"{path} was made with old feature settings; "
                             "re-run the feature script")
        return data[FEATURE_KEY].astype(np.float32)
