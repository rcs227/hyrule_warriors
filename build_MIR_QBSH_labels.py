"""Build frame labels for MIR-QBSH from its .pv pitch-vector files.

The MIR-QBSH MIDI files are the original songs, not transcriptions of each
recording (people sang in their own key and tempo), so we use the .pv files
instead: one pitch value per short frame of the recording, in semitones
(MIDI numbers), with 0 meaning silence.

Each .pv is resampled to our 100 frames per second and rounded to the
nearest semitone. Note starts are only guessed (where the pitch changes), so
the labels are saved with has_onsets=False and training ignores their onset
targets.

To check the .pv files really match the audio, each one is compared with
the pYIN pitch track saved with the features; the fraction of frames within
half a semitone is saved as align_score (same meaning as for HumTrans).

Needs raw_data/MIR-QBSH/MIR-QBSH.zip and the features from
collect_MIR_QBSH_wav.py.

Usage:
    python build_MIR_QBSH_labels.py
"""
import csv
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np

from collect_MIR_QBSH_wav import ZIP_PATH, clip_name
from features import FRAME_S, load_cqt, load_pitch_track


PROJECT_DIR = Path(__file__).resolve().parent
FEATURES_DIR = PROJECT_DIR / "processed_data" / "MIR-QBSH" / "features"
LABELS_DIR = PROJECT_DIR / "processed_data" / "MIR-QBSH" / "labels"
ALIGNMENT_CSV = PROJECT_DIR / "processed_data" / "MIR-QBSH" / "alignment.csv"
EXPECTED_PV_FRAME_S = 256 / 8000      # 32 ms frames at 8 kHz


def parse_pv(text):
    """Pitch values from a .pv file (whitespace separated)."""
    return np.array([float(v) for v in text.split()], dtype=float)


def pv_to_frames(pv, n_frames, pv_frame_s):
    """Resample a pitch vector to our frame rate; 0 = silence."""
    times = np.arange(n_frames) * FRAME_S
    index = np.minimum((times / pv_frame_s).astype(int), len(pv) - 1)
    return pv[index]


def frames_to_labels(pitch_float):
    """Rounded pitch per frame, plus onsets/offsets where the note changes."""
    pitch = np.round(pitch_float).astype(np.int16)
    pitch[pitch_float <= 0] = 0
    previous = np.concatenate([[0], pitch[:-1]])
    following = np.concatenate([pitch[1:], [0]])
    onset = ((pitch > 0) & (pitch != previous)).astype(np.uint8)
    offset = ((pitch > 0) & (pitch != following)).astype(np.uint8)
    return pitch, onset, offset


def agreement(pv_frames, pyin_track):
    """Fraction of frames voiced in both where they agree within 0.5 semitone."""
    both = (pv_frames > 0) & ~np.isnan(pyin_track)
    if not both.any():
        return 0.0
    return float(np.mean(np.abs(pv_frames[both] - pyin_track[both]) < 0.5))


def main():
    if not ZIP_PATH.exists():
        raise SystemExit(f"ZIP file not found: {ZIP_PATH}")
    feature_files = sorted(FEATURES_DIR.glob("*.npz"))
    if not feature_files:
        raise SystemExit("No features; run collect_MIR_QBSH_wav.py first.")

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    rows, frame_rates = [], []

    with ZipFile(ZIP_PATH) as archive:
        pv_members = {}
        for name in archive.namelist():
            if name.lower().endswith(".pv") and "waveFile" in PurePosixPath(name).parts:
                try:
                    pv_members[clip_name(name)] = name
                except ValueError:
                    pass
        print(f"Found {len(pv_members)} .pv files for {len(feature_files)} recordings")

        for path in feature_files:
            if path.stem not in pv_members:
                print(f"{path.stem}: no .pv file, skipped")
                continue
            n_frames = load_cqt(path).shape[1]
            pv = parse_pv(archive.read(pv_members[path.stem]).decode("utf-8", "ignore"))
            if len(pv) == 0:
                continue

            # The .pv covers the whole recording, so its frame length follows
            # from the recording length (expected: 32 ms).
            pv_frame_s = n_frames * FRAME_S / len(pv)
            frame_rates.append(pv_frame_s)
            pv_frames = pv_to_frames(pv, n_frames, pv_frame_s)
            pitch, onset, offset = frames_to_labels(pv_frames)

            pyin_track = load_pitch_track(path)
            score = agreement(pv_frames, pyin_track) if pyin_track is not None else float("nan")
            np.savez_compressed(LABELS_DIR / f"{path.stem}.npz",
                                pitch=pitch, onset=onset, offset=offset,
                                octave_shift=0, lag_s=0.0, align_score=score,
                                has_onsets=False)
            rows.append({"clip": path.stem, "pv_frame_s": f"{pv_frame_s:.4f}",
                         "align_score": f"{score:.3f}"})

    with open(ALIGNMENT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "pv_frame_s", "align_score"])
        writer.writeheader()
        writer.writerows(rows)

    scores = np.array([float(r["align_score"]) for r in rows])
    print(f"\nLabelled {len(rows)} recordings into {LABELS_DIR}")
    print(f"Median .pv frame length {np.median(frame_rates) * 1000:.1f} ms "
          f"(expected {EXPECTED_PV_FRAME_S * 1000:.0f} ms)")
    print(f"Agreement with pYIN: median {np.nanmedian(scores):.2f}, "
          f"{np.mean(scores >= 0.6):.0%} of recordings >= 0.6")


if __name__ == "__main__":
    main()
