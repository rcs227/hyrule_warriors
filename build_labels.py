"""Build frame-level training labels for HumTrans.

Reads the CQT features made by collect_HumTrans_wav.py and the notes in
processed_data/HumTrans/midi_notes.csv, and writes one label file per clip.

The HumTrans MIDI labels are not perfectly aligned with the humming:
  * Octave: many singers (mostly the men) hum one octave away from the MIDI.
  * Lag: the singer starts each note late by a roughly constant amount per
    clip (usually 40-250 ms, up to ~1 s for some singers).
Both are estimated per clip by comparing the hummed pitch (the pYIN track
saved with the features, or the CQT for older files) with the MIDI, and
applied to the labels. The
alignment score is saved so badly aligned clips can be filtered out later.

Usage:
    python build_labels.py                  # label every features/*.npz
    python build_labels.py --plot F01_0028_0002_2_D   # save a check plot
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from features import (BINS_PER_SEMITONE, CQT_FMIN_MIDI, FRAME_S, bin_to_midi,
                      load_cqt, load_pitch_track, midi_to_bin)


PROJECT_DIR = Path(__file__).resolve().parent
FEATURES_DIR = PROJECT_DIR / "processed_data" / "HumTrans" / "features"
NOTES_CSV = PROJECT_DIR / "processed_data" / "HumTrans" / "midi_notes.csv"
LABELS_DIR = PROJECT_DIR / "processed_data" / "HumTrans" / "labels"
ALIGNMENT_CSV = PROJECT_DIR / "processed_data" / "HumTrans" / "alignment.csv"

HUM_RANGE = (36, 84)     # C2-C6: where we look for the hummed pitch
LAG_RANGE_S = (-0.2, 2.0)
OCTAVE_SHIFTS = (-24, -12, 0, 12)


def load_notes(csv_path=NOTES_CSV):
    """Return {clip stem: [(onset_s, offset_s, midi_pitch), ...]}."""
    notes = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            onset = float(row["onset_s"])
            notes[Path(row["file_name"]).stem].append(
                (onset, onset + float(row["duration_s"]), int(row["pitch_midi"]))
            )
    return notes


def cqt_pitch_track(cqt):
    """Rough hummed pitch (MIDI number, NaN when silent) for each CQT frame.

    Sums each bin with its 2nd-4th harmonics so the fundamental wins over
    its overtones, then keeps only frames that are clearly louder than the
    background.
    """
    harmonic_sum = np.zeros_like(cqt)
    for semitones, weight in [(0, 1.0), (12, 0.8), (19, 0.6), (24, 0.4)]:
        shift = semitones * BINS_PER_SEMITONE
        harmonic_sum[:cqt.shape[0] - shift] += weight * cqt[shift:]

    lo, hi = midi_to_bin(HUM_RANGE)
    pitch = bin_to_midi(lo + np.argmax(harmonic_sum[lo:hi], axis=0))

    energy = cqt.max(axis=0)
    floor = np.percentile(energy, 30)
    voiced = energy > floor + 0.25 * (energy.max() - floor)
    return np.where(voiced, pitch, np.nan).astype(float)


def notes_to_frames(notes, n_frames, octave_shift=0, lag_s=0.0):
    """Turn notes into per-frame arrays.

    Returns pitch (MIDI number, 0 = silence), onset and offset flags.
    """
    pitch = np.zeros(n_frames, dtype=np.int16)
    onset = np.zeros(n_frames, dtype=np.uint8)
    offset = np.zeros(n_frames, dtype=np.uint8)

    for start_s, end_s, midi in notes:
        start = int(round((start_s + lag_s) / FRAME_S))
        end = int(round((end_s + lag_s) / FRAME_S))
        start, end = max(start, 0), min(end, n_frames)
        if start >= end:
            continue
        pitch[start:end] = midi + octave_shift
        onset[start] = 1
        offset[end - 1] = 1

    return pitch, onset, offset


def estimate_alignment(hum_pitch, notes):
    """Find the octave shift and lag that best match the hum to the notes.

    The score is the fraction of frames (hummed and labelled) where the hum
    is within half a semitone of the shifted label.
    """
    n = len(hum_pitch)
    ref, _, _ = notes_to_frames(notes, n)
    ref = np.where(ref > 0, ref, np.nan).astype(float)

    # Divide by the whole clip, not just the overlap, so a lag that leaves
    # only a sliver of overlap cannot score well.
    total = min(np.count_nonzero(~np.isnan(hum_pitch)),
                np.count_nonzero(~np.isnan(ref)))
    if total == 0:
        return 0, 0.0, 0.0

    def score(octave_shift, lag_frames):
        shifted = np.full(n, np.nan)
        if lag_frames >= 0:
            shifted[:n - lag_frames] = hum_pitch[lag_frames:]
        else:
            shifted[-lag_frames:] = hum_pitch[:lag_frames]
        ok = ~np.isnan(shifted) & ~np.isnan(ref)
        return float(np.sum(np.abs(shifted[ok] - octave_shift - ref[ok]) < 0.5)) / total

    # Octave and lag are searched together: with a long lag, comparing the
    # unshifted hum to the labels says little about the octave.
    lags = range(int(LAG_RANGE_S[0] / FRAME_S), int(LAG_RANGE_S[1] / FRAME_S) + 1)
    best = max(((score(shift, lag), -abs(shift), shift, lag)
                for shift in OCTAVE_SHIFTS for lag in lags))
    best_score, _, octave_shift, lag_frames = best
    return octave_shift, lag_frames * FRAME_S, best_score


def plot_clip(stem, labels_path, features_path, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cqt = load_cqt(features_path)
    labels = np.load(labels_path)
    pitch = np.where(labels["pitch"] > 0, labels["pitch"], np.nan)
    t = np.arange(cqt.shape[1]) * FRAME_S

    half_bin = 0.5 / BINS_PER_SEMITONE
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.imshow(cqt, origin="lower", aspect="auto", cmap="magma",
              extent=[0, t[-1], CQT_FMIN_MIDI - half_bin,
                      bin_to_midi(cqt.shape[0]) - half_bin])
    ax.plot(t, pitch, c="cyan", lw=2, label="aligned label")
    for x in t[labels["onset"] == 1]:
        ax.axvline(x, c="white", lw=0.5, alpha=0.6)
    ax.set_ylim(HUM_RANGE)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("MIDI pitch")
    ax.set_title(f"{stem}  shift={int(labels['octave_shift']):+d}  "
                 f"lag={float(labels['lag_s']):.2f}s  "
                 f"score={float(labels['align_score']):.2f}")
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    print(f"Plot saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    parser.add_argument("--plot", metavar="STEM",
                        help="only save a check plot for one labelled clip")
    args = parser.parse_args()

    if args.plot:
        plot_clip(args.plot, args.labels_dir / f"{args.plot}.npz",
                  args.features_dir / f"{args.plot}.npz",
                  args.labels_dir / f"{args.plot}.png")
        return

    feature_files = sorted(args.features_dir.glob("*.npz"))
    if not feature_files:
        raise SystemExit(f"No feature files in {args.features_dir}. "
                         "Run collect_HumTrans_wav.py first.")

    all_notes = load_notes()
    args.labels_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for index, path in enumerate(feature_files, start=1):
        stem = path.stem
        if stem not in all_notes:
            print(f"[{index}/{len(feature_files)}] {stem}: no notes, skipped")
            continue

        cqt = load_cqt(path)
        notes = all_notes[stem]
        # Prefer the saved pYIN track; older feature files only have the CQT.
        hum_pitch = load_pitch_track(path)
        pitch_source = "pyin"
        if hum_pitch is None:
            hum_pitch, pitch_source = cqt_pitch_track(cqt), "cqt"
        octave_shift, lag_s, align_score = estimate_alignment(hum_pitch, notes)
        pitch, onset, offset = notes_to_frames(
            notes, cqt.shape[1], octave_shift, lag_s
        )

        np.savez_compressed(
            args.labels_dir / f"{stem}.npz",
            pitch=pitch, onset=onset, offset=offset,
            octave_shift=octave_shift, lag_s=lag_s, align_score=align_score,
        )
        rows.append({"clip": stem, "octave_shift": octave_shift,
                     "lag_s": f"{lag_s:.2f}", "align_score": f"{align_score:.3f}",
                     "pitch_source": pitch_source})
        print(f"[{index}/{len(feature_files)}] {stem}: shift={octave_shift:+d} "
              f"lag={lag_s:.2f}s score={align_score:.2f}")

    with open(args.labels_dir.parent / ALIGNMENT_CSV.name, "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "octave_shift", "lag_s",
                                               "align_score", "pitch_source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nLabelled {len(rows)} clips into {args.labels_dir}")
    print(f"Alignment summary: {args.labels_dir.parent / ALIGNMENT_CSV.name}")


if __name__ == "__main__":
    main()
