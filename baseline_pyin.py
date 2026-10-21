"""Non-deep-learning baseline: pYIN pitch tracking + simple note grouping.

Uses the pYIN pitch track saved in each feature file: rounds every frame to
the nearest semitone, smooths it, and groups runs of the same note into
notes. Scored with evaluation.py against the aligned labels, exactly like
the model, so any model we train should beat these numbers.

Caveat: the labels are aligned using this same pYIN track (build_labels.py),
which may flatter the baseline slightly.

Usage:
    python baseline_pyin.py                 # every labelled test clip
    python baseline_pyin.py --split valid
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from build_labels import FEATURES_DIR, LABELS_DIR, load_notes
from evaluation import METRICS, evaluate_clip, reference_notes, summarize
from features import FRAME_S, load_pitch_track
from splits import get_split


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
MIN_NOTE_S = 0.06


def frames_to_notes(midi, min_note_s=MIN_NOTE_S):
    """Group runs of the same MIDI number (0 = silence) into notes.

    Returns (intervals in seconds, MIDI pitches). Runs shorter than
    min_note_s are dropped.
    """
    intervals, pitches = [], []
    start = 0
    for i in range(1, len(midi) + 1):
        if i == len(midi) or midi[i] != midi[start]:
            if midi[start] > 0 and (i - start) * FRAME_S >= min_note_s - 1e-9:
                intervals.append((start * FRAME_S, i * FRAME_S))
                pitches.append(midi[start])
            start = i
    return np.array(intervals).reshape(-1, 2), np.array(pitches)


def transcribe_pitch_track(pitch_track):
    """Turn a pYIN track (MIDI numbers, NaN = silence) into notes."""
    midi = np.where(np.isnan(pitch_track), 0, np.round(pitch_track))
    midi = median_filter(midi, size=5)   # remove one-frame wobbles
    return frames_to_notes(midi)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    args = parser.parse_args()

    all_notes = load_notes()
    stems = [s for s in get_split(args.split)
             if (LABELS_DIR / f"{s}.npz").exists()]
    if not stems:
        raise SystemExit("No labelled clips; run collect_HumTrans_wav.py "
                         "and build_labels.py first.")

    rows = []
    for stem in stems:
        labels = np.load(LABELS_DIR / f"{stem}.npz")
        ref_intervals, ref_pitches = reference_notes(
            all_notes[stem], int(labels["octave_shift"]), float(labels["lag_s"]))
        est_intervals, est_pitches = transcribe_pitch_track(
            load_pitch_track(FEATURES_DIR / f"{stem}.npz"))
        row = {"clip": stem, "align_score": float(labels["align_score"])}
        row.update(evaluate_clip(ref_intervals, ref_pitches, labels["pitch"],
                                 est_intervals, est_pitches))
        rows.append(row)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"baseline_pyin_{args.split}.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (RESULTS_DIR / f"baseline_pyin_{args.split}.json").write_text(
        json.dumps(summary, indent=2))

    print(f"{args.split}: {summary['n_clips']} clips "
          f"({summary['n_good']} with align_score >= 0.6)")
    for m in METRICS:
        print(f"  {m:11s} {summary['all'][m]:.3f}  ({summary['good'][m]:.3f})")


if __name__ == "__main__":
    main()
