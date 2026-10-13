"""Non-deep-learning baseline: pYIN pitch tracking + simple note grouping.

Streams test-split hums from the HumTrans ZIP, transcribes each with pYIN,
and scores the notes against the aligned MIDI labels with mir_eval.
Any model we train should beat these numbers.

Reference notes get the same per-clip octave shift and lag as the training
labels (see build_labels.py), so this is the evaluation every model uses.

Usage:
    python baseline_pyin.py --n 100     # 100 random test clips
"""
import argparse
import csv
import random
from io import BytesIO
from pathlib import Path

import librosa
import mir_eval
import numpy as np
from remotezip import RemoteZip
from scipy.ndimage import median_filter

from build_labels import (FRAME_S, cqt_pitch_track, estimate_alignment,
                          load_notes, notes_to_frames)
from splits import get_split


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = PROJECT_DIR / "results" / "baseline_pyin.csv"
WAV_ZIP_URL = (
    "https://huggingface.co/datasets/dadinghh2/HumTrans/"
    "resolve/main/all_wav.zip"
)

SAMPLE_RATE = 16_000
HOP_LENGTH = 160
MIN_NOTE_S = 0.06
ONSET_TOLERANCE_S = 0.05
PITCH_TOLERANCE_CENTS = 50


def cqt_features(y, sr):
    """Same CQT as collect_HumTrans_wav.py (used only for alignment)."""
    cqt = librosa.cqt(y=y, sr=sr, hop_length=HOP_LENGTH,
                      fmin=librosa.note_to_hz("C1"), n_bins=84,
                      bins_per_octave=12)
    return np.log1p(np.abs(cqt)).astype(np.float32)


def transcribe(y, sr):
    """Return (intervals in seconds, MIDI pitches) for one hum."""
    f0, voiced, _ = librosa.pyin(y, sr=sr, fmin=60, fmax=1000,
                                 hop_length=HOP_LENGTH)
    midi = np.where(voiced, np.round(librosa.hz_to_midi(f0)), 0)
    midi = median_filter(midi, size=5)   # remove one-frame wobbles

    intervals, pitches = [], []
    start = 0
    for i in range(1, len(midi) + 1):
        if i == len(midi) or midi[i] != midi[start]:
            if midi[start] > 0 and (i - start) * FRAME_S >= MIN_NOTE_S:
                intervals.append((start * FRAME_S, i * FRAME_S))
                pitches.append(midi[start])
            start = i
    return np.array(intervals).reshape(-1, 2), np.array(pitches)


def score_notes(ref_intervals, ref_pitches, est_intervals, est_pitches):
    """Note-level F1 scores in the standard mir_eval settings."""
    ref_hz = librosa.midi_to_hz(ref_pitches)
    est_hz = librosa.midi_to_hz(est_pitches) if len(est_pitches) else np.array([])
    common = dict(onset_tolerance=ONSET_TOLERANCE_S,
                  pitch_tolerance=PITCH_TOLERANCE_CENTS)
    _, _, con = mir_eval.transcription.onset_precision_recall_f1(
        ref_intervals, est_intervals, onset_tolerance=ONSET_TOLERANCE_S)
    _, _, conp, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_hz, est_intervals, est_hz, offset_ratio=None, **common)
    _, _, conpoff, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_hz, est_intervals, est_hz, **common)
    return {"COn_f1": con, "COnP_f1": conp, "COnPOff_f1": conpoff}


def frame_pitch_accuracy(ref_frames, est_intervals, est_pitches):
    """Fraction of labelled frames where the predicted pitch is exactly right."""
    est_frames = np.zeros_like(ref_frames)
    for (start_s, end_s), midi in zip(est_intervals, est_pitches):
        est_frames[int(round(start_s / FRAME_S)):int(round(end_s / FRAME_S))] = midi
    labelled = ref_frames > 0
    return float(np.mean(est_frames[labelled] == ref_frames[labelled]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=100,
                        help="number of random test clips (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    all_notes = load_notes()
    test_stems = get_split("test")
    if args.n:
        test_stems = random.Random(args.seed).sample(test_stems, args.n)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    with RemoteZip(WAV_ZIP_URL, timeout=60) as archive:
        members = {Path(n).stem: n for n in archive.namelist()
                   if n.lower().endswith(".wav")}

        for index, stem in enumerate(test_stems, start=1):
            try:
                y, sr = librosa.load(BytesIO(archive.read(members[stem])),
                                     sr=SAMPLE_RATE, mono=True)
            except Exception as exc:
                print(f"[{index}/{len(test_stems)}] {stem}: skipped ({exc})")
                continue

            notes = all_notes[stem]
            octave_shift, lag_s, align_score = estimate_alignment(
                cqt_pitch_track(cqt_features(y, sr)), notes)
            ref_intervals = np.array([(s + lag_s, e + lag_s) for s, e, _ in notes])
            ref_pitches = np.array([p + octave_shift for _, _, p in notes])
            ref_frames, _, _ = notes_to_frames(
                notes, int(np.ceil(len(y) / HOP_LENGTH)), octave_shift, lag_s)

            est_intervals, est_pitches = transcribe(y, sr)
            row = {"clip": stem, "align_score": round(align_score, 3)}
            row.update(score_notes(ref_intervals, ref_pitches,
                                   est_intervals, est_pitches))
            row["frame_acc"] = frame_pitch_accuracy(
                ref_frames, est_intervals, est_pitches)
            rows.append(row)
            print(f"[{index}/{len(test_stems)}] {stem}: "
                  f"COnP={row['COnP_f1']:.2f} frame_acc={row['frame_acc']:.2f}")

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = ["COn_f1", "COnP_f1", "COnPOff_f1", "frame_acc"]
    good = [r for r in rows if r["align_score"] >= 0.6]
    print(f"\nMean over {len(rows)} test clips "
          f"(and over the {len(good)} with align_score >= 0.6):")
    for m in metrics:
        print(f"  {m:11s} {np.mean([r[m] for r in rows]):.3f}"
              f"  ({np.mean([r[m] for r in good]):.3f})")
    print(f"Per-clip results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
