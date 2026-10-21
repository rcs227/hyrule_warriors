"""Scoring shared by the baseline and the model.

Both are scored against the same reference: the clip's MIDI notes moved by
the octave shift and lag that build_labels.py found for that clip.
"""
import numpy as np
import librosa
import mir_eval

from features import FRAME_S


ONSET_TOLERANCE_S = 0.05
PITCH_TOLERANCE_CENTS = 50
METRICS = ["COn_f1", "COnP_f1", "COnPOff_f1", "frame_acc"]


def reference_notes(notes, octave_shift, lag_s):
    """Aligned reference as (intervals in seconds, MIDI pitches)."""
    intervals = np.array([(s + lag_s, e + lag_s) for s, e, _ in notes]).reshape(-1, 2)
    pitches = np.array([p + octave_shift for _, _, p in notes])
    return intervals, pitches


def score_notes(ref_intervals, ref_pitches, est_intervals, est_pitches):
    """Note-level F1 scores in the standard mir_eval settings.

    COn: note start within 50 ms. COnP: also the right pitch (within 50
    cents). COnPOff: also the note end is close.
    """
    ref_hz = librosa.midi_to_hz(ref_pitches)
    est_hz = librosa.midi_to_hz(est_pitches) if len(est_pitches) else np.array([])
    est_intervals = np.asarray(est_intervals, dtype=float).reshape(-1, 2)
    common = dict(onset_tolerance=ONSET_TOLERANCE_S,
                  pitch_tolerance=PITCH_TOLERANCE_CENTS)
    _, _, con = mir_eval.transcription.onset_precision_recall_f1(
        ref_intervals, est_intervals, onset_tolerance=ONSET_TOLERANCE_S)
    _, _, conp, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_hz, est_intervals, est_hz, offset_ratio=None, **common)
    _, _, conpoff, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_hz, est_intervals, est_hz, **common)
    return {"COn_f1": con, "COnP_f1": conp, "COnPOff_f1": conpoff}


def notes_to_frame_pitches(intervals, pitches, n_frames):
    """Per-frame MIDI number (0 = silence) for a list of notes."""
    frames = np.zeros(n_frames, dtype=np.int16)
    for (start_s, end_s), midi in zip(intervals, pitches):
        frames[int(round(start_s / FRAME_S)):int(round(end_s / FRAME_S))] = midi
    return frames


def frame_pitch_accuracy(ref_frames, est_frames):
    """Fraction of labelled frames where the predicted pitch is exactly right."""
    labelled = ref_frames > 0
    if not labelled.any():
        return float("nan")
    return float(np.mean(est_frames[labelled] == ref_frames[labelled]))


def evaluate_clip(ref_intervals, ref_pitches, ref_frames, est_intervals, est_pitches):
    row = score_notes(ref_intervals, ref_pitches, est_intervals, est_pitches)
    est_frames = notes_to_frame_pitches(est_intervals, est_pitches, len(ref_frames))
    row["frame_acc"] = frame_pitch_accuracy(ref_frames, est_frames)
    return row


def summarize(rows, min_align_score=0.6):
    """Mean of each metric over all clips and over well-aligned clips."""
    good = [r for r in rows if r["align_score"] >= min_align_score]
    return {
        "n_clips": len(rows), "n_good": len(good),
        "all": {m: float(np.nanmean([r[m] for r in rows])) for m in METRICS},
        "good": {m: float(np.nanmean([r[m] for r in good])) for m in METRICS}
        if good else {},
    }
