import numpy as np
import pytest

from build_labels import cqt_pitch_track, estimate_alignment, notes_to_frames
from features import SAMPLE_RATE, compute_cqt

MELODY = [(0.5, 1.0, 60), (1.0, 1.4, 62), (1.6, 2.4, 64), (2.5, 3.0, 60)]


def test_note_becomes_frames():
    pitch, onset, offset = notes_to_frames([(0.5, 1.5, 60)], 300)
    assert (pitch[50:150] == 60).all()
    assert (pitch[:50] == 0).all() and (pitch[150:] == 0).all()
    assert np.flatnonzero(onset).tolist() == [50]
    assert np.flatnonzero(offset).tolist() == [149]


def test_shift_and_lag_move_the_note():
    pitch, onset, _ = notes_to_frames([(0.5, 1.5, 60)], 300,
                                      octave_shift=-12, lag_s=0.3)
    assert (pitch[80:180] == 48).all() and pitch[79] == 0
    assert np.flatnonzero(onset).tolist() == [80]


def test_notes_past_the_end_are_cut_off():
    pitch, onset, offset = notes_to_frames([(0.5, 5.0, 60), (4.0, 4.5, 62)], 100)
    assert len(pitch) == 100 and (pitch[50:] == 60).all()
    assert np.flatnonzero(onset).tolist() == [50]
    assert np.flatnonzero(offset).tolist() == [99]


def test_back_to_back_notes_get_separate_onsets():
    _, onset, _ = notes_to_frames([(0.0, 0.5, 60), (0.5, 1.0, 60)], 100)
    assert np.flatnonzero(onset).tolist() == [0, 50]


@pytest.mark.parametrize("shift, lag_s", [(-12, 0.3), (0, 0.0), (12, 0.9)])
def test_alignment_recovers_known_shift_and_lag(shift, lag_s):
    # A fake "hummed" pitch track: the melody moved by shift and lag_s.
    hum, _, _ = notes_to_frames(MELODY, 400, octave_shift=shift, lag_s=lag_s)
    hum = np.where(hum > 0, hum, np.nan).astype(float)
    est_shift, est_lag, score, _ = estimate_alignment(hum, MELODY)
    assert est_shift == shift
    assert est_lag == pytest.approx(lag_s, abs=0.011)
    assert score > 0.95


def _hum(midi_notes, seconds_each=0.5):
    """Synthesize a simple hum: a tone with a few harmonics per note."""
    t = np.arange(int(SAMPLE_RATE * seconds_each)) / SAMPLE_RATE
    parts = []
    for midi in midi_notes:
        f0 = 440 * 2 ** ((midi - 69) / 12)
        parts.append(sum(a * np.sin(2 * np.pi * k * f0 * t)
                         for k, a in [(1, 1.0), (2, 0.5), (3, 0.3)]))
    silence = np.zeros(SAMPLE_RATE // 2)
    return np.concatenate([silence, *parts, silence]).astype(np.float32) * 0.3


@pytest.mark.parametrize("midi", [45, 57, 69])
def test_cqt_pitch_track_finds_the_hummed_note(midi):
    track = cqt_pitch_track(compute_cqt(_hum([midi], 1.0)))
    middle = track[70:130]          # well inside the 0.5-1.5 s tone
    assert np.nanmedian(middle) == pytest.approx(midi, abs=0.4)
    assert np.isnan(track[:30]).all()   # leading silence is unvoiced


def test_cqt_pitch_track_aligns_to_the_true_melody():
    notes = [(0.5, 1.0, 55), (1.0, 1.5, 57), (1.5, 2.0, 60), (2.0, 2.5, 62)]
    track = cqt_pitch_track(compute_cqt(_hum([55, 57, 60, 62])))
    shift, lag, score, _ = estimate_alignment(track, notes)
    assert shift == 0 and abs(lag) <= 0.05 and score > 0.8


def test_out_of_tune_singer_still_scores_well():
    # Someone humming the right notes 0.45 semitone flat throughout: the
    # labels are correct, so the score should stay high.
    hum, _, _ = notes_to_frames(MELODY, 400, lag_s=0.2)
    hum = np.where(hum > 0, hum - 0.45, np.nan).astype(float)
    shift, lag, score, tuning = estimate_alignment(hum, MELODY)
    assert shift == 0 and lag == pytest.approx(0.2, abs=0.011)
    assert tuning == pytest.approx(-0.45, abs=0.01)
    assert score > 0.95
