import numpy as np

from baseline_pyin import frames_to_notes


def test_runs_become_notes():
    midi = np.array([0] * 10 + [60] * 20 + [62] * 30 + [0] * 5)
    intervals, pitches = frames_to_notes(midi)
    np.testing.assert_allclose(intervals, [[0.10, 0.30], [0.30, 0.60]])
    assert list(pitches) == [60, 62]


def test_short_runs_are_dropped():
    midi = np.array([60] * 3 + [0] * 5 + [64] * 6)
    _, pitches = frames_to_notes(midi, min_note_s=0.06)
    assert list(pitches) == [64]


def test_silence_gives_no_notes():
    intervals, pitches = frames_to_notes(np.zeros(100))
    assert intervals.shape == (0, 2) and len(pitches) == 0
