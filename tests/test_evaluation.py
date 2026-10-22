import numpy as np

from evaluation import evaluate_clip, reference_notes, score_notes


def test_reference_notes_are_shifted():
    intervals, pitches = reference_notes([(0.5, 1.0, 60)], -12, 0.25)
    np.testing.assert_allclose(intervals, [[0.75, 1.25]])
    assert pitches.tolist() == [48]


def test_negative_lag_never_gives_negative_times():
    intervals, pitches = reference_notes([(0.0, 0.1, 60), (0.1, 0.5, 62)], 0, -0.2)
    np.testing.assert_allclose(intervals, [[0.0, 0.3]])
    assert pitches.tolist() == [62]


def test_perfect_transcription_scores_one():
    intervals, pitches = np.array([[0.5, 1.0], [1.0, 1.5]]), np.array([60, 62])
    scores = score_notes(intervals, pitches, intervals, pitches)
    assert scores == {"COn_f1": 1.0, "COnP_f1": 1.0, "COnPOff_f1": 1.0}


def test_wrong_pitch_keeps_onset_score_only():
    ref_i, ref_p = np.array([[0.5, 1.0]]), np.array([60])
    scores = score_notes(ref_i, ref_p, ref_i, np.array([61]))
    assert scores["COn_f1"] == 1.0 and scores["COnP_f1"] == 0.0


def test_empty_transcription_scores_zero():
    ref_frames = np.zeros(200, dtype=np.int16); ref_frames[50:100] = 60
    row = evaluate_clip(np.array([[0.5, 1.0]]), np.array([60]), ref_frames,
                        np.zeros((0, 2)), np.array([]))
    assert row["COnP_f1"] == 0.0 and row["frame_acc"] == 0.0
