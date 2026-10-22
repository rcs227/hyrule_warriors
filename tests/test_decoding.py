import numpy as np
import pretty_midi

from build_labels import notes_to_frames
from dataset import N_CLASSES, midi_to_class
from decoding import decode, notes_to_midi


def perfect_predictions(notes, n_frames):
    """One-hot pitch and onset 'predictions' made from known notes."""
    pitch, onset, _ = notes_to_frames(notes, n_frames)
    probs = np.zeros((n_frames, N_CLASSES))
    probs[np.arange(n_frames), midi_to_class(pitch)] = 1
    return probs, onset.astype(float)


def test_round_trip_recovers_notes():
    notes = [(0.5, 1.0, 60), (1.0, 1.4, 62), (1.6, 2.4, 64)]
    intervals, pitches = decode(*perfect_predictions(notes, 300), smooth=1)
    np.testing.assert_allclose(intervals, [(s, e) for s, e, _ in notes], atol=1e-9)
    assert pitches.tolist() == [60, 62, 64]


def test_repeated_note_is_split_by_onset():
    notes = [(0.5, 1.0, 60), (1.0, 1.5, 60)]
    _, pitches = decode(*perfect_predictions(notes, 200), smooth=1)
    assert pitches.tolist() == [60, 60]


def test_tiny_blips_are_dropped():
    probs, onset = perfect_predictions([(0.5, 1.5, 60), (1.5, 1.53, 62)], 200)
    _, pitches = decode(probs, onset, smooth=1)
    assert pitches.tolist() == [60]


def test_midi_file_contains_the_notes(tmp_path):
    intervals, pitches = np.array([[0.0, 0.5], [0.5, 1.25]]), np.array([60, 67])
    notes_to_midi(intervals, pitches, tmp_path / "out.mid")
    notes = pretty_midi.PrettyMIDI(str(tmp_path / "out.mid")).instruments[0].notes
    assert [(round(n.start, 3), round(n.end, 3), n.pitch) for n in notes] == [
        (0.0, 0.5, 60), (0.5, 1.25, 67)]
