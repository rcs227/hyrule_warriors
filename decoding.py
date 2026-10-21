"""Turn the model's frame-by-frame predictions into notes and MIDI files.

A note starts where the hummed pitch changes, or where the model is
confident a new note begins on the same pitch (a repeated note). It ends at
the next silence, pitch change or note start. Notes shorter than 60 ms are
dropped as noise.
"""
import numpy as np
import pretty_midi
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from dataset import class_to_midi
from features import FRAME_S


MIN_NOTE_S = 0.06
ONSET_THRESHOLD = 0.5


def decode(pitch_probs, onset_probs, onset_threshold=ONSET_THRESHOLD,
           min_note_s=MIN_NOTE_S, smooth=5):
    """pitch_probs (frames, classes), onset_probs (frames,) ->
    (intervals in seconds, MIDI pitches)."""
    midi = class_to_midi(np.argmax(pitch_probs, axis=1))
    if smooth > 1:
        midi = median_filter(midi, size=smooth)
    peaks, _ = find_peaks(np.concatenate([[0], onset_probs, [0]]),
                          height=onset_threshold)
    is_onset = np.zeros(len(midi), dtype=bool)
    is_onset[np.clip(peaks - 1, 0, len(midi) - 1)] = True

    intervals, pitches = [], []
    start = None
    for i in range(len(midi) + 1):
        boundary = (i == len(midi) or midi[i] == 0 or start is None
                    or midi[i] != midi[start] or is_onset[i])
        if not boundary:
            continue
        if start is not None and (i - start) * FRAME_S >= min_note_s - 1e-9:
            intervals.append((start * FRAME_S, i * FRAME_S))
            pitches.append(int(midi[start]))
        start = i if i < len(midi) and midi[i] != 0 else None
    return np.array(intervals, dtype=float).reshape(-1, 2), np.array(pitches, dtype=int)


def notes_to_midi(intervals, pitches, path, program=0):
    """Write notes to a MIDI file (acoustic grand piano by default)."""
    song = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)
    for (start, end), pitch in zip(intervals, pitches):
        instrument.notes.append(pretty_midi.Note(
            velocity=90, pitch=int(pitch), start=float(start), end=float(end)))
    song.instruments.append(instrument)
    song.write(str(path))
