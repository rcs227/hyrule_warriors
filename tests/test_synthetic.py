import numpy as np

from build_labels import cqt_pitch_track, estimate_alignment
from features import SAMPLE_RATE, compute_cqt
from generate_synthetic import random_melody, synthesize


def test_melody_is_well_formed():
    notes, clip_s = random_melody(np.random.default_rng(1))
    assert len(notes) > 5
    for (s, e, p), (s2, _, _) in zip(notes, notes[1:]):
        assert 0 <= s < e <= s2 <= clip_s and 40 <= p <= 72


def test_audio_matches_its_labels():
    # The synthetic hum, read back through the label-alignment code, should
    # need no octave shift or lag: proof the labels are exactly right.
    rng = np.random.default_rng(3)
    notes, clip_s = random_melody(rng)
    audio = synthesize(notes, clip_s, rng)
    assert len(audio) == int(clip_s * SAMPLE_RATE) and np.isfinite(audio).all()
    shift, lag, score, _ = estimate_alignment(cqt_pitch_track(compute_cqt(audio)), notes)
    assert shift == 0 and abs(lag) <= 0.03 and score > 0.7
