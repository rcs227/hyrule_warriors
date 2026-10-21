import numpy as np

from features import (N_BINS, SAMPLE_RATE, bin_to_midi, compute_cqt,
                      features_to_npz_bytes, load_cqt, midi_to_bin)


def test_cqt_shape_and_frame_rate():
    cqt = compute_cqt(np.zeros(SAMPLE_RATE * 2, dtype=np.float32))
    assert cqt.shape == (N_BINS, 201)       # 100 frames per second, centred
    assert cqt.dtype == np.float16


def test_bin_midi_round_trip():
    for midi in (36, 48.0, 60, 83):
        assert bin_to_midi(midi_to_bin(midi)) == midi


def test_npz_round_trip(tmp_path):
    cqt = np.random.rand(N_BINS, 50).astype(np.float16)
    path = tmp_path / "clip.npz"
    path.write_bytes(features_to_npz_bytes(cqt, "clip.wav"))
    np.testing.assert_array_equal(load_cqt(path), cqt.astype(np.float32))
