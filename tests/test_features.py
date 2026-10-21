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


def test_pitch_track_matches_cqt_frames_and_pitch():
    from features import compute_pitch_track
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    y = np.concatenate([np.zeros(SAMPLE_RATE // 2),
                        0.3 * np.sin(2 * np.pi * 220 * t),      # A3 = MIDI 57
                        np.zeros(SAMPLE_RATE // 2)]).astype(np.float32)
    cqt = compute_cqt(y)
    track = compute_pitch_track(y, cqt.shape[1])
    assert track.shape == (cqt.shape[1],)
    assert abs(np.nanmedian(track[70:130].astype(float)) - 57) < 0.2
    assert np.isnan(track[:30].astype(float)).all()


def test_npz_keeps_pitch_track(tmp_path):
    from features import load_pitch_track
    cqt = np.zeros((N_BINS, 10), dtype=np.float16)
    track = np.full(10, np.nan, dtype=np.float16)
    track[3:6] = 60
    (tmp_path / "a.npz").write_bytes(features_to_npz_bytes(cqt, "a", track))
    (tmp_path / "b.npz").write_bytes(features_to_npz_bytes(cqt, "b"))
    np.testing.assert_array_equal(load_pitch_track(tmp_path / "a.npz"), track.astype(np.float32))
    assert load_pitch_track(tmp_path / "b.npz") is None
