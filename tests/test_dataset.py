import numpy as np
import pytest

from build_labels import notes_to_frames
from dataset import (IGNORE, HumDataset, class_to_midi, midi_to_class,
                     widen_onsets)
from features import N_BINS, features_to_npz_bytes


def make_clip(tmp_path, name, n_frames, notes, has_onsets=None):
    cqt = np.random.rand(N_BINS, n_frames).astype(np.float16)
    (tmp_path / f"{name}_f.npz").write_bytes(features_to_npz_bytes(cqt, name))
    pitch, onset, offset = notes_to_frames(notes, n_frames)
    extra = {} if has_onsets is None else {"has_onsets": has_onsets}
    np.savez(tmp_path / f"{name}_l.npz", pitch=pitch, onset=onset, offset=offset, **extra)
    return tmp_path / f"{name}_f.npz", tmp_path / f"{name}_l.npz"


def test_class_mapping_round_trip_and_range():
    midi = np.array([0, 36, 60, 84, 30, 90])
    classes = midi_to_class(midi)
    assert classes.tolist() == [0, 1, 25, 49, IGNORE, IGNORE]
    assert class_to_midi(classes[:4]).tolist() == [0, 36, 60, 84]


def test_widen_onsets():
    onset = np.zeros(10, dtype=np.uint8); onset[[0, 5]] = 1
    assert np.flatnonzero(widen_onsets(onset, 1)).tolist() == [0, 1, 4, 5, 6]


def test_window_shapes_and_pairing(tmp_path):
    clip = make_clip(tmp_path, "a", 500, [(1.0, 2.0, 60), (2.0, 4.0, 64)])
    ds = HumDataset([clip], window=300, seed=0)
    item = ds[0]
    assert item["features"].shape == (N_BINS, 300)
    for key in ("pitch", "onset", "onset_mask"):
        assert item[key].shape == (300,)
    # Features and labels must come from the same position in the clip.
    full_cqt, full_pitch, _, _ = ds.load(0)
    feats = item["features"].numpy()
    start = next(s for s in range(201) if np.array_equal(full_cqt[:, s:s + 300], feats))
    assert np.array_equal(item["pitch"].numpy(), full_pitch[start:start + 300])


def test_short_clip_is_padded_and_masked(tmp_path):
    clip = make_clip(tmp_path, "b", 120, [(0.2, 0.8, 60)])
    item = HumDataset([clip], window=300, seed=0)[0]
    assert item["features"].shape == (N_BINS, 300)
    assert (item["pitch"][120:] == IGNORE).all()
    assert (item["onset_mask"][120:] == 0).all() and (item["onset_mask"][:120] == 1).all()


def test_whole_clip_mode(tmp_path):
    clip = make_clip(tmp_path, "c", 437, [(0.2, 0.8, 60)])
    item = HumDataset([clip], window=None)[0]
    assert item["features"].shape == (N_BINS, 437)


def test_dataset_without_onsets_is_masked(tmp_path):
    clip = make_clip(tmp_path, "d", 400, [(0.2, 0.8, 60)], has_onsets=False)
    item = HumDataset([clip], window=300, seed=0)[0]
    assert (item["onset_mask"] == 0).all()


def test_from_dirs_skips_missing(tmp_path):
    (tmp_path / "f").mkdir(); (tmp_path / "l").mkdir()
    f, l = make_clip(tmp_path, "e", 400, [(0.2, 0.8, 60)])
    f.rename(tmp_path / "f" / "e.npz"); l.rename(tmp_path / "l" / "e.npz")
    ds = HumDataset.from_dirs(tmp_path / "f", tmp_path / "l", ["e", "missing"])
    assert len(ds) == 1
