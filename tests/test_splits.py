from splits import load_split_keys


def test_official_split_sizes():
    keys = load_split_keys()
    assert {k: len(v) for k, v in keys.items()} == {
        "train": 13080, "valid": 765, "test": 769}


def test_no_clip_in_two_splits():
    keys = load_split_keys()
    train, valid, test = (set(keys[k]) for k in ("train", "valid", "test"))
    assert not train & valid and not train & test and not valid & test


def test_no_song_in_two_splits():
    # Clip names are personID_musicID_..., so the song is the 2nd field.
    songs = {k: {s.split("_")[1] for s in v} for k, v in load_split_keys().items()}
    assert not songs["train"] & songs["test"]
    assert not songs["train"] & songs["valid"]
    assert not songs["valid"] & songs["test"]
