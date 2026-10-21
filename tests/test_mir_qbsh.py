import numpy as np

from build_MIR_QBSH_labels import agreement, frames_to_labels, parse_pv, pv_to_frames
from collect_MIR_QBSH_wav import clip_name


def test_parse_pv():
    assert parse_pv("0\n0\n60.5\n61 \n").tolist() == [0, 0, 60.5, 61]


def test_pv_resampled_to_10ms_frames():
    pv = np.array([0, 60, 62, 0])                   # 4 frames of 32 ms
    frames = pv_to_frames(pv, 13, 0.032)            # 0.13 s at 10 ms
    assert frames.tolist() == [0, 0, 0, 0, 60, 60, 60, 62, 62, 62, 0, 0, 0]


def test_labels_from_pitch():
    pitch, onset, offset = frames_to_labels(np.array([0, 59.8, 60.2, 62, 62, 0, -1]))
    assert pitch.tolist() == [0, 60, 60, 62, 62, 0, 0]
    assert np.flatnonzero(onset).tolist() == [1, 3]
    assert np.flatnonzero(offset).tolist() == [2, 4]


def test_agreement():
    pv = np.array([0, 60, 60, 62, 62])
    pyin = np.array([np.nan, 60.2, 59.9, 50.0, np.nan])
    assert agreement(pv, pyin) == 2 / 3


def test_clip_name_is_flat_and_unique():
    member = "MIR-QBSH-corpus/waveFile/year2003/person00001/00013.wav"
    assert clip_name(member) == "year2003_person00001_00013"
    assert clip_name(member.replace(".wav", ".pv")) == "year2003_person00001_00013"
