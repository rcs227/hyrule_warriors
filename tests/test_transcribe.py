import numpy as np
import torch

from generate_synthetic import random_melody, synthesize
from model import HumTranscriber
from transcribe import transcribe


def test_transcribe_runs_end_to_end():
    # An untrained model gives nonsense notes, but the whole path
    # (trim -> features -> model -> decode) must run and return notes.
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    audio = synthesize(*random_melody(rng, clip_s=3.0), rng)
    intervals, pitches = transcribe(audio, HumTranscriber().eval(), torch.device("cpu"))
    assert intervals.shape[1] == 2 and len(intervals) == len(pitches)
    assert (intervals[:, 1] > intervals[:, 0]).all()


def test_silence_gives_no_notes():
    intervals, pitches = transcribe(np.zeros(16000, dtype=np.float32),
                                    HumTranscriber().eval(), torch.device("cpu"))
    assert len(pitches) == 0
