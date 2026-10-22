"""Hum in, notes out (see diagrams/transcription_process.png).

  1. Get audio: a WAV file, or record from the microphone
  2. Clean it: trim silence from both ends
  3. Convert it to CQT features (features.py)
  4. Run the trained model, then decode its predictions into notes
  5. Print the notes and save them as a MIDI file

The model reads the whole recording forwards and backwards (Bi-LSTM), so
notes appear when the recording ends rather than while you hum.

Usage:
    python transcribe.py my_hum.wav                 # -> my_hum.mid
    python transcribe.py --mic 8                    # record 8 s -> hum.mid
    python transcribe.py my_hum.wav --checkpoint checkpoints/model_best.pt
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import torch

from decoding import decode, notes_to_midi
from features import SAMPLE_RATE, compute_cqt, load_audio
from model import HumTranscriber
from train import pick_device, predict


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "checkpoints" / "model_best.pt"


def record(seconds):
    import sounddevice as sd   # only needed for the microphone
    print(f"Recording {seconds} s... hum now!")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    print("Done.")
    return audio[:, 0]


def load_model(checkpoint, device):
    model = HumTranscriber().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.eval()


def transcribe(y, model, device):
    """Audio (16 kHz) -> (intervals in seconds, MIDI pitches)."""
    if np.max(np.abs(y), initial=0) < 1e-3:            # nothing was recorded
        return np.zeros((0, 2)), np.zeros(0, dtype=int)
    y, _ = librosa.effects.trim(y, top_db=40)          # data cleaning
    if len(y) < SAMPLE_RATE // 10:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)
    cqt = compute_cqt(y).astype(np.float32)
    return decode(*predict(model, cqt, device))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("wav", nargs="?", type=Path)
    parser.add_argument("--mic", type=float, metavar="SECONDS",
                        help="record from the microphone instead of a file")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, help="MIDI file to write")
    args = parser.parse_args()
    if (args.wav is None) == (args.mic is None):
        parser.error("give a WAV file or --mic SECONDS")

    device = pick_device()
    model = load_model(args.checkpoint, device)
    y = record(args.mic) if args.mic else load_audio(str(args.wav))
    intervals, pitches = transcribe(y, model, device)

    print(f"{len(pitches)} notes:")
    for (start, end), pitch in zip(intervals, pitches):
        print(f"  {start:6.2f}-{end:6.2f} s  {pretty_midi.note_number_to_name(int(pitch))}")

    out = args.out or (args.wav.with_suffix(".mid") if args.wav else Path("hum.mid"))
    notes_to_midi(intervals, pitches, out)
    print(f"MIDI saved to {out}")


if __name__ == "__main__":
    main()
