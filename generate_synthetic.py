"""Generate synthetic hums with perfectly known labels.

Builds on raw_data/Librosa/test/librosa_hum_demo.ipynb: a tone with
harmonics, vibrato, slow pitch drift, a smooth envelope and a low-pass
filter that makes it sound hummed. Added here for training:
  * random melodies (mostly small steps, some rests and repeated notes)
  * low and high voices, sometimes with a weak fundamental
  * short glides between notes, background noise and random loudness

Because we choose the notes, the labels are exactly right: no alignment
needed. Useful for pre-training and for checking the pipeline end to end.

Usage:
    python generate_synthetic.py --n 2000
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import signal

from build_labels import notes_to_frames
from features import SAMPLE_RATE, compute_cqt, features_to_npz_bytes


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "processed_data" / "Synthetic"
VOICE_RANGES = {"low": (40, 57), "high": (52, 72)}   # MIDI ranges


def random_melody(rng, clip_s=8.0):
    """Notes as (onset_s, offset_s, midi) filling about clip_s seconds."""
    low, high = VOICE_RANGES[rng.choice(list(VOICE_RANGES))]
    pitch = int(rng.integers(low + 3, high - 3))
    notes, t = [], float(rng.uniform(0.2, 0.8))
    while True:
        duration = float(rng.choice([0.15, 0.25, 0.35, 0.5, 0.75, 1.0],
                                    p=[0.1, 0.25, 0.2, 0.25, 0.12, 0.08]))
        if t + duration > clip_s - 0.3:
            break
        notes.append((t, t + duration, pitch))
        t += duration
        if rng.random() < 0.15:
            t += float(rng.uniform(0.1, 0.6))          # a rest
        step = int(rng.choice([0, 1, 2, 3, 4, 5, 7, 12],
                              p=[0.12, 0.2, 0.25, 0.15, 0.1, 0.08, 0.07, 0.03]))
        pitch = int(np.clip(pitch + step * rng.choice([-1, 1]), low, high))
    return notes, clip_s


def synthesize(notes, clip_s, rng, sr=SAMPLE_RATE):
    """Render notes to audio, in the style of the librosa demo notebook."""
    n = int(clip_s * sr)
    t = np.arange(n) / sr
    target = np.zeros(n)            # pitch in MIDI, 0 = silent
    for start, end, midi in notes:
        target[int(start * sr):int(end * sr)] = midi

    # Glide: smooth the pitch steps over 20-60 ms (silence is kept separate).
    voiced = target > 0
    held = target.copy()
    for i in range(1, n):            # hold the last pitch through rests
        if held[i] == 0:
            held[i] = held[i - 1]
    held[held == 0] = held[voiced][0] if voiced.any() else 60
    glide = max(1, int(rng.uniform(0.02, 0.06) * sr))
    held = np.convolve(held, np.ones(glide) / glide, mode="same")

    vibrato = rng.uniform(0.1, 0.4) * np.sin(2 * np.pi * rng.uniform(4, 6.5) * t)
    b, a = signal.butter(2, 2.0 / (sr / 2))
    drift = signal.lfilter(b, a, rng.normal(0, 0.8, n))      # slow wander
    freq = 440.0 * 2 ** ((held + vibrato + drift - 69) / 12)
    phase = 2 * np.pi * np.cumsum(freq) / sr

    fundamental = rng.uniform(0.3, 1.0)   # sometimes weaker than overtones
    harmonics = [fundamental] + list(rng.uniform(0.1, 0.7, 4) * 0.7 ** np.arange(4))
    wave = sum(amp * np.sin(k * phase) for k, amp in enumerate(harmonics, start=1))

    # 10 ms fade in and out of every note (envelope).
    fade = int(0.01 * sr)
    env = signal.convolve(voiced.astype(float), np.ones(fade) / fade, mode="same")
    audio = wave * env

    cutoff = rng.uniform(500, 1500)
    b, a = signal.butter(4, cutoff / (sr / 2))
    audio = signal.lfilter(b, a, audio)
    audio /= np.max(np.abs(audio)) + 1e-9
    noise_db = rng.uniform(-45, -20)
    audio += rng.normal(0, 10 ** (noise_db / 20), n)
    return (audio * rng.uniform(0.2, 0.9)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    features_dir = args.output_dir / "features"
    labels_dir = args.output_dir / "labels"
    features_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        name = f"synth_{i:05d}"
        if (labels_dir / f"{name}.npz").exists():
            continue
        rng = np.random.default_rng([args.seed, i])
        notes, clip_s = random_melody(rng)
        cqt = compute_cqt(synthesize(notes, clip_s, rng))
        pitch, onset, offset = notes_to_frames(notes, cqt.shape[1])
        (features_dir / f"{name}.npz").write_bytes(features_to_npz_bytes(cqt, name))
        np.savez_compressed(labels_dir / f"{name}.npz", pitch=pitch, onset=onset,
                            offset=offset, octave_shift=0, lag_s=0.0,
                            align_score=1.0, has_onsets=True)
        if (i + 1) % 200 == 0:
            print(f"[{i + 1}/{args.n}]", flush=True)

    print(f"Synthetic clips in {args.output_dir}")


if __name__ == "__main__":
    main()
