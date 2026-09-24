from io import BytesIO
from pathlib import Path, PurePosixPath
import json

import librosa
import numpy as np
from remotezip import RemoteZip


# Settings to change
OUTPUT_DIR = Path(__file__).resolve().parent / "processed_data" / "HumTrans" / "features"
MAX_OUTPUT_GB = .013
FEATURE = "both"  # "logmel", "cqt", or "both"

WAV_ZIP_URL = (
    "https://huggingface.co/datasets/dadinghh2/HumTrans/"
    "resolve/main/all_wav.zip"
)

SAMPLE_RATE = 16_000
HOP_LENGTH = 160       # 10 ms
N_FFT = 1024
N_MELS = 128

MAX_OUTPUT_BYTES = int(MAX_OUTPUT_GB * 1_000_000_000)


def make_npz(wav_bytes, member_name):
    """Convert one in-memory WAV to a compressed NPZ byte string."""
    y, sr = librosa.load(BytesIO(wav_bytes), sr=SAMPLE_RATE, mono=True)

    if y.size == 0:
        raise ValueError("WAV contained no audio samples")

    arrays = {}

    if FEATURE in ("logmel", "both"):
        mel_power = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=50,
            fmax=sr / 2,
            power=2.0,
        )
        arrays["log_mel_db"] = librosa.power_to_db(
            mel_power, ref=np.max
        ).astype(np.float32)

    if FEATURE in ("cqt", "both"):
        cqt = librosa.cqt(
            y=y,
            sr=sr,
            hop_length=HOP_LENGTH,
            fmin=librosa.note_to_hz("C1"),
            n_bins=84,
            bins_per_octave=12,
        )
        arrays["cqt_log_magnitude"] = np.log1p(
            np.abs(cqt)
        ).astype(np.float32)

    first_feature = next(iter(arrays.values()))
    frame_times_s = (
        np.arange(first_feature.shape[1], dtype=np.float32)
        * HOP_LENGTH / sr
    )

    metadata = {
        "source_member": member_name,
        "sample_rate": sr,
        "hop_length": HOP_LENGTH,
        "feature": FEATURE,
    }

    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        **arrays,
        frame_times_s=frame_times_s,
        metadata_json=np.array(json.dumps(metadata)),
    )
    return buffer.getvalue()


def existing_output_size():
    return sum(p.stat().st_size for p in OUTPUT_DIR.rglob("*.npz"))


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
used_bytes = existing_output_size()

if used_bytes >= MAX_OUTPUT_BYTES:
    raise SystemExit("Output folder is already at or above the storage limit.")

with RemoteZip(WAV_ZIP_URL, timeout=30) as archive:
    wav_members = [
        name for name in archive.namelist()
        if name.lower().endswith(".wav")
    ]

    for index, member in enumerate(wav_members, start=1):
        relative_output = Path(PurePosixPath(member).name).with_suffix(".npz")
        output_path = OUTPUT_DIR / relative_output

        if output_path.exists():
            print(f"Skipping existing: {output_path}")
            continue

        print(f"[{index}/{len(wav_members)}] Processing {member}")

        try:
            wav_bytes = archive.read(member)
            npz_bytes = make_npz(wav_bytes, member)
        except Exception as exc:
            print(f"Skipped {member}: {exc}")
            continue

        print("Output folder:", OUTPUT_DIR.resolve())
        print("Existing NPZ bytes:", existing_output_size())
        print("Storage cap bytes:", MAX_OUTPUT_BYTES)

        if used_bytes + len(npz_bytes) > MAX_OUTPUT_BYTES:
            print("Storage limit reached; stopping before saving this file.")
            break

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(npz_bytes)
        used_bytes += len(npz_bytes)

        print(
            f"Saved {output_path} "
            f"({used_bytes / 1_000_000_000:.3f} / {MAX_OUTPUT_GB:.3f} GB)"
        )

print("Finished.")