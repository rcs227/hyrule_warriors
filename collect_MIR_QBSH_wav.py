from io import BytesIO
import json
import re
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import librosa
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
ZIP_PATH = PROJECT_DIR / "raw_data" / "MIR-QBSH" / "MIR-QBSH.zip"
OUTPUT_DIR = PROJECT_DIR / "processed_data" / "MIR-QBSH" / "features"

SAMPLE_RATE = 16_000
HOP_LENGTH = 160       # 10 ms at 16 kHz
N_FFT = 1024           # 64 ms window
N_MELS = 128


def get_recording_info(member):
    """Return year folder, person ID, and song ID from a WAV ZIP path."""
    parts = PurePosixPath(member).parts

    try:
        wav_folder_index = parts.index("waveFile")
    except ValueError:
        raise ValueError("Path is not inside waveFile")

    folders = parts[wav_folder_index + 1:-1]
    year_folder = next(
        (p for p in folders if re.fullmatch(r"year\d{4}[ab]?", p)),
        None,
    )
    person_id = next(
        (p for p in folders if re.fullmatch(r"person\d+", p)),
        None,
    )

    if year_folder is None or person_id is None:
        raise ValueError("Could not identify year/person folders")

    song_id = Path(PurePosixPath(member).name).stem
    return year_folder, person_id, song_id


def convert_wav(wav_bytes, source_member):
    """Convert one WAV held in memory to log-mel and CQT arrays."""
    y, sr = librosa.load(BytesIO(wav_bytes), sr=SAMPLE_RATE, mono=True)

    if y.size == 0:
        raise ValueError("WAV contains no audio samples")

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
    log_mel_db = librosa.power_to_db(
        mel_power, ref=np.max
    ).astype(np.float32)

    cqt = librosa.cqt(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
        fmin=librosa.note_to_hz("C1"),
        n_bins=84,             # C1 through B7
        bins_per_octave=12,    # One bin per semitone
    )
    cqt_log_magnitude = np.log1p(
        np.abs(cqt)
    ).astype(np.float32)

    n_frames = log_mel_db.shape[1]
    frame_times_s = (
        np.arange(n_frames, dtype=np.float32) * HOP_LENGTH / sr
    )

    metadata = {
        "source_member": source_member,
        "sample_rate": sr,
        "hop_length_samples": HOP_LENGTH,
        "frame_interval_seconds": HOP_LENGTH / sr,
        "n_fft": N_FFT,
        "n_mels": N_MELS,
        "cqt_fmin": "C1",
        "cqt_bins": 84,
        "cqt_bins_per_octave": 12,
    }

    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        log_mel_db=log_mel_db,
        cqt_log_magnitude=cqt_log_magnitude,
        frame_times_s=frame_times_s,
        metadata_json=np.array(json.dumps(metadata)),
    )
    return buffer.getvalue()


def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"ZIP file not found: {ZIP_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with ZipFile(ZIP_PATH) as archive:
        wav_members = [
            name for name in archive.namelist()
            if (
                "waveFile" in PurePosixPath(name).parts
                and name.lower().endswith(".wav")
            )
        ]

        # Check for duplicate destination paths so one recording cannot
        # silently overwrite another.
        destinations = {}
        for member in wav_members:
            year_folder, person_id, song_id = get_recording_info(member)
            output_path = OUTPUT_DIR / year_folder / person_id / f"{song_id}.npz"
            destinations.setdefault(str(output_path), []).append(member)

        collisions = {
            path: members
            for path, members in destinations.items()
            if len(members) > 1
        }
        if collisions:
            details = "\n".join(
                f"{path}: {members}" for path, members in collisions.items()
            )
            raise ValueError(
                "Multiple WAVs map to the same requested output path. "
                "No files were processed; resolve these collisions first:\n"
                + details
            )

        errors = []

        for index, member in enumerate(wav_members, start=1):
            year_folder, person_id, song_id = get_recording_info(member)
            output_path = OUTPUT_DIR / year_folder / person_id / f"{song_id}.npz"

            if output_path.exists():
                print(f"[{index}/{len(wav_members)}] Skipping existing {output_path}")
                continue

            print(f"[{index}/{len(wav_members)}] Processing {member}")

            try:
                wav_bytes = archive.read(member)
                npz_bytes = convert_wav(wav_bytes, member)

                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(npz_bytes)

            except Exception as exc:
                errors.append((member, str(exc)))
                print(f"  Skipped: {exc}")

    print(f"\nProcessed {len(wav_members) - len(errors)} WAV entries.")
    print(f"Features saved under: {OUTPUT_DIR}")

    if errors:
        print(f"{len(errors)} WAV file(s) could not be processed.")


if __name__ == "__main__":
    main()