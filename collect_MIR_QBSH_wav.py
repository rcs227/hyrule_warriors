import re
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from features import (compute_cqt, compute_pitch_track, features_to_npz_bytes,
                      load_audio)


PROJECT_DIR = Path(__file__).resolve().parent
ZIP_PATH = PROJECT_DIR / "raw_data" / "MIR-QBSH" / "MIR-QBSH.zip"
OUTPUT_DIR = PROJECT_DIR / "processed_data" / "MIR-QBSH" / "features"



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
    """Convert one WAV held in memory to the shared CQT + pYIN features."""
    y = load_audio(wav_bytes)
    cqt = compute_cqt(y)
    return features_to_npz_bytes(cqt, source_member,
                                 compute_pitch_track(y, cqt.shape[1]))


def clip_name(member):
    """Flat, unique name for one recording, e.g. year2003_person00001_00013."""
    year_folder, person_id, song_id = get_recording_info(member)
    return f"{year_folder}_{person_id}_{song_id}"


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
            output_path = OUTPUT_DIR / f"{clip_name(member)}.npz"
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
            output_path = OUTPUT_DIR / f"{clip_name(member)}.npz"

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