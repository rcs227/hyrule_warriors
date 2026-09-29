import csv
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pretty_midi


PROJECT_DIR = Path(__file__).resolve().parent
ZIP_PATH = PROJECT_DIR / "raw_data" / "MIR-QBSH" / "MIR-QBSH.zip"
OUTPUT_CSV = PROJECT_DIR / "processed_data" / "MIR-QBSH" / "midi_notes.csv"


def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"ZIP file not found: {ZIP_PATH}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    midi_count = 0
    note_count = 0
    errors = []

    with ZipFile(ZIP_PATH) as archive, \
         OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "song_id",
                "file_name",
                "pitch_midi",
                "pitch_name",
                "onset_s",
                "duration_s",
            ],
        )
        writer.writeheader()

        midi_members = [
            name for name in archive.namelist()
            if (
                "midiFile" in PurePosixPath(name).parts
                and name.lower().endswith((".mid", ".midi"))
            )
        ]

        for index, member in enumerate(midi_members, start=1):
            print(f"[{index}/{len(midi_members)}] Reading {member}")
            temp_path = None

            try:
                midi_bytes = archive.read(member)

                # pretty_midi loads from a filesystem path, so use one
                # temporary MIDI file at a time.
                with tempfile.NamedTemporaryFile(
                    suffix=".mid", delete=False
                ) as temp_file:
                    temp_file.write(midi_bytes)
                    temp_path = Path(temp_file.name)

                midi = pretty_midi.PrettyMIDI(str(temp_path))
                file_name = PurePosixPath(member).name
                song_id = Path(file_name).stem

                for instrument in midi.instruments:
                    for note in instrument.notes:
                        writer.writerow({
                            "song_id": song_id,
                            "file_name": file_name,
                            "pitch_midi": note.pitch,
                            "pitch_name": pretty_midi.note_number_to_name(
                                note.pitch
                            ),
                            "onset_s": f"{note.start:.6f}",
                            "duration_s": f"{note.end - note.start:.6f}",
                        })
                        note_count += 1

                midi_count += 1

            except Exception as exc:
                errors.append((member, str(exc)))
                print(f"  Skipped: {exc}")

            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

    print(f"\nProcessed {midi_count} MIDI files; wrote {note_count} notes.")
    print(f"CSV saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()