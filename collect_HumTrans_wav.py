"""Download HumTrans hums and save their CQT features.

Streams each WAV from the HumTrans ZIP on Hugging Face (nothing else is
downloaded), computes the CQT defined in features.py plus a pYIN pitch track
(used only to align the labels), and saves both to
processed_data/HumTrans/features/<clip>.npz.

Clips are picked by split so a subset is still a fair sample, e.g. the whole
validation and test sets plus a random part of the training set. Re-running
skips clips that are already saved, so an interrupted run can be resumed.

Usage:
    python collect_HumTrans_wav.py --splits valid test train --train-limit 6000
    python collect_HumTrans_wav.py --splits train            # every train clip
"""
import argparse
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

from remotezip import RemoteZip

from features import (compute_cqt, compute_pitch_track, features_to_npz_bytes,
                      load_audio)
from splits import load_split_keys


OUTPUT_DIR = Path(__file__).resolve().parent / "processed_data" / "HumTrans" / "features"
WAV_ZIP_URL = (
    "https://huggingface.co/datasets/dadinghh2/HumTrans/"
    "resolve/main/all_wav.zip"
)

_archive = None   # one open RemoteZip per worker process


def _open_archive():
    global _archive
    _archive = RemoteZip(WAV_ZIP_URL, timeout=60)


def process_clip(member):
    """Download one WAV and return (clip name, .npz bytes)."""
    y = load_audio(_archive.read(member))
    cqt = compute_cqt(y)
    pitch_track = compute_pitch_track(y, cqt.shape[1])
    return (PurePosixPath(member).stem,
            features_to_npz_bytes(cqt, member, pitch_track))


def choose_clips(splits, train_limit, seed):
    keys = load_split_keys()
    clips = []
    for split in splits:
        stems = keys[split]
        if split == "train" and train_limit:
            stems = random.Random(seed).sample(stems, min(train_limit, len(stems)))
        clips.extend(stems)
    return clips


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--splits", nargs="+", default=["valid", "test", "train"],
                        choices=["train", "valid", "test"])
    parser.add_argument("--train-limit", type=int, default=6000,
                        help="random train clips to keep (0 = all 13080)")
    parser.add_argument("--max-gb", type=float, default=5.0,
                        help="stop once the features folder reaches this size")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(args.max_gb * 1e9)
    used_bytes = sum(p.stat().st_size for p in args.output_dir.glob("*.npz"))

    with RemoteZip(WAV_ZIP_URL, timeout=60) as archive:
        members = {PurePosixPath(n).stem: n for n in archive.namelist()
                   if n.lower().endswith(".wav")}

    wanted = choose_clips(args.splits, args.train_limit, args.seed)
    todo = [members[s] for s in wanted
            if s in members and not (args.output_dir / f"{s}.npz").exists()]
    print(f"{len(wanted)} clips chosen, {len(wanted) - len(todo)} already saved, "
          f"{len(todo)} to process ({used_bytes / 1e9:.2f} / {args.max_gb} GB used)")

    saved, errors = 0, []
    with ProcessPoolExecutor(args.workers, initializer=_open_archive) as pool:
        futures = {pool.submit(process_clip, m): m for m in todo}
        for index, future in enumerate(as_completed(futures), start=1):
            member = futures[future]
            try:
                stem, npz_bytes = future.result()
            except Exception as exc:
                errors.append((member, str(exc)))
                print(f"[{index}/{len(todo)}] Skipped {member}: {exc}")
                continue

            if used_bytes + len(npz_bytes) > max_bytes:
                print("Storage limit reached; stopping.")
                for f in futures:
                    f.cancel()
                break

            (args.output_dir / f"{stem}.npz").write_bytes(npz_bytes)
            used_bytes += len(npz_bytes)
            saved += 1
            if index % 100 == 0 or index == len(todo):
                print(f"[{index}/{len(todo)}] {used_bytes / 1e9:.2f} GB used")

    print(f"\nSaved {saved} clips to {args.output_dir}")
    if errors:
        print(f"{len(errors)} clip(s) failed; re-run to retry them.")


if __name__ == "__main__":
    main()
