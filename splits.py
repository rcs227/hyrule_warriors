"""Train / validation / test split for HumTrans.

Uses the official split shipped with the dataset
(raw_data/HumTrans/train_valid_test_keys.json). It is split by song: no song
appears in more than one split, but every singer appears in all three. This
is the split the HumTrans paper reports results on.

Clips whose labels align poorly with the humming (see build_labels.py) can be
dropped with min_align_score.

Usage:
    python splits.py                     # print clip counts per split
    python splits.py --min-align-score 0.6
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SPLIT_JSON = PROJECT_DIR / "raw_data" / "HumTrans" / "train_valid_test_keys.json"
ALIGNMENT_CSV = PROJECT_DIR / "processed_data" / "HumTrans" / "alignment.csv"

SPLIT_NAMES = {"TRAIN": "train", "VALID": "valid", "TEST": "test"}


def load_split_keys(split_json=SPLIT_JSON):
    """Return {"train": [clip stems], "valid": [...], "test": [...]}."""
    with open(split_json, encoding="utf-8") as f:
        raw = json.load(f)
    return {SPLIT_NAMES[key]: stems for key, stems in raw.items()}


def load_align_scores(alignment_csv=ALIGNMENT_CSV):
    """Return {clip stem: align_score} from build_labels.py's summary."""
    with open(alignment_csv, newline="", encoding="utf-8") as f:
        return {row["clip"]: float(row["align_score"]) for row in csv.DictReader(f)}


def get_split(name, min_align_score=None, alignment_csv=ALIGNMENT_CSV):
    """Clip stems in one split, keeping only labelled clips if a score is set.

    With min_align_score set, clips that have not been labelled yet (missing
    from alignment.csv) are also dropped.
    """
    stems = load_split_keys()[name]
    if min_align_score is None:
        return stems
    scores = load_align_scores(alignment_csv)
    return [s for s in stems if scores.get(s, -1) >= min_align_score]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--min-align-score", type=float)
    args = parser.parse_args()

    for name in SPLIT_NAMES.values():
        stems = get_split(name, args.min_align_score)
        singers = Counter(s.split("_")[0] for s in stems)
        print(f"{name:5s} {len(stems):6d} clips  "
              + " ".join(f"{k}:{v}" for k, v in sorted(singers.items())))


if __name__ == "__main__":
    main()
