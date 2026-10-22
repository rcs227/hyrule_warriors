"""Score a trained model on the HumTrans test split, next to the baseline.

Transcribes every labelled test clip and scores it exactly like
baseline_pyin.py (evaluation.py), then prints both side by side.

Usage:
    python evaluate_model.py --checkpoint checkpoints/model_best.pt
"""
import argparse
import csv
import json
from pathlib import Path

from build_labels import LABELS_DIR, load_notes
from evaluation import METRICS, MIN_ALIGN_SCORE, summarize
from splits import get_split
from train import evaluate_stems, pick_device
from transcribe import load_model


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT_DIR / "checkpoints" / "model_best.pt")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--name", default=None, help="results file name")
    args = parser.parse_args()
    name = args.name or args.checkpoint.stem.replace("_best", "")

    device = pick_device()
    model = load_model(args.checkpoint, device)
    stems = [s for s in get_split(args.split) if (LABELS_DIR / f"{s}.npz").exists()]
    rows = evaluate_stems(model, stems, device, load_notes())

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{name}_{args.split}.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (RESULTS_DIR / f"{name}_{args.split}.json").write_text(json.dumps(summary, indent=2))

    baseline_path = RESULTS_DIR / f"baseline_pyin_{args.split}.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    print(f"{args.split}: {summary['n_clips']} clips "
          f"({summary['n_good']} with align_score >= {MIN_ALIGN_SCORE})\n")
    print(f"{'metric':12s} {'model':>8s} {'baseline':>9s}   (well-aligned clips only)")
    for m in METRICS:
        b = f"{baseline['all'][m]:9.3f}" if baseline else "        -"
        bg = f"{baseline['good'][m]:.3f}" if baseline else "-"
        print(f"{m:12s} {summary['all'][m]:8.3f} {b}   "
              f"({summary['good'][m]:.3f} vs {bg})")


if __name__ == "__main__":
    main()
