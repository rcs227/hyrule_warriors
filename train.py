"""Train the hum transcription model.

Trains on random 3 s windows of the training clips and checks the validation
clips after every epoch, keeping the checkpoint with the best validation
note F1 (COnP). Training stops early when validation stops improving.

Usage:
    python train.py                          # HumTrans train split
    python train.py --extra mir_qbsh synthetic
    python train.py --overfit 50             # sanity check: memorize 50 clips
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from build_labels import FEATURES_DIR, LABELS_DIR, load_notes
from dataset import HumDataset
from decoding import decode
from evaluation import METRICS, evaluate_clip, reference_notes, summarize
from model import HumTranscriber, compute_loss
from splits import get_split


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"
EXTRA_SOURCES = {
    # name: (features dir, labels dir)
    "mir_qbsh": (PROJECT_DIR / "processed_data" / "MIR-QBSH" / "features",
                 PROJECT_DIR / "processed_data" / "MIR-QBSH" / "labels"),
    "synthetic": (PROJECT_DIR / "processed_data" / "Synthetic" / "features",
                  PROJECT_DIR / "processed_data" / "Synthetic" / "labels"),
}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extra_clips(names):
    pairs = []
    for name in names:
        features_dir, labels_dir = EXTRA_SOURCES[name]
        stems = sorted(p.stem for p in labels_dir.glob("*.npz"))
        pairs += HumDataset.from_dirs(features_dir, labels_dir, stems).clips
        print(f"  + {name}: {len(stems)} clips")
    return pairs


@torch.no_grad()
def predict(model, features, device):
    """Whole-clip probabilities: pitch (frames, classes), onset (frames,)."""
    model.eval()
    x = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    pitch_logits, onset_logits = model(x)
    return (torch.softmax(pitch_logits[0], -1).cpu().numpy(),
            torch.sigmoid(onset_logits[0]).cpu().numpy())


def evaluate_stems(model, stems, device, all_notes,
                   features_dir=FEATURES_DIR, labels_dir=LABELS_DIR):
    """Transcribe whole clips and score them like the baseline."""
    dataset = HumDataset.from_dirs(features_dir, labels_dir, stems, window=None)
    rows = []
    for index, (features_path, labels_path) in enumerate(dataset.clips):
        stem = features_path.stem
        cqt = dataset.load(index)[0]
        with np.load(labels_path) as labels:
            ref_frames = labels["pitch"]
            ref_intervals, ref_pitches = reference_notes(
                all_notes[stem], int(labels["octave_shift"]), float(labels["lag_s"]))
            align_score = float(labels["align_score"])
        est_intervals, est_pitches = decode(*predict(model, cqt, device))
        row = {"clip": stem, "align_score": align_score}
        row.update(evaluate_clip(ref_intervals, ref_pitches, ref_frames,
                                 est_intervals, est_pitches))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=300)
    parser.add_argument("--min-align-score", type=float, default=0.6)
    parser.add_argument("--extra", nargs="*", default=[], choices=list(EXTRA_SOURCES))
    parser.add_argument("--patience", type=int, default=8,
                        help="stop after this many epochs without improvement")
    parser.add_argument("--val-clips", type=int, default=200,
                        help="validation clips scored per epoch (0 = all)")
    parser.add_argument("--overfit", type=int, default=0,
                        help="train AND validate on this many train clips")
    parser.add_argument("--init", type=Path, help="start from a checkpoint")
    parser.add_argument("--workers", type=int, default=2,
                        help="data-loading processes (0 = load in the main process)")
    parser.add_argument("--name", default="model")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = pick_device()
    all_notes = load_notes()

    train_stems = get_split("train", args.min_align_score)
    valid_stems = get_split("valid", args.min_align_score)
    if args.overfit:
        # Only a sanity check, so any labelled clips will do.
        pool = train_stems + valid_stems
        train_stems = [s for s in pool if (LABELS_DIR / f"{s}.npz").exists()][:args.overfit]
        valid_stems = train_stems
    elif args.val_clips:
        valid_stems = random.Random(0).sample(valid_stems, min(args.val_clips, len(valid_stems)))

    train_set = HumDataset.from_dirs(FEATURES_DIR, LABELS_DIR, train_stems,
                                     window=args.window, seed=args.seed)
    print(f"Device: {device}. HumTrans train clips: {len(train_set)}")
    if not args.overfit:
        train_set.clips += extra_clips(args.extra)
    print(f"Training clips: {len(train_set)}, validation clips: {len(valid_stems)}")

    # Overfit runs see each clip several times per epoch so they finish fast.
    repeats = max(1, 512 // len(train_set)) if args.overfit else 1
    loader = DataLoader(torch.utils.data.ConcatDataset([train_set] * repeats),
                        batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers,
                        persistent_workers=args.workers > 0)

    model = HumTranscriber().to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
        print(f"Starting from {args.init}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=3)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    best_path = CHECKPOINT_DIR / f"{args.name}_best.pt"
    log_path = RESULTS_DIR / f"{args.name}_training_log.csv"
    log = []
    best, since_best = -1.0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        totals = np.zeros(3)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, pitch_loss, onset_loss = compute_loss(*model(batch["features"]), batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals += [loss.item(), pitch_loss.item(), onset_loss.item()]
        totals /= len(loader)

        summary = summarize(evaluate_stems(model, valid_stems, device, all_notes),
                            min_align_score=0)
        val = summary["all"]
        scheduler.step(val["COnP_f1"])
        entry = {"epoch": epoch, "loss": totals[0], "pitch_loss": totals[1],
                 "onset_loss": totals[2], **{f"val_{m}": val[m] for m in METRICS},
                 "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - start}
        log.append(entry)
        print(f"epoch {epoch:3d}  loss {totals[0]:.3f}  val COnP {val['COnP_f1']:.3f}  "
              f"COn {val['COn_f1']:.3f}  frame_acc {val['frame_acc']:.3f}  "
              f"({entry['seconds']:.0f}s)", flush=True)

        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(entry))
            writer.writeheader()
            writer.writerows(log)

        if val["COnP_f1"] > best:
            best, since_best = val["COnP_f1"], 0
            torch.save(model.state_dict(), best_path)
        else:
            since_best += 1
            if since_best >= args.patience:
                print("No improvement; stopping early.")
                break

    (RESULTS_DIR / f"{args.name}_config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        indent=2))
    print(f"Best validation COnP F1 {best:.3f}; checkpoint {best_path}")


if __name__ == "__main__":
    main()
