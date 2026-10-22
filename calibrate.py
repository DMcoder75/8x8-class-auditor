"""Study the sample dataset: which signals actually separate the quality tiers?

Expected layout (these folders are git-ignored):

    samples/
      excellent/*.mp4  (+ matching .vtt files, optional)
      good/
      average/
      weak/

Usage:
    python calibrate.py samples                  # audio + video signals only (fast)
    python calibrate.py samples --transcribe base   # also transcribe locally (slow)

Writes signals.csv and prints the mean of every signal per tier. Use the table
to keep the signals that separate tiers, drop the ones that don't, and update
THRESHOLDS in signals.py and the rubric in assess.py.
"""
import argparse
import glob
import os

import pandas as pd

from signals import analyze

ORDER = ["weak", "average", "good", "excellent"]


def flatten(label, path, result):
    row = {"label": label, "file": os.path.basename(path), "duration_s": result["duration_s"]}
    for section in ("audio", "video", "text"):
        for k, v in result[section].items():
            if isinstance(v, (int, float, bool)):
                row[f"{section}.{k}"] = float(v)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--transcribe", default=None, help="Whisper size (tiny/base/small)")
    args = ap.parse_args()

    rows = []
    for label in sorted(os.listdir(args.root)):
        folder = os.path.join(args.root, label)
        if not os.path.isdir(folder):
            continue
        for path in sorted(glob.glob(os.path.join(folder, "*.mp4"))):
            vtt = os.path.splitext(path)[0] + ".vtt"
            vtt = vtt if os.path.exists(vtt) else None
            print(f"[{label}] {os.path.basename(path)}")
            result = analyze(path, vtt, None if vtt else args.transcribe)
            rows.append(flatten(label, path, result))

    df = pd.DataFrame(rows)
    df.to_csv("signals.csv", index=False)
    cols = [c for c in df.columns if c not in ("label", "file")]
    summary = df.groupby("label")[cols].mean().T
    summary = summary[[c for c in ORDER if c in summary.columns]
                      + [c for c in summary.columns if c not in ORDER]]
    pd.set_option("display.width", 200)
    print("\nMean of each signal per tier:\n")
    print(summary.round(3).to_string())
    print("\nPer-file values saved to signals.csv")


if __name__ == "__main__":
    main()
