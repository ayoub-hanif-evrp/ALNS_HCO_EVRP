"""Build EVRP-TW-D used split: 100 easiest RC wide train + 10 test (5 wide / 5 narrow)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from src.schneider.data.schneider_txt_convert import convert_one_txt

ROOT = Path(__file__).resolve().parents[1]
RANK_CSV = ROOT / "dataset/EVRPTWD/analysis/cus100_unanchored_RC_easiest_to_hardest.csv"
SOURCE = ROOT / "dataset/EVRPTWD/unanchored/Cus_100/solomon"
USED = ROOT / "dataset/EVRPTWD/used_dataset_instances"
SPLIT_JSON = ROOT / "dataset/EVRPTWD/splits/cus100_rc_easy_wide100_test10_seed42.json"
SEED = 42
N_TRAIN = 100
N_TEST_WIDE = 5
N_TEST_NARROW = 5


def main() -> None:
    df = pd.read_csv(RANK_CSV)
    wide = df[df["tw"] == "wide"].sort_values("ease_score", ascending=False).reset_index(drop=True)
    narrow = df[df["tw"] == "narrow"].sort_values("ease_score", ascending=False).reset_index(drop=True)
    if len(wide) < N_TRAIN + N_TEST_WIDE:
        raise SystemExit(f"Need {N_TRAIN + N_TEST_WIDE} wide instances, found {len(wide)}")
    if len(narrow) < N_TEST_NARROW:
        raise SystemExit(f"Need {N_TEST_NARROW} narrow instances, found {len(narrow)}")

    train = wide.iloc[:N_TRAIN]
    test_wide = wide.iloc[N_TRAIN : N_TRAIN + N_TEST_WIDE]
    test_narrow = narrow.iloc[:N_TEST_NARROW]
    test = pd.concat([test_wide, test_narrow], ignore_index=True)

    train_names = train["stem"].tolist()
    test_names = test["stem"].tolist()
    assert not (set(train_names) & set(test_names))

    selection = {
        "train": "top 100 easiest RC wide (unanchored Cus_100) by ease_score",
        "test_wide": "next 5 easiest RC wide after train (held-out)",
        "test_narrow": "top 5 easiest RC narrow (generalization probe)",
        "note": "Easy warm-start study: learn on easiest wide, probe wide+narrow test.",
    }
    manifest = {
        "dataset": "evrptwd",
        "family": "RC",
        "scale": 100,
        "difficulty": "easy_wide",
        "selection": selection,
        "seed": SEED,
        "n_train": len(train_names),
        "n_test": len(test_names),
        "n_validation": 0,
        "train_ease_mean": float(train["ease_score"].mean()),
        "test_ease_mean": float(test["ease_score"].mean()),
        "train_tw": {"wide": len(train_names), "narrow": 0},
        "test_tw": {"wide": N_TEST_WIDE, "narrow": N_TEST_NARROW},
        "source_solomon": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "train_instances": train_names,
        "test_instances": test_names,
        "eval_instances": test_names,
    }

    # Fresh used folder: keep only selected converted instances.
    USED.mkdir(parents=True, exist_ok=True)
    keep = set(train_names) | set(test_names)
    for child in list(USED.iterdir()):
        if child.name in {"split_manifest.json", "README.md", "train", "test"}:
            continue
        if child.is_dir() and child.name not in keep:
            shutil.rmtree(child)

    missing_src = []
    for name in train_names + test_names:
        src = SOURCE / f"{name}.txt"
        if not src.is_file():
            missing_src.append(name)
            continue
        convert_one_txt(src, USED, overwrite=False)
    if missing_src:
        raise SystemExit(f"Missing source txt for {len(missing_src)} stems e.g. {missing_src[:3]}")

    (USED / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    SPLIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    train_dir = USED / "train"
    test_dir = USED / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    (train_dir / "instances.txt").write_text("\n".join(train_names) + "\n", encoding="utf-8")
    (test_dir / "instances.txt").write_text("\n".join(test_names) + "\n", encoding="utf-8")
    (USED / "README.md").write_text(
        "# Used dataset instances (easy wide)\n\n"
        "RC Cus_100 from EVRP-TW-D **unanchored**, selected for an **easy warm-start** study.\n\n"
        "- **100 train / 10 test** (no validation)\n"
        "- Train: **100 easiest RC wide** by ease score\n"
        "- Test: **5 wide** (next-easiest held-out) + **5 narrow** (easiest narrow)\n"
        "- Seed **42**\n\n"
        "See `split_manifest.json` for the exact lists.\n",
        encoding="utf-8",
    )
    print(
        f"OK | train={len(train_names)} test={len(test_names)} "
        f"| ease_train={manifest['train_ease_mean']:.3f} "
        f"ease_test={manifest['test_ease_mean']:.3f} "
        f"| -> {USED}"
    )


if __name__ == "__main__":
    main()
