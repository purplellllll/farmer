"""Select and download a compact, reproducible CC0 gold-replay corpus.

The public Kaggle index points at daily datasets whose episode manifests are
already sorted by average agent rating.  This utility selects the top
``N`` episode(s) per day, downloads only those JSON replays, and stores each
one as a small ZIP plus a provenance manifest.  It never downloads an entire
daily 20 GiB dataset.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any
import zipfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kaggle_env() -> dict[str, str]:
    """Avoid the workspace Python home leaking into the system Kaggle CLI."""

    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment.pop("UV_INTERNAL__PYTHONHOME", None)
    return environment


def _top_rows(index_path: Path, manifest_dir: Path, top_per_day: int) -> list[dict[str, Any]]:
    with index_path.open(newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))
    all_rows: list[dict[str, Any]] = []
    for index_row in index_rows:
        date = str(index_row["date"])
        slug = str(index_row["daily_dataset_slug"])
        manifest_path = manifest_dir / f"{slug}.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing daily episode manifest: {manifest_path}")
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row.update({"date": date, "dataset_slug": slug})
                for key in ("avg_score", "min_score", "sum_score"):
                    row[key] = float(row[key])
                row["episode_id"] = int(row["episode_id"])
                row["size_bytes"] = int(row["size_bytes"])
                all_rows.append(row)
    selected: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        grouped.setdefault(str(row["date"]), []).append(row)
    for date, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (-row["avg_score"], -row["min_score"], row["episode_id"]))
        for rank, row in enumerate(rows[:top_per_day], start=1):
            row = dict(row)
            row["daily_rank"] = rank
            selected.append(row)
    return selected


def _download_one(row: dict[str, Any], incoming_dir: Path, output_dir: Path) -> dict[str, Any]:
    slug = str(row["dataset_slug"])
    episode_id = int(row["episode_id"])
    raw_dir = incoming_dir / slug
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{episode_id}.json"
    if not raw_path.is_file():
        subprocess.run(
            [
                "kaggle", "datasets", "download", "-d", f"kaggle/{slug}",
                "-f", f"{episode_id}.json", "-p", str(raw_dir), "--unzip", "--force", "--quiet",
            ],
            check=True,
            env=_kaggle_env(),
        )
    if not raw_path.is_file():
        raise FileNotFoundError(f"Kaggle download did not produce {raw_path}")
    archive_name = f"{row['date']}-episode-{episode_id}.json.zip"
    archive_path = output_dir / archive_name
    if not archive_path.is_file():
        temporary = archive_path.with_suffix(archive_path.suffix + ".partial")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.write(raw_path, arcname=raw_path.name)
        temporary.replace(archive_path)
    result = dict(row)
    result.update(
        {
            "source_url": f"https://www.kaggle.com/datasets/kaggle/{slug}",
            "source_file": f"{episode_id}.json",
            "source_sha256": _sha256(raw_path),
            "archive": archive_name,
            "archive_sha256": _sha256(archive_path),
            "archive_bytes": archive_path.stat().st_size,
        }
    )
    return result


def download(index_path: Path, manifest_dir: Path, incoming_dir: Path, output_dir: Path, top_per_day: int, workers: int) -> dict[str, Any]:
    if top_per_day < 1:
        raise ValueError("top_per_day must be positive")
    rows = _top_rows(index_path, manifest_dir, top_per_day)
    output_dir.mkdir(parents=True, exist_ok=True)
    incoming_dir.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_download_one, row, incoming_dir, output_dir) for row in rows]
        for future in as_completed(futures):
            completed.append(future.result())
    completed.sort(key=lambda row: (row["date"], row["daily_rank"]))
    report = {
        "schema_version": "farmer-gold-replay-corpus/v1",
        "license": "CC0-1.0",
        "selection": {
            "source_index": "https://www.kaggle.com/datasets/kaggle/kaggriculture-episodes-index",
            "source_index_sha256": _sha256(index_path),
            "ranking_field": "avg_score",
            "tie_break": ["min_score descending", "episode_id ascending"],
            "top_per_day": top_per_day,
            "daily_dataset_count": len({row["dataset_slug"] for row in completed}),
            "episode_count": len(completed),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "episodes": completed,
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--incoming-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-per-day", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = download(
        args.index.resolve(), args.manifest_dir.resolve(), args.incoming_dir.resolve(),
        args.output_dir.resolve(), args.top_per_day, args.workers,
    )
    print(json.dumps({"ok": True, "episodes": len(report["episodes"]), "output": str(args.output_dir.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
