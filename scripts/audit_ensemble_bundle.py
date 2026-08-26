"""Report serialized size contribution of every model without writing copies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from farmer_ensemble.ensemble import load_bundle


class CountingWriter:
    def __init__(self) -> None:
        self.bytes = 0

    def write(self, value: bytes) -> int:
        size = memoryview(value).nbytes
        self.bytes += size
        return size


def serialized_bytes(value) -> int:
    writer = CountingWriter()
    pickle.Pickler(writer, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    return writer.bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = load_bundle(args.bundle)
    models = {
        item.name: serialized_bytes(item)
        for item in bundle.base_models
    }
    summary = {
        "bundle_bytes_on_disk": args.bundle.stat().st_size,
        "bundle_mib_on_disk": args.bundle.stat().st_size / 2**20,
        "model_serialized_bytes": models,
        "model_serialized_mib": {name: size / 2**20 for name, size in models.items()},
        "stacker_serialized_bytes": serialized_bytes(bundle.stacker),
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
