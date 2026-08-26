"""Build a CPU-safe, vendored Kaggriculture eight-model submission archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import tarfile
import tempfile
import zipfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _force_cpu(bundle: object) -> None:
    for item in bundle.base_models:
        estimator = item.adapter.estimator
        if item.name == "xgboost":
            estimator.set_params(device="cpu", n_jobs=1)
            estimator.get_booster().set_param({"device": "cpu", "nthread": 1})
        elif item.name in {"lightgbm", "extra_trees"}:
            try:
                estimator.set_params(n_jobs=1)
            except (TypeError, ValueError):
                pass
        if item.name in {"ft_transformer", "realmlp"}:
            estimator.device = "cpu"
            interface = getattr(estimator, "alg_interface_", None)
            if hasattr(interface, "device"):
                interface.device = "cpu"
            for sub_interface in getattr(interface, "sub_split_interfaces", []):
                if hasattr(sub_interface, "device"):
                    sub_interface.device = "cpu"
                model = getattr(sub_interface, "model", None)
                if model is not None and hasattr(model, "device"):
                    model.device = "cpu"


def _copy_vendor_wheel(wheel: Path, target: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="farmer-wheel-") as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(extracted)
        for source in extracted.iterdir():
            destination = target / source.name
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)


def build(args: argparse.Namespace) -> dict[str, object]:
    from farmer_ensemble.ensemble import load_bundle

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    model_path = output_dir / "ensemble_bundle.pkl"
    if args.preserve_bundle:
        shutil.copy2(args.bundle.resolve(), model_path)
        model_names = [
            "lightgbm",
            "xgboost",
            "catboost",
            "hgbc",
            "extra_trees",
            "logistic_regression",
            "ft_transformer",
            "realmlp",
        ]
        classes = [0, 5, 6, 7]
    else:
        bundle = load_bundle(args.bundle.resolve())
        _force_cpu(bundle)
        with model_path.open("wb") as handle:
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        model_names = [item.name for item in bundle.base_models]
        classes = [int(value) for value in bundle.classes_]

    shutil.copy2(args.main.resolve(), output_dir / "main.py")
    shutil.copytree(
        args.package.resolve(),
        output_dir / "farmer_ensemble",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for wheel in args.vendor_wheel:
        _copy_vendor_wheel(wheel.resolve(), output_dir)

    manifest = {
        "kind": "kaggriculture-eight-model-router",
        "models": model_names,
        "model_count": len(model_names),
        "vendor_wheels": [wheel.name for wheel in args.vendor_wheel],
        "classes": classes,
        "cpu_forced": not args.preserve_bundle,
        "runtime_cpu_patch": True,
        "model_sha256": _sha256(model_path),
        "model_bytes": model_path.stat().st_size,
    }
    (output_dir / "deployment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    archive_path = args.archive.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            archive.add(path, arcname=path.relative_to(output_dir).as_posix())
    manifest["archive"] = str(archive_path)
    manifest["archive_bytes"] = archive_path.stat().st_size
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--vendor-wheel", type=Path, action="append", default=[])
    parser.add_argument("--preserve-bundle", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
