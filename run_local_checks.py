"""Offline tests with explicit original-source imports; never loads YOLO weights."""
import ast
import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import shutil
import time
import unittest


def main():
    stage = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default=str(stage))
    parser.add_argument("--checks-only", action="store_true", help="Validate without building or deploying")
    args = parser.parse_args()
    source = Path(args.source)
    config_path = stage / "deployment_config.json"
    if not args.checks_only and config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("auto_deploy_after_validation", False):
            from deploy_colab import release
            return release(source)
    if not (source / "trackers" / "tracker.py").is_file():
        raise FileNotFoundError("Pass the project source directory")
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(stage / "colab_support"), str(source)]
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    output = stage / "validation_local"
    output.mkdir(exist_ok=True)
    existing = [output / name for name in ("test_output.txt", "test_results.json") if (output / name).exists()]
    if existing:
        history = output / "history" / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        history.mkdir(parents=True, exist_ok=False)
        for path in existing:
            shutil.copy2(path, history / path.name)
    started = time.perf_counter()
    paths = [p for p in source.rglob("*.py") if ".venv" not in p.parts and ".git" not in p.parts]
    initial = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    python_files = sorted(stage.glob("*.py")) + sorted((stage / "colab_support").glob("*.py")) + sorted((stage / "tests").glob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    notebook = json.loads((stage / "Soccer_Analytics_Colab.ipynb").read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for index, cell in enumerate(cells):
        compile("".join(cell["source"]), f"notebook_step_{index + 1}", "exec")
    with (output / "test_output.txt").open("w", encoding="utf-8") as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        suite = unittest.defaultTestLoader.discover(str(stage / "tests"))
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    changed = [name for name, digest in initial.items()
               if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest]
    report = {"tests_run": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "skipped": len(result.skipped),
              "successful": result.wasSuccessful(), "syntax_python_files": len(python_files),
              "syntax_notebook_cells": len(cells), "original_python_files_verified_unchanged": len(initial),
              "original_files_changed": changed, "elapsed_s": round(time.perf_counter() - started, 2),
              "local_yolo_inference_run": False, "colab_gpu_validation_run": False,
              "real_ocr_accuracy_validated": False,
              "note": "Synthetic detection/OCR; real OpenCV, ByteTrack, original drawing and streaming exports."}
    (output / "test_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() and not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
