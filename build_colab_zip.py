"""Build SoccerProject_Colab_Source.zip from the live local project.

Requires a clean Git checkout so every release identifies a committed source
version. Never changes Git state or writes into original source files.

Usage:
    python build_colab_zip.py D:\\SoccerProject\\SoccerVideoAnalysisDemo
"""
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

PREP_DIR = Path(__file__).resolve().parent
ZIP_PATH = PREP_DIR / "SoccerProject_Colab_Source.zip"
ROOT_IN_ZIP = "SoccerVideoAnalysisDemo"

# Allowlist: only these are packaged. Everything else (.venv, .git, .idea,
# stubs, sports_db, output_videos, CSV_data, JSON_data, temp, Video_Input,
# __pycache__, yolo11l.pt, docs/build ...) is excluded by construction.
PACKAGE_DIRS = [
    "analytics", "camera_motion", "database", "exporters", "homography",
    "player_ball_assigner", "team_assigner", "trackers", "utils",
    "visualization",
]
ROOT_FILES = ["main.py", "streamlit_app.py", "yolo_inference.py", "README.md", ".gitignore"]
EXTRA_GLOBS = ["docs/analytics_validation/*.py", "docs/analytics_validation/*.md",
               "docs/releases/**/*.md", "docs/releases/**/*.json",
               "docs/validation_contracts.md"]
MODEL_FILES = ["models/best (2).pt"]
PREP_FILES = {name: PREP_DIR / name for name in (
    "requirements_colab.txt", "build_colab_zip.py", "deploy_colab.py",
    "drive_identity.py", "make_notebook.py", "run_local_checks.py",
    "deployment_config.json", "Soccer_Analytics_Colab.ipynb",
)}
EMPTY_DIRS = ["stubs", "JSON_data", "CSV_data", "output_videos", "assets",
              "output_heatmaps", "sports_db", "Video_Input"]

FORBIDDEN_NAME = re.compile(
    r"(^|/)(\.env(\..*)?|.*\.(db|sqlite3?|pkl|pem|key|p12|pfx)|credentials.*\.json|"
    r"client_secret.*\.json|.*token.*)$",
    re.IGNORECASE,
)
SECRET_CONTENT = re.compile(
    r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|"
    r"AIza[0-9A-Za-z_-]{30,}|sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|(?i:password|pwd)\s*=\s*['\"][^'\"]+['\"])"
)
MAX_ZIP_MB = 200


def git(project, *args):
    result = subprocess.run(["git", "-C", str(project), *args],
                            capture_output=True, text=True, check=True,
                            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    return result.stdout


def collect(project):
    files = {}
    for folder in PACKAGE_DIRS:
        for path in sorted((project / folder).rglob("*.py")):
            if "__pycache__" not in path.parts:
                files[path.relative_to(project).as_posix()] = path
    for name in ROOT_FILES + MODEL_FILES:
        path = project / name
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")
        files[name] = path
    for pattern in EXTRA_GLOBS:
        for path in sorted(project.glob(pattern)):
            files[path.relative_to(project).as_posix()] = path
    return files


def check_safe(files):
    problems = []
    for rel, path in files.items():
        if FORBIDDEN_NAME.search(rel):
            problems.append(f"forbidden file name: {rel}")
        if path.suffix in {".py", ".md", ".txt", ".json", ".cfg", ".ini", ".toml", ".yaml", ".yml"}:
            match = SECRET_CONTENT.search(path.read_text(encoding="utf-8", errors="ignore"))
            if match:
                problems.append(f"possible secret in {rel}: {match.group(0)[:12]}...")
    if problems:
        raise SystemExit("Refusing to build ZIP:\n  " + "\n  ".join(problems))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_all(project):
    """One allowlist shared by packaging and validation/deployment fingerprints."""
    project_files = collect(Path(project))
    all_files = dict(project_files)
    all_files.update(PREP_FILES)
    for folder in ("colab_support", "tests"):
        for path in sorted((PREP_DIR / folder).rglob("*.py")):
            if "__pycache__" not in path.parts:
                all_files[path.relative_to(PREP_DIR).as_posix()] = path
    all_files["COLAB_SETUP.md"] = PREP_DIR / "COLAB_SETUP.md"
    required_support = {"colab_support/" + name for name in (
        "colab_helpers.py", "chunked_pipeline.py", "field_filter.py", "identity_evidence.py",
        "pipeline_tracker.py", "short_validation.py", "tracking_exports.py", "colab_session.py",
        "drawing_colors.py")}
    missing = required_support - set(all_files)
    if missing:
        raise SystemExit(f"Required support modules missing: {sorted(missing)}")
    return project_files, all_files


def main(project):
    project = Path(project).resolve()
    project_files, all_files = collect_all(project)

    status = git(project, "status", "--porcelain=v1", "--untracked-files=all")
    if status.strip():
        raise SystemExit("Commit validated source changes before building a release; the Git checkout must be clean.")
    untracked_py = [line[3:] for line in status.splitlines()
                    if line.startswith("??") and line.endswith(".py")]
    modified = [line[3:] for line in status.splitlines() if line[:2].strip() in {"M", "MM", "AM", "A"}]
    missing_untracked = [p for p in untracked_py if p not in project_files]
    if missing_untracked:
        raise SystemExit(f"Untracked Python files not covered by the allowlist: {missing_untracked}")

    check_safe(all_files)

    manifest = {
        "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_project": str(project),
        "git_head": git(project, "rev-parse", "HEAD").strip(),
        "git_head_subject": git(project, "log", "-1", "--format=%s").strip(),
        "uncommitted_modified_files": modified,
        "untracked_python_files_included": untracked_py,
        "note": "Clean Git snapshot. The release commit identifies the complete application, Colab support, tests and authoring tools.",
        "files": {
            rel: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for rel, path in sorted(all_files.items())
        },
    }

    temporary_zip = ZIP_PATH.with_suffix(".building.zip")
    with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel, path in sorted(all_files.items()):
            zf.write(path, f"{ROOT_IN_ZIP}/{rel}")
        for folder in EMPTY_DIRS:
            zf.writestr(f"{ROOT_IN_ZIP}/{folder}/.keep", "")
        zf.writestr(f"{ROOT_IN_ZIP}/COLAB_MANIFEST.json", json.dumps(manifest, indent=2))

    size_mb = temporary_zip.stat().st_size / 1024 ** 2
    if size_mb > MAX_ZIP_MB:
        raise SystemExit(f"ZIP is {size_mb:.1f} MB; expected under {MAX_ZIP_MB} MB")

    with zipfile.ZipFile(temporary_zip) as zf:
        if zf.testzip() is not None:
            raise SystemExit("ZIP CRC verification failed")
        for relative, meta in manifest["files"].items():
            actual = hashlib.sha256(zf.read(f"{ROOT_IN_ZIP}/{relative}")).hexdigest()
            if actual != meta["sha256"]:
                raise SystemExit(f"ZIP hash mismatch: {relative}")
    if ZIP_PATH.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = ZIP_PATH.with_name(f"{ZIP_PATH.stem}.backup_{stamp}.zip")
        import shutil
        shutil.copy2(ZIP_PATH, backup)
        print(f"Preserved existing ZIP: {backup}")
    temporary_zip.replace(ZIP_PATH)

    print(f"Wrote {ZIP_PATH} ({size_mb:.1f} MB, {len(all_files)} files)")
    print(f"git HEAD: {manifest['git_head'][:7]} {manifest['git_head_subject']}")
    print("Uncommitted modified files included:", *modified, sep="\n  ")
    print("Untracked Python files included:", *untracked_py, sep="\n  ")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"D:\SoccerProject\SoccerVideoAnalysisDemo")
