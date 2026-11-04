"""Build a stable, single-cell launcher; current UI and analysis live in the ZIP."""
import datetime
import json
from pathlib import Path
import shutil

APPROVED_BINDING = {
    "approved_root_id": "18jlQkG3zSjj-Eogo-uJ4gcuZxX6buBl4",
    "project_folder_id": "1k9LxrKNBogsE_Z-QOqmH6ncecI6nYJ-o",
    "colab_folder_id": "1KC_ybaf_aWRt_4UTt0rc7bbH_-Nty7CC",
    "notebook_id": "1gTBP7t7mw70pOLoTH6XV-RofPR5LA44E",
}
APPROVED_COLAB_PROJECT = "/content/drive/.shortcut-targets-by-id/1k9LxrKNBogsE_Z-QOqmH6ncecI6nYJ-o/Soccer_Analysis_Meena_Kasthuri/SoccerProject"


BOOTSTRAP_HELPERS = r'''
import hashlib, json, shutil, stat, zipfile
from pathlib import Path, PurePosixPath

APPROVED_BINDING = {
    "approved_root_id": "18jlQkG3zSjj-Eogo-uJ4gcuZxX6buBl4",
    "project_folder_id": "1k9LxrKNBogsE_Z-QOqmH6ncecI6nYJ-o",
    "colab_folder_id": "1KC_ybaf_aWRt_4UTt0rc7bbH_-Nty7CC",
    "notebook_id": "1gTBP7t7mw70pOLoTH6XV-RofPR5LA44E",
}
APPROVED_COLAB_PROJECT = "/content/drive/.shortcut-targets-by-id/1k9LxrKNBogsE_Z-QOqmH6ncecI6nYJ-o/Soccer_Analysis_Meena_Kasthuri/SoccerProject"

def validate_release_binding(release):
    binding = release.get("drive_binding")
    if not isinstance(binding, dict) or any(binding.get(key) != value for key, value in APPROVED_BINDING.items()):
        raise RuntimeError("Deployment belongs to an unapproved Drive location or notebook. Deploy to the approved project and rerun; no alternate location will be used.")
    return binding

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def snapshot_release(drive_root, session_root):
    manifest_path = Path(drive_root) / "deployment_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Deployment manifest is not visible in Drive. Desktop/cloud synchronization may still be pending; wait for sync and rerun.")
    try:
        observed = manifest_path.read_bytes()
        release = json.loads(observed)
    except (OSError, ValueError) as exc:
        raise RuntimeError("Deployment manifest cannot be read yet. Cloud synchronization may still be pending; wait for sync and rerun.") from exc
    if not isinstance(release, dict):
        raise RuntimeError("Unsupported deployment manifest; deploy a validated release again.")
    validate_release_binding(release)
    artifact = release.get("zip", {})
    if release.get("schema_version") != 1 or artifact.get("name") != "SoccerProject_Colab_Source.zip":
        raise RuntimeError("Unsupported deployment manifest; deploy a validated release again.")
    source = Path(drive_root) / artifact["name"]
    target = Path(session_root) / artifact["name"]
    if not source.is_file():
        raise RuntimeError("Source ZIP has not reached Drive. Cloud synchronization may still be pending.")
    try:
        shutil.copyfile(source, target)
        matches = (target.stat().st_size == artifact.get("bytes")
                   and file_sha256(target) == artifact.get("sha256")
                   and manifest_path.read_bytes() == observed)
    except OSError as exc:
        raise RuntimeError("Drive files are temporarily unavailable. Cloud synchronization may still be pending; wait for sync and rerun.") from exc
    if not matches:
        raise RuntimeError("Drive release files do not match yet. Cloud synchronization may still be pending; wait for sync, then rerun this cell. No project code was loaded.")
    return release, target

def safe_relative(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Unsafe archive path: " + repr(name))
    parts = name.split("/")
    if name.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise ValueError("Unsafe archive path: " + repr(name))
    return PurePosixPath(name)

def verify_extract(archive_path, destination):
    root_name = "SoccerVideoAnalysisDemo"
    manifest_name = root_name + "/COLAB_MANIFEST.json"
    empty_folders = {"stubs", "JSON_data", "CSV_data", "output_videos", "assets",
                     "output_heatmaps", "sports_db", "Video_Input"}
    destination = Path(destination).resolve()
    project = destination / root_name
    if project.exists():
        raise RuntimeError("Extraction requires a fresh session folder.")
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > 5000 or sum(m.file_size for m in members) > 1024 ** 3:
            raise ValueError("Unexpected archive size")
        seen = set()
        for member in members:
            relative = safe_relative(member.filename)
            if relative.parts[0] != root_name or len(relative.parts) < 2:
                raise ValueError("Unexpected archive root")
            if member.filename.casefold() in seen:
                raise ValueError("Duplicate archive path")
            seen.add(member.filename.casefold())
            mode = (member.external_attr >> 16) & 0xFFFF
            if member.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                raise ValueError("Archive must contain regular files only")
        manifest = json.loads(archive.read(manifest_name))
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("Missing packaged-file manifest")
        required = {"colab_support/colab_session.py", "colab_support/short_validation.py",
                    "colab_support/chunked_pipeline.py", "requirements_colab.txt", "models/best (2).pt"}
        if not required.issubset(files):
            raise ValueError("Incomplete or old source ZIP")
        expected = {manifest_name}
        for relative, metadata in files.items():
            safe_relative(relative)
            name = root_name + "/" + relative
            expected.add(name)
            data = archive.read(name)
            if len(data) != metadata.get("bytes") or hashlib.sha256(data).hexdigest() != metadata.get("sha256"):
                raise ValueError("Packaged-file hash mismatch: " + relative)
        for member in members:
            if member.filename not in expected:
                parts = PurePosixPath(member.filename).parts
                if len(parts) != 3 or parts[1] not in empty_folders or parts[2] != ".keep" or member.file_size:
                    raise ValueError("Unmanifested archive file: " + member.filename)
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    return project, manifest
'''


LAUNCH = r'''
import sys, tempfile, subprocess, importlib.metadata as metadata
from google.colab import drive

old_modules = {"colab_session", "colab_helpers", "chunked_pipeline", "short_validation",
               "pipeline_tracker", "identity_evidence", "field_filter", "trackers",
               "team_assigner", "main"}
if globals().get("_SOCCER_SESSION_STARTED") or old_modules.intersection(sys.modules):
    raise RuntimeError("A Soccer analysis session is already loaded. Use Runtime > Restart session, then run this cell to load the latest synced ZIP. Existing results are preserved.")
drive.mount("/content/drive")
DRIVE_PROJECT = Path(APPROVED_COLAB_PROJECT)
SESSION_ROOT = Path(tempfile.mkdtemp(prefix="soccer_session_", dir="/content"))
release, local_zip = snapshot_release(DRIVE_PROJECT, SESSION_ROOT)
PROJECT_DIR, source_manifest = verify_extract(local_zip, SESSION_ROOT)
print("Verified synced release:", release["release_id"])
print("This is the release currently visible in cloud Drive; Desktop sync may still be pending.")

def installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

binary_packages = ["torch", "torchvision", "numpy", "opencv-python", "opencv-python-headless"]
before = {name: installed_version(name) for name in binary_packages}
if not before["torch"] or not before["torchvision"]:
    raise RuntimeError("Select a standard Colab GPU runtime with torch/torchvision preinstalled.")
constraint_file = SESSION_ROOT / "colab_binary_constraints.txt"
constraint_file.write_text("\n".join(name + "==" + version for name, version in before.items() if version) + "\n")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", str(constraint_file),
                "-r", str(PROJECT_DIR / "requirements_colab.txt"), "easyocr==1.7.2"], check=True)
changed = {name: (before[name], installed_version(name)) for name in binary_packages
           if before[name] != installed_version(name)}
if changed:
    raise RuntimeError("Dependency setup changed binary packages. Use Runtime > Restart session and rerun this cell once. " + str(changed))
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Select Runtime > Change runtime type > GPU, reconnect, then run this cell.")
print("GPU:", torch.cuda.get_device_name(0))
sys.path.insert(0, str(PROJECT_DIR / "colab_support"))
_SOCCER_SESSION_STARTED = True
from colab_session import launch
launch(PROJECT_DIR, DRIVE_PROJECT, SESSION_ROOT, release)
'''


def notebook_data():
    cells = [
        {"cell_type": "markdown", "metadata": {}, "id": "soccer-intro", "source": [
            "# Soccer Video Analysis\n",
            "Select a **GPU runtime**, then **Run all**. Pick a video below and click **Run short analysis**.\n",
            "This permanent notebook loads the approved project through its Drive folder ID and verifies its release binding each new session.\n",
            "Results are saved automatically. Optional field marking enables spectator filtering; unclear jersey numbers stay unknown.\n",
            "Full videos remain disabled until you review the short comparison. Start a new runtime session to load updated code.\n"]},
        {"cell_type": "code", "metadata": {}, "id": "soccer-launch", "source": (BOOTSTRAP_HELPERS + LAUNCH).strip().splitlines(True),
         "outputs": [], "execution_count": None}]
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}


def main():
    config = json.loads(Path(__file__).with_name("deployment_config.json").read_text(encoding="utf-8"))
    if any(config.get(key) != value for key, value in APPROVED_BINDING.items()) or config.get("colab_project") != APPROVED_COLAB_PROJECT:
        raise ValueError("Launcher configuration must match the approved Drive IDs and Colab path")
    destination = Path(__file__).with_name("Soccer_Analytics_Colab.ipynb")
    data = (json.dumps(notebook_data(), indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    if destination.exists() and destination.read_bytes() == data:
        print("Permanent launcher unchanged:", destination)
        return
    if destination.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = destination.with_name(f"{destination.stem}.backup_{stamp}.ipynb")
        shutil.copy2(destination, backup)
        print("Preserved notebook:", backup)
    destination.write_bytes(data)
    print(destination)


if __name__ == "__main__":
    main()
