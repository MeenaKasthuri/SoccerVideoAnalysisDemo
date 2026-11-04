"""Validate, package and deploy to the local Drive Desktop folder, preserving backups.

Cloud synchronization is not observable here. A published hash manifest enables
Colab to refuse partially synced releases. No original-source writes or Git mutations.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import uuid
import zipfile

from drive_identity import verify_local_drive_binding

PREP = Path(__file__).resolve().parent
ZIP_NAME = "SoccerProject_Colab_Source.zip"
NOTEBOOK_NAME = "Soccer_Analytics_Colab.ipynb"
MANIFEST_NAME = "deployment_manifest.json"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def within(root, relative):
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError(f"Path outside deployment directory: {relative}")
    return candidate


def verify_zip(path, expected=None):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC check failed")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members")
        for info in archive.infolist():
            p = PurePosixPath(info.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in info.filename or ":" in info.filename:
                raise ValueError("Unsafe ZIP member")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("ZIP symlinks are not supported")
        root = "SoccerVideoAnalysisDemo/"
        manifest = json.loads(archive.read(root + "COLAB_MANIFEST.json"))
        files = manifest["files"]
        for relative, meta in files.items():
            content = archive.read(root + relative)
            if len(content) != meta["bytes"] or hashlib.sha256(content).hexdigest() != meta["sha256"]:
                raise ValueError(f"ZIP content mismatch: {relative}")
        allowed = {root + name for name in files} | {root + "COLAB_MANIFEST.json"}
        empty = {"stubs", "JSON_data", "CSV_data", "output_videos", "assets",
                 "output_heatmaps", "sports_db", "Video_Input"}
        for name in names:
            if name not in allowed and not (name in {root + folder + "/.keep" for folder in empty}
                                            and archive.read(name) == b""):
                raise ValueError(f"Unmanifested ZIP member: {name}")
        actual = {name: meta["sha256"] for name, meta in files.items()}
        if expected is not None and actual != expected:
            raise ValueError("ZIP does not match the source files that passed validation")
        return manifest


def fingerprint(files):
    return {name: sha256(path) for name, path in sorted(files.items())}


def assert_unchanged(files, expected):
    if fingerprint(files) != expected:
        raise RuntimeError("Sources changed during validation/build; rerun deployment")


def notebook_content(notebook):
    """Compare authored cells, ignoring execution outputs and Colab metadata."""
    return [(cell["cell_type"], "".join(cell["source"])) for cell in notebook["cells"]]


def publish(prep, destination, checks, expected_package=None, expected_notebook=None, config=None):
    """Publish only validated artifacts. Tests inject temp folders; never touch results."""
    prep, destination = Path(prep).resolve(), Path(destination).absolute()
    config = config if config is not None else json.loads((PREP / "deployment_config.json").read_text(encoding="utf-8"))
    binding = verify_local_drive_binding(config, destination)
    if not destination.is_dir():
        raise FileNotFoundError(f"Drive Desktop folder unavailable: {destination}")
    if not checks.get("successful") or checks.get("errors") or checks.get("failures"):
        raise RuntimeError("Failed validation cannot be deployed")
    source_zip = within(prep, ZIP_NAME)
    source_notebook = within(prep, NOTEBOOK_NAME)
    zip_hash = sha256(source_zip)
    manifest = verify_zip(source_zip, expected_package)
    notebook_bytes = source_notebook.read_bytes()
    notebook_hash = hashlib.sha256(notebook_bytes).hexdigest()
    if expected_notebook is not None and notebook_hash != expected_notebook:
        raise RuntimeError("Notebook changed after validation")
    notebook = json.loads(notebook_bytes)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), NOTEBOOK_NAME, "exec")
    release_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    backup = within(destination, "deployment_backups/" + release_id)
    backup.mkdir(parents=True, exist_ok=False)
    preserved = {}
    for name in (ZIP_NAME, NOTEBOOK_NAME, MANIFEST_NAME):
        target = within(destination, name)
        if target.exists():
            saved = within(backup, name)
            shutil.copy2(target, saved)
            if sha256(target) != sha256(saved):
                raise RuntimeError(f"Backup verification failed: {name}")
            preserved[name] = sha256(saved)
    temp_zip = within(destination, f".{ZIP_NAME}.{release_id}.uploading")
    shutil.copy2(source_zip, temp_zip)
    if sha256(temp_zip) != zip_hash:
        raise RuntimeError("Copied ZIP hash mismatch; previous release remains backed up")
    verify_zip(temp_zip, expected_package or {name: meta["sha256"] for name, meta in manifest["files"].items()})
    os.replace(temp_zip, within(destination, ZIP_NAME))
    target_notebook = within(destination, NOTEBOOK_NAME)
    saved_notebook_bytes = target_notebook.read_bytes() if target_notebook.exists() else None
    notebook_updated = saved_notebook_bytes != notebook_bytes
    if notebook_updated and saved_notebook_bytes is not None:
        try:
            saved_notebook = json.loads(saved_notebook_bytes)
            notebook_updated = notebook_content(saved_notebook) != notebook_content(notebook)
        except (ValueError, KeyError, TypeError):
            pass
    deployed_notebook_hash = notebook_hash if notebook_updated else hashlib.sha256(saved_notebook_bytes).hexdigest()
    if notebook_updated:
        # In-place content update keeps the permanent file instead of renaming it.
        with target_notebook.open("wb") as dst:
            dst.write(notebook_bytes)
            dst.flush()
            os.fsync(dst.fileno())
    if sha256(within(destination, ZIP_NAME)) != zip_hash or sha256(target_notebook) != deployed_notebook_hash:
        raise RuntimeError("Local deployment hash check failed; previous files remain in backup")
    release = {"schema_version": 1, "release_id": release_id,
               "drive_binding": {key: binding[key] for key in ("approved_root_id", "project_folder_id", "colab_folder_id", "notebook_id")},
               "built_at": manifest["built_at"],
               "deployed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "zip": {"name": ZIP_NAME, "sha256": zip_hash, "bytes": within(destination, ZIP_NAME).stat().st_size},
               "notebook": {"name": NOTEBOOK_NAME, "sha256": deployed_notebook_hash,
                            "template_sha256": notebook_hash},
               "local_validation": checks, "cloud_sync_status": "pending_or_unknown"}
    temporary_manifest = within(destination, f".{MANIFEST_NAME}.{release_id}.uploading")
    if verify_local_drive_binding(config, destination) != binding:
        raise RuntimeError("Drive identity changed while preparing deployment; release marker was not published")
    temporary_manifest.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, within(destination, MANIFEST_NAME))
    if json.loads(within(destination, MANIFEST_NAME).read_text()) != release:
        raise RuntimeError("Deployment marker verification failed")
    result = {**release, "local_deployment_verified": True, "backup": str(backup),
              "backup_hashes": preserved, "notebook_updated": notebook_updated,
              "packaged_files": len(manifest["files"]), "drive_project": str(destination)}
    (prep / "deployment_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def release(source=None):
    config = json.loads((PREP / "deployment_config.json").read_text(encoding="utf-8"))
    source = Path(source or config["source_project"])
    source = (PREP / source).resolve() if not source.is_absolute() else source.resolve()
    destination = Path(config["drive_project"]).absolute()
    verify_local_drive_binding(config, destination)
    if destination == PREP or destination == source or destination.is_relative_to(source):
        raise ValueError("Source and deployment destination must remain separate")
    if not destination.is_dir():
        raise FileNotFoundError(f"Start Drive Desktop and reconnect: {destination}")
    lock_path = PREP / ".deploy.lock"
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError:
        raise RuntimeError("Another deployment may be running; inspect .deploy.lock before retrying") from None
    with lock:
        lock.write(str(os.getpid()))
    try:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "GIT_OPTIONAL_LOCKS": "0"}
        def run(script, *args):
            subprocess.run([sys.executable, "-B", str(PREP / script), *map(str, args)],
                           cwd=PREP, env=env, check=True)
        run("make_notebook.py")
        import build_colab_zip as builder
        _, package = builder.collect_all(source)
        control = {"@prep/" + p.name: p for p in PREP.glob("*.py")}
        for name in (NOTEBOOK_NAME, "deployment_config.json"):
            control["@prep/" + name] = PREP / name
        if (PREP / "AGENTS.md").is_file():
            control["@prep/AGENTS.md"] = PREP / "AGENTS.md"
        checked_files = {**package, **control}
        checked_hashes = fingerprint(checked_files)
        package_hashes = fingerprint(package)
        run("run_local_checks.py", source, "--checks-only")
        assert_unchanged(checked_files, checked_hashes)
        checks = json.loads((PREP / "validation_local/test_results.json").read_text())
        run("build_colab_zip.py", source)
        assert_unchanged(checked_files, checked_hashes)
        # Compare membership too, not only contents of initially discovered files.
        if set(builder.collect_all(source)[1]) != set(package):
            raise RuntimeError("Package membership changed during validation")
        result = publish(PREP, destination, checks, package_hashes, checked_hashes["@prep/" + NOTEBOOK_NAME], config=config)
        print(json.dumps({key: result[key] for key in (
            "release_id", "local_deployment_verified", "backup", "packaged_files",
            "notebook_updated", "cloud_sync_status")}, indent=2))
        print("Local Drive deployment verified. Cloud synchronization may still be pending.")
        return 0
    finally:
        # Only this process's explicitly named transient lock is removed.
        lock_path.unlink()


if __name__ == "__main__":
    raise SystemExit(release(sys.argv[1] if len(sys.argv) > 1 else None))
