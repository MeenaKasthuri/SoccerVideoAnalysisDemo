# Release provenance and validation

The verified historical deployment is `20261028T032449Z_8ee99bdd`. Its source ZIP
SHA-256 is `b9cbce539bd358078e05337e686f4ef1307e915ffec89791e11629ac4b36fb05`
(37,515,790 bytes). The model `models/best (2).pt` has SHA-256
`58e84d483751f8f37db0f05f0655f5799a68b49e0805c7fe9d8dc136a6b062cc`.
The historical manifests in the adjacent release directory describe that release,
not later changes in this checkout.

That release recorded local HEAD `efbc3623bfe6de3e448e0d8a39929f43e2e7a6eb`,
seven modified tracked files and `utils/label_utils.py` as an untracked file.
Its contents therefore could not be recovered by checking out that commit alone.
All 69 manifest-listed files were verified against their recorded bytes and hashes
before importing the release into this branch.

The published fork master at reconciliation was
`897b7860e2096e5e73e5785e50ca1c61325921cd` (October 7, 2026).
Andrew's upstream master was `7df9675a8a87e90a1552881ac150cf0211dfc41c`
(August 11, 2026). The import preserves the local commit ancestry and existing
repository files. No backlog data, new results or input videos are added by the
release tooling.

The notebook on Drive has three appended diagnostic cells. Its original markdown
and bootstrap source match the committed two-cell launcher template. Preserve the
Drive notebook before any replacement; executed notebook bytes are expected to
differ from the unexecuted launcher because of outputs and metadata.

## Validate a checkout

Install `requirements_colab.txt` over the compatible Colab environment, or use an
existing compatible Python environment. Then run:

```powershell
python -B run_local_checks.py --checks-only
```

This command uses the current checkout by default. The committed configuration
disables automatic deployment. Tests use synthetic detections/OCR where documented;
passing tests does not establish player, ball, identity, OCR or football accuracy.
Do not run long local YOLO jobs. Manual review and GPU comparisons remain separate
validation steps.

## Build a release

Commit validated changes first, then run:

```powershell
python -B build_colab_zip.py .
```

The builder rejects a dirty Git checkout, includes the commit and per-file hashes,
and uses an allowlist that excludes backlog files, source videos, prior results,
databases, caches and credentials. Deployment authoring modules are included so
the packaged test imports are present. Keep the generated ZIP outside Git.

`requirements_colab.txt` pins Ultralytics and Supervision. Capture the complete
runtime package versions, Python/CUDA/device, model hash, video hash, source commit,
configuration and time window with each real comparison. Colab-provided binary
packages vary and are not fully specified by the requirements file alone.

Live publication requires a separately configured Drive Desktop metadata path and
the existing approved folder/notebook binding. The safe committed configuration
does not contain that private local cache path. A successful local file copy does
not prove Google Drive cloud synchronization or a successful Colab GPU run.
