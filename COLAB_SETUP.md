# Automatic Colab deployment

The permanent notebook is inside Martin's approved project hierarchy:
**Meena Kasthuri / Soccer_Analysis_Meena_Kasthuri / SoccerProject**.
The approved root ID is `18jlQkG3zSjj-Eogo-uJ4gcuZxX6buBl4` and its existing
project child is `1k9LxrKNBogsE_Z-QOqmH6ncecI6nYJ-o`. The notebook retains ID
`1gTBP7t7mw70pOLoTH6XV-RofPR5LA44E`.

Windows Drive Desktop path:
`G:\.shortcut-targets-by-id\1k9LxrKNBogsE_Z-QOqmH6ncecI6nYJ-o\Soccer_Analysis_Meena_Kasthuri\SoccerProject\Soccer_Analytics_Colab.ipynb`.
The Colab launcher uses
`/content/drive/.shortcut-targets-by-id/1k9LxrKNBogsE_Z-QOqmH6ncecI6nYJ-o/Soccer_Analysis_Meena_Kasthuri/SoccerProject`.
Deployment checks cached IDs and parents; the launcher checks the published
approved-ID binding before loading code. An unavailable approved path stops the
run; it never falls back to a private MyDrive copy. Mount the Drive account with
access to that existing shared-folder shortcut.
Open that same file from Google Drive in Colab. Keep a GPU runtime selected, choose
**Runtime > Run all**, select a video, and click **Run short analysis**. Each new
session reads the current `deployment_manifest.json`, copies and verifies the latest
ZIP, extracts into a fresh session directory and loads its current analysis UI.
You do not need to upload the ZIP or replace the notebook manually.

The notebook checks ZIP size/hash, its embedded manifest and safe extraction paths.
If local Drive changes have not fully reached the cloud, it stops with a synchronization
message. Wait for Drive Desktop to finish syncing, then rerun the launcher. A notebook
that is already running keeps its loaded code; start a fresh runtime for a new release.
A fully consistent older cloud snapshot cannot reveal a newer unsynced local release;
compare the displayed release ID with `deployment_report.json` when unsure.

## Everyday analysis

Videos are discovered under the approved `SoccerProject/Video_Input/`, including subfolders.
Each run saves fresh results under `colab_outputs/`; existing videos and results are
never deleted. Short analysis compares baseline and improved processing on about two
seconds, using at least two consecutive chunks, and produces video/frame comparisons,
tracking CSV/JSON, events, decision logs, continuity diagnostics and review artifacts.

Choose **Start (s)** and a short duration (2-5 seconds) to inspect later action.
**Optional: mark field** calibrates that selected source frame; later camera views
need their own reviewed keyframes. Earlier polygons are never silently reused on
unrelated sampled views. Review approval is tied to the exact sampled window too.

**Compare all 5 videos** runs opening, middle and late short windows from every
source using the same model hash, baseline detector settings and seed. Exactly
five source videos must be visible in Video_Input. Each comparison includes at
least two consecutive chunks. A failed clip stays failed while other clips run;
the suite saves suite_summary.json and individual review artifacts to a fresh
folder. This button never starts a full run or grants review approval. Inspect
crossings, occlusions, field edges, camera motion and readable-number crops before
deciding whether identification is reliable. Tracker-ID counts alone are not accuracy.

Improved tracking now uses weak person detections for active-track recovery;
weak boxes cannot create new IDs or add color/OCR votes. Andrew's color extraction
is retained. Contradictory shirt evidence clears stale number evidence, and all
raw IDs remain separate. Ball thresholds and the common baseline are preserved.

Basic analysis works without editing notebook code or marking a field. When no saved
field calibration exists, off-field filtering is explicitly disabled and no spectator
quality claim is made. The optional calibration control saves a polygon for that video
content so later sessions can reuse it. Camera propagation is conservative and expires
after 120 frames unless later reviewed keyframes are available; uncertainty retains
detections. Jersey OCR is optional and unreadable numbers remain unknown.

Full-video processing remains disabled until a short GPU comparison for the selected
video passes and you explicitly review its result. Approval is tied to video content,
configuration and release. No local test establishes real soccer accuracy or GPU success.

## Automatic completion workflow

All development stays in `D:\SoccerProject\SoccerVideoAnalysisDemo_Colab`.
Its `AGENTS.md` tells future coding sessions to finish validated changes by deploying.
Both of these commands perform the complete workflow:

```powershell
& '..\SoccerVideoAnalysisDemo\.venv\Scripts\python.exe' -B .\deploy_colab.py
& '..\SoccerVideoAnalysisDemo\.venv\Scripts\python.exe' -B .\run_local_checks.py
```

For manual use, double-click `Deploy_Colab.cmd`. This is a validation-triggered workflow,
not a background watcher: unsaved or unfinished changes are never copied automatically.
Intermediate checks can use `run_local_checks.py --checks-only`; finish with deployment.
Disable the automatic check-command deployment by setting
`auto_deploy_after_validation` to `false` in `deployment_config.json`.

Deployment performs these steps:

1. Generate the stable launcher and run lightweight offline checks.
2. Confirm that the source files did not change during testing.
3. Rebuild the ZIP from the read-only original source plus Colab-only support modules.
4. Verify archive CRCs, every packaged hash and the validated source fingerprint.
5. Back up the existing synced notebook, ZIP and deployment manifest under
   `SoccerProject/deployment_backups/<release-id>/` inside the approved hierarchy.
6. Copy/verify the ZIP; update notebook contents in place only if the launcher changed.
7. Publish `deployment_manifest.json` last and write local `deployment_report.json`.

No original Windows source, Git history, input videos or previous analysis results are
changed. The builder also preserves its previous local ZIP; the notebook generator
preserves its previous notebook. Test logs are retained in `validation_local/history/`.
File locks or unavailable Drive Desktop cause a visible failure rather than a false
success report. A crash can leave `.deploy.lock`; inspect the recorded process before
removing that one transient lock and retrying.

## Deployment verification

`deployment_report.json` records the release ID, tests, deployed hashes, verified backups
and destination. `cloud_sync_status` is intentionally `pending_or_unknown`: matching
local G: hashes verifies the local deployment, not Google's server synchronization.
The notebook's published path is preserved; remote Drive file identity and browser-side
notebook execution are not asserted by local filesystem checks.
