# Soccer Video Analysis Platform

Piedra Alta Sistemas soccer video analytics, originally developed by Andrew
Brandenburg and continued by Meena Kasthuri.

The application uses YOLO detections and ByteTrack identities to export player,
referee and ball observations, with team evidence, possession heuristics, event
records and visualizations. These outputs require reference-based validation
before they can support football-performance conclusions.

## Current architecture

- `trackers/`, `team_assigner/`, `player_ball_assigner/` and `analytics/` contain
  detection, tracking, assignment and football observation/event logic.
- `colab_support/` provides the Colab application, persistent tracking across
  decoding chunks, baseline/improved comparisons, review gates and exports.
- `Soccer_Analytics_Colab.ipynb` is the stable launcher. Runtime behavior lives in
  Python modules, rather than in growing notebook cells.
- `build_colab_zip.py`, `deploy_colab.py` and `drive_identity.py` package committed
  source and guard the approved Drive destination. Automatic deployment is off in
  the committed configuration.
- `tests/` contains offline regression and integration checks. The desktop
  Streamlit/database entry points remain available as inherited application paths.

## Reproduce the Colab source

The preserved historical release is `20261028T032449Z_8ee99bdd`. Its verified ZIP
contains a working-tree snapshot, including unpublished source changes; checking
out its old internal HEAD alone does not reproduce it. This branch imports that
snapshot, the missing authoring tools and subsequent regression fixes.
See [release provenance](docs/releases/README.md).

The tested local Python version is 3.13.3. `requirements_colab.txt` pins
`ultralytics==8.4.131` and `supervision==0.30.1` over a compatible Colab runtime.
Colab supplies PyTorch, OpenCV and other binary dependencies. Capture their actual
versions and CUDA/device information with every real run; these two pins alone
are not a full environment lock. Do not upgrade Supervision without checking its
ByteTrack API compatibility.

```powershell
python -m pip install -r requirements_colab.txt
python -B run_local_checks.py --checks-only
```

For Colab, open the stable launcher against the approved shared project folder.
For a new source package, commit validated changes and run:

```powershell
python -B build_colab_zip.py .
```

The builder requires a clean checkout, records the release commit and file
hashes, and excludes input videos, backlog data and previous results. Deployment
requires the separately configured existing Drive binding and cloud readback.
The preserved production release is not automatically replaced by this branch.

## Event and camera contracts

`EventModel.build_events` consumes the complete ordered sequence after chunks
are assembled. A continuous divided-ball observation becomes one bounded episode
with inclusive start/end frames, duration and boundary-censoring flags. Teammate
owner transfers produce an ownership-proxy touch, while only known opposing
teams produce `possession_change`. A proximity observation does not prove a pass,
physical touch or genuine contest.

`CameraMotionEstimator.update_camera` returns displacement since its previous
call, or zero when an increment cannot be measured. It always advances the image
reference after an accepted frame. Zero is an unavailable estimate, not proof of
a stationary camera. The inherited trajectory overlay still has limitations in
how it consumes sampled motion estimates; corrected estimator tests do not prove
camera-compensated trajectory accuracy.

## Validation status

Keep three claims separate: software correctness, computer-vision accuracy, and
football-analytics accuracy. Offline tests use synthetic detections/OCR where
documented and do not establish real detector, tracker, OCR or event accuracy.
Historical short Colab runs are execution evidence for their recorded release,
not GPU validation of later code changes.

The fixed 70-pixel ownership threshold remains a baseline heuristic. Pitch
coordinates remain unreliable without per-video/camera-segment calibration and
independent validation. Field filtering alone is not metric homography.
Optional ownership experiments and offline event/spatial assessment tools are
described in [validation contracts](docs/validation_contracts.md). Their defaults
and explicit review gates prevent provisional experiments from becoming accepted
accuracy claims.
Baseline/improved comparisons must use identical labeled source frames and record
video/model hashes, commit, configuration, device, errors and elapsed time.

The current backlog is maintained separately in the existing shared Excel
workbook. Historical Markdown backlog material in this repository is not the
current acceptance record. No new backlog data belongs in source releases.
