# November 2026 technical handover

The maintained review is [draft PR #1](https://github.com/MeenaKasthuri/SoccerVideoAnalysisDemo/pull/1).
The October 28 operational baseline is release `20261028T032449Z_8ee99bdd`.
The November source and GPU receipts are pinned to their respective commits.
See [release provenance](releases/README.md) and [validation contracts](validation_contracts.md).

## Reproduce the tested source

1. Check out `d1d665983d7f3a3b559aea4bd3af8ae31f11523b` to reproduce the November 4 validation.
2. In a compatible Python environment, install `requirements_colab.txt`.
3. Run `python -B run_local_checks.py --checks-only` from the repository root.
4. To build a new package, commit approved source changes and run `python -B build_colab_zip.py .` from a clean checkout.
5. Preserve the deployed source archive and manifest. Any later Colab promotion requires the approved Drive binding and a cloud readback.

The pinned candidate ZIP has 88 manifest entries and SHA-256
`b004eb78e098af2b6bbb04c2661d0b56c108c2677cea91d43653b2543238858b`.
The November 10 recheck of those exact archive bytes passed all 184 offline tests.
The tests exercise software contracts and exports with synthetic detection and OCR
where documented.

## Interpret the evidence

The project recorded 30 short baseline and improved Tesla T4 pairs across three
source versions: 20 at `f6e68e2`, five ownership experiments at `d19b21a`,
and five trajectory-candidate pairs at `d1d6659`. The latter cover 615 frames
per mode and 107.7 versus 113.4 seconds of recorded pipeline time. OCR and field
filtering were off. Use the saved run receipts to compare the same version and
source frames.

The dense player diagnostics use 50 correlated opening frames and earlier
`f6e68e2` predictions. Human tracking, team assignment, ownership and football
event acceptance remains open. Two measured field-calibration trials failed
the proposed independent-checkpoint limits. Metric pitch locations, speed and
distance stay gated. Agree acceptance targets before extending to longer or
full-video comparisons.

## Maintain the project

The shared Drive project and existing Excel backlog hold access-controlled
inputs, run artifacts, and the current acceptance record. The draft PR is the
code-review entry point. Preserve legacy models and examples until the project
owner confirms their role. Keep additional footage, generated results, workbook
data and credentials out of source packages.

For legacy training, configure a project-owned `ROBOFLOW_API_KEY` through the
environment or Colab Secrets before running the notebook. For the inherited
database workflow, provision the successor's SQL Server connection separately.
