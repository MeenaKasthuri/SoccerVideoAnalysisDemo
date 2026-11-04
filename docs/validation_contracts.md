# Ownership, event-reference and spatial validation

These tools support validation. Their presence and passing software tests do not
establish football accuracy. The default ownership rule remains the 70-pixel
nearest-foot heuristic. Source videos, labels and current backlog stay outside
the repository and source release.

## Controlled ownership comparison

The chunked runner accepts an optional `improvements.ball_ownership` object:

```json
{"distance_mode": "height_ratio", "max_player_ball_height_ratio": 0.75,
 "abstain_ambiguous": true, "require_opposing_teams_for_contest": true}
```

The normalized rule divides the minimum distance between the ball-box centre and
either lower player-box corner by player-box height. Both ranking and cutoff
use that ratio. The 0.75 cutoff is an experimental hypothesis requiring labeled
comparison; uniform resize invariance does not prove robustness to camera zoom,
pose, occlusion or perspective. Truncated boxes and explicitly unreliable scale
are rejected in this mode. Missing reliability flags do not prove valid scale.

The observation records candidates and distinguishes ambiguous proximity from
unassigned proximity. Neither means that a ball is physically free. Opposing-team
proximity can gate a contest candidate, but still does not prove a genuine duel.
Every run summary records the effective settings. No configuration promotes
`physical_control_verified` to true.

## Event correspondence

`colab_support/event_validation.py` evaluates bounded divided-ball intervals by
inclusive frame overlap and point events by a stated time tolerance. It uses
one-to-one maximum-cardinality matching, then timing/overlap quality. Explicit
complete coverage per event type is required; absent coverage yields unavailable
metrics. Uncertain reference intervals and clipped predictions are reported as
exclusions. This is type-and-time correspondence only: it does not check the
correct player, team, action interpretation or physical contact. Review status
is retained and `accuracy_validated` remains false.

## Per-camera metric calibration

`colab_support/spatial_calibration.py` is an offline assessment tool, not wired
into the production tracking pipeline. Each configuration binds original source
SHA-256, image size, pitch dimensions and bounded camera intervals. A keyframe
fit cannot propagate beyond that frame; a static interval requires recorded
review. Panning or zooming intervals need further validated temporal calibration.

Four noncollinear fit landmarks are necessary but insufficient. At least two
distinct independent checkpoints, configured reprojection limits, verified pitch
dimensions, matching source identity and recorded review are required before
metric coordinates can be valid. Fit points cannot be reused as checkpoints.
Candidate coordinates retain explicit invalid reasons when these gates fail.
The proposed 3-pixel RMSE and 5-pixel maximum limits are review choices, not an
established accuracy guarantee. An airborne ball is outside the ground-plane
model even with a valid field homography.

Run `python -B run_local_checks.py --checks-only` for the offline software suite.
Actual inference must separately record source/model hashes, release commit,
effective configuration, device, timing, errors and reference review status.
