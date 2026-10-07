# Sports Analytics Platform Backlog

# Current Development Roadmap

The project is now moving from performance profiling of the inherited
computer-vision pipeline toward validated football intelligence.

## Priority 1 — Reliable Detection & Validation

- Multi-video validation — High / Planned
- 960-resolution anomaly investigation — High / Planned
- Player/referee/ball detection-quality metrics — High / Planned
- Low-quality/amateur footage validation — High / Planned
- GPU benchmark — Planned when suitable hardware is available

## Priority 2 — Structured Events & Analytics

- CSV Export — Done
- Event Data Model — Done
- Touch events — Done
- Possession-change events — Done
- Divided-ball / contested-ball events — Done
- Ball recovery — Planned
- Pass detection — Planned
- Interception detection — Future
- Shot-event detection — Planned

## Priority 3 — Analytics Validation

- Manual event validation — Done (100-frame validation v1)
- Touch validation — Done for initial segment
- Possession-change validation — Done for initial segment
- Divided-ball validation — Done for initial segment
- Longer/multi-video analytics validation — Planned

## Priority 4 — Football Intelligence

- Team spatial width — Done
- Team spatial depth — Done
- Team compactness — Done
- Team centroid — Done
- Temporal comparison — High / Planned
- Defensive-line positioning — Planned
- Team spacing / occupied area — Planned

## Priority 5 — Tactical Visualization

- Synchronized match video + 2D tactical pitch — High / Planned
- Player/ball trajectories in synchronized view — Planned
- Possession and event overlays — Planned
- Team-shape metric overlays — Planned
- Broadcast-style analytics visualization — Planned

## Future Match Intelligence

- Soccer Match Summary dashboard — Planned
- Shot location and distance — Planned
- Shot angle — Planned
- Goalkeeper position during shots — Planned
- Defender pressure around shooter — Planned
- Shot outcome — Planned
- Possession sequence before shot — Planned
- Expected Goals (xG) — Future, after validated shot-event detection

### Development Dependency

Reliable tracking  
→ Structured events  
→ Validated events  
→ Football intelligence  
→ Shot detection and context  
→ Expected Goals  
→ Match Summary dashboard

This document is based on the inherited project backlog and is maintained as a living backlog for continued development.

## Project Foundation

| ID | Priority | Status | Task |
|---|---|---|---|
| EPIC-1.1 | High | Done | Repository structure |
| EPIC-1.2 | High | Done | Environment setup |
| EPIC-1.3 | Medium | In Progress | Architecture documentation |

## Computer Vision

| ID | Priority | Status | Task |
|---|---|---|---|
| EPIC-2.1 | High | Done | Player detection |
| EPIC-2.2 | High | Done | Referee detection |
| EPIC-2.3 | High | In Progress | Ball detection |
| EPIC-2.4 | High | Done | Tracking stability |
| EPIC-2.5 | High | Planned | Tracker evaluation |
| EPIC-2.6 | Medium | Done | Low-quality benchmark |

## Data & Analytics

| ID | Priority | Status | Task |
|---|---|---|---|
| EPIC-3.1 | High | In Progress | JSON export |
| EPIC-3.2 | High | Done | CSV export |
| EPIC-3.3 | High | Done | Event data model |
| EPIC-3.4 | Medium | Done | Player trajectories |
| EPIC-3.5 | Medium | Done | Heatmaps |
| EPIC-3.6 | Medium | Done | Team possession |
| EPIC-3.7 | Medium | Done | Individual possession |
| EPIC-3.8 | Medium | Done | Divided-ball detection |
| EPIC-3.9 | Medium | Planned | Temporal comparison |

## Tactical Visualization

| ID | Priority | Status | Task |
|---|---|---|---|
| EPIC-4.1 | Medium | In Progress | Professional overlays study |
| EPIC-4.2 | Medium | Planned | Broadcast graphics prototype |
| EPIC-4.3 | Low | Planned | Tactical annotations |

## Multi-Sport Architecture

| ID | Priority | Status | Task |
|---|---|---|---|
| EPIC-5.1 | High | Planned | Platform modularization |
| EPIC-5.2 | High | Planned | Reusable interfaces |
| EPIC-5.3 | Medium | Planned | Soccer module |
| EPIC-5.4 | Medium | Planned | American football analysis |
| EPIC-5.5 | Low | Planned | Baseball analysis |

## Demo & User Experience

| ID | Priority | Status | Task |
|---|---|---|---|
| EPIC-6.1 | Medium | Planned | Streamlit interface |
| EPIC-6.2 | Medium | Planned | Demo dashboard |
| EPIC-6.3 | Low | Planned | Project webpage |

## Documentation

| ID | Priority | Status | Task |
|---|---|---|---|
| EPIC-7.1 | High | In Progress | Weekly engineering log |
| EPIC-7.2 | Medium | Planned | AI usage log |
| EPIC-7.3 | High | Planned | Final report |

## Additional / Stretch Work

| ID | Priority | Status | Task |
|---|---|---|---|
| EPIC-8.1 | Low | Future | Camera calibration |
| EPIC-8.2 | Low | Future | Tactical metrics |
| EPIC-8.3 | Low | Future | Event prediction |
| EPIC-8.4 | Medium | In Progress | Database usability improvements |
| EPIC-8.5 | Medium | In Progress | Camera panning |
| EPIC-8.6 | Medium | Done | Homography / 2D mapping |
| EPIC-8.7 | Medium | Done | 2D pitch video |
| EPIC-8.8 | Medium | Done | Ball control / possession |
| EPIC-8.9 | Medium | Done | Player touches |

## Performance Work Added During Current Development

| ID | Status | Task |
|---|---|---|
| PERF-1 | Done | Performance profiling |
| PERF-2 | Done | Detection vs tracking timing |
| PERF-3 | Done | Camera-motion optimization |
| PERF-4 | Done | Resolution benchmark: 640 / 960 / 1280 |
| PERF-5 | Done | True YOLO batch inference experiment |
| PERF-6 | Done | Full-video CPU validation |
| PERF-7 | Planned | GPU acceleration benchmark |
| PERF-8 | Planned | CPU/GPU resource utilization |
| PERF-9 | Future | Real-time processing investigation |

## Recent Completed Work

### EPIC-3.2 — CSV Export
Implemented reusable frame-level CSV export for player, referee, and ball tracking data and validated the output through the main application flow.

### EPIC-3.3 — Event Data Model
Implemented structured touch and possession-change events with unique IDs, frame numbers, timestamps, player/team information, pitch coordinates, and JSON export.

### EPIC-3.8 — Divided-Ball Detection
Implemented a rule-based first version that identifies frames with at least two players within the existing 70-pixel ball-distance threshold. The nearest player remains the possession owner for backward compatibility, while contested frames are recorded with all nearby player IDs, teams, distances, frame metadata, and ball pitch coordinates and exported as `divided_ball` events.

Synthetic validation confirms that two nearby players are detected, a single nearby player is not classified as contested, and nearest-player assignment is unchanged. The cached 100-frame flow produced 3 `divided_ball` events at frames 0, 1, and 21 out of 12 total events. Generated annotated evidence was visually inspected at frame 0 and confirmed two players near the ball. The 70-pixel threshold remains a rule-based approximation and should be tuned or replaced with a calibrated pitch-distance rule for broader match conditions.

## Football Intelligence

| Task | Status | Notes |
|---|---|---|
| Team width / depth / compactness | Done | Validated on enriched 100-frame tracking output. |
| Temporal comparison | Planned | High priority follow-up. |
| Defensive-line positioning | Planned |  |
| Occupied-area / spacing metrics | Planned |  |

### Team Shape Metrics v1
Implemented per-frame team spatial width, depth, centroid, compactness, and player-count metrics. Exported JSON and CSV results and generated a team-width-over-time chart from the enriched 100-frame tracking output. Width and depth currently refer to the pitch-coordinate X and Y spans; the pitch-axis convention remains to be verified before using definitive football lateral/longitudinal labels.

## Current Recommendation

Maintain the CPU benchmark results as the performance baseline. The next performance phase should investigate GPU acceleration while preserving CPU compatibility and comparing processing time, FPS, detection quality, and resource utilization.
