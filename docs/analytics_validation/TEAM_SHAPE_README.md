# Team Shape Metrics v1

## Objective

Extend the existing tracking and pitch-coordinate data into basic
football-intelligence metrics describing team spatial organization.

## Metrics

The first version calculates:

- Spatial width
- Spatial depth
- Team centroid
- Compactness
- Player count

Metrics are calculated independently for each team on every frame.

## Validation Data

The raw tracking stub contains only bounding boxes and therefore cannot
directly produce team-shape metrics.

The analyzer was first validated using a small enriched synthetic fixture.

Real 100-frame validation was then performed using the existing enriched
`tracking_output.json`, which contains team assignments and pitch coordinates.

## Results

### Team 1

- Average spatial width: 88.36
- Average spatial depth: 40.33
- Average compactness: 26.71

### Team 2

- Average spatial width: 87.71
- Average spatial depth: 56.57
- Average compactness: 33.74

## Interpretation

The metrics demonstrate that existing tracking observations can be converted
into structured team-shape information.

Team 2 showed a larger average depth and compactness value during this
100-frame segment, indicating a more spatially dispersed shape according to
the current pitch-coordinate system.

## Limitations

The current implementation defines width from the pitch-coordinate X span
and depth from the pitch-coordinate Y span.

The pitch-axis convention should be confirmed before interpreting these
dimensions as definitive football lateral width and longitudinal depth.

Metrics are also sensitive to missing players, incorrect team assignment,
tracking errors, and homography inaccuracies.

## Recommendation

1. Validate pitch-coordinate axis orientation.
2. Add temporal smoothing to team-shape metrics.
3. Compare attacking and defensive team shape over longer periods.
4. Integrate team-shape metrics with synchronized tactical visualization.
5. Extend later toward defensive-line positioning and occupied-area metrics.
