# Analytics Validation — 100-Frame Segment

## Objective

Manually review a short 100-frame segment and compare system-generated
football events with visual evidence.

## Events Reviewed

- Touch
- Possession change
- Divided ball

## Key Findings

### Divided-Ball Detection

Three divided-ball detections were generated at frames 0, 1, and 21.

All three were visually supported during manual review.

Consecutive divided-ball observations at frames 0 and 1 currently represent
frame-level observations rather than one consolidated contested-ball episode.

### Possession Changes

Possession changes at frames 21 and 22 appear to be false positives caused by
rapid ownership switching during a contested-ball sequence.

This suggests that possession-change events may require temporal stability,
such as requiring ownership to remain with a new player for multiple
consecutive frames before generating an event.

### Touch Events

Several touch events were visually supported, while some remained unclear
because ball control could not be confidently determined from the available
visual evidence.

### Spatial Coordinate Issue

The event at frame 90 appears visually valid, but its pitch coordinate includes
a negative pitch_x value.

This indicates that event correctness and pitch-coordinate correctness should
be validated independently.

## Missed-Event Review

The complete 100-frame segment was manually reviewed for additional clear
touch, possession-change, and divided-ball events. No additional clearly
identifiable missed events were found in this short validation segment.

## Recommendation

1. Add temporal stabilization to possession ownership before generating a
   possession-change event.
2. Merge consecutive divided-ball observations into contested-ball episodes.
3. Add pitch-coordinate boundary validation.
4. Continue analytics validation on additional and longer video segments.
