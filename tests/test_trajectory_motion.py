"""Trajectory camera-position integration, failure boundaries and batch parity."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from trackers.tracker import Tracker
from camera_motion.camera_motion import CameraMotionEstimator


class Motion:
    def __init__(self, failed=()):
        self.index=0; self.failed=set(failed); self.calls=[]
    def update_camera(self, frame):
        i=self.index;self.index+=1;self.calls.append(i)
        self.last_update_valid=i not in self.failed and i>0
        return (2.,1.) if self.last_update_valid else (0.,0.)


def tracker(motion):
    obj=object.__new__(Tracker)
    obj.track_history={};obj.ball_history=[];obj.last_ball_bbox=None
    obj.motion_estimator=motion
    obj.smooth_bbox=lambda key,box:box
    obj.draw_ellipse=lambda frame,*a,**k:frame
    obj.draw_ball_trail=lambda frame,*a,**k:frame
    obj.draw_team_ball_control=lambda frame,*a,**k:frame
    return obj


def annotate(obj, start, count):
    frames=[np.zeros((160,240,3),np.uint8) for _ in range(count)]
    tracks={"players":[{1:{"bbox":[30+2*i,30+i,50+2*i,80+i]}} for i in range(start,start+count)],
            "referees":[{} for _ in frames],"ball":[{} for _ in frames]}
    with contextlib.redirect_stdout(io.StringIO()):
        return obj.draw_annotations(frames,tracks,np.zeros(count))


class TrajectoryMotionTests(unittest.TestCase):
    def test_stationary_ground_point_has_constant_stabilized_image_position(self):
        obj=tracker(Motion());annotate(obj,0,12)
        self.assertEqual(obj.motion_estimator.calls,list(range(12)))
        self.assertEqual(obj.annotation_camera_offset,(22.,11.))
        self.assertEqual(obj.track_history[1],[(40.,80.)]*12)

    def test_arbitrary_batch_boundaries_match_a_single_annotation_call(self):
        whole=tracker(Motion());annotate(whole,0,12)
        split=tracker(Motion())
        for start,count in ((0,3),(3,2),(5,6),(11,1)):annotate(split,start,count)
        self.assertEqual(split.annotation_camera_offset,whole.annotation_camera_offset)
        self.assertEqual(split.track_history,whole.track_history)

    def test_failed_increment_breaks_the_trail_and_starts_a_new_segment(self):
        obj=tracker(Motion(failed={4}));annotate(obj,0,7)
        self.assertEqual(obj.annotation_camera_offset,(4.,2.))
        self.assertEqual(obj.track_history[1],[(48.,84.)]*3)

    def test_nonfinite_increment_cannot_corrupt_positions_or_bridge_history(self):
        motion=Motion();obj=tracker(motion);annotate(obj,0,3)
        obj.motion_estimator=SimpleNamespace(last_update_valid=True,update_camera=lambda frame:(float('nan'),0.))
        annotate(obj,3,1)
        self.assertEqual(obj.annotation_camera_offset,(0.,0.))
        self.assertEqual(obj.track_history[1],[(46.,83.)])

    def test_drawn_history_is_reprojected_into_current_image_position(self):
        obj=tracker(Motion())
        with patch('trackers.tracker.cv2.line',wraps=cv2.line) as line:
            annotate(obj,0,3)
        self.assertEqual(line.call_args.args[1:3],((44,82),(44,82)))

    def test_real_optical_flow_handles_known_translation_without_model_inference(self):
        rng=np.random.default_rng(42)
        image=rng.integers(0,256,(160,240,3),dtype=np.uint8)
        estimator=CameraMotionEstimator();self.assertFalse(estimator.last_update_valid)
        estimator.update_camera(image);self.assertFalse(estimator.last_update_valid)
        moved=cv2.warpAffine(image,np.float32([[1,0,3],[0,1,2]]),(240,160))
        dx,dy=estimator.update_camera(moved)
        self.assertTrue(estimator.last_update_valid)
        self.assertAlmostEqual(dx,3,delta=.2);self.assertAlmostEqual(dy,2,delta=.2)
        with patch('camera_motion.camera_motion.cv2.goodFeaturesToTrack',return_value=None):
            estimator.update_camera(moved)
        self.assertFalse(estimator.last_update_valid)


if __name__=='__main__':unittest.main()
