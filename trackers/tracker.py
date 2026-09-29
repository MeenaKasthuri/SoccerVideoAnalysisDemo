from ultralytics import YOLO
import supervision as sv
print(sv.__version__)
import pickle
import os
import json
import numpy as np
import pandas as pd
import cv2
import sys
sys.path.append('../')
import inspect
import traceback
import time
from itertools import combinations
from utils import get_center_of_bbox, get_bbox_width
from camera_motion.camera_motion import CameraMotionEstimator
from homography.homography import HomographyGenerator

# Tracker

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        print("Loaded model:", model_path)
        print("Model names:", self.model.names)
        print("TRACKER CREATED")
        print(inspect.signature(sv.ByteTrack))
        self.tracker = sv.ByteTrack(
            # lost_track_buffer=300,
            # minimum_matching_threshold=0.2,
            # frame_rate=30
        )
        print("TRACKER INSTANCE:", id(self))
        print("track_activation_threshold =", self.tracker.track_activation_threshold)
        print("minimum_matching_threshold =", self.tracker.minimum_matching_threshold)
        print("minimum_consecutive_frames =", self.tracker.minimum_consecutive_frames)
        print("det_thresh =", self.tracker.det_thresh)
        print("max_time_lost =", self.tracker.max_time_lost)

        print(type(self.tracker))
        print(dir(self.tracker))
        self.track_history = {}
        self.pretrack_history = {}
        self.last_ball_bbox = None
        self.previous_ball_center = None
        self.smoothed_ball = None
        self.ball_alpha = 0.7
        self.previous_ids = set()
        self.motion_estimator = CameraMotionEstimator()
        self.ball_history = []


    def frame_blur_score(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def interpolate_ball_positions(self, ball_positions):

        extracted = []

        for x in ball_positions:
            bbox = x.get(1, {}).get("bbox", None)

            if bbox is None or len(bbox) != 4:
                extracted.append([np.nan, np.nan, np.nan, np.nan])
            else:
                extracted.append(bbox)

        # No ball detections anywhere in the video
        if all(np.isnan(row).all() for row in extracted):
            print("No ball detections found in video")
            return ball_positions

        df_ball_positions = pd.DataFrame(
            extracted,
            columns=["x1", "y1", "x2", "y2"]
        )

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [
            {1: {"bbox": x}}
            for x in df_ball_positions.to_numpy().tolist()
        ]

        return ball_positions

    def detect_frames(self, frames, use_batch=False):
        """
        Run YOLO detection on video frames.

        Parameters
        ----------
        frames : list
            Video frames to process.

        use_batch : bool
            False = standard single-frame inference.
            True = experimental true batch inference.

        Returns
        -------
        list
            YOLO detection results in original frame order.
        """

        # -------------------------------------------------
        # Standard single-frame inference
        # -------------------------------------------------
        if not use_batch:
            detections = []

            for frame_num, frame in enumerate(frames):
                blur = self.frame_blur_score(frame)

                conf = 0.22 if blur < 200 else 0.25

                result = self.model.predict(
                    frame,
                    conf=conf,
                    iou=0.35,
                    imgsz=1280,
                    verbose=False
                )

                detections.append(result[0])

            return detections

        # -------------------------------------------------
        # Experimental true batch inference
        # -------------------------------------------------
        batch_size = 16
        detections = []

        for i in range(0, len(frames), batch_size):
            print(f"Batch starting at frame {i}")

            batch = frames[i:i + batch_size]

            # Preserve original frame order
            batch_detections = [None] * len(batch)

            # Maintain existing blur-based confidence logic
            low_conf_frames = []
            low_conf_indices = []

            high_conf_frames = []
            high_conf_indices = []

            for index, frame in enumerate(batch):
                blur = self.frame_blur_score(frame)

                if blur < 200:
                    low_conf_frames.append(frame)
                    low_conf_indices.append(index)
                else:
                    high_conf_frames.append(frame)
                    high_conf_indices.append(index)

            # Batch inference using confidence 0.22
            if low_conf_frames:
                results = self.model.predict(
                    low_conf_frames,
                    conf=0.22,
                    iou=0.35,
                    imgsz=1280,
                    verbose=False
                )

                for index, result in zip(low_conf_indices, results):
                    batch_detections[index] = result

            # Batch inference using confidence 0.25
            if high_conf_frames:
                results = self.model.predict(
                    high_conf_frames,
                    conf=0.25,
                    iou=0.35,
                    imgsz=1280,
                    verbose=False
                )

                for index, result in zip(high_conf_indices, results):
                    batch_detections[index] = result

            detections.extend(batch_detections)

        return detections

    def smooth_bbox(self, key, bbox, alpha=0.6):

        if key not in self.pretrack_history:
            self.pretrack_history[key] = bbox
            return bbox

        prev = self.pretrack_history[key]

        smoothed = [
            alpha * bbox[0] + (1 - alpha) * prev[0],
            alpha * bbox[1] + (1 - alpha) * prev[1],
            alpha * bbox[2] + (1 - alpha) * prev[2],
            alpha * bbox[3] + (1 - alpha) * prev[3],
        ]

        self.pretrack_history[key] = smoothed
        return smoothed

    def get_object_tracker(
        self,
        frames,
        read_from_stub=False,
        stub_path=None,
        use_batch=False
    ):
        print("TRACKER INSTANCE IN METHOD:", id(self))
        print(__file__)
        print("RUNNING TRACKER VERSION JUNE-9-TEST")
        # tracking_export = []

        print("read_from_stub =", read_from_stub)
        print("stub exists =", os.path.exists(stub_path) if stub_path else False)

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            print(">>> LOADING FROM STUB")
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            print(">>> LOADED STUB")
            return tracks

        print(">>> STARTING DETECTION")

        detection_start = time.perf_counter()

        detections = self.detect_frames(
            frames,
            use_batch=use_batch
        )

        detection_time = time.perf_counter() - detection_start
        print(f"[DETAIL] YOLO detection only: {detection_time:.2f} seconds")

        tracking_start = time.perf_counter()

        print(">>> DETECTION COMPLETE")
        print("NUM DETECTIONS:", len(detections))

        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }

        #loops over frames one by one
        for frame_num, detection in enumerate(detections):
            try:
                print("FRAME:", frame_num)
                # print("Tracker instance:", id(self.tracker))

                cls_names = detection.names
                # print(detection.names)
                cls_names_inv = {v: k for k, v in cls_names.items()}

                # Convert to supervision detection format
                detection_supervision = sv.Detections.from_ultralytics(detection)

                #Separate the ball from players and referees before tracking
                tracking_mask = (
                        (detection_supervision.class_id == cls_names_inv["player"]) |
                        (detection_supervision.class_id == cls_names_inv["goalkeeper"]) |
                        (detection_supervision.class_id == cls_names_inv["referee"])
                )

                tracking_detections = detection_supervision[tracking_mask]
                # print("Detections before tracking:", len(detection_supervision))
                # print(
                #     "tracking boxes:",
                #     tracking_detections.xyxy[:5]
                # )
                print(
                    f"Frame {frame_num}: "
                    f"Player detections = {len(tracking_detections)}"
                )

                # Track Objects
                print("BEFORE TRACKER")
                # print("Before tracker:", len(tracking_detections))
                if frame_num in [745, 746]:
                    centers = []

                    for box in tracking_detections.xyxy:
                        cx = (box[0] + box[2]) / 2
                        cy = (box[1] + box[3]) / 2
                        centers.append((round(float(cx), 1), round(float(cy), 1)))


                    # print(f"Frame {frame_num} centers:")
                    # print(centers)
                    # current_ids = set(int(det[4]) for det in detection_with_tracks)
                    #
                    # print(f"Frame {frame_num} IDs:")
                    # print(current_ids)
                    # print(detection_with_tracks[0])
                    # print(type(detection_with_tracks))
                    # surviving = current_ids.intersection(self.previous_ids)

                    # print("Previous IDs:", sorted(self.previous_ids))
                    # print("Current IDs:", sorted(current_ids))
                    # print("Surviving IDs:", sorted(surviving))
                boxes = tracking_detections.xyxy

                duplicates = 0
                #Debugging Code
                for i, j in combinations(range(len(boxes)), 2):
                    box1 = boxes[i]
                    box2 = boxes[j]

                    x1 = max(box1[0], box2[0])
                    y1 = max(box1[1], box2[1])
                    x2 = min(box1[2], box2[2])
                    y2 = min(box1[3], box2[3])

                    inter = max(0, x2 - x1) * max(0, y2 - y1)

                    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

                    union = area1 + area2 - inter

                    iou = inter / union if union > 0 else 0

                    if iou > 0.5:
                        duplicates += 1


                # print("Duplicate pairs:", duplicates)
                detection_with_tracks = self.tracker.update_with_detections(tracking_detections)
                print("=== TRACKER OUTPUT STRUCTURE DEBUG ===")
                d = detection_with_tracks[0]
                for i, v in enumerate(d):
                    print(i, v)
                print("======================================")
                if frame_num == 0:
                    print("TRACKER SAMPLE OUTPUT:", detection_with_tracks[0])
                    print("RAW SAMPLE:", detection_with_tracks[:3])
                    for i in range(min(3, len(detection_with_tracks))):
                        print(detection_with_tracks[i])

                print(
                    "TRACKER STATE:",
                    "tracked =", len(self.tracker.tracked_tracks),
                    "lost =", len(self.tracker.lost_tracks),
                    "removed =", len(self.tracker.removed_tracks)
                )
                print("TRACKER OBJECT ID:", id(self.tracker))

                print("AFTER TRACKER")
                print("TYPE:", type(detection_with_tracks[0]))
                print("SAMPLE:", detection_with_tracks[0])
                # print("After tracker:", len(detection_with_tracks))
                # print("Tracker IDs:", detection_with_tracks.tracker_id)
                # print("Tracker frame_id:", self.tracker.frame_id)

                current_ids = set()

                for det in detection_with_tracks:
                    track_id = int(det[4])
                    current_ids.add(track_id)

                surviving = current_ids.intersection(self.previous_ids)

                if current_ids:
                    print(
                        f"Frame {frame_num}",
                        "Min ID:", min(current_ids),
                        "Max ID:", max(current_ids)
                    )
                else:
                    print(
                        f"Frame {frame_num}: No active tracks"
                    )
                # if frame_num in [745, 746]:
                #     print("Tracker IDs:")
                #     print(sorted(current_ids))

                print(
                    f"Frame {frame_num}: "
                    f"{len(surviving)} IDs survived"
                )

                self.previous_ids = current_ids


            except Exception as e:
                print("ERROR AT FRAME:", frame_num)
                traceback.print_exc()
                break



            #Uses a dictionary format
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            #Debugging Code:
            if frame_num < 5:
                print(
                    f"Frame {frame_num}: "
                    f"{len(tracks['players'][frame_num])} players saved"
                )

            frame_data = {
                "frame": frame_num,
                "players": [],
                "referees": [],
                "ball": None
            }

            # for det in detection_supervision:
            #     if det[3] == cls_names_inv["player"]:
            #         print(float(det[2]))

            #loop over detections with the players and referees
            seen_track_ids = set()
            for det in detection_with_tracks:
                bbox = det[0].tolist()  # already a (x1,y1,x2,y2) array
                cls_id = int(det[3])
                track_id = int(det[4])

                if track_id in seen_track_ids:
                    continue
                seen_track_ids.add(track_id)

                #Debugging code
                # print(
                #     "Track:",
                #     track_id,
                #     "Class:",
                #     cls_id,
                #     "Name:",
                #     cls_names[cls_id]
                # )
                #ID Change Diagnostic (debugging)
                # if cls_id == cls_names_inv["player"]:
                #     print(
                #         f"Frame {frame_num}: "
                #         f"Track {track_id} "
                #         f"BBox {bbox}"
                #     )

                # Saves Players
                if cls_id == cls_names_inv["player"]:
                    # if frame_num in [745, 746]:
                    #     print(
                    #         f"SAVING PLAYER "
                    #         f"{track_id}"
                    #     )
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                    frame_data["players"].append({
                        "track_id": track_id,
                        "bbox": bbox
                    })

                # Saves Referees
                elif cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

                    frame_data["referees"].append({
                        "track_id": track_id,
                        "bbox": bbox
                    })
            print("Detections after tracking:", len(detection_with_tracks))


            # Process ball detections
            best_ball = None
            best_conf = 0

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                confidence = frame_detection[2]
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:

                    if confidence > best_conf:
                        best_conf = confidence
                        best_ball = bbox

            # Reject impossible jumps
            if best_ball is not None:

                x_center = (
                                   best_ball[0] +
                                   best_ball[2]
                           ) / 2

                y_center = (
                                   best_ball[1] +
                                   best_ball[3]
                           ) / 2

                if self.previous_ball_center is not None:

                    prev_x, prev_y = (
                        self.previous_ball_center
                    )

                    distance = np.sqrt(
                        (x_center - prev_x) ** 2 +
                        (y_center - prev_y) ** 2
                    )

                    # Reject crazy movement
                    if distance > 200:
                        best_ball = None

                # Update previous location
                if best_ball is not None:
                    self.previous_ball_center = (
                        x_center,
                        y_center
                    )

            # Save ball
            if best_ball is not None:
                tracks["ball"][frame_num][1] = {"bbox": best_ball}

                frame_data["ball"] = {
                    "bbox": best_ball
                }
            # Populate frame data
            # tracking_export.append(frame_data)




        # print("EXPORT LENGTH:", len(tracking_export))
        tracking_time = time.perf_counter() - tracking_start
        print(f"[DETAIL] Tracking/post-processing only: {tracking_time:.2f} seconds")

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        #Debugging code to see if JSON file is being written
        # print(">>> ABOUT TO WRITE JSON")
        #
        # output_path = os.path.join(
        #     os.getcwd(),
        #     "JSON_data",
        #     "tracking_output.json"
        # )
        #
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        #
        # print("Saving JSON to:", output_path)
        #
        # with open(output_path, "w") as f:
        #     json.dump(
        #         tracking_export,
        #         f,
        #         indent=4,
        #         default=json_converter
        #     )
        #
        # print("JSON WRITE COMPLETE")
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id = None, draw_id = True):
        y2 = int(bbox[3])
        #centers the circle to the center of the bounding box
        x_center, _ = get_center_of_bbox(bbox)
        x_center = int(x_center)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center = (x_center, y2),
            axes = (int(width), int(0.35 * width)),
            angle = 0.0,
            startAngle = -45,
            endAngle = 235,
            color = color,
            thickness = 2,
            lineType = cv2.LINE_4
        )
        # Creates rectangles on the ellipses to display numbers to tell players apart
        #Creates the rectangles
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None and draw_id:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            #Creates the numbers inside the rectangles
            x1_text = x1_rect + 12
            if track_id > 90:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0), #black
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y - 5],  # point toward ball
            [x - 10, y - 25],  # top left
            [x + 10, y - 25]
        ],  dtype=np.int32)
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)
        return frame
    #Draws the ball trail using the past locations stored in its history
    def draw_ball_trail(
            self,
            frame,
            color=(0, 255, 255)
    ):

        for i in range(
                1,
                len(self.ball_history)
        ):
            cv2.line(
                frame,
                self.ball_history[i - 1],
                self.ball_history[i],
                color,
                2
            )

        return frame
    # Draws team ball control
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi transparent rectangle
        overlay = frame.copy()
        height, width = frame.shape[:2]

        x1 = int(width * 0.70)
        y1 = int(height * 0.85)

        x2 = int(width * 0.98)
        y2 = int(height * 0.98)

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            -1
        )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[: frame_num + 1]
        # Count possession frames
        team_1_num_frames = (
            team_ball_control_till_frame[
                team_ball_control_till_frame == 1
                ].shape[0]
        )

        team_2_num_frames = (
            team_ball_control_till_frame[
                team_ball_control_till_frame == 2
                ].shape[0]
        )

        total_frames = (
                team_1_num_frames +
                team_2_num_frames
        )

        # Avoid division by zero
        if total_frames > 0:
            team_1 = (
                    team_1_num_frames /
                    total_frames
            )

            team_2 = (
                    team_2_num_frames /
                    total_frames
            )
        else:
            team_1 = 0
            team_2 = 0

        # Write the statistics into the rectangle
        # Team 1
        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1 * 100:.2f}%",
            (x1 + 20, y1 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3
        )
        # Team 2
        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2 * 100:.2f}%",
            (x1 + 20, y1 + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3
        )

        return frame


    def draw_trajectory(
            self,
            frame,
            track_id,
            bbox,
            color=(0, 255, 255),
            history_length=50,
            offset_x=0,
            offset_y=0
    ):
        # Bottom-center of bbox
        x = int((bbox[0] + bbox[2]) / 2)
        y = int(bbox[3])

        world_x = x - offset_x
        world_y = y - offset_y

        if track_id not in self.track_history:
            self.track_history[track_id] = []

        self.track_history[track_id].append(
            (world_x, world_y)
        )

        # Keep only recent positions
        self.track_history[track_id] = \
            self.track_history[track_id][-history_length:]

        history = self.track_history[track_id]
        # Draw lines
        for i in range(1, len(history)):
            x1 = int(history[i - 1][0] + offset_x)
            y1 = int(history[i - 1][1] + offset_y)

            x2 = int(history[i][0] + offset_x)
            y2 = int(history[i][1] + offset_y)

            thickness = int(1 + (4 * i / len(history)))

            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
        return frame

    #Predicts the direction the ball is going in
    def predict_ball_direction(self):

        if len(self.ball_history) < 5:
            return None

        vx_total = 0
        vy_total = 0

        for i in range(
                1,
                len(self.ball_history)
        ):
            vx_total += (
                    self.ball_history[i][0]
                    - self.ball_history[i - 1][0]
            )

            vy_total += (
                    self.ball_history[i][1]
                    - self.ball_history[i - 1][1]
            )

        vx = vx_total / (len(self.ball_history) - 1)
        vy = vy_total / (len(self.ball_history) - 1)

        x, y = self.ball_history[-1]

        future_x = x + vx * 10
        future_y = y + vy * 10

        return (
            (x, y),
            (int(future_x), int(future_y))
        )


    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []

        camera_motion_total = 0.0
        player_drawing_total = 0.0
        referee_drawing_total = 0.0
        ball_drawing_total = 0.0
        team_control_total = 0.0

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # Optimize camera-motion estimation by recalculating every 5 frames
            # and reusing the previous offset between calculations.
            if frame_num % 5 == 0:
                camera_start = time.perf_counter()

                offset_x, offset_y = self.motion_estimator.update_camera(frame)

                camera_motion_total += time.perf_counter() - camera_start

                self.last_camera_offset = (offset_x, offset_y)

            else:
                offset_x, offset_y = getattr(
                    self,
                    "last_camera_offset",
                    (0.0, 0.0)
                )

            # print(
            #     f"Frame {frame_num}: "
            #     f"dx={offset_x:.1f}, "
            #     f"dy={offset_y:.1f}"
            # )

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            #Draw Players
            for track_id, player in player_dict.items():
                #Draws the circles under the players in the color of their respective team, if not team color is found, use red
                # Smooth bbox
                bbox = self.smooth_bbox(
                    f"player_{track_id}",
                    player["bbox"]
                )

                # Draw team-colored circle
                color = player.get(
                    "team_color",
                    (0, 0, 255)
                )

                frame = self.draw_ellipse(
                    frame,
                    bbox,
                    color,
                    track_id,
                    draw_id=True
                )
                #Draw player trajectory lines
                frame = self.draw_trajectory(
                    frame,
                    track_id,
                    bbox,
                    color,
                    offset_x = offset_x,
                    offset_y = offset_y
                )

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, bbox, (0, 0, 255))

                # print(player)



            # Draw Referees
            for track_id, referee in referee_dict.items():

                bbox = self.smooth_bbox(
                    f"ref_{track_id}",
                    referee["bbox"]
                )
                # Draws the circles under the referees in yellow
                frame = self.draw_ellipse(
                    frame,
                    bbox,
                    (0, 255, 255),
                    track_id,
                    draw_id=False
                )

            # Draw Ball
            frame = self.draw_ball_trail(frame)
            for track_id, ball in ball_dict.items():
                bbox = ball["bbox"]
                self.last_ball_bbox = bbox

            # Keep last 30 ball locations
            if self.last_ball_bbox is not None:
                x1, y1, x2, y2 = self.last_ball_bbox

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                self.ball_history.append(
                    (center_x, center_y)
                )

                self.ball_history = self.ball_history[-30:]

            # Draw the ball predictions
            ball_prediction = self.predict_ball_direction()

            if ball_prediction is not None:
                start_point, end_point = ball_prediction
                cv2.arrowedLine(
                    frame,
                    start_point,
                    end_point,
                    (0,255,255),
                    3)

            #Draw green triangle over ball
            if self.last_ball_bbox is not None:
                frame = self.draw_triangle(
                    frame,
                    self.last_ball_bbox,
                    (0, 255, 0)
                )


            #Draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_video_frames.append(frame)

            #Debugging Code
            # current_ids = set(player_dict.keys())
            #
            # new_ids = current_ids - self.previous_ids
            #
            # if len(new_ids) > 0:
            #     print(f"Frame {frame_num}: New IDs {new_ids}")
            #
            # self.previous_ids = current_ids

            # if not hasattr(self, "all_player_ids"):
            #     self.all_player_ids = set()
            #
            # for track_id in player_dict.keys():
            #     self.all_player_ids.add(track_id)

            # print(
            #     f"Frame {frame_num}: "
            #     f"Players={len(player_dict)} "
            #     f"Unique IDs Seen={len(self.all_player_ids)}"
            # )

            # Clean up dead tracks:
            active_ids = set(player_dict.keys())

            for track_id in list(self.track_history.keys()):

                if track_id not in active_ids:
                    self.track_history[track_id] = \
                        self.track_history[track_id][-20:]

        print(f"[DETAIL] Camera motion estimation: {camera_motion_total:.2f} seconds")

        return output_video_frames



