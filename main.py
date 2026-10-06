from utils.video_utils import read_video, save_video
from trackers import Tracker
import cv2
import os
import numpy as np
import streamlit as st
from team_assigner import TeamAssigner
from sklearn.cluster import KMeans
from player_ball_assigner import PlayerBallAssigner
from utils.tracking_data import  load_tracking_data
from analytics.player_histories import build_player_histories
from camera_motion.camera_motion import CameraMotionEstimator
from database.sql_lite import SQLiteManager
from database.sql_server import SQLServerManager
from analytics.heatmaps import  HeatmapAnalyzer
from homography.homography import HomographyGenerator
from exporters.json_exporter import export_tracking_json
from exporters.csv_exporter import export_tracking_csv
from exporters.event_exporter import export_events_json
from visualization import pitch_visualizer
from visualization.pitch_visualizer import PitchVisualizer
from analytics.positioning import Position_Generator
from analytics.possession_analyzer import PossessionAnalyzer
from analytics.touch_analyzer import TouchAnalyzer
from analytics.distance_analyzer import DistanceAnalyzer
from analytics.history_builder import HistoryBuilder
from analytics.event_model import EventModel
from analytics.team_zone_analyzer import TeamZoneAnalyzer
from analytics.player_statistics_builder import PlayerStatisticsBuilder
from visualization.tactical_visualizer import TacticalVisualizer
import time

# Main
def collect_player_colors(
        tracks,
        video_frames,
        team_assigner,
        min_colors=20):

    player_colors = []

    for frame_num, players in enumerate(tracks["players"]):

        if len(players) == 0:
            continue

        frame = video_frames[frame_num]

        for _, player in players.items():
            bbox = player["bbox"]

            color = team_assigner.get_player_color(
                frame,
                bbox
            )

            player_colors.append(color)

        if len(player_colors) >= min_colors:
            break

    return player_colors




def main(
        video_path,
        output_folder = "output_videos",
        options = None
):
    total_start = time.perf_counter()
    if options is None:
        options = {
            "pitch": True,
            "heatmap": True,
            "zones": True,
            "hulls": True,
            "overlay_heatmap": True,
            "json": True,
            "csv": True,
            "database": True
        }
    ###-----RESULTS DICT-----###
    results = {}
    ###-----RESULTS DICT END-----###
    print("STEP 1 - entering main")

    # PERFORMANCE TIMER - VIDEO LOADING
    video_load_start = time.perf_counter()

    video_frames, fps = read_video(video_path)

    video_load_time = time.perf_counter() - video_load_start

    print("FPS:", fps)
    print(f"[TIME] Video loading: {video_load_time:.2f} seconds")
    print("STEP 2 - video loaded")

    # Create unique stub file name per video
    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    stub_path = f"stubs/{video_name}_tracks.pkl"


    #Initialize the tracker
        # Initialize the tracker
    tracker = Tracker('models/best (2).pt')
    print("STEP 3 - tracker initialized")

    # PERFORMANCE TIMER - DETECTION AND TRACKING
    tracking_start = time.perf_counter()

    tracks = tracker.get_object_tracker(
        video_frames,
        read_from_stub=False,
        stub_path=stub_path
    )

    tracking_time = time.perf_counter() - tracking_start

    print(f"[TIME] Detection + Tracking: {tracking_time:.2f} seconds")
    print("Players in frame 0:", len(tracks["players"][0]))
    print("STEP 4 - tracking complete")
    # Match_analysis dictionary
    match_analysis = {}

    image_points = [
        [174, 914],
        [1752, 901],
        [1456, 182],
        [432, 205]
    ]

    field_points = [
        [0, 68],
        [105, 68],
        [105, 0],
        [0, 0]
    ]

    # Homography
    homography = HomographyGenerator()

    homography.set_points(
        image_points,
        field_points
    )

    homography.compute_homography()

    tracks = homography.transform_tracks(tracks)


    first_frame = tracks["players"][0]

    for track_id, player in first_frame.items():
        print(f"Player {track_id}")

        print("Image:",
              player["bbox"])

        print("Pitch:",
              player["pitch_x"],
              player["pitch_y"])

        print("----------------")


    if options["database"]:

        # PERFORMANCE TIMER - INITIAL DATABASE OPERATIONS
        database_initial_start = time.perf_counter()

        #Implement SQL Lite database:
        #Implement SQL Lite database:
        # Create database (SQL Lite)
        sqlite_db = SQLiteManager()

        # Create database (SQL Server)
        sqlserver_db = SQLServerManager()
        # Create SQLite tables
        sqlite_db.create_tables()

        # Create match in SQL Server
        sqlserver_match_id = sqlserver_db.create_match(
            video_name=video_path,
            fps=fps,
            width=1920,
            height=1080
        )

        # Create match in SQLite
        sqlite_match_id = sqlite_db.create_match(
            video_name=video_path,
            fps=fps,
            width=1920,
            height=1080
        )


        # Create tables
        sqlite_db.create_tables()

        #Insert Players
        databases = [
            (sqlite_db, sqlite_match_id),
            (sqlserver_db, sqlserver_match_id)
        ]

        for frame_num, player_dict in enumerate(tracks["players"]):

            for track_id, player in player_dict.items():

                for db, match_id in databases:
                    db.insert_player(

                        frame_num,
                        track_id,
                        player["bbox"],
                        match_id,
                        player["pitch_x"],
                        player["pitch_y"],
                        player.get("team"),
                        player.get("team_color"),
                        player.get("has_ball", False)

                    )

        #Insert Ball
        for frame_num, ball_dict in enumerate(tracks["ball"]):

            if 1 in ball_dict:
                for db, match_id in databases:
                    db.insert_ball(

                        frame_num,
                        ball_dict[1]["bbox"],
                        match_id,
                        ball_dict[1]["pitch_x"],
                        ball_dict[1]["pitch_y"]

                    )

        sqlite_db.save()
        sqlserver_db.save()

        database_initial_time = time.perf_counter() - database_initial_start
        print(f"[TIME] Initial database operations: {database_initial_time:.2f} seconds")



        # PERFORMANCE TIMER - ANALYTICS
    analytics_start = time.perf_counter()

    player_histories = build_player_histories(
        tracks
    )
    print(
        "Tracked players:",
        len(player_histories)
    )


    print(
        "Player 1 samples:",
        player_histories[1][:5]
    )

    # #Uses the code in heatmaps/HeatmapMaker to build the player heatmap
    # heatmap_maker = HeatmapMaker()
    # PITCH_WIDTH = 1050
    # PITCH_HEIGHT = 680
    #
    # # heatmap_width = video_frames[0].shape[1]
    # # heatmap_height = video_frames[0].shape[0]
    # #Debugging code
    # for item in player_histories[1][:5]:
    #     print("-----------------PLAYER_HISTORIES_ITEM----------------------")
    #     print(item)
    #
    # player_heatmap = heatmap_maker.build_player_heatmap(
    #     player_histories[1],
    #     PITCH_WIDTH,
    #     PITCH_HEIGHT
    # )
    # team_heatmap = heatmap_maker.build_team_heatmap(
    #     player_histories,
    #     PITCH_WIDTH,
    #     PITCH_HEIGHT
    # )
    #
    # heatmap_maker.save_heatmap(
    #     team_heatmap,
    #     "output_heatmaps/team_heatmap_low.png"
    # )
    #
    # heatmap_maker.save_heatmap(
    #     player_heatmap,
    #     "output_heatmaps/player1_heatmap_low.png"
    # )

    # Interpolate ball positions
    print("BALL FRAME 0:", tracks["ball"][0])
    print("BALL FRAME 1:", tracks["ball"][1])
    print("BALL FRAME 2:", tracks["ball"][2])
    tracks['ball'] = tracker.interpolate_ball_positions(tracks["ball"])

    for frame_ball in tracks["ball"]:

        if 1 not in frame_ball:
            continue

        ball = frame_ball[1]

        bbox = ball["bbox"]

        image_x = (bbox[0] + bbox[2]) / 2
        image_y = (bbox[1] + bbox[3]) / 2

        pitch_x, pitch_y = homography.transform_homography(
            image_x,
            image_y
        )

        ball["pitch_x"] = pitch_x
        ball["pitch_y"] = pitch_y

    # Assign player teams
    team_assigner = TeamAssigner()

    player_colors = collect_player_colors(
        tracks,
        video_frames,
        team_assigner
    )

    print("Collected colors:", len(player_colors))

    if len(player_colors) < 5:
        print("Not enough data for team clustering")
        print(f"[TIME] Team assignment + analytics: {time.perf_counter() - analytics_start:.2f} seconds")
        print(f"[TIME] TOTAL PROCESSING TIME: {time.perf_counter() - total_start:.2f} seconds")
        return

    team_assigner.assign_team_color_from_colors(player_colors)

    # max_players = max(len(players) for players in tracks["players"])
    # print("Maximum players detected in a frame:", max_players)
    #
    # all_player_colors = []
    #
    # for frame_num, players in enumerate(tracks["players"]):
    #     if len(players) >= 2:
    #         for _, player in players.items():
    #             bbox = player["bbox"]
    #             color = team_assigner.get_player_color(video_frames[frame_num], bbox)
    #             all_player_colors.append(color)
    #
    #     if len(all_player_colors) >= 10:
    #         break
    #
    # team_assigner.kmeans = KMeans(n_clusters=2, n_init=10).fit(all_player_colors)
    #
    # team_assigner.team_colors = {
    #     1: team_assigner.kmeans.cluster_centers_[0],
    #     2: team_assigner.kmeans.cluster_centers_[1],
    # }


    print("Team colors:", team_assigner.team_colors)

    # Loop over each player and assign them to correct team
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            # tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            tracks['players'][frame_num][player_id]['team_color'] = (
                team_assigner.team_colors.get(team, (0, 0, 255))
            )

    # Assign ball acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    tracks["divided_ball"] = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox', None)

        if ball_bbox is None:
            tracks["divided_ball"].append(None)
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
            continue

        divided_ball = player_assigner.detect_divided_ball(player_track, ball_bbox)
        if divided_ball is None:
            tracks["divided_ball"].append(None)
        else:
            ball = tracks['ball'][frame_num][1]
            tracks["divided_ball"].append({
                "player_ids": divided_ball["player_ids"],
                "distances": divided_ball["distances"],
                "teams": [player_track[player_id].get("team")
                          for player_id in divided_ball["player_ids"]],
                "pitch_x": ball.get("pitch_x"),
                "pitch_y": ball.get("pitch_y")
            })

        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            player_team = player_track[assigned_player]['team']
            team_ball_control.append(player_team)

        else:
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(0)
    team_ball_control = np.array(team_ball_control)

    # Event data model

    event_model = EventModel(fps)

    event_data = event_model.build_events(
        tracks
    )

    match_analysis["events"] = event_data

    export_events_json(
        event_data,
        "JSON_data/events.json"
    )

    print("-----EVENT DATA MODEL-----")
    print("Total events:", len(event_data))

    for event in event_data[:5]:
        print(event)

    #History Builder
    history_builder = HistoryBuilder()

    player_histories = history_builder.build_player_history(tracks)

    team_histories = history_builder.build_team_history(
        player_histories
    )

    print("TEAM HISTORIES DEBUG")

    for team, history in team_histories.items():
        print("TEAM:", team)
        print("NUMBER OF POSITIONS:", len(history))
        print("FIRST ENTRY:", history[0])

    # Possession analysis
    print("-----CHECK FOR FRAMES WITH POSSESSION-----")
    for player_id, player in tracks["players"][0].items():
        print(player)
    possession_analyzer = PossessionAnalyzer()

    possession_data = possession_analyzer.calculate_possession(
        tracks
    )
    print(possession_data)

    match_analysis["possessions"] = possession_data

    #Touch analysis
    print("-----TOUCH ANALYSIS FOR PLAYERS-----")
    for player_id, player in tracks["players"][0].items():
        print(player)
    touch_analyzer = TouchAnalyzer()
    touch_data = touch_analyzer.calculate_touches(
        tracks)
    print(touch_data)

    match_analysis["touches"] = touch_data

    #JSON implementation

    if options["json"]:

        export_tracking_json(
            tracks,
            "JSON_data/tracking_output.json"
        )
        results["JSON"] = \
            "JSON_data/tracking_output.json"

    # CSV implementation

    if options["csv"]:

        export_tracking_csv(
            tracks,
            "CSV_data/tracking_output.csv"
        )

        results["CSV"] = \
            "CSV_data/tracking_output.csv"

    analytics_time = time.perf_counter() - analytics_start
    print(f"[TIME] Team assignment + analytics: {analytics_time:.2f} seconds")


    # #Save cropped image of a player
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]
    #
    #     # Crop bbox from frame
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    #
    #     # Save the cropped image
    #     cv2.imwrite(f'output_videos/cropped_image.jpg', cropped_image)
    #     break

    # print("video frames:", len(video_frames))
    # print("player tracks:", len(tracks["players"]))
    # print("ball tracks:", len(tracks["ball"]))
    # print("ref tracks:", len(tracks["referees"]))
    # print("team ball control:", len(team_ball_control))
    #Draw output
    ##Draw Object Tracks
    annotation_start = time.perf_counter()
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    annotation_time = time.perf_counter() - annotation_start
    print(f"[TIME] Annotation drawing: {annotation_time:.2f} seconds")

    print("Input frames:", len(video_frames))
    print("Output frames:", len(output_video_frames))

    # Get average position
    position_generator = Position_Generator()
    print("TEST PLAYER")
    print(tracks["players"][0][1])
    player_positions = position_generator.collect_player_positions(
        tracks
    )
    average_positions = position_generator.calculate_average_positions(
        player_positions
    )
    match_analysis["average_positions"] = average_positions

    # print("---DEBUGGING---")
    # for track_id, position in average_positions.items():
    #
    #     print(track_id, position)
    #
    #     if track_id >= 5:
    #         break

    # Pitch Visualization
    tactical_visualizer = TacticalVisualizer()
    pitch_visualizer = PitchVisualizer()
    print("---BALL TRACKS---")
    print(tracks["ball"][0])

    if options["pitch"]:

        pitch_start = time.perf_counter()

        pitch_frames = tactical_visualizer.build_pitch_video(
            tracks,
            video_frames
        )

        save_video(
            pitch_frames,
            os.path.join(
                output_folder,
                "pitch_view.mp4"
            ),
            fps
        )
        results["Pitch View"] = \
            "pitch_view.mp4"

        print(f"[TIME] Pitch video: {time.perf_counter() - pitch_start:.2f} seconds")

    #Output for average positions:
    average_pitch = pitch_visualizer.create_pitch()

    print("MATCH ANALYSIS:")
    print(match_analysis)

    print("MATCH ANALYSIS KEYS:")
    print(match_analysis.keys())
    average_pitch = pitch_visualizer.draw_average_positions(
        average_pitch,
        match_analysis["average_positions"]
    )
    cv2.imwrite(
        "assets/average_positions.png",
        average_pitch
    )

    #Distance Analyzer
    distance_analyzer = DistanceAnalyzer()
    match_analysis["distance"] = \
        distance_analyzer.calculate_distance(
            player_histories
        )

    #Zone analyzer
    zone_analyzer = TeamZoneAnalyzer()

    team_centers = zone_analyzer.calculate_team_centers(
        team_histories
    )

    team_hulls = zone_analyzer.calculate_convex_hulls(
        team_histories
    )

    match_analysis["team_zones"] = {

        "team_centers": team_centers,

        "team_hulls": team_hulls

    }

    ###-----ZONE VIDEO DRAWING LOOP-----###
    if options["zones"]:
        zone_start = time.perf_counter()
        zone_frames = tactical_visualizer.build_zone_video(

            tracks,

            video_frames

        )
        #Save Video
        save_video(
            zone_frames,
            os.path.join(
                output_folder,
                "team_zones_video.mp4"
            ),
            fps
        )
        results["Zone View"] = \
            "team_zones_video.mp4"
        print(f"[TIME] Zone video: {time.perf_counter() - zone_start:.2f} seconds")
    ####------END OF ZONE VIDEO BLOCK------####

    ####------ZONE IMAGE DRAWING------####
    zone_pitch = pitch_visualizer.create_pitch()
    zone_pitch = pitch_visualizer.draw_team_centers(
        zone_pitch,
        match_analysis["team_zones"]["team_centers"]
    )
    print("TEAM ZONE ANALYSIS:")
    print(match_analysis["team_zones"]["team_hulls"].keys())

    zone_pitch = pitch_visualizer.draw_team_hulls(
        zone_pitch,
        match_analysis["team_zones"]["team_hulls"]
    )

    cv2.imwrite(
        "assets/team_zones_hulls.png",
        zone_pitch
    )
    #####------END OF ZONE IMAGE DRAWING BLOCK------####

    #####------ZONE HULL OVERLAY VIDEO BLOCK------####
    if options["hulls"]:
        hull_start = time.perf_counter()
        overlay_frames = tactical_visualizer.build_overlay_hull_video(
            tracks,
            video_frames
        )

        save_video(
            overlay_frames,
            os.path.join(
                output_folder,
                "overlay_hulls.mp4"
            ),
            fps
        )
        results["Hull Overlay"] = \
            "overlay_hulls.mp4"
        print(f"[TIME] Hull overlay: {time.perf_counter() - hull_start:.2f} seconds")
    #####------END OF ZONE HULL OVERLAY VIDEO BLOCK------####

    #####------HEATMAP VIDEO BLOCK-----#####
    if options["heatmap"]:

        heatmap_start = time.perf_counter()

        heatmap_frames = tactical_visualizer.build_heatmap_video(
            tracks,
            video_frames
        )
        save_video(
            heatmap_frames,
            os.path.join(
                output_folder,
                "team_heatmap_video.mp4"
            ),
            fps
        )
        results["Team Heatmaps"] = \
            "team_heatmap_video.mp4"
        print(f"[TIME] Heatmap video: {time.perf_counter() - heatmap_start:.2f} seconds")

    ########-----HEATMAPS IMAGE BLOCK-----#########
    heatmap_analyzer = HeatmapAnalyzer()

    match_analysis["heatmaps"] = \
        heatmap_analyzer.calculate_team_heatmaps(
            team_histories
        )

    heatmap_pitch = pitch_visualizer.create_pitch()
    heatmap_pitch = pitch_visualizer.draw_team_heatmaps(

        heatmap_pitch,
        match_analysis["heatmaps"]

    )
    cv2.imwrite(

        "output_heatmaps/team_heatmaps.png",
        heatmap_pitch

    )
    ######-----HEATMAPS IMAGE BLOCK END-----######

    ######-----HEATMAP OVERLAY VIDEO BLOCK-----######
    if options["overlay_heatmap"]:

        overlay_heatmap_start = time.perf_counter()

        overlay_heatmap_frames = tactical_visualizer.build_overlay_heatmap_video(
            tracks,
            video_frames
        )

        save_video(
            overlay_heatmap_frames,
            os.path.join(
                output_folder,
                "overlay_heatmaps.mp4"
            ),
            fps
        )
        results["Heatmap Overlay"] = \
            "overlay_heatmaps.mp4"
        print(f"[TIME] Heatmap overlay: {time.perf_counter() - overlay_heatmap_start:.2f} seconds")
    ######-----HEATMAP OVERLAY VIDEO BLOCK END-----######

    #STATISTICS BUILDER CALL
    final_database_start = time.perf_counter()
    statistics_builder = PlayerStatisticsBuilder()

    match_analysis["player_statistics"] = \
        statistics_builder.build_player_statistics(
            match_analysis
        )

    #Implement statistics and analysis into the database
    if options["database"]:
        for track_id, stats in match_analysis["player_statistics"].items():
            sqlite_db.insert_player_statistics(
                sqlite_match_id,
                stats
            )

            sqlserver_db.insert_player_statistics(
                sqlserver_match_id,
                stats
            )

        sqlite_db.save()
        sqlserver_db.save()
        sqlite_db.close()
        sqlserver_db.close()

        results["SQLite"] = "sports_db/soccer_tracking.db"
        results["SQL Server"] = "Updated"

    print(f"[TIME] Final database/statistics: {time.perf_counter() - final_database_start:.2f} seconds")





    # cv2.imwrite(
    #     "assets/pitch_test.png",
    #     pitch
    # )

    #Save Video
    final_video_start = time.perf_counter()
    save_video(
        output_video_frames,
        os.path.join(
            output_folder,
            "output_video_Soccer.avi"
        ),
        fps
    )
    print(f"[TIME] Final annotated video: {time.perf_counter() - final_video_start:.2f} seconds")
    print(f"[TIME] TOTAL PROCESSING TIME: {time.perf_counter() - total_start:.2f} seconds")
    print(results)
    return results


if __name__ == "__main__":

    main(

        "Video_Input/Soccer_Test_Video.mp4"

    )