class EventModel:
    def __init__(self, fps):
        if fps is None or fps <= 0:
            raise ValueError("FPS must be greater than 0")

        self.fps = fps

    def build_events(self, tracks):
        events = []

        previous_owner = None
        event_counter = 1

        for frame_num, frame_players in enumerate(tracks["players"]):

            current_owner = None
            current_player = None

            # Find the player who currently has the ball
            for track_id, player in frame_players.items():

                if player.get("has_ball", False):
                    current_owner = track_id
                    current_player = player
                    break

            # Create events only when ownership changes
            if current_owner is not None and current_owner != previous_owner:

                timestamp = round(
                    frame_num / self.fps,
                    3
                )

                # -------------------------
                # Touch event
                # -------------------------

                events.append({
                    "event_id": f"EVT-{event_counter:06d}",
                    "event_type": "touch",
                    "frame": frame_num,
                    "timestamp": timestamp,
                    "player_id": current_owner,
                    "team_id": current_player.get("team"),
                    "pitch_x": current_player.get("pitch_x"),
                    "pitch_y": current_player.get("pitch_y"),
                    "related_player_id": previous_owner,
                    "details": {
                        "has_ball": True
                    }
                })

                event_counter += 1

                # -------------------------
                # Possession change event
                # -------------------------

                if previous_owner is not None:

                    events.append({
                        "event_id": f"EVT-{event_counter:06d}",
                        "event_type": "possession_change",
                        "frame": frame_num,
                        "timestamp": timestamp,
                        "player_id": current_owner,
                        "team_id": current_player.get("team"),
                        "pitch_x": current_player.get("pitch_x"),
                        "pitch_y": current_player.get("pitch_y"),
                        "related_player_id": previous_owner,
                        "details": {
                            "previous_player_id": previous_owner,
                            "new_player_id": current_owner
                        }
                    })

                    event_counter += 1

                previous_owner = current_owner

        return events
