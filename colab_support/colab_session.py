"""Latest-release Colab UI. Pure helpers remain testable without Colab or inference."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import uuid


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mts"}
OUTPUT_FOLDERS = {"colab_outputs", "output_videos", "results", "backups", ".trash", ".trash-1000"}


def discover_videos(video_root):
    root = Path(video_root).resolve()
    if not root.is_dir():
        return []
    found = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if (path.suffix.lower() in VIDEO_SUFFIXES and path.is_file() and not path.is_symlink()
                and not any(p.lower() in OUTPUT_FOLDERS or p.startswith(".") for p in relative.parts)
                and root in path.resolve().parents):
            found.append(path)
    return sorted(found, key=lambda p: p.relative_to(root).as_posix().lower())


def source_identity(video, video_root):
    path, root = Path(video).resolve(), Path(video_root).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("Select a file inside Video_Input")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError("Video changed while reading. Wait for Drive synchronization and retry.")
    return {"relative_path": path.relative_to(root).as_posix(), "bytes": after.st_size,
            "sha256": digest.hexdigest()}


def stable_key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def calibration_path(drive_root, identity):
    return Path(drive_root) / "colab_settings" / "field_calibrations" / (stable_key(identity) + ".json")


def valid_polygon(polygon):
    return (isinstance(polygon, list) and len(polygon) >= 3
            and all(isinstance(p, (list, tuple)) and len(p) == 2
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            and math.isfinite(v) and 0 <= v <= 1 for v in p) for p in polygon))


def load_configuration(drive_root, identity, enable_ocr=True):
    config = {"filter_field": False, "field_keyframes": {}, "enable_ocr": bool(enable_ocr),
              "suppress_duplicates": True}
    path = calibration_path(drive_root, identity)
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        keyframes = saved.get("keyframes", {saved.get("frame", 0): saved.get("polygon")})
        if (saved.get("source") != identity or not isinstance(keyframes, dict) or not keyframes
                or any(not str(frame).isdigit() or not valid_polygon(polygon)
                       for frame, polygon in keyframes.items())):
            raise ValueError("Saved field calibration is invalid for this video; mark the field again.")
        config.update(filter_field=True, field_keyframes={int(frame): polygon for frame, polygon in keyframes.items()})
    return config


def review_key(identity, release, config, sample=None):
    return stable_key({"source": identity, "release_id": release.get("release_id"),
                       "zip_sha256": release.get("zip", {}).get("sha256"), "configuration": config,
                       "sample": sample or {"start_s": 0, "duration_s": 2}})


def full_run_allowed(result, saved_key, current_key, approved, notes):
    required_outputs = {"annotated_video", "tracking_json", "tracking_csv", "events_json"}
    return (approved is True and bool(str(notes).strip()) and bool(saved_key) and saved_key == current_key
            and isinstance(result, dict) and result.get("status") == "technical_checks_passed_review_required"
            and set(result.get("runs", {})) == {"baseline", "improved"}
            and all(run.get("status") == "ok"
                    and str(run.get("inference_device", "")).startswith("cuda")
                    and len(run.get("chunk_starts", [])) >= 2
                    and required_outputs.issubset(run.get("validation", {}))
                    and all(isinstance(check, dict) and check.get("ok") is True
                            for check in run["validation"].values())
                    for run in result["runs"].values()))


def short_chunk_size(info, duration_s=2):
    frames = min(int(info["frames"]), round(float(info["fps"]) * duration_s))
    if frames < 10:
        raise ValueError("Select a video with at least ten frames for two consecutive chunks.")
    return min(30, max(5, (frames // 2 // 5) * 5))


def _save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def _pick_polygon(frame, title):
    import base64
    import cv2
    from google.colab import output
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Could not encode the first video frame")
    script = r'''new Promise(resolve => {
      const root=document.createElement('div');
      const title=document.createElement('p');title.textContent=TITLE;root.appendChild(title);
      const canvas=document.createElement('canvas');canvas.style.maxWidth='100%';canvas.style.cursor='crosshair';root.appendChild(canvas);
      const undo=document.createElement('button');undo.textContent='Undo';root.appendChild(undo);
      const save=document.createElement('button');save.textContent='Save field';root.appendChild(save);
      const cancel=document.createElement('button');cancel.textContent='Cancel';root.appendChild(cancel);
      const status=document.createElement('p');root.appendChild(status);document.body.appendChild(root);
      const img=new Image();let points=[];
      function draw(){const c=canvas.getContext('2d');c.drawImage(img,0,0,canvas.width,canvas.height);
        c.strokeStyle='#ffff00';c.fillStyle='#ffff00';c.lineWidth=3;c.beginPath();
        points.forEach((p,i)=>{if(i)c.lineTo(p[0]*canvas.width,p[1]*canvas.height);else c.moveTo(p[0]*canvas.width,p[1]*canvas.height);});
        if(points.length>2)c.closePath();c.stroke();
        points.forEach(p=>{c.beginPath();c.arc(p[0]*canvas.width,p[1]*canvas.height,4,0,Math.PI*2);c.fill();});
        status.textContent=points.length+' points. Outline the visible pitch in order, including its image-edge continuation.';}
      img.onload=()=>{canvas.width=Math.min(1000,img.width);canvas.height=Math.round(img.height*canvas.width/img.width);draw();};
      img.src=IMAGE_URL;
      canvas.onclick=e=>{const r=canvas.getBoundingClientRect();points.push([(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]);draw();};
      undo.onclick=()=>{points.pop();draw();};
      save.onclick=()=>{if(points.length<3)return;save.disabled=true;cancel.disabled=true;resolve(points);};
      cancel.onclick=()=>{save.disabled=true;cancel.disabled=true;resolve(null);};
    })'''
    data = "data:image/jpeg;base64," + base64.b64encode(encoded).decode()
    return output.eval_js(script.replace("TITLE", json.dumps(title)).replace("IMAGE_URL", json.dumps(data)))


def launch(project_dir, drive_project_dir, session_root, release_manifest):
    """Display one selected-video workflow; never start inference automatically."""
    import ipywidgets as widgets
    from IPython.display import Image, display
    from google.colab import output
    import colab_helpers as ch
    ch.prepare_project(project_dir)
    ch.install_import_stubs()
    from short_validation import run_short_comparison
    import chunked_pipeline as cp

    output.enable_custom_widget_manager()
    drive_root, local_root = Path(drive_project_dir), Path(session_root)
    video_root = drive_root / "Video_Input"
    outputs = drive_root / "colab_outputs" / "player_validation"
    selection = widgets.Dropdown(description="Video:", layout=widgets.Layout(width="95%"))
    refresh = widgets.Button(description="Refresh videos")
    ocr = widgets.Checkbox(value=True, description="Read visible jersey numbers", indent=False)
    start = widgets.BoundedFloatText(value=0, min=0, max=1000000, step=0.5, description="Start (s):")
    duration = widgets.BoundedFloatText(value=2, min=2, max=5, step=0.5, description="Length (s):")
    calibrate = widgets.Button(description="Optional: mark field", layout=widgets.Layout(width="190px"))
    short = widgets.Button(description="Run short analysis", button_style="primary", layout=widgets.Layout(width="190px"))
    suite = widgets.Button(description="Compare all 5 videos", layout=widgets.Layout(width="190px"))
    reviewed = widgets.Checkbox(value=False, description="I reviewed the short comparison", disabled=True, indent=False)
    notes = widgets.Textarea(description="Review notes:", placeholder="Record false positives, missed players and incorrect exclusions, or none observed.",
                             layout=widgets.Layout(width="95%"), disabled=True)
    full = widgets.Button(description="Run full video", button_style="warning", disabled=True)
    log = widgets.Output()
    state = {"result": None, "key": None, "directory": None, "busy": False}

    def invalidate(change=None):
        state.update(result=None, key=None, directory=None)
        reviewed.value = False
        reviewed.disabled = True
        notes.value = ""
        notes.disabled = True
        full.disabled = True

    def refresh_videos(button=None):
        found = discover_videos(video_root)
        selection.options = [(p.relative_to(video_root).as_posix(), str(p)) for p in found]
        short.disabled = calibrate.disabled = suite.disabled = not found
        invalidate()
        if not found:
            with log:
                print("No videos visible yet. Put videos in", video_root, "and click Refresh videos after Drive sync.")

    def approval_changed(change=None):
        full.disabled = state["busy"] or not (state["result"] and reviewed.value and notes.value.strip())

    def busy(value):
        state["busy"] = value
        for control in (selection, refresh, ocr, start, duration, calibrate, short, suite):
            control.disabled = value or (not selection.options and control in (calibrate, short, suite))
        reviewed.disabled = value or state["result"] is None
        notes.disabled = value or state["result"] is None
        approval_changed()

    def selected_configuration():
        if not selection.value:
            raise ValueError("Select a video first")
        path = Path(selection.value)
        print("Checking video identity and saved field settings...")
        identity = source_identity(path, video_root)
        config = load_configuration(drive_root, identity, ocr.value)
        if config["filter_field"]:
            from field_filter import FieldFilter
            for source_frame, polygon in config["field_keyframes"].items():
                if not FieldFilter({"field_keyframes": {0: polygon}}).update(ch.read_frame(path, source_frame), 0)["filtering_active"]:
                    raise ValueError("Saved field polygon is invalid. Use Optional: mark field.")
            print("Saved field calibration enabled; filtering stops when camera motion becomes uncertain or calibration expires.")
        else:
            print("Field filtering OFF: no saved boundary. Spectator-filter quality is unknown. Optional: mark field to enable it.")
        return path, identity, config

    def sample_settings():
        return {"start_s": float(start.value), "duration_s": float(duration.value)}

    def prepare_ocr():
        if ocr.value:
            import easyocr
            import torch
            print("Preparing jersey OCR once (downloads recognition weights if needed)...")
            reader = easyocr.Reader(["en"], gpu=True)
            del reader
            torch.cuda.empty_cache()

    def mark_field(button):
        busy(True)
        with log:
            try:
                identity = source_identity(selection.value, video_root)
                source_frame = round(start.value * ch.video_info(selection.value)["fps"])
                frame = ch.read_frame(selection.value, source_frame)
                polygon = _pick_polygon(frame, f"Mark source frame {source_frame}; include boundary players, goalkeepers and referees.")
                if polygon is None:
                    print("Field marking cancelled.")
                    return
                from field_filter import FieldFilter
                if not valid_polygon(polygon) or not FieldFilter({"field_keyframes": {0: polygon}}).update(frame, 0)["filtering_active"]:
                    raise ValueError("Use a convex polygon with vertices in order. Field calibration was not saved.")
                if source_identity(selection.value, video_root) != identity:
                    raise RuntimeError("Video changed; field calibration was not saved.")
                path = calibration_path(drive_root, identity)
                keyframes = load_configuration(drive_root, identity, ocr.value)["field_keyframes"] if path.exists() else {}
                keyframes[source_frame] = polygon
                if path.exists():
                    shutil.copy2(path, path.with_name(path.stem + ".backup_" + uuid.uuid4().hex[:12] + ".json"))
                _save_json(path, {"source": identity, "keyframes": keyframes})
                invalidate()
                print(f"Field saved for this exact video at source frame {source_frame}; reused only from reviewed keyframes.")
            except Exception as exc:
                print("Field marking failed:", exc)
            finally:
                busy(False)

    def run_short(button):
        invalidate()
        busy(True)
        with log:
            log.clear_output(wait=True)
            local_dir = destination = None
            try:
                path, identity, config = selected_configuration()
                info = ch.video_info(path)
                sample = sample_settings()
                remaining = info["duration_s"] - sample["start_s"]
                if remaining <= 0:
                    raise ValueError("Start must be inside the selected video.")
                chunk_size = short_chunk_size(info, min(sample["duration_s"], remaining))
                prepare_ocr()
                label = ch.safe_stem(path) + "_short_" + uuid.uuid4().hex[:12]
                local_dir, destination = local_root / "comparisons" / label, outputs / label
                print(f"Running from {sample['start_s']:g}s for up to {sample['duration_s']:g}s, baseline then improved, with at least two consecutive chunks.")
                result = run_short_comparison(path, local_dir, config, chunk_size=chunk_size,
                                              **sample, require_gpu=True, min_available_gb=2.0)
                if source_identity(path, video_root) != identity:
                    raise RuntimeError("Video changed during validation. Wait for synchronization and rerun.")
                key = review_key(identity, release_manifest, config, sample)
                _save_json(local_dir / "session_release.json", {"release": release_manifest, "source": identity,
                           "configuration": config, "sample": sample, "review_key": key})
                state.update(result=result, key=key, directory=destination)
                print("Short technical checks passed. Accuracy still requires your review.")
                display(Image(filename=str(local_dir / "review" / "contact_sheet.jpg"), width=1100))
                print("Red boxes are rule exclusions, not verified spectators. Review crossing players, edges, goalkeepers and referees.")
                print("Diagnostic counts:", json.dumps({k: result[k] for k in ("detections", "field_filter", "ocr")}, indent=2))
                print("Teams:", result["improved"]["team_observations"], "Unknown:", result["improved"]["unknown_team_observations"])
                print("Numbers:", result["improved"]["jersey_number_observations"], "Unknown:", result["improved"]["unknown_jersey_number_observations"])
                print("Review contact_sheet.jpg, baseline_vs_improved.avi and manual_review.csv in the saved folder.")
            except Exception as exc:
                invalidate()
                print("Short analysis stopped:", exc)
            finally:
                try:
                    if local_dir is not None and local_dir.exists():
                        shutil.copytree(local_dir, destination, dirs_exist_ok=False)
                        print("Saved results:", destination)
                except Exception as exc:
                    invalidate()
                    print("Drive save failed; local results are preserved at", local_dir, "Error:", exc)
                busy(False)

    def run_full(button):
        busy(True)
        with log:
            local_dir = destination = None
            try:
                path, identity, config = selected_configuration()
                key = review_key(identity, release_manifest, config, sample_settings())
                if not full_run_allowed(state["result"], state["key"], key, reviewed.value, notes.value):
                    invalidate()
                    raise RuntimeError("Run and review a short GPU comparison for this exact video, release and configuration first.")
                approval = {"approved": True, "review_key": key, "notes": notes.value,
                            "review_directory": str(state["directory"]), "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
                _save_json(state["directory"] / ("human_review_" + uuid.uuid4().hex[:12] + ".json"), approval)
                label = ch.safe_stem(path) + "_full_" + uuid.uuid4().hex[:12]
                local_dir, destination = local_root / "full_runs" / label, outputs / label
                print("Full run requested after short review. Field propagation expires after 120 frames without later calibrated keyframes.")
                result = cp.run_chunked(path, local_dir, chunk_size=30, require_gpu=True,
                                        min_available_gb=2.0, improvements=config)
                _save_json(local_dir / "session_release.json", {"release": release_manifest, "source": identity,
                           "configuration": config, "approval": approval})
                if result.get("status") != "ok":
                    raise RuntimeError(result.get("error", "Full run failed"))
                valid, checks = ch.validate_outputs(result["outputs"], result["frames"])
                if not valid:
                    raise RuntimeError("Full output verification failed: " + str(checks))
                print("Full run technical checks passed.")
            except Exception as exc:
                print("Full analysis stopped:", exc)
            finally:
                try:
                    if local_dir is not None and local_dir.exists():
                        shutil.copytree(local_dir, destination, dirs_exist_ok=False)
                        print("Saved results:", destination)
                except Exception as exc:
                    print("Drive save failed; local results are preserved at", local_dir, "Error:", exc)
                busy(False)

    def run_suite(button):
        from comparison_suite import run_short_suite
        invalidate()
        busy(True)
        with log:
            local_dir = destination = None
            try:
                label = "five_video_short_" + uuid.uuid4().hex[:12]
                local_dir, destination = local_root / "comparisons" / label, outputs / label
                def configuration_for_video(path):
                    identity = source_identity(path, video_root)
                    return load_configuration(drive_root, identity, ocr.value)
                prepare_ocr()
                print("Comparing opening, middle and late short clips for all five videos with one baseline.")
                result = run_short_suite(discover_videos(video_root), local_dir, configuration_for_video,
                                         duration_s=duration.value, require_gpu=True, min_available_gb=2.0,
                                         expected_videos=5)
                _save_json(local_dir / "session_release.json", {"release": release_manifest})
                print("Suite status:", result["status"], "Accuracy and identity continuity still require labeled review.")
            except Exception as exc:
                print("Comparison suite stopped:", exc)
            finally:
                try:
                    if local_dir is not None and local_dir.exists():
                        shutil.copytree(local_dir, destination, dirs_exist_ok=False)
                        print("Saved results:", destination)
                except Exception as exc:
                    print("Drive save failed; local results are preserved at", local_dir, "Error:", exc)
                busy(False)

    selection.observe(invalidate, names="value")
    ocr.observe(invalidate, names="value")
    start.observe(invalidate, names="value")
    duration.observe(invalidate, names="value")
    reviewed.observe(approval_changed, names="value")
    notes.observe(approval_changed, names="value")
    refresh.on_click(refresh_videos)
    calibrate.on_click(mark_field)
    short.on_click(run_short)
    suite.on_click(run_suite)
    full.on_click(run_full)
    refresh_videos()
    display(widgets.VBox([selection, widgets.HBox([refresh, calibrate]), ocr, widgets.HBox([start, duration]),
                         widgets.HBox([short, suite]),
                         widgets.HTML("<p>Full analysis stays off until you review the short comparison and record notes.</p>"),
                         reviewed, notes, full, log]))
    print("Loaded release:", release_manifest["release_id"], "| Videos:", video_root)
