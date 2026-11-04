"""Offline launcher integrity and review gates; no GPU, pip, OCR or video inference."""
import hashlib
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "colab_support")]
import make_notebook
from colab_session import (calibration_path, discover_videos, full_run_allowed,
                           load_configuration, review_key, short_chunk_size, source_identity)

bootstrap = {}
exec(compile(make_notebook.BOOTSTRAP_HELPERS, "launcher_helpers", "exec"), bootstrap)


class ColabSessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def archive(self, extra=None, changed=False):
        root = "SoccerVideoAnalysisDemo/"
        files = {name: b"offline fixture" for name in (
            "colab_support/colab_session.py", "colab_support/short_validation.py",
            "colab_support/chunked_pipeline.py", "requirements_colab.txt", "models/best (2).pt")}
        manifest = {"files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                              for name, data in files.items()}}
        path = self.root / "source.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in files.items():
                archive.writestr(root + name, b"corrupted" if changed else data)
            archive.writestr(root + "COLAB_MANIFEST.json", json.dumps(manifest))
            archive.writestr(root + "Video_Input/.keep", "")
            if extra is not None:
                archive.writestr(extra, "")
        return path

    def test_single_launcher_cell_compiles_and_is_deterministic(self):
        notebook = make_notebook.notebook_data()
        self.assertEqual(notebook, make_notebook.notebook_data())
        cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(cells), 1)
        compile("".join(cells[0]["source"]), "launcher", "exec")
        self.assertIn("_SOCCER_SESSION_STARTED", "".join(cells[0]["source"]))

    def test_extract_verifies_and_uses_new_root(self):
        project, manifest = bootstrap["verify_extract"](self.archive(), self.root / "extract")
        self.assertEqual(len(manifest["files"]), 5)
        self.assertTrue((project / "colab_support/colab_session.py").is_file())
        with self.assertRaisesRegex(RuntimeError, "fresh"):
            bootstrap["verify_extract"](self.root / "source.zip", self.root / "extract")

    def test_extract_rejects_corruption_before_writing(self):
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            bootstrap["verify_extract"](self.archive(changed=True), self.root / "extract")
        self.assertFalse((self.root / "extract" / "SoccerVideoAnalysisDemo").exists())

    def test_extract_rejects_unsafe_and_unmanifested_entries(self):
        for unsafe in ("../outside.py", "SoccerVideoAnalysisDemo/../outside.py",
                       "SoccerVideoAnalysisDemo\\outside.py", "C:/outside.py",
                       "SoccerVideoAnalysisDemo/extra.py", "SoccerVideoAnalysisDemo//odd.py"):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                bootstrap["verify_extract"](self.archive(extra=unsafe), self.root / "extract")

    def test_extract_rejects_symlink(self):
        info = zipfile.ZipInfo("SoccerVideoAnalysisDemo/link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaisesRegex(ValueError, "regular"):
            bootstrap["verify_extract"](self.archive(extra=info), self.root / "extract")

    def deployment(self):
        drive, session = self.root / "drive", self.root / "session"
        drive.mkdir()
        session.mkdir()
        archive = drive / "SoccerProject_Colab_Source.zip"
        archive.write_bytes(b"fixture")
        manifest = {"schema_version": 1, "release_id": "test", "drive_binding": make_notebook.APPROVED_BINDING, "zip": {
            "name": archive.name, "bytes": archive.stat().st_size, "sha256": bootstrap["file_sha256"](archive)}}
        (drive / "deployment_manifest.json").write_text(json.dumps(manifest))
        return drive, session, manifest

    def test_release_snapshot_and_sync_mismatch(self):
        drive, session, manifest = self.deployment()
        observed, local = bootstrap["snapshot_release"](drive, session)
        self.assertEqual(observed, manifest)
        self.assertEqual(local.read_bytes(), b"fixture")
        (drive / "SoccerProject_Colab_Source.zip").write_bytes(b"different")
        with self.assertRaisesRegex(RuntimeError, "synchronization may still be pending"):
            bootstrap["snapshot_release"](drive, session)

    def test_launcher_uses_only_approved_id_bound_path(self):
        config = json.loads((ROOT / "deployment_config.json").read_text())
        self.assertEqual(config["colab_project"], make_notebook.APPROVED_COLAB_PROJECT)
        self.assertEqual(bootstrap["APPROVED_BINDING"], make_notebook.APPROVED_BINDING)
        self.assertIn('DRIVE_PROJECT = Path(APPROVED_COLAB_PROJECT)', make_notebook.LAUNCH)
        self.assertNotIn('MyDrive/SoccerProject', make_notebook.LAUNCH)

    def test_release_identity_mismatch_stops_before_copying_code(self):
        for key in make_notebook.APPROVED_BINDING:
            with self.subTest(key=key):
                drive, session, manifest = self.deployment()
                manifest["drive_binding"] = {**manifest["drive_binding"], key: "unapproved"}
                (drive / "deployment_manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, "unapproved"):
                    bootstrap["snapshot_release"](drive, session)
                self.assertFalse((session / "SoccerProject_Colab_Source.zip").exists())
                import shutil
                shutil.rmtree(drive)
                shutil.rmtree(session)

    def test_legacy_unbound_release_is_rejected(self):
        drive, session, manifest = self.deployment()
        del manifest["drive_binding"]
        (drive / "deployment_manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError, "unapproved"):
            bootstrap["snapshot_release"](drive, session)

    def test_release_manifest_change_during_copy_stops_load(self):
        drive, session, _ = self.deployment()
        original = bootstrap["shutil"].copyfile
        def changed(source, destination):
            result = original(source, destination)
            path = drive / "deployment_manifest.json"
            path.write_text(path.read_text() + " ")
            return result
        with patch.object(bootstrap["shutil"], "copyfile", changed):
            with self.assertRaisesRegex(RuntimeError, "synchronization may still be pending"):
                bootstrap["snapshot_release"](drive, session)

    def test_partial_manifest_has_clear_sync_message(self):
        drive, session, _ = self.deployment()
        (drive / "deployment_manifest.json").write_text('{"schema_version":')
        with self.assertRaisesRegex(RuntimeError, "synchronization may still be pending"):
            bootstrap["snapshot_release"](drive, session)

    def test_video_discovery_recursive_excludes_results(self):
        for relative in ("a.mp4", "match/b.MOV", "results/annotated.mp4", "colab_outputs/c.mp4", ".hidden/d.mp4", "readme.txt"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"video")
        self.assertEqual([p.relative_to(self.root).as_posix() for p in discover_videos(self.root)], ["a.mp4", "match/b.MOV"])

    def test_calibration_is_content_and_path_bound_and_off_without_seed(self):
        videos = self.root / "Video_Input"
        videos.mkdir()
        path = videos / "match.mp4"
        path.write_bytes(b"version one")
        identity = source_identity(path, videos)
        self.assertFalse(load_configuration(self.root, identity)["filter_field"])
        calibration = calibration_path(self.root, identity)
        calibration.parent.mkdir(parents=True)
        calibration.write_text(json.dumps({"source": identity, "polygon": [[0, 0], [1, 0], [1, 1]]}))
        self.assertTrue(load_configuration(self.root, identity)["filter_field"])
        path.write_bytes(b"version two")
        changed = source_identity(path, videos)
        self.assertNotEqual(calibration_path(self.root, changed), calibration)
        self.assertFalse(load_configuration(self.root, changed)["filter_field"])

    def test_full_gate_requires_gpu_two_chunks_review_and_same_source_config_release(self):
        identity = {"sha256": "video", "relative_path": "a.mp4"}
        release = {"release_id": "one", "zip": {"sha256": "source"}}
        config = {"enable_ocr": True}
        key = review_key(identity, release, config)
        result = {"status": "technical_checks_passed_review_required", "runs": {
            label: {"inference_device": "cuda:0", "chunk_starts": [0, 30], "status": "ok",
                    "validation": {name: {"ok": True} for name in ("annotated_video", "tracking_json", "tracking_csv", "events_json")}}
            for label in ("baseline", "improved")}}
        self.assertTrue(full_run_allowed(result, key, key, True, "Reviewed"))
        self.assertFalse(full_run_allowed(result, key, key, False, "Reviewed"))
        self.assertFalse(full_run_allowed(result, key, key, True, ""))
        for source, build, conf in (({**identity, "sha256": "other"}, release, config),
                                     (identity, {**release, "release_id": "two"}, config),
                                     (identity, release, {"enable_ocr": False})):
            self.assertFalse(full_run_allowed(result, key, review_key(source, build, conf), True, "Reviewed"))
        result["runs"]["baseline"]["inference_device"] = "cpu"
        self.assertFalse(full_run_allowed(result, key, key, True, "Reviewed"))

    def test_sample_changes_invalidate_review_and_failed_artifacts_block_full_run(self):
        key = review_key({}, {}, {}, {"start_s": 0, "duration_s": 2})
        self.assertNotEqual(key, review_key({}, {}, {}, {"start_s": 3, "duration_s": 2}))
        self.assertNotEqual(key, review_key({}, {}, {}, {"start_s": 0, "duration_s": 5}))
        good = {"status": "ok", "inference_device": "cuda:0", "chunk_starts": [0, 30],
                "validation": {name: {"ok": True} for name in ("annotated_video", "tracking_json", "tracking_csv", "events_json")}}
        for broken in ({**good, "status": "failed"}, {**good, "validation": {}},
                       {**good, "validation": {"annotated_video": {"ok": False}}}):
            result = {"status": "technical_checks_passed_review_required", "runs": {"baseline": good, "improved": broken}}
            self.assertFalse(full_run_allowed(result, key, key, True, "Reviewed"))

    def test_calibration_retains_source_frame_and_multiple_reviewed_keyframes(self):
        identity = {"sha256": "video"}
        path = calibration_path(self.root, identity)
        path.parent.mkdir(parents=True)
        polygon = [[0, 0], [1, 0], [1, 1]]
        path.write_text(json.dumps({"source": identity, "polygon": polygon, "frame": 60}))
        self.assertEqual(load_configuration(self.root, identity)["field_keyframes"], {60: polygon})
        path.write_text(json.dumps({"source": identity, "keyframes": {"60": polygon, "120": polygon}}))
        self.assertEqual(set(load_configuration(self.root, identity)["field_keyframes"]), {60, 120})

    def test_short_chunks_cover_low_and_high_fps(self):
        for fps in (10, 15, 25, 30, 60, 120):
            chunk = short_chunk_size({"frames": 1000, "fps": fps})
            self.assertEqual(chunk % 5, 0)
            self.assertLess(chunk, fps * 2)
            self.assertLessEqual(chunk, 30)
        with self.assertRaises(ValueError):
            short_chunk_size({"frames": 5, "fps": 30})

    def test_disabled_field_filter_allows_basic_validation_without_quality_claim(self):
        from short_validation import validation_failures
        summary = {"ocr": {}, "field_filter": {"calibrated_or_propagated_frames": 0}}
        self.assertEqual(validation_failures(summary, {"filter_field": False}), [])
        self.assertEqual(summary["field_filter"]["spectator_filter_evaluation_status"], "disabled_no_spectator_quality_claim")
        self.assertTrue(validation_failures(summary, {"filter_field": True}))
        self.assertTrue(validation_failures(summary, {"filter_field": False, "enable_ocr": True}))


if __name__ == "__main__":
    unittest.main()
