"""Offline release checks: tiny temporary artifacts, never Drive or YOLO."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import deploy_colab as deploy
from drive_identity import APPROVED_BINDING


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="soccer_deploy_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prep = self.root / "prep"
        self.drive = self.root / "drive"
        self.prep.mkdir()
        self.drive.mkdir()
        self.files = {"colab_support/example.py": b"VALUE = 1\n"}
        self.expected = {name: hashlib.sha256(data).hexdigest()
                         for name, data in self.files.items()}
        self.write_zip()
        self.notebook = {"cells": [{"cell_type": "code", "source": ["VALUE = 1\n"],
                                    "metadata": {}, "outputs": [], "execution_count": None}],
                         "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        (self.prep / deploy.NOTEBOOK_NAME).write_text(json.dumps(self.notebook), encoding="utf-8")
        self.checks = {"successful": True, "tests_run": 1, "errors": 0, "failures": 0}
        self.binding = dict(APPROVED_BINDING)
        self.guard = mock.patch.object(deploy, "verify_local_drive_binding", return_value=self.binding).start()
        self.addCleanup(mock.patch.stopall)

    def write_zip(self, files=None, content_overrides=None, extras=None):
        files = self.files if files is None else files
        manifest = {"built_at": "2026-10-23T12:00:00", "files": {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in files.items()}}
        with zipfile.ZipFile(self.prep / deploy.ZIP_NAME, "w") as archive:
            for name, data in files.items():
                archive.writestr("SoccerVideoAnalysisDemo/" + name,
                                 (content_overrides or {}).get(name, data))
            archive.writestr("SoccerVideoAnalysisDemo/COLAB_MANIFEST.json", json.dumps(manifest))
            for name, data in (extras or {}).items():
                archive.writestr(name, data)
        return manifest

    def prior_release(self):
        previous = {deploy.ZIP_NAME: b"previous working zip", deploy.NOTEBOOK_NAME: b"previous notebook",
                    deploy.MANIFEST_NAME: b'{"release_id":"previous"}\n'}
        for name, content in previous.items():
            (self.drive / name).write_bytes(content)
        return previous

    def test_failed_checks_create_no_copy_or_backups(self):
        previous = self.prior_release()
        for checks in ({"successful": False}, {**self.checks, "errors": 1},
                       {**self.checks, "failures": 1}):
            with self.subTest(checks=checks), self.assertRaises(RuntimeError):
                deploy.publish(self.prep, self.drive, checks, self.expected)
        self.assertEqual(set(previous), {path.name for path in self.drive.iterdir()})
        for name, content in previous.items():
            self.assertEqual(content, (self.drive / name).read_bytes())

    def test_identity_guard_rejects_before_any_destination_write(self):
        previous = self.prior_release()
        self.guard.side_effect = ValueError("Unapproved Drive identity")
        with self.assertRaisesRegex(ValueError, "Unapproved"):
            deploy.publish(self.prep, self.drive, self.checks, self.expected)
        self.assertEqual(set(previous), {path.name for path in self.drive.iterdir()})
        for name, content in previous.items():
            self.assertEqual(content, (self.drive / name).read_bytes())

    def test_manifest_binds_existing_approved_notebook_and_folder_ids(self):
        result = deploy.publish(self.prep, self.drive, self.checks, self.expected)
        marker = json.loads((self.drive / deploy.MANIFEST_NAME).read_text())
        self.assertEqual(marker["drive_binding"], self.binding)
        self.assertEqual(result["drive_binding"], self.binding)
        self.assertGreaterEqual(self.guard.call_count, 2)
        self.assertEqual(self.guard.call_args.args[1], self.drive.resolve())

    def test_identity_change_before_marker_publication_stops_release(self):
        previous = self.prior_release()
        self.guard.side_effect = [self.binding, {**self.binding, "notebook_id": "different"}]
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            deploy.publish(self.prep, self.drive, self.checks, self.expected)
        self.assertEqual(previous[deploy.MANIFEST_NAME], (self.drive / deploy.MANIFEST_NAME).read_bytes())

    def test_zip_expected_source_fingerprint_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "source files"):
            deploy.verify_zip(self.prep / deploy.ZIP_NAME, {"different.py": "0" * 64})

    def test_zip_tampered_content_rejected(self):
        self.write_zip(content_overrides={"colab_support/example.py": b"VALUE = 9\n"})
        with self.assertRaisesRegex(ValueError, "content mismatch"):
            deploy.verify_zip(self.prep / deploy.ZIP_NAME, self.expected)

    def test_zip_unmanifested_and_unsafe_paths_rejected(self):
        for name in ("SoccerVideoAnalysisDemo/unlisted.py", "../escape.py", "/absolute.py",
                     "C:/escape.py", "SoccerVideoAnalysisDemo/..\\escape.py"):
            with self.subTest(name=name):
                self.write_zip(extras={name: b"VALUE = 9\n"})
                with self.assertRaises(ValueError):
                    deploy.verify_zip(self.prep / deploy.ZIP_NAME, self.expected)

    def test_zip_empty_output_placeholders_allowed_but_data_rejected(self):
        self.write_zip(extras={"SoccerVideoAnalysisDemo/CSV_data/.keep": b""})
        deploy.verify_zip(self.prep / deploy.ZIP_NAME, self.expected)
        self.write_zip(extras={"SoccerVideoAnalysisDemo/CSV_data/.keep": b"unexpected data"})
        with self.assertRaisesRegex(ValueError, "Unmanifested"):
            deploy.verify_zip(self.prep / deploy.ZIP_NAME, self.expected)

    def test_backup_preserves_previous_files_and_never_touches_results(self):
        previous = self.prior_release()
        results = self.drive / "colab_outputs" / "existing_run" / "events.json"
        results.parent.mkdir(parents=True)
        results.write_bytes(b'{"keep":"previous results"}\n')
        result = deploy.publish(self.prep, self.drive, self.checks, self.expected)
        backup = Path(result["backup"])
        for name, content in previous.items():
            self.assertEqual(content, (backup / name).read_bytes())
            self.assertEqual(hashlib.sha256(content).hexdigest(), result["backup_hashes"][name])
        self.assertEqual(b'{"keep":"previous results"}\n', results.read_bytes())
        self.assertEqual(deploy.sha256(self.prep / deploy.ZIP_NAME),
                         deploy.sha256(self.drive / deploy.ZIP_NAME))
        self.assertEqual(deploy.sha256(self.prep / deploy.NOTEBOOK_NAME),
                         deploy.sha256(self.drive / deploy.NOTEBOOK_NAME))
        self.assertTrue(result["local_deployment_verified"])
        self.assertEqual("pending_or_unknown", result["cloud_sync_status"])

    def test_release_marker_written_only_after_destination_hashes_match(self):
        previous = self.prior_release()
        original_replace = os.replace
        events = []

        def checked_replace(source, destination):
            destination = Path(destination)
            events.append(destination.name)
            if destination.name == deploy.MANIFEST_NAME:
                self.assertEqual(previous[deploy.MANIFEST_NAME], destination.read_bytes())
                marker = json.loads(Path(source).read_text(encoding="utf-8"))
                self.assertEqual(marker["zip"]["sha256"], deploy.sha256(self.drive / deploy.ZIP_NAME))
                self.assertEqual(marker["notebook"]["sha256"], deploy.sha256(self.drive / deploy.NOTEBOOK_NAME))
                self.assertEqual(marker["zip"]["bytes"], (self.drive / deploy.ZIP_NAME).stat().st_size)
            return original_replace(source, destination)

        with mock.patch.object(deploy.os, "replace", side_effect=checked_replace):
            deploy.publish(self.prep, self.drive, self.checks, self.expected)
        self.assertEqual(deploy.MANIFEST_NAME, events[-1])

    def test_copy_failure_leaves_previous_marker_and_backup(self):
        previous = self.prior_release()
        original_copy = deploy.shutil.copy2

        def fail_upload(source, destination, *args, **kwargs):
            if Path(source) == self.prep / deploy.ZIP_NAME:
                raise OSError("Simulated Drive unavailable during copy")
            return original_copy(source, destination, *args, **kwargs)

        with mock.patch.object(deploy.shutil, "copy2", side_effect=fail_upload), self.assertRaises(OSError):
            deploy.publish(self.prep, self.drive, self.checks, self.expected)
        for name, content in previous.items():
            self.assertEqual(content, (self.drive / name).read_bytes())
        backups = list((self.drive / "deployment_backups").iterdir())
        self.assertEqual(1, len(backups))
        for name, content in previous.items():
            self.assertEqual(content, (backups[0] / name).read_bytes())
        self.assertFalse((self.prep / "deployment_report.json").exists())

    def test_destination_hash_failure_does_not_publish_new_marker(self):
        previous = self.prior_release()
        original_hash = deploy.sha256
        original_replace = os.replace
        replaced = []

        def record_replace(source, destination):
            result = original_replace(source, destination)
            replaced.append(Path(destination))
            return result

        def wrong_destination_hash(path):
            if Path(path) == self.drive / deploy.ZIP_NAME and self.drive / deploy.ZIP_NAME in replaced:
                return "0" * 64
            return original_hash(path)

        with mock.patch.object(deploy.os, "replace", side_effect=record_replace), \
                mock.patch.object(deploy, "sha256", side_effect=wrong_destination_hash), \
                self.assertRaisesRegex(RuntimeError, "hash check failed"):
            deploy.publish(self.prep, self.drive, self.checks, self.expected)
        self.assertEqual(previous[deploy.MANIFEST_NAME], (self.drive / deploy.MANIFEST_NAME).read_bytes())

    def test_unchanged_permanent_notebook_not_rewritten_next_release(self):
        first = deploy.publish(self.prep, self.drive, self.checks, self.expected)
        notebook = self.drive / deploy.NOTEBOOK_NAME
        os.utime(notebook, ns=(1_000_000_000_000, 1_000_000_000_000))
        before = notebook.stat()
        second = deploy.publish(self.prep, self.drive, self.checks, self.expected)
        self.assertFalse(second["notebook_updated"])
        self.assertEqual(before.st_mtime_ns, notebook.stat().st_mtime_ns)
        self.assertNotEqual(first["backup"], second["backup"])
        self.assertTrue(Path(first["backup"]).is_dir())
        self.assertTrue(Path(second["backup"]).is_dir())

    def test_preserve_saved_notebook_outputs_when_launcher_unchanged(self):
        saved = json.loads(json.dumps(self.notebook))
        saved["cells"][0]["execution_count"] = 4
        saved["cells"][0]["outputs"] = [{"output_type": "stream", "name": "stdout",
                                         "text": ["Saved analysis session\n"]}]
        saved["cells"][0]["metadata"] = {"executionInfo": {"status": "ok"}}
        saved["metadata"] = {"colab": {"provenance": [], "name": "Soccer analysis"}}
        notebook = self.drive / deploy.NOTEBOOK_NAME
        saved_bytes = (json.dumps(saved, indent=2) + "\n").encode("utf-8")
        notebook.write_bytes(saved_bytes)
        os.utime(notebook, ns=(1_000_000_000_000, 1_000_000_000_000))
        before = notebook.stat()
        result = deploy.publish(self.prep, self.drive, self.checks, self.expected)
        marker = json.loads((self.drive / deploy.MANIFEST_NAME).read_text(encoding="utf-8"))
        self.assertFalse(result["notebook_updated"])
        self.assertEqual(saved_bytes, notebook.read_bytes())
        self.assertEqual(before.st_mtime_ns, notebook.stat().st_mtime_ns)
        self.assertEqual(hashlib.sha256(saved_bytes).hexdigest(), marker["notebook"]["sha256"])
        self.assertEqual(deploy.sha256(self.prep / deploy.NOTEBOOK_NAME),
                         marker["notebook"]["template_sha256"])

    def test_changed_sources_rejected(self):
        source = self.prep / "source.py"
        source.write_bytes(b"VALUE = 1\n")
        files = {"source.py": source}
        expected = deploy.fingerprint(files)
        deploy.assert_unchanged(files, expected)
        source.write_bytes(b"VALUE = 2\n")
        with self.assertRaisesRegex(RuntimeError, "Sources changed"):
            deploy.assert_unchanged(files, expected)

    def test_notebook_changed_since_checks_rejected_before_copy(self):
        previous = self.prior_release()
        expected_notebook = deploy.sha256(self.prep / deploy.NOTEBOOK_NAME)
        self.notebook["cells"][0]["source"] = ["VALUE = 2\n"]
        (self.prep / deploy.NOTEBOOK_NAME).write_text(json.dumps(self.notebook), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Notebook changed"):
            deploy.publish(self.prep, self.drive, self.checks, self.expected, expected_notebook)
        self.assertFalse((self.drive / "deployment_backups").exists())
        for name, content in previous.items():
            self.assertEqual(content, (self.drive / name).read_bytes())

    def test_notebook_copy_uses_exact_captured_validated_bytes(self):
        self.prior_release()
        validated_bytes = (self.prep / deploy.NOTEBOOK_NAME).read_bytes()
        expected_notebook = hashlib.sha256(validated_bytes).hexdigest()
        original_copy = deploy.shutil.copy2

        def mutate_during_backup(source, destination, *args, **kwargs):
            if Path(source) == self.drive / deploy.ZIP_NAME:
                (self.prep / deploy.NOTEBOOK_NAME).write_bytes(b"invalid unvalidated bytes")
            return original_copy(source, destination, *args, **kwargs)

        with mock.patch.object(deploy.shutil, "copy2", side_effect=mutate_during_backup):
            result = deploy.publish(self.prep, self.drive, self.checks, self.expected, expected_notebook)
        self.assertEqual(validated_bytes, (self.drive / deploy.NOTEBOOK_NAME).read_bytes())
        self.assertEqual(expected_notebook, result["notebook"]["sha256"])

    def test_source_zip_changed_after_initial_verification_rejected(self):
        previous = self.prior_release()
        original_verify = deploy.verify_zip
        verified = []

        def mutate_after_initial_verify(path, expected=None):
            result = original_verify(path, expected)
            if not verified:
                verified.append(path)
                self.write_zip(files={"colab_support/example.py": b"VALUE = 2\n"})
            return result

        with mock.patch.object(deploy, "verify_zip", side_effect=mutate_after_initial_verify), \
                self.assertRaises((RuntimeError, ValueError)):
            deploy.publish(self.prep, self.drive, self.checks, self.expected)
        self.assertEqual(previous[deploy.MANIFEST_NAME], (self.drive / deploy.MANIFEST_NAME).read_bytes())


if __name__ == "__main__":
    unittest.main()
