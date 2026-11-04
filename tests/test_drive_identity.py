"""Offline identity checks using a realistic, read-only SQLite metadata fixture."""
from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import drive_identity as identity


class DriveIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.mount = self.root / "fake_drive"
        self.destination = (self.mount / ".shortcut-targets-by-id" / identity.PROJECT_FOLDER_ID
                            / identity.PROJECT_FOLDER_TITLE / identity.COLAB_FOLDER_TITLE)
        self.destination.mkdir(parents=True)
        (self.destination / identity.NOTEBOOK_NAME).write_text("existing notebook", encoding="utf-8")
        self.database = self.root / "metadata_sqlite_db"
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.executescript(
                "CREATE TABLE items(stable_id INTEGER, id TEXT, local_title TEXT, "
                "is_folder INTEGER, trashed INTEGER, is_tombstone INTEGER);"
                "CREATE TABLE stable_parents(item_stable_id INTEGER,parent_stable_id INTEGER);")
            connection.executemany("INSERT INTO items VALUES(?,?,?,?,0,0)", [
                (1, identity.PROJECT_FOLDER_ID, identity.PROJECT_FOLDER_TITLE, 1),
                (2, identity.COLAB_FOLDER_ID, identity.COLAB_FOLDER_TITLE, 1),
                (3, identity.NOTEBOOK_ID, identity.NOTEBOOK_NAME, 0),
            ])
            connection.executemany("INSERT INTO stable_parents VALUES(?,?)", [(2, 1), (3, 2)])
        self.config = {**identity.PINNED_IDS, "drive_mount": str(self.mount),
                       "drive_project": str(self.destination), "drive_metadata_db": str(self.database)}
        self.mount_patch = patch.object(identity, "DRIVE_MOUNT", self.mount)
        self.mount_patch.start()

    def tearDown(self):
        self.mount_patch.stop()
        self.temp.cleanup()

    def sql(self, statement, parameters=()):
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute(statement, parameters)

    def verify(self, config=None, destination=None):
        return identity.verify_local_drive_binding(config or self.config, destination)

    def test_valid_binding_without_root_in_cache_is_read_only_and_not_cloud_proof(self):
        before = hashlib.sha256(self.database.read_bytes()).hexdigest()
        binding = self.verify(destination=self.destination)
        self.assertEqual({key: binding[key] for key in identity.PINNED_IDS}, identity.PINNED_IDS)
        self.assertTrue(binding["identity_verified_locally"])
        self.assertEqual(binding["cloud_sync_status"], "pending_or_unknown")
        self.assertEqual(before, hashlib.sha256(self.database.read_bytes()).hexdigest())
        self.assertEqual(sorted(path.name for path in self.root.iterdir()),
                         ["fake_drive", "metadata_sqlite_db"])

    def test_every_configured_id_is_pinned(self):
        for key in identity.PINNED_IDS:
            for bad in (None, "same-name-other-id"):
                with self.subTest(key=key, bad=bad):
                    config = {**self.config, key: bad}
                    with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                        self.verify(config)

    def test_live_duplicate_id_is_rejected(self):
        self.sql("INSERT INTO items SELECT 4,id,local_title,is_folder,0,0 FROM items WHERE stable_id=2")
        with self.assertRaisesRegex(RuntimeError, "missing or ambiguous"):
            self.verify()

    def test_same_name_other_id_under_pinned_parent_is_rejected(self):
        self.sql("INSERT INTO items VALUES(4,'another-id',?,1,0,0)", (identity.COLAB_FOLDER_TITLE,))
        self.sql("INSERT INTO stable_parents VALUES(4,1)")
        with self.assertRaisesRegex(RuntimeError, "local name is missing or ambiguous"):
            self.verify()

    def test_changed_colab_parent_is_rejected(self):
        self.sql("UPDATE stable_parents SET parent_stable_id=99 WHERE item_stable_id=2")
        with self.assertRaisesRegex(RuntimeError, "parent changed"):
            self.verify()

    def test_changed_notebook_parent_is_rejected(self):
        self.sql("UPDATE stable_parents SET parent_stable_id=1 WHERE item_stable_id=3")
        with self.assertRaisesRegex(RuntimeError, "parent changed"):
            self.verify()

    def test_extra_parent_is_rejected(self):
        self.sql("INSERT INTO stable_parents VALUES(2,99)")
        with self.assertRaisesRegex(RuntimeError, "parent changed"):
            self.verify()

    def test_each_item_title_change_is_rejected(self):
        for stable in (1, 2, 3):
            with self.subTest(stable_id=stable):
                with closing(sqlite3.connect(self.database)) as connection:
                    old = connection.execute("SELECT local_title FROM items WHERE stable_id=?", (stable,)).fetchone()[0]
                self.sql("UPDATE items SET local_title='Renamed' WHERE stable_id=?", (stable,))
                with self.assertRaisesRegex(RuntimeError, "name/type changed"):
                    self.verify()
                self.sql("UPDATE items SET local_title=? WHERE stable_id=?", (old, stable))

    def test_trashed_or_tombstoned_item_is_rejected(self):
        for flag in ("trashed", "is_tombstone"):
            with self.subTest(flag=flag):
                self.sql(f"UPDATE items SET {flag}=1 WHERE stable_id=3")
                with self.assertRaisesRegex(RuntimeError, "missing or ambiguous"):
                    self.verify()
                self.sql(f"UPDATE items SET {flag}=0 WHERE stable_id=3")

    def test_wrong_folder_type_is_rejected(self):
        self.sql("UPDATE items SET is_folder=1 WHERE stable_id=3")
        with self.assertRaisesRegex(RuntimeError, "name/type changed"):
            self.verify()

    def test_missing_or_unreadable_metadata_is_rejected_without_creating_db(self):
        missing = self.root / "missing_db"
        with self.assertRaisesRegex(RuntimeError, "metadata is unavailable"):
            self.verify({**self.config, "drive_metadata_db": str(missing)})
        self.assertFalse(missing.exists())
        corrupt = self.root / "corrupt_db"
        corrupt.write_text("not SQLite", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "metadata is unreadable"):
            self.verify({**self.config, "drive_metadata_db": str(corrupt)})

    def test_old_my_drive_path_and_destination_override_are_rejected(self):
        old = self.mount / "My Drive" / identity.COLAB_FOLDER_TITLE
        old.mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "Deployment path"):
            self.verify({**self.config, "drive_project": str(old)})
        with self.assertRaisesRegex(RuntimeError, "destination differs"):
            self.verify(destination=old)

    def test_unapproved_mount_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Drive mount"):
            self.verify({**self.config, "drive_mount": str(self.root / "other")})

    def test_missing_existing_notebook_is_rejected(self):
        (self.destination / identity.NOTEBOOK_NAME).unlink()
        with self.assertRaisesRegex(RuntimeError, "pinned notebook is unavailable"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
