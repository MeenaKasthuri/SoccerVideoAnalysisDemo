"""Read-only identity checks against Drive Desktop's local metadata cache.

This verifies a local path binding, never cloud synchronization or cloud ancestry.
The approved root's ancestry above PROJECT_FOLDER_ID is checked separately via
Drive. The root is intentionally not required to exist in the Desktop cache.
No credentials, tokens, network calls, or metadata writes are used here.
"""
from contextlib import closing
from pathlib import Path
import sqlite3


APPROVED_ROOT_ID = "18jlQkG3zSjj-Eogo-uJ4gcuZxX6buBl4"
PROJECT_FOLDER_ID = "1k9LxrKNBogsE_Z-QOqmH6ncecI6nYJ-o"
COLAB_FOLDER_ID = "1KC_ybaf_aWRt_4UTt0rc7bbH_-Nty7CC"
NOTEBOOK_ID = "1gTBP7t7mw70pOLoTH6XV-RofPR5LA44E"
PROJECT_FOLDER_TITLE = "Soccer_Analysis_Meena_Kasthuri"
COLAB_FOLDER_TITLE = "SoccerProject"
NOTEBOOK_NAME = "Soccer_Analytics_Colab.ipynb"
DRIVE_MOUNT = Path("G:\\")
APPROVED_BINDING = {
    "approved_root_id": APPROVED_ROOT_ID,
    "project_folder_id": PROJECT_FOLDER_ID,
    "colab_folder_id": COLAB_FOLDER_ID,
    "notebook_id": NOTEBOOK_ID,
}
PINNED_IDS = APPROVED_BINDING


def validate_config_binding(config):
    """Validate the four explicit IDs without touching Drive or its cache."""
    for key, expected in APPROVED_BINDING.items():
        if config.get(key) != expected:
            raise RuntimeError(f"Drive identity configuration mismatch: {key}")
    return dict(APPROVED_BINDING)


def verify_local_drive_binding(config, destination=None):
    """Fail closed unless configured IDs, cached parents, and local paths agree.

    ``destination`` lets the deployment caller bind the directory it will write,
    as well as the directory named in the configuration. The existing notebook
    must already be present; a same-name replacement is not accepted as proof of
    preserving the pinned notebook's identity.
    """
    validate_config_binding(config)
    mount = Path(config.get("drive_mount", ""))
    if mount.absolute() != DRIVE_MOUNT.absolute():
        raise RuntimeError("Drive mount must remain the approved G: Desktop mount")
    canonical = (DRIVE_MOUNT / ".shortcut-targets-by-id" / PROJECT_FOLDER_ID
                 / PROJECT_FOLDER_TITLE / COLAB_FOLDER_TITLE)
    configured = Path(config.get("drive_project", ""))
    if configured.absolute() != canonical.absolute():
        raise RuntimeError("Deployment path does not match the approved folder identity")
    if destination is not None and Path(destination).absolute() != canonical.absolute():
        raise RuntimeError("Deployment destination differs from the approved folder identity")
    if not canonical.is_dir() or not (canonical / NOTEBOOK_NAME).is_file():
        raise RuntimeError("Approved Drive Desktop folder or pinned notebook is unavailable")
    metadata_path = config.get("drive_metadata_db")
    if not metadata_path or not Path(metadata_path).is_file():
        raise RuntimeError("Drive Desktop metadata is unavailable; identity not verified")

    try:
        with closing(sqlite3.connect(Path(metadata_path).resolve().as_uri() + "?mode=ro",
                                     uri=True, timeout=5)) as connection:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")

            def item(file_id, expected_title, folder):
                rows = connection.execute(
                    "SELECT stable_id,id,local_title,is_folder FROM items "
                    "WHERE id=? AND trashed=0 AND is_tombstone=0", (file_id,)).fetchall()
                if len(rows) != 1:
                    raise RuntimeError("Drive item is missing or ambiguous: " + file_id)
                row = rows[0]
                if (row[0] is None or row[2] != expected_title
                        or row[3] != int(folder)):
                    raise RuntimeError("Drive item name/type changed: " + file_id)
                return row

            def direct_parent(child, expected):
                # Require exactly one edge and a live, unique parent. A dangling
                # edge or extra parent must not disappear behind an inner join.
                rows = connection.execute(
                    "SELECT parent_stable_id FROM stable_parents WHERE item_stable_id=?",
                    (child[0],)).fetchall()
                if len(rows) != 1 or rows[0][0] != expected[0]:
                    raise RuntimeError("Drive item parent changed or is ambiguous: " + child[1])

            def unique_local_name(parent, child):
                rows = connection.execute(
                    "SELECT i.id FROM stable_parents p JOIN items i "
                    "ON i.stable_id=p.item_stable_id "
                    "WHERE p.parent_stable_id=? AND i.local_title=? "
                    "AND i.trashed=0 AND i.is_tombstone=0",
                    (parent[0], child[2])).fetchall()
                if rows != [(child[1],)]:
                    raise RuntimeError("Drive local name is missing or ambiguous: " + child[2])

            project = item(PROJECT_FOLDER_ID, PROJECT_FOLDER_TITLE, True)
            colab = item(COLAB_FOLDER_ID, COLAB_FOLDER_TITLE, True)
            notebook = item(NOTEBOOK_ID, NOTEBOOK_NAME, False)
            direct_parent(colab, project)
            direct_parent(notebook, colab)
            unique_local_name(project, colab)
            unique_local_name(colab, notebook)
    except sqlite3.Error as exc:
        raise RuntimeError("Drive Desktop metadata is unreadable; identity not verified") from exc

    return {**PINNED_IDS, "identity_verified_locally": True,
            "cloud_sync_status": "pending_or_unknown"}
