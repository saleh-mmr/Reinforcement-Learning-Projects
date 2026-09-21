from pathlib import Path


def ensure_parent_dir(path):
    """
    Ensure the parent directory of `path` exists. If it doesn't, create it (including
    intermediate directories).

    This is a no-op when `path` has no parent (e.g., saving to the current directory).
    """
    parent = Path(path).parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
