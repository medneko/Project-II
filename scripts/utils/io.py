from pathlib import Path


def ensure_dir(path):
    """Create a directory and return it as a Path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def safe_makedirs(path):
    return ensure_dir(path)


def check_output_path(path, overwrite=False):
    """Validate that an output file can be written safely."""
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {target}. Pass --overwrite to replace it."
        )
    if target.parent:
        ensure_dir(target.parent)
    return target


def legacy_default_output(path):
    print(
        f"WARNING: using default output {path}; prefer explicit --out/--outdir"
    )
    return Path(path)
