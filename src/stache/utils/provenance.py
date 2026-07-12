"""Best-effort primitive provenance for persisted scientific outputs."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import platform
import subprocess


def collect_provenance() -> dict[str, object]:
    """Collect dependency versions and local Git identity when available."""

    dependencies: dict[str, str] = {"python": platform.python_version()}
    for package in (
        "stache",
        "numpy",
        "stable-baselines3",
        "torch",
        "pyyaml",
    ):
        try:
            dependencies[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    result: dict[str, object] = {"dependencies": dependencies}
    repository = next(
        (
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / ".git").exists()
        ),
        None,
    )
    if repository is None:
        return result
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return result
    if revision.returncode == 0 and status.returncode == 0:
        result["git"] = {
            "commit": revision.stdout.strip(),
            "dirty": bool(status.stdout),
        }
    return result


__all__ = ["collect_provenance"]
