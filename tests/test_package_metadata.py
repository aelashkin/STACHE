"""Distribution metadata contracts."""

from importlib import metadata

import stache


def test_runtime_version_matches_installed_distribution_metadata() -> None:
    assert stache.__version__ == metadata.version("stache")
