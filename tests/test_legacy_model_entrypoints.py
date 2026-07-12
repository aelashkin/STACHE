"""Historical model workflows collect trust before archive loading."""

from __future__ import annotations

import pytest

from stache.explainability import evaluate, rr_bfs


@pytest.mark.parametrize("module", [evaluate, rr_bfs])
def test_historical_model_entrypoint_requires_explicit_trust(module: object) -> None:
    with pytest.raises(SystemExit) as missing:
        module._parse_entrypoint_args([])
    assert missing.value.code == 2

    with pytest.raises(SystemExit) as invalid:
        module._parse_entrypoint_args(["--not-a-real-option"])
    assert invalid.value.code == 2

    accepted = module._parse_entrypoint_args(
        ["--acknowledge-trusted-model"]
    )
    assert accepted.acknowledge_trusted_model is True
