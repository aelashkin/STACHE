import pytest


def pytest_collection_modifyitems(config, items):
    # Only keep test_taxi_robustness_region.py
    items[:] = [item for item in items if 'test_taxi_robustness_region.py' in str(item.fspath)]
