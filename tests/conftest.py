import pytest


@pytest.fixture(autouse=True)
def _reset_amp():
    """Trainer sets the process-wide MLP autocast dtype (config ``amp``, bf16 by default); reset it after
    every test so later numerical tests run in full precision regardless of test order."""
    yield
    from supermimic.world_model import dynamics

    dynamics.AMP_DTYPE = None
