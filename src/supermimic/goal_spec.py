"""Goal-layout settings shared by the environment, the imagination and the networks (no heavy imports).

``anchor_obs`` (IMPROVEMENT_PLAN F2): reference anchor pose relative to the robot anchor appended to
the goal: "none" (pure reference goal), "ori" (rot6d, 6, IMU-observable), "pos_ori" (pos 3 + rot6d 6,
needs state estimation). ``goal_features`` (F6): per-frame features "v1" (68) or "traj" (77).
"""

from __future__ import annotations

ANCHOR_OBS_DIMS = {"none": 0, "ori": 6, "pos_ori": 9}
GOAL_FEAT_DIMS = {"v1": 68, "traj": 77}


def anchor_mode(value) -> str:
    """Normalise the ``anchor_obs`` setting; old configs stored a bool (True = pos + ori)."""
    if value is True:
        return "pos_ori"
    if value is False or value is None:
        return "none"
    value = str(value)
    if value not in ANCHOR_OBS_DIMS:
        raise ValueError(f"anchor_obs must be one of {tuple(ANCHOR_OBS_DIMS)}, got {value!r}")
    return value


def goal_dim(future_frames: int, goal_features: str = "v1", anchor_obs="none") -> int:
    return future_frames * GOAL_FEAT_DIMS[goal_features] + ANCHOR_OBS_DIMS[anchor_mode(anchor_obs)]
