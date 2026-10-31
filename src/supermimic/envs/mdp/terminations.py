"""Paper terminations: root (pelvis) height deviation and tilt; clip end as time-out."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def root_height_error(cmd: MultiClipMotionCommand, relative_to_ground: bool = False) -> torch.Tensor:
    """(N) reference minus robot pelvis height; ``relative_to_ground``: heights above the ground under each pelvis
    (horizontal drift onto another stair tread is then not a height error)."""
    ref, rob = cmd.root_pos_w, cmd.robot_root_pos_w
    e = ref[:, 2] - rob[:, 2]
    if relative_to_ground and cmd.tmap is not None:
        e = e + cmd.tmap.ground(rob[:, 0:2]) - cmd.tmap.ground(ref[:, 0:2])
    return e


def bad_root_height(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.25, relative_to_ground: bool = False) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return root_height_error(cmd, relative_to_ground).abs() > threshold


def bad_root_tilt(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.8) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    g = cmd.robot.data.gravity_vec_w
    ref = quat_apply_inverse(cmd.root_quat_w, g)[:, 2]
    rob = quat_apply_inverse(cmd.robot_root_quat_w, g)[:, 2]
    return (ref - rob).abs() > threshold


def clip_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return cmd.clip_end
