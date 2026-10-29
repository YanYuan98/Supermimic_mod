"""SuperMimic tracking environment on mjlab (plan Phase 2.3).

Differences from mjlab's BeyondMimic re-implementation (``mjlab.tasks.tracking``):
  * observations: paper Table A.1 proprioception only (no anchor pose, no base lin vel),
    10-frame history (term-major), plus a separate ``goal`` group with K future frames;
  * action: ``NoisyJointPositionAction`` (clean action observed, noise only executed);
  * multi-clip command with return-based reset sampling and clip-end time-outs;
  * rewards: self-collision term removed, joint-limit weight -2;
  * terminations: pelvis height / tilt vs reference, no end-effector term, clip-end time-out;
  * events: no push; friction and encoder bias re-sampled per episode; COM at startup;
  * a ``ReplayRecorder`` streams transitions into a GPU replay buffer.

Terrain (``terrain`` = plane | mixed, docs/VISION_TERRAIN_PLAN.md §2.3, §2.7): an extra ``terrain``
observation group (the noisy 17x11 policy height scan); ``mixed`` replaces the ground plane by the
curriculum terrain grid (+ flat outer border), raises the MuJoCo Warp capacities and installs the height-field
contact-normal fix (``terrain.mjwarp_patch``). ``none`` (default) builds exactly the flat environment.
"""

from __future__ import annotations

import dataclasses

import os
from dataclasses import dataclass, field

import mujoco
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp as track_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from supermimic.data.recorder import ReplayRecorderCfg
from supermimic.envs.actions import NoisyJointPositionActionCfg
from supermimic.envs.mdp import observations as sm_obs
from supermimic.envs.mdp import terminations as sm_term
from supermimic.envs.mdp.commands import MultiClipMotionCommandCfg
from supermimic.goal_spec import anchor_mode
from supermimic.terrain.depth import DepthCfg
from supermimic.terrain.heightmap import ScanNoiseCfg, TerrainMapCfg

class QuietSimulationCfg(SimulationCfg):
    """SimulationCfg that disables MuJoCo Warp's per-step overflow printf (a kernel build-time constant,
    so it must be set before the CUDA graphs are captured)."""

    def apply_wp_opt(self, wp_opt) -> None:
        super().apply_wp_opt(wp_opt)
        wp_opt.warn_overflow = 0


TRACKED_BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
ANCHOR_BODY = "torso_link"
ROOT_BODY = "pelvis"
FOOT_GEOM_REGEX = r"^(left|right)_foot[1-7]_collision$"


@dataclass
class SuperMimicEnvCfgArgs:
    npz_files: list[str]
    num_envs: int = 4096
    future_frames: int = 10
    history_length: int = 10
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"  # return | uniform | start
    replay_capacity: int = 1000
    record_transitions: bool = True
    seed: int | None = 0
    root_height_threshold: float = 0.25
    root_tilt_threshold: float = 0.8
    joint_limit_weight: float = -2.0
    bin_frames: int = 50
    compact_library: bool = False
    anchor_obs: str = "ori"  # "none" | "ori" (6, IMU-observable) | "pos_ori" (9, needs state estimation)
    goal_features: str = "v1"  # per-frame goal features: "v1" (68) | "traj" (77)
    self_collision: bool = True  # False: mjlab's FULL_COLLISION_WITHOUT_SELF (as supermimic_vibed)
    debug_vis: bool = False  # draw the reference ghost in the viewer (scripts/play.py)
    extra_command_kwargs: dict = field(default_factory=dict)
    # Terrain (VISION_TERRAIN_PLAN): "none" (main's flat env) | "plane" | "mixed".
    terrain: str = "none"
    terrain_seed: int = 0
    terrain_size: float = 12.0
    terrain_rows: int = 10
    terrain_border: float = 4.0
    terrain_res: float = 0.025
    terrain_level: int = 2  # initial curriculum level (max spawn row)
    terrain_cache_dir: str | None = "logs/terrain_cache"
    spawn_margin: float = 0.5
    scan_noise: ScanNoiseCfg = field(default_factory=ScanNoiseCfg)  # applied only with obs_noise (or scan_noise_always)
    scan_noise_always: bool = False  # apply scan_noise even with obs_noise off (benchmark robustness, plan M5-A)
    depth: DepthCfg | None = None  # rendered policy depth frames (terrain plane | mixed); noise follows obs_noise
    # MuJoCo Warp capacity overrides (None = main's njmax 250 / nconmax 35 / ccd_iterations 50).
    njmax: int | None = None
    nconmax: int | None = None
    ccd_iterations: int | None = None


def make_env_cfg(a: SuperMimicEnvCfgArgs) -> ManagerBasedRlEnvCfg:
    # ---------------------------------------------------------------- observations
    noise = (lambda lo, hi: Unoise(n_min=lo, n_max=hi)) if a.obs_noise else (lambda lo, hi: None)
    policy_terms = {
        "joint_pos": ObservationTermCfg(func=base_mdp.joint_pos_rel, params={"biased": True}, noise=noise(-0.01, 0.01)),
        "joint_vel": ObservationTermCfg(func=base_mdp.joint_vel_rel, noise=noise(-0.5, 0.5)),
        "base_ang_vel": ObservationTermCfg(func=base_mdp.base_ang_vel, noise=noise(-0.2, 0.2)),
        "projected_gravity": ObservationTermCfg(func=base_mdp.projected_gravity, noise=noise(-0.05, 0.05)),
        "actions": ObservationTermCfg(func=base_mdp.last_action),
    }
    # Reference anchor pose relative to the robot anchor, appended to the goal (IMPROVEMENT_PLAN F2).
    mode = anchor_mode(a.anchor_obs)
    anchor_terms = {}
    if mode == "pos_ori":
        anchor_terms["anchor_pos"] = ObservationTermCfg(func=track_mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=noise(-0.25, 0.25))
    if mode in ("ori", "pos_ori"):
        anchor_terms["anchor_ori"] = ObservationTermCfg(func=track_mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=noise(-0.05, 0.05))
    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
            history_length=a.history_length,
            flatten_history_dim=True,
        ),
        "goal": ObservationGroupCfg(
            terms={
                "future_frames": ObservationTermCfg(func=sm_obs.goal_future_frames, params={"command_name": "motion"}),
                **anchor_terms,
            },
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
        ),
    }
    if a.terrain not in ("none", "plane", "mixed"):
        raise ValueError(f"unknown terrain {a.terrain!r}")
    terrain_kwargs = {}
    if a.terrain != "none":
        # The scan noise model (latency, offsets, dropout) lives in the command (shared with the imagination).
        observations["terrain"] = ObservationGroupCfg(
            terms={"scan": ObservationTermCfg(func=sm_obs.terrain_scan, params={"command_name": "motion"})},
            concatenate_terms=True,
            enable_corruption=False,
        )
        terrain_kwargs = dict(
            terrain=a.terrain,
            terrain_level=a.terrain_level,
            terrain_map=TerrainMapCfg(res=a.terrain_res, cache_dir=a.terrain_cache_dir),
            scan_noise=a.scan_noise if (a.obs_noise or a.scan_noise_always) else ScanNoiseCfg(enabled=False),
            depth=None if a.depth is None else (a.depth if (a.obs_noise or a.scan_noise_always) else dataclasses.replace(a.depth, noise=False)),
            spawn_margin=a.spawn_margin,
        )

    # ---------------------------------------------------------------- actions
    actions = {
        "joint_pos": NoisyJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=G1_ACTION_SCALE,
            use_default_offset=True,
            noise_std=a.action_noise_std,
        )
    }

    # ---------------------------------------------------------------- commands
    pose_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)}
    vel_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2), "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)}
    commands = {
        "motion": MultiClipMotionCommandCfg(
            npz_files=[str(f) for f in a.npz_files],
            anchor_body_name=ANCHOR_BODY,
            root_body_name=ROOT_BODY,
            body_names=TRACKED_BODIES,
            entity_name="robot",
            future_frames=a.future_frames,
            goal_features=a.goal_features,
            bin_frames=a.bin_frames,
            pose_range=pose_range if a.reset_noise else {},
            velocity_range=vel_range if a.reset_noise else {},
            joint_position_range=(-0.1, 0.1) if a.reset_noise else (0.0, 0.0),
            sampling_mode=a.sampling_mode,  # type: ignore[arg-type]
            compact_library=a.compact_library,
            debug_vis=a.debug_vis,
            **terrain_kwargs,
            **a.extra_command_kwargs,
        )
    }

    # ---------------------------------------------------------------- events (plan D8)
    events: dict[str, EventTermCfg] = {}
    if a.domain_randomization:
        events["base_com"] = EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "operation": "add",
                "ranges": {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
            },
        )
        events["encoder_bias"] = EventTermCfg(
            mode="reset",
            func=dr.encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
        )
        events["foot_friction"] = EventTermCfg(
            mode="reset",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_REGEX),
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,
            },
        )
    else:
        # Evaluation protocol: foot friction fixed at 1.0, no other randomization.
        events["foot_friction"] = EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_REGEX),
                "operation": "abs",
                "ranges": (1.0, 1.0),
                "shared_random": True,
            },
        )

    # ---------------------------------------------------------------- rewards (Table A.2)
    rewards = {
        "motion_global_root_pos": RewardTermCfg(func=track_mdp.motion_global_anchor_position_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.3}),
        "motion_global_root_ori": RewardTermCfg(func=track_mdp.motion_global_anchor_orientation_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.4}),
        "motion_body_pos": RewardTermCfg(func=track_mdp.motion_relative_body_position_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.3}),
        "motion_body_ori": RewardTermCfg(func=track_mdp.motion_relative_body_orientation_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.4}),
        "motion_body_lin_vel": RewardTermCfg(func=track_mdp.motion_global_body_linear_velocity_error_exp, weight=1.0, params={"command_name": "motion", "std": 1.0}),
        "motion_body_ang_vel": RewardTermCfg(func=track_mdp.motion_global_body_angular_velocity_error_exp, weight=1.0, params={"command_name": "motion", "std": 3.14}),
        "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.1),
        "joint_limit": RewardTermCfg(func=base_mdp.joint_pos_limits, weight=a.joint_limit_weight, params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
    }

    # ---------------------------------------------------------------- terminations
    terminations = {
        "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
        "clip_end": TerminationTermCfg(func=sm_term.clip_end, params={"command_name": "motion"}, time_out=True),
        "root_height": TerminationTermCfg(func=sm_term.bad_root_height, params={"command_name": "motion", "threshold": a.root_height_threshold}),
        "root_tilt": TerminationTermCfg(func=sm_term.bad_root_tilt, params={"command_name": "motion", "threshold": a.root_tilt_threshold}),
    }

    recorders = {}
    if a.record_transitions:
        recorders["replay"] = ReplayRecorderCfg(params={"capacity": a.replay_capacity, "foot_geom_regex": FOOT_GEOM_REGEX, "com_body_name": "torso_link"})

    robot_cfg = get_g1_robot_cfg()
    if not a.self_collision:
        from mjlab.asset_zoo.robots.unitree_g1.g1_constants import FULL_COLLISION_WITHOUT_SELF

        robot_cfg.collisions = (FULL_COLLISION_WITHOUT_SELF,)
    if a.terrain == "mixed":
        from supermimic.terrain import make_terrain_generator_cfg

        gen = make_terrain_generator_cfg("mixed", seed=a.terrain_seed, size=a.terrain_size, num_rows=a.terrain_rows, border=a.terrain_border)
        terrain = TerrainEntityCfg(terrain_type="generator", terrain_generator=gen)
        robot_cfg.spec_fn = _raised_spec_fn(robot_cfg.spec_fn)
    else:
        terrain = TerrainEntityCfg(terrain_type="plane")
    mujoco_kwargs = {} if a.ccd_iterations is None else {"ccd_iterations": a.ccd_iterations}
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=terrain, num_envs=a.num_envs, entities={"robot": robot_cfg}),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        recorders=recorders,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="torso_link", distance=2.8, fovy=55.0, elevation=-5.0, azimuth=120.0),
        sim=QuietSimulationCfg(
            nconmax=35 if a.nconmax is None else a.nconmax,
            njmax=250 if a.njmax is None else a.njmax,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20, **mujoco_kwargs),
        ),
        decimation=4,
        episode_length_s=a.episode_length_s,
        seed=a.seed,
    )
    return cfg


def _raised_spec_fn(spec_fn, dz: float = 2.0):
    """Robot spec with the floating base 2 m higher. mjlab validates the host ``MjData(model)`` (qpos0: robot at
    the world origin) against ``nconmax``; on a grid with an odd number of rows the origin is a tile centre (e.g.
    a pyramid top), the default pose then intersects the terrain and MuJoCo Warp refuses the model. qpos0 only
    seeds the state before the first reset: the motion command writes every root state at reset."""

    def fn():
        spec = spec_fn()
        root = next(b for b in spec.worldbody.bodies if b.joints and b.joints[0].type == mujoco.mjtJoint.mjJNT_FREE)
        root.pos = [root.pos[0], root.pos[1], root.pos[2] + dz]
        return spec

    return fn


def make_env(a: SuperMimicEnvCfgArgs, device: str = "cuda:0") -> ManagerBasedRlEnv:
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    if a.terrain == "mixed" and os.environ.get("SUPERMIMIC_HFIELD_PATCH", "1") != "0":
        from supermimic.terrain.mjwarp_patch import install_hfield_normal_filter

        install_hfield_normal_filter()  # before the first step: mjlab captures the CUDA graphs then
    return ManagerBasedRlEnv(make_env_cfg(a), device=device)
