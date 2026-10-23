"""All training configuration in one dataclass tree (tyro-friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.policy.nets import CriticCfg, PolicyCfg
from supermimic.world_model.wm_trainer import WMTrainerCfg
from supermimic.world_model.world_model import WorldModelCfg


@dataclass
class DataCfg:
    npz_dir: Path = Path("data/npz/lafan1")
    clips: list[str] | None = None  # None = all npz in npz_dir
    limit: int | None = None
    manifest: Path | None = None  # csv with name,keep (filter_bones_seed.py output)
    library: Path | None = None  # packed compact library (.pt from build_library.py); overrides npz_dir


@dataclass
class EnvCfg:
    num_envs: int = 4096
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"
    replay_capacity: int = 1000
    future_frames: int = 10
    compact_library: bool = False
    # Reference-relative anchor pose in the goal (IMPROVEMENT_PLAN F2): "none" (pure reference goal),
    # "ori" (reference anchor orientation in the robot anchor frame, 6-d, IMU-observable; default),
    # "pos_ori" (+ anchor position, 9-d, needs state estimation).
    anchor_obs: str = "ori"
    # Per-frame goal features (F6): "v1" (68-d local reference quantities) or "traj" (77-d: + root
    # displacement and rotation relative to the current target frame's heading frame).
    goal_features: str = "traj"
    # supermimic_vibed option: False drops robot self-collisions (feet / body vs ground only).
    self_collision: bool = False


@dataclass
class ScheduleCfg:
    # One iteration = one world-model update + one policy/critic update (Algorithm A.1: "Both omega and
    # theta update every iteration"), preceded by 1/UTD vectorised env steps (fractional accumulator).
    total_iterations: int = 320_000
    updates_per_env_step: float = 0.4  # UTD
    policy_updates_per_wm_update: int = 1
    warmup_env_steps: int = 40  # env steps collected before the first iteration (fills k-step windows)
    wm_warmup_iterations: int = 500  # world-model-only iterations before the first policy update
    lr_final_ratio: float = 0.03
    lr_decay_iterations: int = 250_000
    log_interval: int = 50  # iterations
    checkpoint_interval: int = 4000  # iterations
    max_hours: float | None = None


@dataclass
class TrainConfig:
    run_name: str = "debug"
    log_root: Path = Path("logs")
    seed: int = 0
    device: str = "cuda:0"
    data: DataCfg = field(default_factory=DataCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    resume: Path | None = None
    amp: str = "bf16"  # "bf16" (autocast for MLP matmuls; ~31% faster on RTX 5090) | "none" (TF32 matmuls, as in the paper)
