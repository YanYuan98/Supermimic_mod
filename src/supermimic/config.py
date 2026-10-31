"""All training configuration in one dataclass tree (tyro-friendly)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.policy.nets import CriticCfg, PolicyCfg
from supermimic.terrain.depth import DepthCfg
from supermimic.terrain.heightmap import ScanNoiseCfg
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
    # World-model-only training (VISION_TERRAIN_PLAN M2): the policy collects data but is frozen (no
    # policy / critic updates, input normalisers not updated); the world model keeps training.
    wm_only: bool = False


@dataclass
class TerrainCurriculumCfg:
    """Global difficulty level L (plan §2.7): spawn rows ~ U{0..L}; L += 1 when the terrain episodes
    finished since the last check have mean normalised return > promote_return and failure rate < promote_max_fail."""

    start_level: int = 2
    check_every: int = 2000  # iterations
    # Terrain-episode statistics exclude short clip-end episodes (MultiClipMotionCommand._track_terrain_episodes).
    # Frozen flat policy, means of the 50-iteration logging windows: L = 2 return 0.82 (flat column 0.87), L = 9 0.75;
    # the pooled failure rate of a 2000-iteration check is lower than the window means (time-out waves).
    promote_return: float = 0.75
    promote_max_fail: float = 0.10
    min_episodes: int = 200  # summed over ranks; fewer -> no decision at this check


@dataclass
class TerrainCfg:
    """Terrain + height-scan extension (docs/VISION_TERRAIN_PLAN.md).

    kind: "none" = main's flat environment, no terrain inputs anywhere (default); "plane" = the
    terrain-capable models (policy scan, critic / world-model terrain features) on mjlab's flat plane
    (regression); "mixed" = the 9-column x ``num_rows`` curriculum grid of plan §2.7 with spawn sampling,
    reference lift and the difficulty curriculum."""

    kind: Literal["none", "plane", "mixed"] = "none"
    seed: int = 0  # terrain generator seed; explicit so every rank builds the same map
    # World-model terrain condition (plan §0): foot (28) | scan (187) | both (215) | none (flat WM) | foot_fine (80:
    # 5 x 3 sole clearances + 5 x 5 patch at 5 cm per foot, plan §7 mitigation).
    wm_terrain: Literal["none", "foot", "scan", "both", "foot_fine"] = "foot"
    scan_noise: ScanNoiseCfg = field(default_factory=ScanNoiseCfg)  # policy scan only (sim + imagination)
    # Policy terrain input: "scan" (noisy 17x11 height scan) | "depth" (rendered D435 frames, terrain/depth.py).
    # The world model and the critic keep their noise-free terrain features either way.
    policy_obs: Literal["scan", "depth"] = "scan"
    # Central flat platform of the curriculum tiles ("mixed"). stairs_platform: flat top width in m (None = 7.4 m with 6
    # steps per side; 1.5 = 16 steps per side in a 0.45 m frame, measured 1.47 m). slope_platform: mjlab's
    # HfPyramidSloped platform_width, which is NOT the flat width (None = 3.0 -> 5.2 m flat; 0.85 -> 1.44-1.47 m on rows
    # >= 5, 1.7 m on row 1). Blocks keep their 1.0 m platform; rough / wave / perlin have none.
    stairs_platform: float | None = None
    slope_platform: float | None = None
    # Pelvis-height termination (simulator, world model, imagination): "reference" = absolute height vs the
    # terrain-lifted reference (previous runs); "ground" = heights above the ground under each pelvis, so horizontal
    # drift onto another tread or slope height is not a failure (the global position reward still penalises it).
    height_termination: Literal["reference", "ground"] = "reference"
    depth: DepthCfg = field(default_factory=DepthCfg)
    grad: bool = False  # True: let gradients flow through the terrain slope (bilinear weights); default detached
    curriculum: TerrainCurriculumCfg = field(default_factory=TerrainCurriculumCfg)
    size: float = 12.0  # tile edge, m
    num_rows: int = 10  # difficulty rows
    border: float = 4.0  # flat outer border, m
    res: float = 0.025  # height-map resolution, m
    spawn_margin: float = 0.5  # m kept between the reference path bbox and the tile edge
    # MuJoCo Warp capacities for "mixed" (plane / none keep main's 250 / 35 / 50). mjlab's G1 rough velocity
    # task uses njmax 1500 / nconmax 70 / ccd_iterations 500. Measured with 4096 envs (flat policy, L = 2 and 9):
    # max nefc 187, ~9 contacts per world, no nefc / contact / ccd / hfield / epa overflow at 600 / 70 / 50..500;
    # the EPA work buffers scale with ccd_iterations (Warp memory 7 / 9 / 24 GB at 50 / 100 / 500), and 500 does
    # not fit next to the imagination on a 32 GB GPU. Rollout outcomes did not change between 50 and 500.
    njmax: int = 600
    nconmax: int = 70
    ccd_iterations: int = 100


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
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    resume: Path | None = None
    # Warm start from a (flat) checkpoint: weights + normalisers only, new terrain input columns of the
    # policy decoder / critic / world model zero-initialised, new modules fresh, optimisers fresh, iteration 0.
    init_from: Path | None = None
    amp: str = "bf16"  # "bf16" (autocast for MLP matmuls; ~31% faster on RTX 5090) | "none" (TF32 matmuls, as in the paper)


def terrain_env_kwargs(tcfg) -> dict:
    """``SuperMimicEnvCfgArgs`` terrain fields from a ``TerrainCfg`` (or its dict from a checkpoint config);
    {} for kind "none" (main's environment). The MuJoCo Warp capacity overrides apply to "mixed" only."""
    t = tcfg if isinstance(tcfg, dict) else asdict(tcfg)
    if t.get("kind", "none") == "none":
        return {}
    out = dict(
        terrain=t["kind"],
        terrain_seed=t["seed"],
        terrain_size=t["size"],
        terrain_rows=t["num_rows"],
        terrain_border=t["border"],
        terrain_res=t["res"],
        terrain_level=t["curriculum"]["start_level"],
        spawn_margin=t["spawn_margin"],
        scan_noise=ScanNoiseCfg(**t["scan_noise"]),
    )
    if t.get("height_termination", "reference") != "reference":
        out["height_termination"] = t["height_termination"]
    if t.get("policy_obs", "scan") == "depth":
        out["depth"] = DepthCfg(**t["depth"])
    if t["kind"] == "mixed":
        out.update(njmax=t["njmax"], nconmax=t["nconmax"], ccd_iterations=t["ccd_iterations"])
        out.update(terrain_stairs_platform=t.get("stairs_platform"), terrain_slope_platform=t.get("slope_platform"))
    return out
