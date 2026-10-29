"""Static terrain: generator presets, the global height map and its analytic heads (scan, foot
features, scan noise), spawn sampling and terrain-lifted references (docs/VISION_TERRAIN_PLAN.md)."""

from supermimic.terrain.heightmap import (  # noqa: F401
    FOOT_FEAT_DIM,
    SCAN_DIM,
    WM_TERRAIN_DIMS,
    FootGeometry,
    GridSpec,
    HeightGrid,
    ScanNoiseCfg,
    ScanNoiseParams,
    TerrainMap,
    TerrainMapCfg,
    init_scan_history,
    noisy_scan,
    push_scan,
    resample_scan_noise_,
    sample_scan_noise,
)
from supermimic.terrain.inputs import CRITIC_TERRAIN_DIM, TerrainInputs  # noqa: F401
from supermimic.terrain.presets import TERRAIN_KINDS, TerrainLayout, make_terrain_generator_cfg  # noqa: F401
from supermimic.terrain.reference import (  # noqa: F401
    SpawnSample,
    default_window_frames,
    lifted_reference,
    path_bbox,
    reference_lift,
    sample_spawn,
    spawn_lift,
)
