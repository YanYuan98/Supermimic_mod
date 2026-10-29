"""Time one policy update's imagination (forward + backward) eager vs torch.compile (plan 3.6).

``--terrain plane|mixed`` adds the terrain inputs (VISION_TERRAIN_PLAN §2.6: per-step noisy policy scan, the
world model's ``--wm-terrain`` condition, batched critic terrain features) to measure their overhead; mixed
uses a compiled terrain-only model of the ``mixed`` grid with the start origins at the map centre.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import torch
import tyro


def main(
    batch: int = 2048, horizon: int = 32, modes: tuple[str, ...] = ("eager", "default"), reps: int = 3, clips: int = 1, amp: str = "none",
    terrain: Literal["none", "plane", "mixed"] = "none", wm_terrain: Literal["none", "foot", "scan", "both"] = "foot",
) -> None:  # fmt: skip
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.data.replay_buffer import StartBatch
    from supermimic.envs.mdp.motion_library import MotionLibrary, discover_npz
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.imagination.compiled import ImagHeads, ImagStep
    from supermimic.imagination.objective import policy_objective
    from supermimic.imagination.rollout import imagine
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.policy.nets import Policy, PolicyCfg
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    from supermimic.world_model import dynamics

    dynamics.AMP_DTYPE = {"none": None, "bf16": torch.bfloat16}[amp]
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda:0"
    model = load_g1_model()
    files = discover_npz(Path("data/npz/lafan1"), limit=clips)
    q0 = torch.zeros(29, device=dev)
    lib = MotionLibrary(files, TRACKED_BODIES, q0, device=dev)
    fk = G1Kinematics(model, device=dev)
    lim = torch.as_tensor(model.jnt_range[1:], dtype=torch.float32, device=dev)
    heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, lim, q0)
    tin = None
    if terrain != "none":
        from supermimic.terrain import SCAN_DIM, TerrainInputs, TerrainMap, make_terrain_generator_cfg
        from supermimic.terrain.presets import compile_terrain

        if terrain == "plane":
            tmap = TerrainMap.plane(dev)
        else:
            gen = make_terrain_generator_cfg("mixed", seed=0)
            tmap = TerrainMap.build(compile_terrain(gen)[0], gen, dev)
        tin = TerrainInputs(tmap, fk, wm_terrain)
    wm = WorldModel(WorldModelCfg(terrain=wm_terrain if tin is not None else "none"), heads, dev, terrain=tin)
    policy = Policy(PolicyCfg(terrain_dim=SCAN_DIM if tin is not None else 0)).to(dev)

    # Synthetic start batch from library reference states.
    f = torch.randint(0, lib.num_frames - 100, (batch,), device=dev)
    qpos, qvel = lib.ref_qpos_qvel(f)
    s0 = torch.cat([qpos, qvel], -1)
    start = StartBatch(
        state=s0,
        xi=torch.zeros(batch, 33, device=dev) + torch.tensor([0.75] + [0.0] * 32, device=dev),
        frame=f,
        prev_action=torch.zeros(batch, 29, device=dev),
        hist_states=s0[None].expand(10, batch, 71).contiguous(),
        hist_actions=torch.zeros(10, batch, 29, device=dev),
        origin=torch.zeros(batch, 3, device=dev),
    )
    for mode in modes:
        cm = None if mode == "eager" else mode
        step_fn = ImagStep(wm, policy, q0, anchor_obs="none", compile_mode=cm, terrain=tin)
        heads_fn = ImagHeads(wm, critic_features=True, compile_mode=cm, terrain=tin)
        wm.requires_grad_(False)
        times = []
        for i in range(reps + 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ro = imagine(lib, start, horizon, step_fn, heads_fn)
            J = policy_objective(ro, None, 0.99)
            (-J).backward()
            policy.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if i > 0:
                times.append(dt)
            else:
                print(f"[{mode}] first call (incl. compile): {dt:.1f}s")
        print(f"[{mode}] terrain={terrain} wm_terrain={wm_terrain if tin is not None else 'none'} batch={batch} H={horizon}: {sum(times)/len(times):.3f} s per imagination fwd+bwd; peak mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    tyro.cli(main)
