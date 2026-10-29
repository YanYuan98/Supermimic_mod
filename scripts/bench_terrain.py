"""Terrain benchmarks (plan §2.1, §4B, M1):

  1. rasterisation of the ``mixed`` grid (standalone MuJoCo Warp RenderContext, 2.5 cm, cache off);
  2. compiled lookups for 4096 envs x (187 scan + 28 foot) points, incl. the two leg FK chains;
  3. ``mujoco_warp.rays`` throughput for 4096 pinhole cameras x 64x48 = 3072 rays (depth student gate);
  4. per-type terrain statistics at rows 0/3/6/9 (compare with the table in plan §2.7).

Usage (driver 570 machines: ``source scripts/env_cuda12_driver.sh`` first):
    CUDA_VISIBLE_DEVICES=1 python scripts/bench_terrain.py [--seed 0] [--rows 10] [--envs 4096]
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from supermimic.data.motion_convert import load_g1_model
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.terrain.heightmap import FOOT_FEAT_DIM, SCAN_DIM, TerrainMap, TerrainMapCfg
from supermimic.terrain.presets import TerrainLayout, compile_terrain, make_terrain_generator_cfg


def bench(fn, iters: int = 200, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


def random_states(tm: TerrainMap, layout: TerrainLayout, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Standing-ish states spread over the tile grid; root relative to a random origin."""
    g = torch.Generator(device=device).manual_seed(0)
    hx, hy = layout.grid_half_extent
    origin = torch.zeros(n, 3, device=device)
    origin[:, 0] = (torch.rand(n, device=device, generator=g) * 2 - 1) * (hx - 1)
    origin[:, 1] = (torch.rand(n, device=device, generator=g) * 2 - 1) * (hy - 1)
    s = torch.zeros(n, 71, device=device)
    s[:, 0:2] = (torch.rand(n, 2, device=device, generator=g) * 2 - 1) * 0.5
    s[:, 2] = tm.ground(origin[:, 0:2] + s[:, 0:2]) + 0.78
    yaw = torch.rand(n, device=device, generator=g) * 2 * math.pi
    s[:, 3], s[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    s[:, 7:36] = (torch.rand(n, 29, device=device, generator=g) * 2 - 1) * 0.2
    return s, origin


def bench_lookup(tm: TerrainMap, layout: TerrainLayout, n: int, device) -> dict:
    fk = G1Kinematics(load_g1_model(), device=device)
    s, origin = random_states(tm, layout, n, device)
    grid = tm.grid

    def heads(s, origin, grid):
        root = s[..., 0:3] + origin
        return tm.scan(root, s[..., 3:7], grid), tm.foot_features(s, origin, fk, grid)

    pts = torch.cat([tm.scan_points(s[:, 0:3] + origin, s[:, 3:7]), tm.foot_points(s, origin, fk).sole[..., 0:2].flatten(1, 2)], 1)
    pts = torch.cat([pts, pts[:, :18]], 1)  # 187 + 10 + 18 = 215 points
    assert pts.shape == (n, SCAN_DIM + FOOT_FEAT_DIM, 2)

    def lookup(p, grid):
        return tm.height(p, grid)

    out = {}
    c_lookup = torch.compile(lookup, dynamic=False)
    c_heads = torch.compile(heads, dynamic=False)
    out["lookup_eager_ms"] = bench(lambda: lookup(pts, grid))
    t = time.perf_counter()
    c_lookup(pts, grid)
    torch.cuda.synchronize()
    out["lookup_compile_s"] = time.perf_counter() - t
    out["lookup_compiled_ms"] = bench(lambda: c_lookup(pts, grid))
    sc, ff = heads(s, origin, grid)
    out["heads_eager_ms"] = bench(lambda: heads(s, origin, grid), iters=50)
    t = time.perf_counter()
    sc2, ff2 = c_heads(s, origin, grid)
    torch.cuda.synchronize()
    out["heads_compile_s"] = time.perf_counter() - t
    out["heads_compiled_ms"] = bench(lambda: c_heads(s, origin, grid))
    out["heads_compiled_vs_eager_maxabs"] = max((sc - sc2).abs().max().item(), (ff - ff2).abs().max().item())
    return out


def bench_rays(mj_model, layout: TerrainLayout, n_cam: int, width: int, height: int, fovy_deg: float, device) -> dict:
    """``mujoco_warp.rays`` with the standalone nworld=1 context and pnt/vec of shape (1, n_cam * w * h)."""
    import mujoco
    import mujoco_warp as mjwarp
    import warp as wp

    from supermimic.kinematics.rotations import quat_apply

    dev = torch.device(device)
    mjd = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mjd)
    g = torch.Generator(device=dev).manual_seed(1)
    hx, hy = layout.grid_half_extent
    pos = torch.zeros(n_cam, 3, device=dev)
    pos[:, 0] = (torch.rand(n_cam, device=dev, generator=g) * 2 - 1) * (hx - 2)
    pos[:, 1] = (torch.rand(n_cam, device=dev, generator=g) * 2 - 1) * (hy - 2)
    pos[:, 2] = 1.3
    # Camera looking along +x of a random heading, pitched 45 deg down (MuJoCo camera: -z forward, +y up).
    yaw = torch.rand(n_cam, device=dev, generator=g) * 2 * math.pi
    pitch = math.radians(45.0)
    q_yaw = torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], -1)
    # Pinhole rays as mjlab's PinholeCameraPatternCfg, camera frame -> body frame (x fwd, y left, z up), pitched down.
    tv = math.tan(math.radians(fovy_deg) / 2)
    th = tv * width / height
    u = (torch.arange(width, device=dev) + 0.5) / width * 2 - 1
    v = 1 - (torch.arange(height, device=dev) + 0.5) / height * 2
    gu, gv = torch.meshgrid(u, v, indexing="xy")
    d_cam = torch.stack([gu.flatten() * th, gv.flatten() * tv, -torch.ones(width * height, device=dev)], -1)
    d_body = torch.stack([-d_cam[:, 2], -d_cam[:, 0], d_cam[:, 1]], -1)  # forward x, left y, up z
    cp, sp = math.cos(pitch), math.sin(pitch)
    d_body = torch.stack([cp * d_body[:, 0] + sp * d_body[:, 2], d_body[:, 1], -sp * d_body[:, 0] + cp * d_body[:, 2]], -1)
    d_body = d_body / d_body.norm(dim=-1, keepdim=True)
    nr = width * height
    vec = quat_apply(q_yaw[:, None, :].expand(n_cam, nr, 4), d_body[None].expand(n_cam, nr, 3)).reshape(1, -1, 3).contiguous()
    pnt = pos[:, None, :].expand(n_cam, nr, 3).reshape(1, -1, 3).contiguous()
    n = pnt.shape[1]
    dist = torch.empty(1, n, device=dev)
    geomid = torch.empty(1, n, device=dev, dtype=torch.int32)
    normal = torch.empty(1, n, 3, device=dev)
    bex = torch.full((n,), -1, device=dev, dtype=torch.int32)
    vec6 = wp.types.vector(length=6, dtype=float)
    out = {}
    with wp.ScopedDevice(f"cuda:{dev.index or 0}"), wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(dev))):
        m = mjwarp.put_model(mj_model)
        d = mjwarp.put_data(mj_model, mjd, nworld=1)
        rc = mjwarp.create_render_context(mj_model, nworld=1, cam_active=[False] * mj_model.ncam, enabled_geom_groups=[0], use_textures=False, use_shadows=False)
        args = (m, d, wp.from_torch(pnt, dtype=wp.vec3), wp.from_torch(vec, dtype=wp.vec3), vec6(-1, 0, 0, 0, 0, 0), True, wp.from_torch(bex), wp.from_torch(dist), wp.from_torch(geomid), wp.from_torch(normal, dtype=wp.vec3))
        out["rays_ms"] = bench(lambda: mjwarp.rays(*args, rc=rc), iters=20, warmup=3)
    out["num_rays"] = n
    out["hit_frac"] = (dist >= 0).float().mean().item()
    out["mrays_per_s"] = n / out["rays_ms"] / 1e3
    return out


def tile_stats(tm: TerrainMap, layout: TerrainLayout, rows=(0, 3, 6, 9), sub: int = 4) -> list[str]:
    """Per type and row: z range, p99 slope of smooth cells, max adjacent jump at 0.1 m inside the tile
    (0.5 m frame excluded) and anywhere in the tile, dominant step rise (stairs). ``sub`` = 0.1 m / res."""
    h = tm.grid.h.float()
    spec = tm.spec
    n_tile = round(layout.tile_size[0] / spec.res)
    nb = round(layout.border / spec.res)  # border cells (same on both axes)
    lines = []
    for c, name in enumerate(layout.col_names):
        cells = []
        for r in rows:
            if r >= layout.num_rows:
                continue
            i0, j0 = nb + r * n_tile, nb + c * n_tile
            t = h[i0 : i0 + n_tile, j0 : j0 + n_tile]
            ts = t[sub // 2 :: sub, sub // 2 :: sub]  # 0.1 m sub-grid
            gx, gy = (ts[1:] - ts[:-1]).abs(), (ts[:, 1:] - ts[:, :-1]).abs()
            jumps = torch.cat([gx.flatten(), gy.flatten()])
            smooth = jumps[jumps < 0.04]  # 36 % slope = 3.6 cm per 0.1 m
            slope = (torch.quantile(smooth, 0.99).item() / (sub * spec.res) * 100) if smooth.numel() else 0.0
            k = 5  # 0.5 m frame in 0.1 m cells
            inner = torch.cat([gx[k:-k, k:-k].flatten(), gy[k:-k, k:-k].flatten()])
            fine = torch.cat([(t[1:] - t[:-1]).abs().flatten(), (t[:, 1:] - t[:, :-1]).abs().flatten()])
            steps = fine[fine > 0.005]
            rise = torch.mode((steps * 1000).round()).values.item() / 10 if steps.numel() else 0.0
            cells.append(f"r{r}: z {t.min().item():+.2f}/{t.max().item():+.2f} slope {slope:3.0f}% jump {inner.max().item():.2f}/{jumps.max().item():.2f}" + (f" rise {rise:.0f}cm" if "stairs" in name else ""))
        lines.append(f"{name:11s} " + " | ".join(cells))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--cam", type=str, default="64x48")
    ap.add_argument("--fovy", type=float, default=58.0)
    ap.add_argument("--device", type=str, default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False

    gen_cfg = make_terrain_generator_cfg("mixed", seed=a.seed, num_rows=a.rows)
    t = time.perf_counter()
    mj_model, origins = compile_terrain(gen_cfg)
    t_compile = time.perf_counter() - t
    layout = TerrainLayout.from_generator_cfg(gen_cfg)
    layout.check_origins(origins)
    raster = []
    for _ in range(2):  # first call includes warp module loading / BVH kernels
        t = time.perf_counter()
        tm = TerrainMap.build(mj_model, gen_cfg, dev, TerrainMapCfg(cache_dir=None))
        raster.append((time.perf_counter() - t, tm.info))
    lk = bench_lookup(tm, layout, a.envs, dev)
    w, hgt = (int(x) for x in a.cam.split("x"))
    rays = bench_rays(mj_model, layout, a.envs, w, hgt, a.fovy, dev)

    spec = tm.spec
    mem = sum(x.numel() * x.element_size() for x in tm.grid) / 2**20
    print("=" * 100)
    print(f"terrain mixed seed={a.seed}: {layout.num_rows} rows x {layout.num_cols} cols of {layout.tile_size[0]:.0f} m tiles + {layout.border:.0f} m border; "
          f"compile {t_compile:.2f} s, ngeom {mj_model.ngeom} ({int((mj_model.geom_type == 6).sum())} boxes, {mj_model.nhfield} hfields)")
    print(f"raster grid {spec.nx} x {spec.ny} @ {spec.res} m = {spec.nx * spec.ny / 1e6:.1f} M cells, h+g fp16 {mem:.0f} MiB, misses {tm.info['misses']}")
    for k, (tt, info) in enumerate(raster):
        print(f"  build #{k + 1}: total {tt:.2f} s (context {info['context_s']:.2f} s, rays+fill {info['raster_s'] - info['context_s']:.3f} s, blur {info['blur_s'] * 1e3:.1f} ms)")
    print(f"lookups, {a.envs} envs:")
    print(f"  height only, {SCAN_DIM + FOOT_FEAT_DIM} pts/env: eager {lk['lookup_eager_ms']:.3f} ms, compiled {lk['lookup_compiled_ms']:.3f} ms (compile {lk['lookup_compile_s']:.1f} s)")
    print(f"  scan(187) + foot_features(28) incl. 2 leg FK chains: eager {lk['heads_eager_ms']:.3f} ms, compiled {lk['heads_compiled_ms']:.3f} ms "
          f"(compile {lk['heads_compile_s']:.1f} s, |compiled - eager| {lk['heads_compiled_vs_eager_maxabs']:.1e})")
    print(f"mujoco_warp.rays, {a.envs} x {w}x{hgt} = {rays['num_rays'] / 1e6:.1f} M rays (pinhole fovy {a.fovy}, 45 deg down, 1.3 m): "
          f"{rays['rays_ms']:.2f} ms/call, {rays['mrays_per_s']:.0f} Mrays/s, hit {rays['hit_frac'] * 100:.1f}%")
    print("per-type stats (z min/max m; p99 slope of smooth 0.1 m cells; max 0.1 m jump interior/whole tile m; stairs: dominant rise):")
    for line in tile_stats(tm, layout):
        print("  " + line)


if __name__ == "__main__":
    main()
