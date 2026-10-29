"""Test videos of one terrain policy: full-screen segments, 2x2 grid pages and title cards.

Builds on ``render_compare.py`` (same G1 model + terrain map, follow camera, translucent reference ghost re-anchored at
the robot pelvis xy / yaw, keypoint error, root drift and the training-termination failure test). Inputs are
``record_rollouts.py --terrain mixed`` rollouts, which may come from different tiles of the same map (the map is
rebuilt once from ``metadata_json["terrain"]``; each clip is drawn at its recorded world origin).

Spec (json): {"output", "font_dir", "items": [...]}, each item one of
  {"type": "card", "duration_s", "title", "lines": [...]}
  {"type": "single", "duration_s", "seg": SEG}
  {"type": "grid", "duration_s", "title" (summary only; precede the page with a card), "panels": [SEG x 4]}
  {"type": "side", "duration_s", "panels": [SEG x 2]}   (two synchronised 960x1080 panels, e.g. one clip on two terrains)
with SEG = {"rollout", "clip", "start_s", "azimuth" (default 135), "distance" (default 3.0), "elevation" (default -18)}.
Rollouts of one video must share the map (``metadata_json["terrain"]`` up to col / row), e.g. the narrow-platform
stairs variant (``record_rollouts.py --stairs-platform``) in its own video. On stairs, the overlay also counts the
stair levels the feet stood on (``foot_levels``) and the pelvis level changes (``stair_crossings``).

  MUJOCO_GL=egl python scripts/render_terrain_video.py --spec spec.json --only 3   # one item -> <out>_parts/03.mp4
  python scripts/render_terrain_video.py --spec spec.json --concat                 # parts -> output (+ summary json)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import render_compare as rc
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, DT = 1920, 1080, 30, 0.02
TERRAIN_ZH = {"flat": "平地", "stairs_up": "上台阶（凹坑，中心低）", "stairs_down": "下台阶（凸台，中心高）", "slope_up": "上坡（凹坑）",
              "slope_down": "下坡（凸台）", "rough": "随机粗糙", "wave": "波浪", "perlin": "Perlin 噪声", "blocks": "离散方块"}  # fmt: skip


def terrain_param(col: str, row: int, rows: int = 10, stairs_platform: float | None = None) -> str:
    """Key parameter of a tile of the mixed grid (supermimic.terrain.presets.mixed_sub_terrains, d = row / (rows - 1));
    ``stairs_platform``: the narrow-platform stairs variant (steps per side / platform from presets.stairs_layout)."""
    d = row / (rows - 1)
    if col in ("stairs_up", "stairs_down"):
        if stairs_platform is None:
            return f"级高 {100 * (0.02 + 0.13 * d):.1f} cm · 每侧 6 级 · 踏面 30 cm"
        from supermimic.terrain.presets import stairs_layout

        n, plat = stairs_layout(platform_width=stairs_platform)
        return f"级高 {100 * step_height(row, rows):.1f} cm · 每侧 {n} 级 · 踏面 30 cm · 中央平台 {plat:.1f} m × {plat:.1f} m · 总高 {(n + 1) * step_height(row, rows):.2f} m"
    if col in ("slope_up", "slope_down"):
        return f"最大坡度 {36 * d:.0f}%（{math.degrees(math.atan(0.36 * d)):.0f}°）"
    return {"flat": "无起伏（对照）", "rough": f"随机起伏 0–{6 * d:.1f} cm", "wave": f"振幅 ±{15 * d:.1f} cm",
            "perlin": f"高度 0–{20 * d:.1f} cm", "blocks": f"块高 ±{2 + 10 * d:.1f} cm"}[col]  # fmt: skip


def step_height(row: int, rows: int = 10) -> float:
    return 0.02 + 0.13 * row / (rows - 1)


def style_terrain(model, terrain: dict) -> None:
    """Neutral terrain look for the video: stair / box tops alternate two greys per step level (step edges readable),
    height fields get the checker material of render_compare; near / far planes in metres (the large map inflates
    ``stat.extent``, which put the default near plane at 2.6 m and clipped the stairs in front of the camera)."""
    tb = np.bincount(model.geom_bodyid).argmax()  # the terrain body holds most geoms
    grid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "grid")
    size, rows = terrain["terrain_size"], terrain["terrain_rows"]
    boxes: dict[tuple[int, int], list[int]] = {}
    for g in np.flatnonzero(model.geom_bodyid == tb):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD:
            model.geom_matid[g] = grid
            model.geom_rgba[g] = (1.0, 1.0, 1.0, 1.0)
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            tile = tuple(int(np.floor((model.geom_pos[g, i] + n * size / 2) / size)) for i, n in ((0, rows), (1, 9)))
            boxes.setdefault(tile, []).append(int(g))
    for gs in boxes.values():  # level = rank of the box top among the distinct tops of its tile
        tops = np.round(model.geom_pos[gs, 2] + model.geom_size[gs, 2], 3)
        lvl = np.searchsorted(np.unique(tops), tops)
        for g, lv in zip(gs, lvl):
            model.geom_matid[g] = -1
            model.geom_rgba[g] = (0.88, 0.87, 0.84, 1.0) if lv % 2 else (0.68, 0.70, 0.74, 1.0)
    model.vis.headlight.ambient[:] = 0.45
    model.vis.map.znear = 0.05 / model.stat.extent
    model.vis.map.zfar = 120.0 / model.stat.extent


class Ground:
    """Terrain-only copy of the map for ground-height queries (mj_ray straight down)."""

    def __init__(self, terrain: dict):
        from supermimic.terrain import make_terrain_generator_cfg
        from supermimic.terrain.presets import compile_terrain

        gen = make_terrain_generator_cfg(terrain["terrain"], seed=terrain["terrain_seed"], size=terrain["terrain_size"], num_rows=terrain["terrain_rows"], border=terrain["terrain_border"], stairs_platform=terrain.get("stairs_platform"))
        self.m, _ = compile_terrain(gen)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.gid = np.zeros(1, dtype=np.int32)

    def height(self, x: float, y: float) -> float:
        dist = mujoco.mj_ray(self.m, self.d, np.array([x, y, 5.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, self.gid)
        return 5.0 - dist if dist >= 0 else 0.0


def stair_crossings(ground: Ground, track, row: int, stride: int = 2, hold: int = 5) -> np.ndarray:
    """Cumulative number of stair steps crossed by the robot pelvis (per sample): ground height under the pelvis
    quantised by the tile's step height, a level change counts once it holds for ``hold`` strided samples."""
    sh = step_height(row)
    idx = np.arange(0, track.n, stride)
    lvl = np.array([round(ground.height(*track.rp[s, :2]) / sh) for s in idx])
    count, cur, cand, run = np.zeros(len(idx), dtype=np.int64), lvl[0], lvl[0], 0
    total = 0
    for i, v in enumerate(lvl):
        if v == cur:
            run = 0
        elif v == cand:
            run += 1
            if run >= hold:
                total += abs(int(v - cur))
                cur, run = v, 0
        else:
            cand, run = v, 1
        count[i] = total
    return np.repeat(count, stride)[: track.n]


FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def foot_levels(model, ground: Ground, track, row: int, lo: int, hi: int, contact: float = 0.03, hold: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Stair level under each foot while it stands (samples lo..hi-1): the support height is the highest ground under
    the 7 sole capsules, the foot stands if its lowest capsule is within ``contact`` of it for ``hold`` samples; level
    = support height / step height (0 = the flat frame, +-k = k rises up / down). Returns ((n, 2) level or a large
    sentinel when the foot is in the air, (n,) cumulative number of distinct levels stood on since ``lo``)."""
    sh = step_height(row)
    d = mujoco.MjData(model)
    geoms = [np.flatnonzero((model.geom_bodyid == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)) & (model.geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE)) for b in FOOT_BODIES]
    AIR = 10**6
    raw = np.full((track.n, 2), AIR, dtype=np.int64)
    for s in range(lo, hi):
        rc.set_qpos(model, d, track.rp[s], track.rq[s], track.jp[s])
        for f, gs in enumerate(geoms):
            ax = d.geom_xmat[gs].reshape(-1, 3, 3)[:, :, 2] * model.geom_size[gs, 1:2]  # capsule half axis (world)
            sole = float(np.min(d.geom_xpos[gs, 2] - np.abs(ax[:, 2]) - model.geom_size[gs, 0]))
            sup = max(ground.height(*d.geom_xpos[g, :2]) for g in gs)
            if sole - sup < contact:
                raw[s, f] = int(round(sup / sh))
    lv = raw.copy()  # debounce: a stance must hold ``hold`` samples on one level
    for f in range(2):
        for s in range(lo, hi):
            if raw[s, f] != AIR and not all(raw[q, f] == raw[s, f] for q in range(max(lo, s - hold + 1), s + 1)):
                lv[s, f] = AIR
    seen: set[int] = set()
    count = np.zeros(track.n, dtype=np.int64)
    for s in range(lo, hi):
        seen.update(int(v) for v in lv[s] if v != AIR)
        count[s] = len(seen)
    count[hi:] = count[hi - 1] if hi > lo else 0
    return lv, count


class Seg:
    _cache: dict[str, tuple] = {}

    def __init__(self, cfg: dict, model, body_ids, ground: Ground, duration_s: float):
        path = cfg["rollout"]
        if path not in Seg._cache:
            roll = np.load(path, allow_pickle=False)
            Seg._cache[path] = (roll, json.loads(str(roll["metadata_json"])))
        roll, meta = Seg._cache[path]
        i = next(j for j, m in enumerate(meta["segments"]) if m["clip"] == cfg["clip"])
        self.cfg, self.clip = cfg, cfg["clip"]
        self.col, self.row = meta["terrain"]["col"], meta["terrain"]["row"]
        self.platform = meta["terrain"].get("stairs_platform")
        self.t = rc.Track(roll, i, world=True)
        self.depth = roll[f"s{i}_depth"] if f"s{i}_depth" in roll.files else None  # policy depth frames (depth policies)
        self.lo = int(round(cfg["start_s"] / DT))
        self.hi = min(self.lo + int(round(duration_s / DT)), self.t.n)
        assert self.hi > self.lo, (cfg, self.t.n)
        rc.keypoint_errors(model, self.t, body_ids, max(0, self.lo - 50), self.hi)
        self.fail = self.t.first_fail
        self.cause, self.fell = None, False
        if self.fail is not None:  # which test fired, and did the robot physically go down (pelvis < 0.4 m above ground within 5 s)
            f, t = self.fail, self.t
            tilt = abs(rc.gravity_z(t.gq[f]) - rc.gravity_z(t.rq[f]))
            self.cause = "倾角误差 > 0.8" if tilt > 0.8 else ("骨盆比参考高 > 0.25 m" if t.gp[f, 2] < t.rp[f, 2] else "骨盆比参考低 > 0.25 m")
            self.fell = bool(min(t.rp[q, 2] - ground.height(*t.rp[q, :2]) for q in range(f, min(t.n, f + 250), 5)) < 0.4)
        self.cross = stair_crossings(ground, self.t, self.row) if self.col.startswith("stairs") else None
        self.levels = foot_levels(model, ground, self.t, self.row, self.lo, self.hi) if self.col.startswith("stairs") else None
        if self.col.startswith("stairs"):  # frame + steps + platform
            from supermimic.terrain.presets import stairs_layout

            self.levels_total = stairs_layout(platform_width=self.platform)[0] + 2
        self.lookat = None

    @property
    def param(self) -> str:
        return terrain_param(self.col, self.row, stairs_platform=self.platform)

    def summary(self) -> dict:
        t = self.t
        lv = None if self.levels is None else sorted({int(v) for v in self.levels[0][self.lo : self.hi].ravel() if v < 10**6})
        return {"clip": self.clip, "terrain": self.col, "row": self.row, "stairs_platform": self.platform, "param": self.param, "window_s": [round(self.lo * DT, 2), round(self.hi * DT, 2)],
                "clip_len_s": round(t.n * DT, 2), "outcome": "SUCCESS" if self.fail is None else "FAIL", "first_failure_s": None if self.fail is None else round(self.fail * DT, 2), "fail_cause": self.cause, "physical_fall": self.fell,
                "fail_in_window": self.fail is not None and self.lo <= self.fail < self.hi, "keypoint_err_mm_window_mean": round(float(np.nanmean(t.err_mm[self.lo:self.hi])), 1),
                "stair_crossings_window": None if self.cross is None else int(self.cross[self.hi - 1] - self.cross[self.lo]),
                "foot_levels_window": lv, "foot_levels_count": None if lv is None else len(lv)}  # fmt: skip


def make_fonts(font_dir: Path, k: float) -> dict:
    F = lambda name, size: ImageFont.truetype(str(font_dir / name), max(10, int(round(size * k))))
    return {"big": F("NotoSansCJKsc-Bold.otf", 34), "body": F("NotoSansCJKsc-Regular.otf", 24), "small": F("NotoSansCJKsc-Regular.otf", 18),
            "stat": F("NotoSansCJKsc-Regular.otf", 24), "badge": F("NotoSansCJKsc-Bold.otf", 40), "fail": F("NotoSansCJKsc-Bold.otf", 52)}  # fmt: skip


def draw_overlay(canvas: Image.Image, x0: int, y0: int, pw: int, ph: int, seg: Seg, s: int, fonts: dict, full: bool) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    k = pw / W
    t = seg.t
    pad = int(16 * k) + 4
    # Top-left: terrain, row, parameter, clip and time.
    param = seg.param
    if seg.platform is not None and " · 中央平台" in param:  # two lines for the longer narrow-stairs parameter
        a, b = param.split(" · 中央平台", 1)
        param_lines = [(a, fonts["body"]), ("中央平台" + b, fonts["body"])]
    else:
        param_lines = [(param, fonts["body"])]
    lines = [(f"{TERRAIN_ZH[seg.col]} · 第 {seg.row} 行（难度 {seg.row / 9:.2f}）", fonts["big"]), *param_lines,
             (f"{seg.clip}   clip 时间 {rc.fmt_t(s * DT)} / {rc.fmt_t(t.n * DT)}", fonts["body"])]  # fmt: skip
    widths = [draw.textlength(txt, font=f) for txt, f in lines]
    heights = [f.size * 1.32 for _, f in lines]
    bx, by = x0 + pad, y0 + pad
    draw.rounded_rectangle([bx, by, bx + max(widths) + 2 * pad, by + sum(heights) + 1.4 * pad], radius=int(10 * k) + 2, fill=(20, 24, 32, 185))
    yy = by + 0.6 * pad
    for (txt, f), h in zip(lines, heights):
        draw.text((bx + pad, yy), txt, font=f, fill=(255, 255, 255))
        yy += h
    # Top-right: outcome of the whole clip in this recorded rollout.
    ok = seg.fail is None
    badge = "成功 SUCCESS" if ok else f"失败 FAIL @ {rc.fmt_t(seg.fail * DT)}"
    bw = draw.textlength(badge, font=fonts["badge"])
    bh = fonts["badge"].size * 1.45
    rx1 = x0 + pw - pad
    if bx + max(widths) + 2 * pad + pad > rx1 - bw - 2 * pad:  # narrow panel: badge below the info box
        by = by + sum(heights) + 1.4 * pad + pad
    draw.rounded_rectangle([rx1 - bw - 2 * pad, by, rx1, by + bh], radius=int(10 * k) + 2, fill=(30, 140, 70, 230) if ok else (200, 40, 40, 230))
    draw.text((rx1 - bw - pad, by + 0.12 * bh), badge, font=fonts["badge"], fill=(255, 255, 255))
    if full:
        sub = "本次回放：整条 clip 未触发终止" if ok else f"本次回放：{seg.cause}{'，摔倒' if seg.fell else '，未摔倒'}"
        sw = draw.textlength(sub, font=fonts["small"])
        draw.rounded_rectangle([rx1 - sw - pad, by + bh + 4, rx1, by + bh + 8 + fonts["small"].size * 1.4], radius=6, fill=(20, 24, 32, 170))
        draw.text((rx1 - sw - pad / 2, by + bh + 6), sub, font=fonts["small"], fill=(255, 255, 255))
    # Bottom-left: live tracking stats.
    w0 = max(0, s - 49)
    err = float(np.nanmean(t.err_mm[w0 : s + 1]))
    failed = seg.fail is not None and s >= seg.fail
    stats = [(f"关键点误差（1 s 平均）  {err:6.1f} mm", (20, 20, 20)), (f"根部水平偏移            {t.drift[s]:6.2f} m", (20, 20, 20))]
    if seg.levels is not None:
        lv, cnt = seg.levels
        top = seg.levels_total - 1  # level of the central platform (0 = the flat frame)
        name = lambda v: "腾空" if v >= 10**6 else ("中央平台" if abs(int(v)) == top else ("外框" if v == 0 else f"第 {abs(int(v))} 层"))
        cur = "  ".join(f"{side} {name(v)}" for side, v in zip(("左脚", "右脚"), lv[s]))
        stats.append((f"双脚踩过的台阶层          {int(cnt[s]):4d} / {seg.levels_total} 层（本段）", (20, 20, 20)))
        stats.append((f"当前  {cur}", (60, 60, 60)))
    if seg.cross is not None:
        stats.append((f"骨盆跨越台阶              {int(seg.cross[s] - seg.cross[seg.lo]):4d} 级（本段）", (20, 20, 20)))
    stats.append((f"失败：{rc.fmt_t(seg.fail * DT)} {seg.cause}" if failed else "状态：正常跟踪", (200, 30, 30) if failed else (20, 120, 50)))
    if failed:
        stats.append(("评测在此终止；录制关闭了终止，继续回放", (110, 110, 110)))
    lh = fonts["stat"].size * 1.3
    sx, sy = x0 + pad, y0 + ph - pad - len(stats) * lh - pad
    sw = max(draw.textlength(txt, font=fonts["stat"]) for txt, _ in stats)
    draw.rounded_rectangle([sx, sy, sx + sw + 2 * pad, sy + len(stats) * lh + pad], radius=int(10 * k) + 2, fill=(255, 255, 255, 215))
    for li, (txt, col) in enumerate(stats):
        draw.text((sx + pad, sy + 0.5 * pad + li * lh), txt, font=fonts["stat"], fill=col)
    if full:  # error sparkline of the last 10 s and the top-view path
        gx1, gy1 = x0 + pw - pad, y0 + ph - pad
        gw, gh = int(420 * k), int(150 * k)
        gx0, gy0 = gx1 - gw, gy1 - gh
        draw.rounded_rectangle([gx0 - pad, gy0 - 34 * k - pad, gx1, gy1], radius=10, fill=(255, 255, 255, 215))
        draw.text((gx0, gy0 - 34 * k - 0.5 * pad), "关键点误差（近 10 s）", font=fonts["small"], fill=(30, 30, 30))
        a = max(0, s - 499)
        e = t.err_mm[a : s + 1]
        top = 150.0 if np.nanmax(e) < 150 else 400.0
        for ref in (50, 100) if top <= 150 else (100, 200, 300):
            yr = gy1 - 0.5 * pad - (gh - pad) * ref / top
            draw.line([(gx0, yr), (gx1 - pad, yr)], fill=(200, 200, 200), width=1)
            draw.text((gx0 + 2, yr - 18 * k - 2), f"{ref}", font=fonts["small"], fill=(140, 140, 140))
        if len(e) > 1:
            xs = gx0 + (gw - pad) * (np.arange(len(e)) + (500 - len(e))) / 499
            ys = gy1 - 0.5 * pad - (gh - pad) * np.clip(np.nan_to_num(e), 0, top) / top
            draw.line(list(zip(xs.tolist(), ys.tolist())), fill=(235, 120, 20), width=3)
        rc.draw_minimap(draw, {"small": fonts["small"]}, x0 + pw - rc.MAP - pad - 8, int(by + bh + 70 * k + 40), t, seg.lo, s)
    draw_depth_inset(canvas, x0, y0, pw, ph, seg, s, fonts, k, full)
    if seg.cfg.get("label"):  # policy tag under the info box (side-by-side comparisons)
        tag = seg.cfg["label"]
        tw = draw.textlength(tag, font=fonts["badge"])
        ty = y0 + pad + sum(heights) + 1.4 * pad + pad
        col = tuple(seg.cfg.get("label_color", (40, 90, 200)))
        draw.rounded_rectangle([x0 + pad, ty, x0 + pad + tw + 2 * pad, ty + fonts["badge"].size * 1.45], radius=int(10 * k) + 2, fill=(*col, 230))
        draw.text((x0 + 2 * pad, ty + 0.12 * fonts["badge"].size * 1.45), tag, font=fonts["badge"], fill=(255, 255, 255))
    if failed:
        msg = "摔倒 / 终止" if seg.fell else "触发终止（未摔倒）"
        mw = draw.textlength(msg, font=fonts["fail"])
        draw.text((x0 + (pw - mw) / 2, y0 + ph * 0.30), msg, font=fonts["fail"], fill=(215, 30, 30, 240), stroke_width=3, stroke_fill=(255, 255, 255, 200))


_DEPTH_LUT = None


def draw_depth_inset(canvas: Image.Image, x0: int, y0: int, pw: int, ph: int, seg: Seg, s: int, fonts: dict, k: float, full: bool) -> None:
    """Right-side inset: the depth frame the policy sees at step s (turbo, near red / far blue, holes black)."""
    global _DEPTH_LUT
    if seg.depth is None:
        return
    if _DEPTH_LUT is None:
        import matplotlib

        lut = (matplotlib.colormaps["turbo"](np.linspace(0.95, 0.1, 255))[:, :3] * 255).astype(np.uint8)  # 1 = near -> red
        _DEPTH_LUT = np.concatenate([np.zeros((1, 3), np.uint8), lut])
    fr = seg.depth[min(s, len(seg.depth) - 1)]
    rgb = _DEPTH_LUT[fr]
    scale = max(5, int(round(11 * k)))
    img = Image.fromarray(rgb).resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.NEAREST)
    draw = ImageDraw.Draw(canvas, "RGBA")
    lines = [f"策略输入：深度 {rgb.shape[1]}×{rgb.shape[0]}", "近红 远蓝 0.3–3 m · 黑 = 空洞/自遮挡"]
    lw = max(draw.textlength(t, font=fonts["small"]) for t in lines)
    lh = fonts["small"].size * 1.3
    pad = int(16 * k) + 4
    bw = max(img.width, lw) + 16
    bx1 = x0 + pw - pad
    bx = bx1 - bw
    by = y0 + int(ph * (0.42 if full else 0.17))
    draw.rounded_rectangle([bx, by, bx1, by + 2 * lh + img.height + 16], radius=8, fill=(20, 24, 32, 205))
    for i, t in enumerate(lines):
        draw.text((bx + 8, by + 4 + i * lh), t, font=fonts["small"], fill=(255, 255, 255))
    canvas.paste(img, (int(bx + (bw - img.width) / 2), int(by + 2 * lh + 8)))


def render_seg_frame(renderer, model, robot, ghost, opt, seg: Seg, f: int) -> tuple[np.ndarray, int]:
    s = min(seg.lo + int(round(f / FPS / DT)), seg.hi - 1)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = float(seg.cfg.get("distance", 3.0))
    cam.elevation = float(seg.cfg.get("elevation", -18.0))
    cam.azimuth = float(seg.cfg.get("azimuth", 135.0))
    target = seg.t.lookat(s)
    seg.lookat = target if seg.lookat is None else seg.lookat + 0.15 * (target - seg.lookat)
    return rc.render_panel(renderer, model, robot, ghost, cam, opt, seg.t, s, seg.lookat), s


def card_frame(item: dict, fonts: dict) -> np.ndarray:
    canvas = Image.new("RGB", (W, H), (24, 28, 36))
    draw = ImageDraw.Draw(canvas)
    lines = [(item["title"], fonts["title"], (255, 255, 255))] + [(ln, fonts["card"], (215, 222, 232)) for ln in item.get("lines", [])]
    total = sum(f.size * 1.6 for _, f, _ in lines)
    y = (H - total) / 2
    for txt, f, col in lines:
        tw = draw.textlength(txt, font=f)
        draw.text(((W - tw) / 2, y), txt, font=f, fill=col)
        y += f.size * 1.6
    return np.asarray(canvas)


def writer(path: Path):
    return imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=8, ffmpeg_params=["-preset", "medium"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--only", type=int, nargs="*", default=None, help="render only these item indices into <out>_parts/")
    p.add_argument("--concat", action="store_true", help="concatenate the rendered parts into the output")
    p.add_argument("--test-frames", type=int, default=None)
    a = p.parse_args()
    spec = json.loads(a.spec.read_text())
    out = Path(spec["output"])
    parts = out.parent / (out.stem + "_parts")
    parts.mkdir(parents=True, exist_ok=True)
    items = spec["items"]
    if a.concat:
        lst = parts / "list.txt"
        lst.write_text("".join(f"file '{(parts / f'{i:02d}.mp4').resolve()}'\n" for i in range(len(items))))
        import imageio_ffmpeg

        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
        summ = [json.loads((parts / f"{i:02d}.json").read_text()) for i in range(len(items))]
        t0 = 0.0
        for sm in summ:
            sm["video_start_s"] = round(t0, 2)
            t0 += sm["video_duration_s"]
        (out.parent / (out.stem + "_summary.json")).write_text(json.dumps(summ, ensure_ascii=False, indent=1))
        print("saved", out, f"{t0:.1f} s")
        return
    font_dir = Path(spec["font_dir"])
    F = lambda name, size: ImageFont.truetype(str(font_dir / name), size)
    card_fonts = {"title": F("NotoSansCJKsc-Bold.otf", 56), "card": F("NotoSansCJKsc-Regular.otf", 32)}
    rc.PW, rc.PH = W, H  # offscreen buffer for full-screen panels
    terrain = None
    for it in items:
        for sc in ([it["seg"]] if it["type"] == "single" else it.get("panels", [])):
            roll = np.load(sc["rollout"], allow_pickle=False)
            terrain = {k: v for k, v in json.loads(str(roll["metadata_json"]))["terrain"].items() if k not in ("col", "row")}
            break
        if terrain:
            break
    model = rc.build_model(terrain)
    style_terrain(model, terrain)
    ground = Ground(terrain)
    body_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in rc.TRACKED])
    robot, ghost = mujoco.MjData(model), mujoco.MjData(model)
    opt = mujoco.MjvOption()
    opt.geomgroup[3:] = 0
    full_fonts, grid_fonts, side_fonts = make_fonts(font_dir, 1.0), make_fonts(font_dir, 0.62), make_fonts(font_dir, 0.74)
    todo = a.only if a.only is not None else list(range(len(items)))
    with mujoco.Renderer(model, height=H, width=W) as big, mujoco.Renderer(model, height=H // 2, width=W // 2) as small, mujoco.Renderer(model, height=H, width=W // 2) as half:
        for ii in todo:
            it = items[ii]
            dur = float(it["duration_s"])
            count = int(round(dur * FPS)) if a.test_frames is None else min(a.test_frames, int(round(dur * FPS)))
            segs = []
            if it["type"] == "single":
                segs = [Seg(it["seg"], model, body_ids, ground, dur)]
            elif it["type"] in ("grid", "side"):
                segs = [Seg(sc, model, body_ids, ground, dur) for sc in it["panels"]]
            if it["type"] != "card":
                count = min(count, min(int(round((sg.hi - sg.lo) * DT * FPS)) for sg in segs))
            wr = writer(parts / f"{ii:02d}.mp4")
            for f in range(count):
                if it["type"] == "card":
                    frame = card_frame(it, card_fonts)
                elif it["type"] == "single":
                    img, s = render_seg_frame(big, model, robot, ghost, opt, segs[0], f)
                    canvas = Image.fromarray(img)
                    draw_overlay(canvas, 0, 0, W, H, segs[0], s, full_fonts, full=True)
                    frame = np.asarray(canvas)
                elif it["type"] == "side":
                    canvas = Image.new("RGB", (W, H))
                    for k, sg in enumerate(segs):
                        img, s = render_seg_frame(half, model, robot, ghost, opt, sg, f)
                        canvas.paste(Image.fromarray(img), (k * (W // 2), 0))
                        draw_overlay(canvas, k * (W // 2), 0, W // 2, H, sg, s, side_fonts, full=True)
                    ImageDraw.Draw(canvas).line([(W // 2, 0), (W // 2, H)], fill=(30, 30, 30), width=4)
                    frame = np.asarray(canvas)
                else:
                    canvas = Image.new("RGB", (W, H))
                    for k, sg in enumerate(segs):
                        img, s = render_seg_frame(small, model, robot, ghost, opt, sg, f)
                        x0, y0 = (k % 2) * (W // 2), (k // 2) * (H // 2)
                        canvas.paste(Image.fromarray(img), (x0, y0))
                        draw_overlay(canvas, x0, y0, W // 2, H // 2, sg, s, grid_fonts, full=False)
                    d = ImageDraw.Draw(canvas, "RGBA")
                    d.line([(W // 2, 0), (W // 2, H)], fill=(30, 30, 30), width=4)
                    d.line([(0, H // 2), (W, H // 2)], fill=(30, 30, 30), width=4)
                    frame = np.asarray(canvas)
                wr.append_data(frame)
            wr.close()
            summ = {"item": ii, "type": it["type"], "title": it.get("title"), "video_duration_s": round(count / FPS, 3), "segments": [sg.summary() for sg in segs]}
            (parts / f"{ii:02d}.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
            print(json.dumps(summ, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
