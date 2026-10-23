"""Side-by-side comparison video of two G1 tracking policies on the same clips.

Inputs are two rollout npz files with the layout of ``scripts/record_rollouts.py`` (also produced by
rgmt_mjlab's ``montage_rollout.py``): per clip ``s{i}_root_pos / root_quat / joint_pos`` and the
reference ``s{i}_ref_*``, sample k = state after k control steps vs reference frame k (50 Hz).

Both panels use the same G1 model, camera, ghost and metrics (computed here from the recorded qpos):
  * keypoint error: mean distance of the 14 tracked bodies to the reference re-anchored at the robot
    pelvis xy and yaw (as the BeyondMimic body-position reward);
  * root drift: horizontal distance between robot and reference pelvis;
  * failure: the paper's training termination (pelvis height error > 0.25 m or tilt error > 0.8).

Needs MuJoCo EGL, PIL, imageio-ffmpeg and mjlab (G1 asset); e.g. with rgmt_mjlab's environment:
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=3 python scripts/render_compare.py --spec logs/compare_rgmt/video_spec.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, DT = 1920, 1080, 30, 0.02
HEADER, FOOTER = 196, 132
PW, PH = W // 2, H - HEADER - FOOTER  # one panel
GHOST_RGBA = (0.2, 0.6, 1.0, 0.22)
TRACKED = ("pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link", "right_knee_link",
           "right_ankle_roll_link", "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
           "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link")  # fmt: skip
MAP = 200


def build_model():
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    from mjlab.entity import Entity

    spec = Entity(get_g1_robot_cfg()).spec
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.82, 0.84, 0.86],
                     rgb2=[0.70, 0.72, 0.75], width=512, height=512, mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.55, 0.57, 0.6])  # fmt: skip
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.93, 0.95, 0.98], rgb2=[0.75, 0.80, 0.88], width=512, height=3072)  # fmt: skip
    mat = spec.add_material(name="grid", texrepeat=[4, 4], texuniform=True, reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1], material="grid", contype=0, conaffinity=0)
    spec.worldbody.add_light(pos=[0, 0, 6], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6], castshadow=False)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = PW, PH
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    model = spec.compile()
    assert model.nq == 36, "expected the 29-DoF G1 with a free root"
    return model


def yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def gravity_z(q):
    """z component of world gravity (0, 0, -1) expressed in the body frame of quaternion q (wxyz)."""
    w, x, y, z = q
    return -(1 - 2 * (x * x + y * y))


def aligned_ghost(robot_pos, robot_quat, ref_pos, ref_quat):
    """Reference root moved to the robot pelvis xy and yaw (z, roll, pitch stay the reference's)."""
    delta = yaw(robot_quat) - yaw(ref_quat)
    rot = np.array([np.cos(delta / 2), 0.0, 0.0, np.sin(delta / 2)])
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, rot, np.asarray(ref_quat, dtype=np.float64))
    return np.array([robot_pos[0], robot_pos[1], ref_pos[2]]), quat


def set_qpos(model, data, pos, quat, joints):
    data.qpos[:3], data.qpos[3:7], data.qpos[7:] = pos, quat, joints
    mujoco.mj_kinematics(model, data)


class Track:
    """One policy's rollout of one clip plus the per-sample metrics."""

    def __init__(self, roll, i):
        k = f"s{i}_"
        self.rp, self.rq, self.jp = roll[k + "root_pos"], roll[k + "root_quat"], roll[k + "joint_pos"]
        self.gp, self.gq, self.gj = roll[k + "ref_root_pos"], roll[k + "ref_root_quat"], roll[k + "ref_joint_pos"]
        self.n = len(self.rp)
        self.drift = np.linalg.norm(self.rp[:, :2] - self.gp[:, :2], axis=-1)
        dz = np.abs(self.gp[:, 2] - self.rp[:, 2])
        tilt = np.abs(np.array([gravity_z(q) for q in self.gq]) - np.array([gravity_z(q) for q in self.rq]))
        bad = np.flatnonzero((dz > 0.25) | (tilt > 0.8))
        self.first_fail = int(bad[0]) if bad.size else None
        self.err_mm = np.full(self.n, np.nan)


def keypoint_errors(model, track, body_ids, lo, hi):
    a, b = mujoco.MjData(model), mujoco.MjData(model)
    for s in range(lo, hi):
        set_qpos(model, a, track.rp[s], track.rq[s], track.jp[s])
        gp, gq = aligned_ghost(track.rp[s], track.rq[s], track.gp[s], track.gq[s])
        set_qpos(model, b, gp, gq, track.gj[s])
        track.err_mm[s] = 1000.0 * np.linalg.norm(a.xpos[body_ids] - b.xpos[body_ids], axis=-1).mean()


def render_panel(renderer, model, robot, ghost, cam, opt, track, s, lookat):
    set_qpos(model, robot, track.rp[s], track.rq[s], track.jp[s])
    gpos, gquat = aligned_ghost(track.rp[s], track.rq[s], track.gp[s], track.gq[s])
    set_qpos(model, ghost, gpos, gquat, track.gj[s])
    cam.lookat[:] = lookat
    renderer.update_scene(robot, camera=cam, scene_option=opt)
    scene = renderer.scene
    before = scene.ngeom
    mujoco.mjv_addGeoms(model, ghost, opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
    for g in range(before, scene.ngeom):
        scene.geoms[g].rgba[:] = GHOST_RGBA
    return renderer.render()


def draw_minimap(draw, fonts, x0, y0, track, lo, s):
    rxy, gxy = track.rp[lo : s + 1, :2], track.gp[lo : s + 1, :2]
    allp = np.concatenate((track.rp[lo:, :2][: 3001], track.gp[lo:, :2][: 3001]))
    lo_xy, hi_xy = allp.min(0), allp.max(0)
    span = max(float((hi_xy - lo_xy).max()), 1.0) * 1.15
    c = (lo_xy + hi_xy) / 2

    def px(xy):
        u = (xy - c) / span + 0.5
        return [tuple(p) for p in np.stack((x0 + u[:, 0] * MAP, y0 + (1 - u[:, 1]) * MAP), -1)]

    draw.rectangle([x0 - 8, y0 - 30, x0 + MAP + 8, y0 + MAP + 8], fill=(255, 255, 255, 210))
    draw.text((x0 - 2, y0 - 29), f"俯视路径（框宽 {span:.1f} m）", font=fonts["small"], fill=(30, 30, 30))
    for pts, color in ((px(gxy), (40, 110, 220, 255)), (px(rxy), (235, 120, 20, 255))):
        if len(pts) > 1:
            draw.line(pts, fill=color, width=3)
        if pts:
            x, y = pts[-1]
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color)


def fmt_t(sec):
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True, help="video spec json (sides, segments, output)")
    p.add_argument("--test-frames", type=int, default=None, help="render only N frames per segment")
    a = p.parse_args()
    spec = json.loads(a.spec.read_text())
    global HEADER, FOOTER, PH
    HEADER, FOOTER = int(spec.get("header_height", HEADER)), int(spec.get("footer_height", FOOTER))
    PH = H - HEADER - FOOTER
    sides = spec["sides"]
    rolls = [np.load(s["rollout"], allow_pickle=False) for s in sides]
    metas = [json.loads(str(r["metadata_json"])) for r in rolls]
    font_dir = Path(spec["font_dir"])
    F = lambda name, size: ImageFont.truetype(str(font_dir / name), size)  # noqa: E731
    fonts = {"title": F("NotoSansCJKsc-Bold.otf", 34), "body": F("NotoSansCJKsc-Regular.otf", 21), "seg": F("NotoSansCJKsc-Bold.otf", 30),
             "small": F("NotoSansCJKsc-Regular.otf", 17), "stat": F("NotoSansCJKsc-Regular.otf", 22), "fail": F("NotoSansCJKsc-Bold.otf", 44)}  # fmt: skip
    model = build_model()
    body_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in TRACKED])
    assert (body_ids >= 0).all()
    robot, ghost = mujoco.MjData(model), mujoco.MjData(model)
    opt = mujoco.MjvOption()
    opt.geomgroup[3:] = 0
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation = 3.0, -10.0
    out = Path(spec["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    seg_out = out.parent / (out.stem + "_segments")
    seg_out.mkdir(exist_ok=True)
    summary = []
    with mujoco.Renderer(model, height=PH, width=PW) as renderer, imageio.get_writer(
        str(out), fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=8, ffmpeg_params=["-preset", "medium"]
    ) as writer:  # fmt: skip
        for si, seg in enumerate(spec["segments"]):
            idx = [next(j for j, m in enumerate(meta["segments"]) if m["clip"] == seg["clip"]) for meta in metas]
            tracks = [Track(r, j) for r, j in zip(rolls, idx)]
            lo = int(round(seg["start_s"] / DT))
            hi = min(lo + int(round(seg["duration_s"] / DT)), *(t.n for t in tracks))
            for t in tracks:
                keypoint_errors(model, t, body_ids, lo, hi)
            count = int(round((hi - lo) * DT * FPS))
            if a.test_frames:
                count = min(count, a.test_frames)
            cam.azimuth = float(seg.get("azimuth", 135.0))
            lookats = [np.array([t.rp[lo, 0], t.rp[lo, 1], 0.8]) for t in tracks]
            seg_writer = imageio.get_writer(str(seg_out / f"{si + 1}_{seg['clip']}.mp4"), fps=FPS, codec="libx264", quality=8,
                                            pixelformat="yuv420p", macro_block_size=8)  # fmt: skip
            for f in range(count):
                s = min(lo + int(round(f / FPS / DT)), hi - 1)
                canvas = Image.new("RGB", (W, H), (246, 247, 249))
                draw = ImageDraw.Draw(canvas, "RGBA")
                for k, (t, side) in enumerate(zip(tracks, sides)):
                    target = np.array([t.rp[s, 0], t.rp[s, 1], 0.8])
                    lookats[k] = target if f == 0 else lookats[k] + 0.15 * (target - lookats[k])
                    img = render_panel(renderer, model, robot, ghost, cam, opt, t, s, lookats[k])
                    x0 = k * PW
                    canvas.paste(Image.fromarray(img), (x0, HEADER))
                    # Header: project, resources, envs, training time.
                    draw.rectangle([x0, 0, x0 + PW, HEADER], fill=tuple(side["color_bg"]))
                    draw.text((x0 + 24, 12), side["name"], font=fonts["title"], fill=(255, 255, 255))
                    for li, line in enumerate(side["lines"]):
                        draw.text((x0 + 24, 62 + 32 * li), line, font=fonts["body"], fill=(240, 244, 250))
                    # Live stats (window-local running mean of the last second).
                    w0 = max(lo, s - 49)
                    err = float(np.nanmean(t.err_mm[w0 : s + 1]))
                    failed = t.first_fail is not None and s >= t.first_fail
                    sx, sy = x0 + 18, HEADER + PH - 112
                    draw.rectangle([sx - 8, sy - 8, sx + 470, sy + 98], fill=(255, 255, 255, 215))
                    draw.text((sx, sy), f"关键点误差（航向对齐）  {err:6.1f} mm", font=fonts["stat"], fill=(20, 20, 20))
                    draw.text((sx, sy + 30), f"根部水平偏移              {t.drift[s]:6.2f} m", font=fonts["stat"], fill=(20, 20, 20))
                    status = f"失败：{fmt_t(t.first_fail * DT)} 触发终止条件" if failed else "状态：正常跟踪"
                    draw.text((sx, sy + 60), status, font=fonts["stat"], fill=(200, 30, 30) if failed else (20, 120, 50))
                    if failed:
                        draw.text((x0 + PW // 2 - 150, HEADER + 24), "摔倒 / 终止", font=fonts["fail"], fill=(210, 30, 30, 235))
                    draw_minimap(draw, fonts, x0 + PW - MAP - 22, HEADER + 44, t, lo, s)
                draw.line([(PW, 0), (PW, H)], fill=(40, 40, 40), width=3)
                # Footer: trajectory name (+ optional clip time and protocol note).
                fy = H - FOOTER
                draw.rectangle([0, fy, W, H], fill=(32, 36, 44))
                if spec.get("footer_details", True):
                    clip_t = s * DT
                    draw.text((24, fy + 10), f"{si + 1}/{len(spec['segments'])}  难度：{seg['difficulty']}  ·  {seg['clip']}（{seg['desc']}）", font=fonts["seg"], fill=(255, 255, 255))
                    draw.text((24, fy + 54), f"clip 时间 {fmt_t(clip_t)}（本段 {fmt_t(lo * DT)}–{fmt_t(hi * DT)}，clip 全长 {fmt_t(min(t.n for t in tracks) * DT)}）    两侧均从 clip 第 0 帧起确定性回放，截取本段", font=fonts["body"], fill=(225, 230, 238))
                    draw.text((24, fy + 92), spec["note"], font=fonts["small"], fill=(185, 192, 205))
                else:
                    name = seg["clip"]
                    tw = draw.textlength(name, font=fonts["seg"])
                    draw.text(((W - tw) / 2, fy + (FOOTER - 42) / 2), name, font=fonts["seg"], fill=(255, 255, 255))
                frame = np.asarray(canvas)
                writer.append_data(frame)
                seg_writer.append_data(frame)
            seg_writer.close()
            summary.append({"clip": seg["clip"], "difficulty": seg.get("difficulty"), "window_s": [lo * DT, hi * DT], "sides": [
                {"name": side["name"], "first_failure_s": None if t.first_fail is None else round(t.first_fail * DT, 2),
                 "keypoint_err_mm_window_mean": round(float(np.nanmean(t.err_mm[lo:hi])), 1),
                 "root_drift_m_window_mean": round(float(t.drift[lo:hi].mean()), 2)} for side, t in zip(sides, tracks)]})  # fmt: skip
            print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
    (out.parent / (out.stem + "_summary.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
