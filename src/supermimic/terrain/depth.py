"""Rendered depth camera for the single-stage depth policy (depth-direct plan).

The terrain is static and known, so the image the policy sees is a pure function of the robot pose:
the same ``DepthSensor.observe`` renders it from the simulator's qpos (data collection) and from the
world model's predicted states (imagination). Every policy input is computed the way the robot computes it:

* camera: G1 head D435 on ``torso_link`` (URDF ``d435_joint``: offset (0.0576, 0.0175, 0.4199), pitch 47.6 deg
  down), 32 x 24 pinhole rays over 87 x 58 deg, the top ``crop_top`` rows dropped (beyond the far clip);
* render: one Warp kernel per call. Terrain = ray march on the 2.5D height map ``h`` (5 cm steps + bisection);
  robot = the model's collision capsules / spheres of the limbs (true pose, *perturbed* extrinsics);
* self-occlusion mask: the same capsules inflated by ``mask_inflate``, cast from the *nominal* camera (what the
  robot can compute from its joint encoders and the calibrated mount); masked pixels become holes;
* sensor effects (``noise``): z < near -> hole, range-dependent Gaussian noise, edge holes, pixel dropout;
* preprocessing: z clipped to [near, far], normalised to about [-1, 1], holes -> 0, plus a validity channel;
* per frame also g_cam: gravity in the nominal camera frame (IMU + waist FK + calibrated mount);
* timing: one frame is sampled every ``sample_every`` control steps (25 Hz) with a per-episode phase; the
  image is rendered from the state ``lag`` (+ 0/1 jitter) control steps old; a dropped frame repeats the last.

No gradient flows through the renderer (the custom op has no autograd formula; inputs are detached).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import mujoco
import numpy as np
import torch

from supermimic.kinematics.rotations import matrix_from_quat, quat_apply, quat_from_angle_axis, quat_mul

if TYPE_CHECKING:
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.terrain.heightmap import HeightGrid, TerrainMap

DEPTH_FEAT_EXTRA = 3  # g_cam appended to every frame feature


@dataclass
class DepthCfg:
    # Camera (G1 URDF d435_joint on torso_link; confirm against the real mount).
    width: int = 32
    height: int = 24
    crop_top: int = 4  # rows dropped at the top of the image (look beyond the far clip)
    hfov_deg: float = 87.0
    vfov_deg: float = 58.0
    mount_pos: tuple[float, float, float] = (0.0576235, 0.01753, 0.41987)
    mount_pitch: float = 0.8307767239493009  # rad, positive = looking down
    near: float = 0.3
    far: float = 3.0
    # Ray march on the height map.
    march_step: float = 0.05
    bisect_iters: int = 5
    # Robot body in the image: capsules rendered with the true extrinsics, masked with the nominal ones.
    render_body: bool = True
    mask_inflate: float = 0.025
    # Sensor noise (only with ``noise``; evaluation turns it off).
    noise: bool = True
    sigma0: float = 0.005  # m
    sigma1: float = 0.01  # 1/m: sigma = sigma0 + sigma1 * z^2
    edge_threshold: float = 0.08  # m, |second difference| of the depth along x or y (0 on planes: ~1 cm)
    edge_hole_prob: float = 0.5
    pixel_dropout: float = 0.02
    ext_pos_noise: float = 0.01  # m, per episode, uniform half-width
    ext_rot_noise_deg: float = 2.5  # per episode, uniform half-width per axis
    grav_noise: float = 0.05  # g_cam, uniform half-width (as the projected-gravity observation)
    # Timing: one frame every ``sample_every`` control steps, rendered ``lag`` (+ jitter) steps late.
    sample_every: int = 2
    lag_min: int = 1
    lag_max: int = 3
    lag_jitter: bool = True  # + U{0, 1} per frame (30 Hz camera sampled at 25 Hz)
    frame_drop: float = 0.02  # probability that a sample repeats the previous frame
    # Frame history read by the policy: ``slots`` frames, ``slot_stride`` samples apart (newest first).
    slots: int = 16
    slot_stride: int = 3

    @property
    def out_h(self) -> int:
        return self.height - self.crop_top

    @property
    def num_pixels(self) -> int:
        return self.out_h * self.width

    @property
    def hist_frames(self) -> int:
        """Frames kept so that the oldest slot is still available."""
        return (self.slots - 1) * self.slot_stride + 1

    @property
    def state_hist(self) -> int:
        """qpos history length for the latency (lag up to lag_max + jitter)."""
        return self.lag_max + (2 if self.lag_jitter else 1)

    @property
    def start_hist(self) -> int:
        """Logged states an imagined rollout needs to rebuild the frame history before its first step."""
        return self.sample_every * self.hist_frames + self.state_hist + 1

    def slot_index(self) -> list[int]:
        """History rows (oldest first) read by the policy; the newest frame is the last row."""
        F = self.hist_frames
        return [F - 1 - self.slot_stride * k for k in reversed(range(self.slots))]


class DepthParams(NamedTuple):
    """Per-episode camera parameters, batch shape (B,)."""

    ext_dp: torch.Tensor  # (B, 3) extrinsic position error, torso frame
    ext_dq: torch.Tensor  # (B, 4) extrinsic rotation error (camera frame), wxyz
    lag: torch.Tensor  # (B,) long, control steps
    phase: torch.Tensor  # (B,) long in [0, sample_every)


def _rpy_quat(rpy: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion of R = Rz(yaw) Ry(pitch) Rx(roll)."""
    ax = torch.eye(3, device=rpy.device)
    q = [quat_from_angle_axis(rpy[..., i], ax[i].expand_as(rpy)) for i in range(3)]
    return quat_mul(q[2], quat_mul(q[1], q[0]))


def sample_depth_params(batch: int, cfg: DepthCfg, device, generator: torch.Generator | None = None) -> DepthParams:
    def u(shape, half):
        return (torch.rand(shape, device=device, generator=generator) * 2 - 1) * half

    phase = torch.randint(0, cfg.sample_every, (batch,), device=device, generator=generator)
    if not cfg.noise:
        unit = torch.zeros(batch, 4, device=device)
        unit[:, 0] = 1.0
        return DepthParams(torch.zeros(batch, 3, device=device), unit, torch.full((batch,), cfg.lag_min, device=device, dtype=torch.long), phase)
    dq = _rpy_quat(u((batch, 3), math.radians(cfg.ext_rot_noise_deg)))
    lag = torch.randint(cfg.lag_min, cfg.lag_max + 1, (batch,), device=device, generator=generator)
    return DepthParams(u((batch, 3), cfg.ext_pos_noise), dq, lag, phase)


def resample_depth_params_(params: DepthParams, env_ids: torch.Tensor, cfg: DepthCfg) -> None:
    new = sample_depth_params(env_ids.numel(), cfg, params.lag.device)
    for dst, src in zip(params, new):
        dst[env_ids] = src


# ---------------------------------------------------------------------------------------------- Warp kernel

_WP = None


def _warp():
    """Build the Warp kernel once (lazy: importing this module must not initialise Warp)."""
    global _WP
    if _WP is not None:
        return _WP
    import warp as wp

    wp.config.quiet = True
    wp.init()

    @wp.func
    def hgt(h: wp.array2d(dtype=wp.float16), x: float, y: float, x0: float, y0: float, inv: float, nx: int, ny: int):
        # Bilinear between cell centres (as ``heightmap.bilinear``); 0 outside the grid.
        u = (x - x0) * inv
        v = (y - y0) * inv
        if u < 0.0 or v < 0.0 or u > float(nx) or v > float(ny):
            return 0.0
        fu = u - 0.5
        fv = v - 0.5
        i = wp.clamp(int(wp.floor(fu)), 0, nx - 2)
        j = wp.clamp(int(wp.floor(fv)), 0, ny - 2)
        ax = wp.clamp(fu - float(i), 0.0, 1.0)
        ay = wp.clamp(fv - float(j), 0.0, 1.0)
        a = float(h[i, j])
        b = float(h[i + 1, j])
        c = float(h[i, j + 1])
        d = float(h[i + 1, j + 1])
        return (1.0 - ax) * ((1.0 - ay) * a + ay * c) + ax * ((1.0 - ay) * b + ay * d)

    @wp.func
    def capsule(ro: wp.vec3, rd: wp.vec3, pa: wp.vec3, pb: wp.vec3, r: float):
        # First positive hit distance of the ray with a capsule (sphere if pa == pb), else -1.
        ba = pb - pa
        oa = ro - pa
        baba = wp.dot(ba, ba)
        t = float(-1.0)
        if baba > 1.0e-12:
            bard = wp.dot(ba, rd)
            baoa = wp.dot(ba, oa)
            rdoa = wp.dot(rd, oa)
            oaoa = wp.dot(oa, oa)
            a = baba - bard * bard
            if a > 1.0e-10:
                b = baba * rdoa - baoa * bard
                c = baba * oaoa - baoa * baoa - r * r * baba
                hh = b * b - a * c
                if hh >= 0.0:
                    tc = (-b - wp.sqrt(hh)) / a
                    y = baoa + tc * bard
                    if y > 0.0 and y < baba and tc > 0.0:
                        return tc
        # End spheres (or the whole capsule when the ray is parallel to its axis).
        for k in range(2):
            ce = pa
            if k == 1:
                ce = pb
            oc = ro - ce
            b2 = wp.dot(rd, oc)
            c2 = wp.dot(oc, oc) - r * r
            h2 = b2 * b2 - c2
            if h2 >= 0.0:
                ts = -b2 - wp.sqrt(h2)
                if ts > 0.0 and (t < 0.0 or ts < t):
                    t = ts
        return t

    @wp.kernel
    def rays(
        h: wp.array2d(dtype=wp.float16),
        x0: float,
        y0: float,
        inv: float,
        nx: int,
        ny: int,
        cam_o: wp.array(dtype=wp.vec3),
        cam_R: wp.array(dtype=wp.mat33),
        nom_o: wp.array(dtype=wp.vec3),
        nom_R: wp.array(dtype=wp.mat33),
        dirs: wp.array(dtype=wp.vec3),
        npix: int,
        cap_a: wp.array(dtype=wp.vec3),
        cap_b: wp.array(dtype=wp.vec3),
        cap_r: wp.array(dtype=float),
        ncap: int,
        inflate: float,
        tmin: float,
        dt: float,
        far: float,
        nbisect: int,
        depth: wp.array(dtype=float),
        mask: wp.array(dtype=float),
    ):
        tid = wp.tid()
        c = tid // npix
        dc = dirs[tid - c * npix]
        o = cam_o[c]
        d = cam_R[c] * dc
        tmax = far / dc[0] + dt
        # Terrain: march until below the surface, then bisect.
        t = tmin
        prev = tmin
        hit = int(0)
        while t <= tmax:
            p = o + d * t
            if p[2] <= hgt(h, p[0], p[1], x0, y0, inv, nx, ny):
                hit = 1
                break
            prev = t
            t += dt
        tt = float(1.0e9)
        if hit == 1:
            lo = prev
            hi = t
            for _k in range(nbisect):
                m = 0.5 * (lo + hi)
                p = o + d * m
                if p[2] <= hgt(h, p[0], p[1], x0, y0, inv, nx, ny):
                    hi = m
                else:
                    lo = m
            tt = hi
        # Robot body with the true extrinsics; self-occlusion mask with the nominal ones (inflated).
        on = nom_o[c]
        dn = nom_R[c] * dc
        mk = float(0.0)
        for k in range(ncap):
            pa = cap_a[c * ncap + k]
            pb = cap_b[c * ncap + k]
            r = cap_r[k]
            tc = capsule(o, d, pa, pb, r)
            if tc > 0.0 and tc < tt:
                tt = tc
            if mk == 0.0:
                if capsule(on, dn, pa, pb, r + inflate) > 0.0:
                    mk = 1.0
        zd = tt * dc[0]
        if zd > far:
            zd = far
        depth[tid] = zd
        mask[tid] = mk

    _WP = (wp, rays)
    return _WP


@torch.library.custom_op("supermimic::depth_rays", mutates_args=())
def depth_rays(
    h: torch.Tensor,
    spec: list[float],
    cam_o: torch.Tensor,
    cam_R: torch.Tensor,
    nom_o: torch.Tensor,
    nom_R: torch.Tensor,
    dirs: torch.Tensor,
    cap_a: torch.Tensor,
    cap_b: torch.Tensor,
    cap_r: torch.Tensor,
    march: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """z-depth (B, P) clipped at far (misses = far) and the self-occlusion mask (B, P) in {0, 1}.

    h (nx, ny) fp16; spec [x0, y0, 1/res, nx, ny]; cam_* (B, 3) / (B, 3, 3) world; dirs (P, 3) camera-frame
    unit rays (x forward); cap_a / cap_b (B, C, 3) world capsule end points, cap_r (C,); march
    [tmin, dt, far, n_bisect, inflate]."""
    wp, kernel = _warp()
    B, P, C = cam_o.shape[0], dirs.shape[0], cap_r.shape[0]
    dev = cam_o.device
    depth = torch.empty(B * P, device=dev, dtype=torch.float32)
    mask = torch.empty(B * P, device=dev, dtype=torch.float32)
    f32 = lambda x: x.detach().to(torch.float32).contiguous()  # noqa: E731
    cap_a_, cap_b_ = f32(cap_a).reshape(-1, 3), f32(cap_b).reshape(-1, 3)
    if C == 0:  # Warp needs non-empty arrays
        cap_a_ = cap_b_ = torch.zeros(1, 3, device=dev)
        cap_r = torch.zeros(1, device=dev)
    with wp.ScopedDevice(wp.device_from_torch(dev)):
        stream = wp.stream_from_torch(torch.cuda.current_stream(dev))
        wp.launch(
            kernel,
            dim=B * P,
            inputs=[
                wp.from_torch(h.detach().contiguous(), dtype=wp.float16),
                float(spec[0]), float(spec[1]), float(spec[2]), int(spec[3]), int(spec[4]),
                wp.from_torch(f32(cam_o), dtype=wp.vec3), wp.from_torch(f32(cam_R), dtype=wp.mat33),
                wp.from_torch(f32(nom_o), dtype=wp.vec3), wp.from_torch(f32(nom_R), dtype=wp.mat33),
                wp.from_torch(f32(dirs), dtype=wp.vec3), P,
                wp.from_torch(cap_a_, dtype=wp.vec3), wp.from_torch(cap_b_, dtype=wp.vec3), wp.from_torch(f32(cap_r)), C,
                float(march[4]), float(march[0]), float(march[1]), float(march[2]), int(march[3]),
                wp.from_torch(depth), wp.from_torch(mask),
            ],
            stream=stream,
        )
    return depth.view(B, P), mask.view(B, P)


@depth_rays.register_fake
def _(h, spec, cam_o, cam_R, nom_o, nom_R, dirs, cap_a, cap_b, cap_r, march):
    B, P = cam_o.shape[0], dirs.shape[0]
    return cam_o.new_empty(B, P), cam_o.new_empty(B, P)


# ---------------------------------------------------------------------------------------------- sensor


class BodyCapsules(NamedTuple):
    body: torch.Tensor  # (C,) long, FK body index
    a: torch.Tensor  # (C, 3) end point in the body frame
    b: torch.Tensor  # (C, 3)
    r: torch.Tensor  # (C,)


_SKIP_GEOMS = ("torso_collision", "head_collision")  # the camera sits on them


def body_capsules(model: mujoco.MjModel, fk: G1Kinematics, device) -> BodyCapsules:
    """Collision capsules / spheres of the robot (limbs, feet, pelvis); the torso and head are skipped."""
    body, a, b, r = [], [], [], []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not name.endswith("_collision") or name in _SKIP_GEOMS:
            continue
        typ = int(model.geom_type[g])
        if typ not in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE):
            continue
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
        half = float(model.geom_size[g][1]) if typ == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        axis = R.reshape(3, 3)[:, 2] * half
        body.append(fk.body_index(bname))
        a.append(model.geom_pos[g] - axis)
        b.append(model.geom_pos[g] + axis)
        r.append(float(model.geom_size[g][0]))
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)  # noqa: E731
    return BodyCapsules(torch.as_tensor(body, dtype=torch.long, device=device), t(a), t(b), t(r))


class DepthObs(NamedTuple):
    img: torch.Tensor  # (B, 2, out_h, W): normalised depth (holes 0), validity
    g_cam: torch.Tensor  # (B, 3) gravity in the nominal camera frame (with IMU noise)
    cam_pos: torch.Tensor  # (B, 3) true camera position, world (visibility of the auxiliary target)
    cam_R: torch.Tensor  # (B, 3, 3) true camera orientation, world


class DepthSensor:
    """Renders policy depth frames from qpos; a plain object (no module buffers), shared by sim and imagination."""

    def __init__(self, cfg: DepthCfg, fk: G1Kinematics, tmap: TerrainMap, model: mujoco.MjModel | None = None):
        if model is None:
            from supermimic.data.motion_convert import load_g1_model

            model = load_g1_model()
        self.cfg, self.fk, self.tmap = cfg, fk, tmap
        dev = self.device = fk.device
        self.torso = fk.body_index("torso_link")
        p = cfg.mount_pitch
        self.mount_q = torch.tensor([math.cos(p / 2), 0.0, math.sin(p / 2), 0.0], device=dev)  # about +y: x tilts down
        self.mount_t = torch.tensor(cfg.mount_pos, device=dev)
        # Camera-frame rays (x forward, y left, z up), rows top -> bottom, the top crop_top rows dropped.
        th, tv = math.tan(math.radians(cfg.hfov_deg) / 2), math.tan(math.radians(cfg.vfov_deg) / 2)
        u = (torch.arange(cfg.width, device=dev) + 0.5) / cfg.width * 2 - 1
        v = 1 - (torch.arange(cfg.height, device=dev) + 0.5) / cfg.height * 2
        v = v[cfg.crop_top :]
        gv, gu = torch.meshgrid(v, u, indexing="ij")
        d = torch.stack([torch.ones_like(gu), -gu * th, gv * tv], -1).reshape(-1, 3)
        self.dirs = (d / d.norm(dim=-1, keepdim=True)).contiguous()
        self.tan_h, self.tan_v = th, tv
        self.tan_top = (1 - 2 * cfg.crop_top / cfg.height) * tv  # top edge of the kept rows
        self.caps = body_capsules(model, fk, dev)
        if not cfg.render_body:
            self.caps = BodyCapsules(*(x[:0] for x in self.caps))
        self.slot_idx = torch.tensor(cfg.slot_index(), device=dev)
        self._mid, self._half = 0.5 * (cfg.far + cfg.near), 0.5 * (cfg.far - cfg.near)
        self._dummy_h = torch.zeros(2, 2, dtype=torch.float16, device=dev)

    # -------------------------------------------------------------- geometry

    def _spec(self) -> list[float]:
        s = self.tmap.spec
        if s is None:  # plane: every lookup is off the (dummy) grid -> height 0
            return [1.0e9, 1.0e9, 1.0, 2.0, 2.0]
        return [s.x0, s.y0, 1.0 / s.res, float(s.nx), float(s.ny)]

    def cameras(self, qpos_w: torch.Tensor, params: DepthParams):
        """True and nominal camera poses + world capsules from world qpos (B, 36). Elementwise quaternion algebra
        only (no matmuls: TF32 would round the poses to ~1e-3)."""
        out = self.fk.forward(qpos_w)
        pos, quat = out["pos"], out["quat"]
        qt, pt = quat[:, self.torso], pos[:, self.torso]
        nom_q = quat_mul(qt, self.mount_q.expand_as(qt))
        nom_o = pt + quat_apply(qt, self.mount_t.expand_as(pt))
        cam_q = quat_mul(nom_q, params.ext_dq)
        cam_o = pt + quat_apply(qt, self.mount_t + params.ext_dp)
        cb = self.caps.body
        qb, pb = quat[:, cb], pos[:, cb]  # (B, C, ·)
        cap_a = pb + quat_apply(qb, self.caps.a.expand_as(pb))
        cap_b = pb + quat_apply(qb, self.caps.b.expand_as(pb))
        return cam_o, matrix_from_quat(cam_q), nom_o, matrix_from_quat(nom_q), cap_a, cap_b

    def render_raw(self, qpos_w: torch.Tensor, params: DepthParams, grid: HeightGrid | None = None):
        """(z-depth (B, P), mask (B, P), cam_o, cam_R, nom_R) without sensor effects."""
        cfg = self.cfg
        cam_o, cam_R, nom_o, nom_R, cap_a, cap_b = self.cameras(qpos_w.detach(), params)
        g = self.tmap.grid if grid is None else grid
        h = self._dummy_h if g is None else g.h
        z, mask = torch.ops.supermimic.depth_rays(
            h, self._spec(), cam_o, cam_R, nom_o, nom_R, self.dirs, cap_a, cap_b, self.caps.r,
            [0.02, cfg.march_step, cfg.far, float(cfg.bisect_iters), cfg.mask_inflate],
        )
        return z, mask, cam_o, cam_R, nom_R

    # -------------------------------------------------------------- sensor model

    def observe(self, qpos: torch.Tensor, origin: torch.Tensor, params: DepthParams, grid: HeightGrid | None = None, noise: bool | None = None) -> DepthObs:
        """Policy frame of the state with qpos (B, 36) (root relative to ``origin`` (B, 3))."""
        cfg = self.cfg
        noise = cfg.noise if noise is None else noise
        qw = torch.cat([qpos[..., 0:3] + origin, qpos[..., 3:]], dim=-1)
        z, mask, cam_o, cam_R, nom_R = self.render_raw(qw, params, grid)
        B = z.shape[0]
        H, W = cfg.out_h, cfg.width
        hole = mask > 0.5
        if noise:
            hole = hole | (torch.rand_like(z) < cfg.pixel_dropout)
            edge = depth_edges(z.view(B, H, W), cfg.edge_threshold)
            hole = hole | (edge.view(B, -1) & (torch.rand_like(z) < cfg.edge_hole_prob))
            z = z + torch.randn_like(z) * (cfg.sigma0 + cfg.sigma1 * z * z)
        valid = (z >= cfg.near) & ~hole
        zn = ((z.clamp(cfg.near, cfg.far) - self._mid) / self._half) * valid
        img = torch.stack([zn, valid.to(zn.dtype)], dim=1).view(B, 2, H, W)
        g = -nom_R[:, 2, :]  # world -z in the nominal camera frame: R^T (0, 0, -1)
        if noise:
            g = g + (torch.rand_like(g) * 2 - 1) * cfg.grav_noise
        return DepthObs(img, g, cam_o, cam_R)

    def frame_draws(self, params: DepthParams, noise: bool | None = None):
        """Per-frame randomness: (lag incl. jitter (B,) long, dropped (B,) bool)."""
        cfg = self.cfg
        noise = cfg.noise if noise is None else noise
        lag = params.lag
        if noise and cfg.lag_jitter:
            lag = lag + (torch.rand_like(lag, dtype=torch.float32) < 0.5).long()
        drop = (torch.rand_like(lag, dtype=torch.float32) < cfg.frame_drop) if noise else torch.zeros_like(lag, dtype=torch.bool)
        return lag.clamp(0, cfg.state_hist - 1), drop

    def is_sample_step(self, step: torch.Tensor, params: DepthParams) -> torch.Tensor:
        return ((step + params.phase) % self.cfg.sample_every) == 0

    # -------------------------------------------------------------- auxiliary-target visibility

    def visible(self, pts_w: torch.Tensor, cam_pos: torch.Tensor, cam_R: torch.Tensor) -> torch.Tensor:
        """(B, Q) points inside the view frustum (kept rows, [near, far]) of any of the cameras (S, B, ·).
        Occlusion by the terrain is not tested."""
        cfg = self.cfg
        rel = pts_w[None] - cam_pos[:, :, None, :]  # (S, B, Q, 3)
        loc = torch.einsum("sbji,sbqj->sbqi", cam_R, rel)
        x, y, z = loc.unbind(-1)
        inside = (x > cfg.near) & (x < cfg.far) & (y.abs() < x * self.tan_h) & (z < x * self.tan_top) & (z > -x * self.tan_v)
        return inside.any(0)


def depth_edges(z: torch.Tensor, threshold: float) -> torch.Tensor:
    """(B, H, W) depth discontinuities: |second difference| along x or y above ``threshold`` (both sides of a jump)."""
    edge = torch.zeros_like(z, dtype=torch.bool)
    edge[:, :, 1:-1] |= (z[:, :, 2:] - 2 * z[:, :, 1:-1] + z[:, :, :-2]).abs() > threshold
    edge[:, 1:-1, :] |= (z[:, 2:, :] - 2 * z[:, 1:-1, :] + z[:, :-2, :]).abs() > threshold
    return edge


def push_rows(hist: torch.Tensor, entry: torch.Tensor, where: torch.Tensor) -> torch.Tensor:
    """hist (F, B, ...) oldest -> newest; append ``entry`` (B, ...) for the envs with ``where`` (B,)."""
    shifted = torch.cat([hist[1:], entry[None]], dim=0)
    w = where.reshape(1, -1, *([1] * (hist.dim() - 2)))
    return torch.where(w, shifted, hist)
