"""GAE targets, policy objective, and a short end-to-end training smoke run."""

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"


def test_gae_targets_match_hand_computation():
    from supermimic.imagination.objective import gae_targets

    H, n = 4, 3
    g, lam = 0.9, 0.8
    torch.manual_seed(0)
    r = torch.rand(H, n)
    d = torch.rand(H, n) * 0.3
    V = torch.rand(H + 1, n)
    out = gae_targets(r, d, V, g, lam)
    R = V[H].clone()
    for t in range(H - 1, -1, -1):
        R = r[t] + g * (1 - d[t]) * ((1 - lam) * V[t + 1] + lam * R)
        np.testing.assert_allclose(out[t].numpy(), R.numpy(), atol=1e-6)
    # lambda = 1, d = 0 -> Monte-Carlo discounted return with bootstrap.
    out1 = gae_targets(r, torch.zeros_like(d), V, g, 1.0)
    mc = r[0] + g * r[1] + g**2 * r[2] + g**3 * r[3] + g**4 * V[4]
    np.testing.assert_allclose(out1[0].numpy(), mc.numpy(), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")
def test_training_smoke(tmp_path):
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    cfg = TrainConfig(run_name="smoke", log_root=tmp_path, seed=1)
    cfg.data.clips = ["walk1_subject1"]
    cfg.env.num_envs = 128
    cfg.env.replay_capacity = 200
    cfg.world_model.hidden = (256, 256)
    cfg.policy.encoder_hidden = (256, 128)
    cfg.policy.decoder_hidden = (256, 128)
    cfg.wm_trainer.batch_size = 256
    cfg.wm_trainer.batches_per_update = 2
    cfg.policy_trainer.batch_size = 128
    cfg.policy_trainer.horizon = 8
    cfg.schedule.total_iterations = 24  # iterations = updates (2.5 env steps each)
    cfg.schedule.warmup_env_steps = 20
    cfg.schedule.wm_warmup_iterations = 8
    cfg.schedule.log_interval = 8
    cfg.schedule.checkpoint_interval = 24
    tr = Trainer(cfg)
    try:
        tr.run()
        assert tr.iteration == 24 and tr.num_updates == 24 - 8
        assert tr.wm.in_norm.count > 0 and tr.policy.decoder.norm.count > 0  # v3 defaults: running normalisers
        assert tr.env_steps == 20 + 60  # warm-up + 24 iterations * 2.5
        assert (tmp_path / "smoke" / "ckpt_latest.pt").exists()
        # Resume path works.
        cfg2 = TrainConfig(**{**cfg.__dict__, "resume": tmp_path / "smoke" / "ckpt_latest.pt", "run_name": "smoke2"})
        tr.env.close()
        tr2 = Trainer(cfg2)
        assert tr2.iteration == 24 and tr2.env_steps == 80
        # A resumed run refills the (unsaved) replay buffer before its first update.
        cfg2.schedule.total_iterations = 28
        tr2.run()
        assert tr2.iteration == 28 and tr2.env_steps == 80 + 20 + 10
        tr2.env.close()
    finally:
        pass


@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")
def test_training_smoke_v2_options(tmp_path):
    """The pre-v3 configuration still trains: fixed normalisation scales, no LayerNorm, learned
    termination head, SiLU policy with fixed input scales, no start margin, self-collisions on,
    68-d goal frames, and the critic without privileged tracking errors."""
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    cfg = TrainConfig(run_name="smoke_v2", log_root=tmp_path, seed=2)
    cfg.data.clips = ["walk1_subject1"]
    cfg.env.num_envs = 128
    cfg.env.replay_capacity = 200
    cfg.env.self_collision = True
    cfg.env.goal_features = "v1"
    cfg.world_model.hidden = (256, 256)
    cfg.world_model.normalization = "fixed"
    cfg.world_model.layer_norm = False
    cfg.world_model.termination = "learned"
    cfg.policy.encoder_hidden = (256, 128)
    cfg.policy.decoder_hidden = (256, 128)
    cfg.policy.input_norm = "fixed"
    cfg.policy.activation = "silu"
    cfg.critic.error_features = False
    cfg.wm_trainer.batch_size = 256
    cfg.wm_trainer.batches_per_update = 2
    cfg.policy_trainer.batch_size = 128
    cfg.policy_trainer.horizon = 8
    cfg.policy_trainer.start_margin = 0
    cfg.schedule.total_iterations = 16
    cfg.schedule.warmup_env_steps = 20
    cfg.schedule.wm_warmup_iterations = 4
    cfg.schedule.log_interval = 8
    cfg.schedule.checkpoint_interval = 16
    tr = Trainer(cfg)
    try:
        tr.run()
        assert tr.iteration == 16 and tr.num_updates == 12
        assert not hasattr(tr.wm, "in_norm") and tr.policy.decoder.norm is None
    finally:
        tr.env.close()
