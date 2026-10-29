"""Warm start of the terrain-capable networks from a flat checkpoint (docs/VISION_TERRAIN_PLAN.md §2.4-2.5).

Every tensor of the flat state dict is copied unchanged, except the first Linear layer of the policy
decoder, the critic and the world-model dynamics, whose weight gains zero-initialised input columns for the
new terrain inputs: the initial outputs equal the flat model's, and the new columns still receive gradients.
Modules that do not exist in the flat model (``TerrainEncoder``, ``DepthModules``, the world model's ``terr_norm``) keep their
fresh initialisation. Anything else that does not match is an error (e.g. a different network size).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# First Linear layer of each network and where its new input columns go: decoder [z_motion | z_terrain | obs],
# critic [... | terrain], world model [in_norm(feat, a, xi) | terr_norm(c)].
DECODER_IN = "decoder.mlp.net.0.weight"
CRITIC_IN = "mlp.net.0.weight"
WM_IN = "dynamics.mlp.net.0.weight"
NEW_PREFIXES = {"policy": ("terrain.", "depth."), "critic": (), "wm": ("terr_norm.",)}


def expand_input_columns(old: torch.Tensor, new_in: int, insert_at: int) -> torch.Tensor:
    """(out, in_old) -> (out, new_in) with ``new_in - in_old`` zero columns inserted before column ``insert_at``."""
    out, in_old = old.shape
    d = new_in - in_old
    assert d >= 0 and 0 <= insert_at <= in_old, (old.shape, new_in, insert_at)
    return torch.cat([old[:, :insert_at], old.new_zeros(out, d), old[:, insert_at:]], dim=1)


@torch.no_grad()
def load_expanded(module: nn.Module, flat_sd: dict[str, torch.Tensor], first_layer: str, insert_at: int, new_prefixes: tuple[str, ...]) -> list[str]:
    """Load ``flat_sd`` into ``module``; returns the keys left at their fresh initialisation."""
    own = module.state_dict()
    unexpected = sorted(set(flat_sd) - set(own))
    if unexpected:
        raise KeyError(f"flat checkpoint has keys the model does not: {unexpected[:8]}")
    fresh, out = [], {}
    for k, v in own.items():
        if k not in flat_sd:
            if not k.startswith(new_prefixes):
                raise KeyError(f"model key {k!r} missing from the flat checkpoint")
            fresh.append(k)
            out[k] = v
            continue
        src = flat_sd[k].to(device=v.device, dtype=v.dtype)
        if k == first_layer and src.shape != v.shape:
            src = expand_input_columns(src, v.shape[1], insert_at)
        if src.shape != v.shape:
            raise ValueError(f"{k}: checkpoint {tuple(src.shape)} vs model {tuple(v.shape)} (network sizes differ?)")
        out[k] = src
    module.load_state_dict(out, strict=True)
    return fresh


def warm_start(policy: nn.Module, critic: nn.Module, wm: nn.Module, ck: dict) -> dict[str, list[str]]:
    """Load a flat checkpoint dict (``ck["policy" / "critic" / "wm"]``) into terrain-capable networks."""
    lat = policy.cfg.latent_dim
    return {
        "policy": load_expanded(policy, ck["policy"], DECODER_IN, lat, NEW_PREFIXES["policy"]),
        "critic": load_expanded(critic, ck["critic"], CRITIC_IN, ck["critic"][CRITIC_IN].shape[1], NEW_PREFIXES["critic"]),
        "wm": load_expanded(wm, ck["wm"], WM_IN, ck["wm"][WM_IN].shape[1], NEW_PREFIXES["wm"]),
    }
