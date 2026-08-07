"""Tests for Transformer search-move guide (P11)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.search_move_guide import OPS, SearchMoveGuideNet, build_global_features, build_node_features
from src.search.neural_guide import NeuralCheckpointError
from src.search.transformer_search_guide import TransformerSearchMoveGuide, load_search_move_guide

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "dataset" / "Schneider" / "evrptw_instances" / "c101C5"


def test_search_move_net_forward_shapes():
    net = SearchMoveGuideNet(embedding_dim=32, num_heads=4, num_layers=1, ff_dim=64)
    nodes = torch.randn(20, 12)
    g = torch.randn(16)
    emb = net.encode(nodes)
    assert emb.shape == (20, 32)
    logits = net.forward_ops(emb, g)
    assert logits.shape == (len(OPS),)
    cust = net.forward_customers(emb, g, [1, 2, 3])
    assert cust.shape == (3,)


@pytest.mark.skipif(not SMALL.is_dir(), reason="c101C5 absent")
def test_transformer_search_guide_roundtrip(tmp_path):
    from src.data.instance_loader import load_instance

    inst = load_instance(SMALL)
    routes = [([inst.depot_index, inst.depot_index], [None, None]) for _ in range(3)]
    # Put a couple customers on route 0 if possible.
    custs = [i for i, n in enumerate(inst.nodes) if n.is_customer][:3]
    if custs:
        routes[0] = ([inst.depot_index, custs[0], inst.depot_index], [None, None, None])

    net = SearchMoveGuideNet(embedding_dim=32, num_heads=4, num_layers=1, ff_dim=64)
    ckpt = tmp_path / "transformer_search_guide.pt"
    torch.save({
        "format": "transformer_search_guide_v1",
        "state_dict": net.state_dict(),
        "arch": {"embedding_dim": 32, "num_heads": 4, "num_layers": 1, "ff_dim": 64, "dropout": 0.1},
        "ops": list(OPS),
    }, ckpt)

    cfg = {"hybrid_search": {"neural_guidance": {
        "enabled": True, "deprecated_bc_guide": False, "checkpoint": str(ckpt),
    }}}
    guide = load_search_move_guide(cfg, device="cpu")
    guide.set_solution_context(routes)
    ops = list(OPS)
    ranked = guide.rank_destroy_operators(ops, instance=inst, routes=routes)
    assert set(ranked) == set(ops)
    pool = custs or [i for i, n in enumerate(inst.nodes) if n.is_customer][:5]
    rem = guide.rank_customers_for_removal(inst, pool, routes=routes)
    assert len(rem) == len(pool)
    tel = guide.telemetry.as_dict()
    assert tel["neural_participation"] is True
    assert tel["encode_calls"] >= 1
    assert guide.checkpoint_sha256


def test_rejects_bc_checkpoint_name(tmp_path):
    bad = tmp_path / "bc_best_policy.pt"
    bad.write_bytes(b"x")
    cfg = {"hybrid_search": {"neural_guidance": {
        "enabled": True, "deprecated_bc_guide": False, "checkpoint": str(bad),
    }}}
    with pytest.raises(NeuralCheckpointError):
        load_search_move_guide(cfg)


def test_rejects_deprecated_bc_flag(tmp_path):
    ckpt = tmp_path / "transformer_search_guide.pt"
    net = SearchMoveGuideNet(embedding_dim=16, num_heads=2, num_layers=1, ff_dim=32)
    torch.save({"state_dict": net.state_dict(), "arch": {"embedding_dim": 16, "num_heads": 2, "num_layers": 1, "ff_dim": 32}}, ckpt)
    cfg = {"hybrid_search": {"neural_guidance": {
        "enabled": True, "deprecated_bc_guide": True, "checkpoint": str(ckpt),
    }}}
    with pytest.raises(NeuralCheckpointError):
        load_search_move_guide(cfg)
