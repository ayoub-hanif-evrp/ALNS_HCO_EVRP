"""Train Transformer search-move guide from ALNS move demos."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.search_move_guide import OPS, OP_TO_IDX, SearchMoveGuideNet, listwise_ce


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="outputs/demonstrations/schneider21_search_moves.pt")
    ap.add_argument("--out", default="outputs/tgfps_p11/transformer_search_guide.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    demos_path = ROOT / args.demos
    payload = torch.load(demos_path, map_location="cpu", weights_only=False)
    samples = list(payload["samples"])
    assert payload.get("format") == "tgfps_search_move_v1"
    print(f"Loaded {len(samples)} samples from {demos_path}", flush=True)

    # Prefer lex-improving accepted moves as positive op targets; fall back to accepted.
    pos = [s for s in samples if s.get("lex_improved")]
    if len(pos) < 20:
        pos = [s for s in samples if s.get("accepted")]
    neg = [s for s in samples if not s.get("accepted")]
    print(f"positives={len(pos)} negatives={len(neg)}", flush=True)

    arch = {"embedding_dim": 64, "num_heads": 4, "num_layers": 2, "ff_dim": 128, "dropout": 0.1}
    net = SearchMoveGuideNet(**arch).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def _batch(epoch_samples: list[dict]):
        random.shuffle(epoch_samples)
        return epoch_samples

    t0 = time.perf_counter()
    history = []
    for ep in range(1, args.epochs + 1):
        net.train()
        total = 0.0
        n = 0
        # Mix positives heavily.
        epoch_data = pos * 3 + random.sample(neg, min(len(neg), max(len(pos), 1) * 2))
        for s in _batch(epoch_data):
            node = s["node_features"].to(device)
            g = s["global_features"].to(device)
            emb = net.encode(node)
            op_logits = net.forward_ops(emb, g)
            target_op = int(s["op_idx"])
            loss_op = listwise_ce(op_logits, target_op)

            # Customer ranking: removed customers of improving destroy moves are positives.
            loss_cust = emb.sum() * 0.0
            removed = [int(x) for x in (s.get("removed") or [])]
            cust_ids = [int(x) for x in (s.get("customer_ids") or [])]
            if removed and cust_ids and s.get("lex_improved"):
                logits = net.forward_customers(emb, g, cust_ids)
                id_to_i = {c: i for i, c in enumerate(cust_ids)}
                pos_i = [id_to_i[c] for c in removed if c in id_to_i]
                neg_i = [i for i, c in enumerate(cust_ids) if c not in set(removed)]
                if pos_i and neg_i:
                    # Sample negatives for speed.
                    neg_i = random.sample(neg_i, min(32, len(neg_i)))
                    pos_score = logits[pos_i].mean()
                    neg_score = logits[neg_i].mean()
                    loss_cust = F.softplus(neg_score - pos_score)

            # For non-improving samples, push probability away from that op slightly.
            if not s.get("accepted"):
                loss_op = loss_op * 0.25 + F.softplus(op_logits[target_op]) * 0.05

            loss = loss_op + 0.5 * loss_cust
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            n += 1
        avg = total / max(n, 1)
        history.append({"epoch": ep, "loss": avg})
        print(f"epoch {ep:02d} loss={avg:.4f}", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "format": "transformer_search_guide_v1",
        "state_dict": net.state_dict(),
        "arch": arch,
        "ops": list(OPS),
        "demos": str(demos_path),
        "n_samples": len(samples),
        "train_seconds": time.perf_counter() - t0,
        "history": history,
        "note": "Search-move guide — not node BC / not bc_best_policy.pt",
    }
    torch.save(ckpt, out)
    # Also write last alias.
    (out.parent / "transformer_search_guide_last.pt").write_bytes(out.read_bytes())
    meta = {
        "checkpoint": str(out),
        "n_samples": len(samples),
        "epochs": args.epochs,
        "final_loss": history[-1]["loss"] if history else None,
        "train_seconds": ckpt["train_seconds"],
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
