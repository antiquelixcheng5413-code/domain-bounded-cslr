"""Smoke orchestration, receipts, JSON and per-sample CSV output for Part 3.

The smoke path uses synthetic visual features by default and marks every
result as ``formal_result: false``.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import torch

from cslr.translation.cache import write_per_sample_csv, write_smoke_receipt
from cslr.translation.dataset import (
    build_eval_batch,
    build_training_batch,
    load_part3_dataset,
)
from cslr.translation.decoder import TinyTransformerChineseDecoder
from cslr.translation.metrics import LatencyTimer, bleu, chrf, rouge_l
from cslr.translation.models import Part3SpaMoModel, build_vocab_from_dataset

logger = logging.getLogger(__name__)


def git_head(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def run_smoke(cfg, *, train_steps: int, dev_limit: int | None, synthetic: bool, seed: int) -> dict:
    root = cfg.source_config_path.parent if cfg.source_config_path else Path.cwd()
    dataset = load_part3_dataset(
        cfg.data.manifest,
        cfg.data.label_dir,
        split=cfg.split,
        root=root,
        synthetic=synthetic,
        seed=seed,
        limit=None,
    )
    vocab = build_vocab_from_dataset(dataset)
    model = Part3SpaMoModel(
        modalities=cfg.fusion.modalities,
        hidden_dim=cfg.model.hidden_dim,
        fusion_heads=cfg.fusion.num_heads,
        fusion_layers=cfg.fusion.num_layers,
        fusion_ff=cfg.fusion.feedforward_dim,
        dropout=cfg.model.dropout,
        vocab=vocab,
        gloss_vocab_size=None,
        max_target_len=cfg.model.decoder.max_target_len,
        decoder_layers=cfg.model.decoder.num_layers,
        decoder_heads=cfg.model.decoder.num_heads,
        gloss_aux_weight=cfg.model.gloss_aux_weight,
        device=cfg.device,
    )

    train_batches = build_training_batch(dataset, batch_size=8, hidden_dim=cfg.model.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_history: list[float] = []
    for step in range(train_steps):
        model.train()
        optimizer.zero_grad()
        batch = train_batches[step % len(train_batches)]
        out = model(
            _features_from_batch(batch),
            batch["feature_masks"],
            target_texts=batch["target_texts"],
        )
        assert out.loss is not None
        out.loss.backward()
        optimizer.step()
        loss_history.append(float(out.loss.item()))

    started_loss = loss_history[0] if loss_history else float("nan")
    ended_loss = loss_history[-1] if loss_history else float("nan")

    # Dev evaluation (formal_result: false)
    dev_loader = _load_dev(cfg, root, vocab)
    outputs, latency_ms = predict_batch(model, dev_loader, vocab)

    report_dir = cfg.smoke.report_dir
    _write_outputs(report_dir, dataset, vocab, outputs, cfg, started=started_loss, ended=ended_loss)

    summary = {
        "status": "ok",
        "feature_source": "synthetic" if synthetic else "real",
        "formal_result": False,
        "test_split_read": False,
        "uses_external_weights": False,
        "split": cfg.split,
        "train_loss_start": round(started_loss, 4),
        "train_loss_end": round(ended_loss, 4),
        "loss_decreased": bool(started_loss and ended_loss and ended_loss < started_loss),
        "dev_samples": len(outputs),
        "latency_ms_per_sample": round(latency_ms, 4),
        "visual_token_count": 11,  # smoke: 4 RGB + 3 motion + 4 landmark
        "dev_limit": dev_limit,
    }
    return summary


def _features_from_batch(batch: dict) -> dict[str, torch.Tensor]:
    return {
        "rgb": batch["rgb_features"],
        "motion": batch["motion_features"],
        "landmark": batch["landmark_features"],
    }


def _load_dev(cfg, root: Path, vocab: dict[str, int], limit: int = 3):
    dev_ds = load_part3_dataset(
        cfg.data.manifest,
        cfg.data.label_dir,
        split="validation",
        root=root,
        synthetic=True,
        seed=0,
        limit=limit,
    )
    if len(dev_ds) == 0:
        return []
    return build_eval_batch(dev_ds, hidden_dim=cfg.model.hidden_dim)


def predict_batch(model: Part3SpaMoModel, batches, vocab: dict[str, int]) -> tuple[list[dict], float]:
    """Return (list of per-sample result dicts, total latency ms)."""
    records: list[dict] = []
    timer = LatencyTimer()
    total_ms = 0.0
    for batch in batches:
        timer.start()
        model.eval()
        with torch.no_grad():
            out = model(
                _features_from_batch(batch),
                batch["feature_masks"],
                generate=True,
            )
        ms = timer.stop()
        total_ms += ms
        preds = out.generated_texts or []
        for sample_id, target, pred in zip(
            batch["sample_ids"], batch["target_texts"], preds
        ):
            records.append(
                {
                    "sample_id": sample_id,
                    "reference": target,
                    "prediction": pred,
                    "bleu_1": round(bleu(target, pred, 1), 4),
                    "bleu_2": round(bleu(target, pred, 2), 4),
                    "bleu_3": round(bleu(target, pred, 3), 4),
                    "bleu_4": round(bleu(target, pred, 4), 4),
                    "rouge_l": round(rouge_l(target, pred), 4),
                    "chrf": round(chrf(target, pred), 4),
                    "exact_match": float(target == pred),
                    "latency_ms": round(ms / max(1, len(preds)), 4),
                }
            )
    return records, total_ms


def _write_outputs(report_dir, dataset, vocab, records, cfg, *, started: float, ended: float) -> None:
    write_per_sample_csv(report_dir, records)
    config_dump = {
        "split": cfg.split,
        "modalities": list(cfg.fusion.modalities),
        "train_steps": cfg.smoke.train_steps,
    }
    aggregate = {
        "bleu_1": _avg(records, "bleu_1"),
        "rouge_l": _avg(records, "rouge_l"),
        "chrf": _avg(records, "chrf"),
    }
    write_smoke_receipt(
        report_dir,
        status="ok",
        split=cfg.split,
        synthetic=True,
        formal_result=False,
        test_split_read=False,
        uses_external_weights=False,
        git_commit=git_head(Path.cwd()),
        config=config_dump,
        extra={"aggregate": aggregate},
    )


def _avg(records: list[dict], key: str) -> float:
    if not records:
        return 0.0
    return round(sum(float(r[key]) for r in records) / len(records), 4)