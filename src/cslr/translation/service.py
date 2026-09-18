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
from cslr.translation.models import (
    Part3SpaMoModel,
    build_model_from_config,
    build_vocab_from_dataset,
)
from cslr.translation.realdata import (
    build_real_eval_batch,
    build_real_training_batches,
    load_real_features,
)

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


def _to_device(batch: dict, device: str) -> dict:
    dst = torch.device(device)
    out = dict(batch)
    for key, value in batch.items():
        if isinstance(value, dict):
            out[key] = {
                k: (v.to(dst) if isinstance(v, torch.Tensor) else v)
                for k, v in value.items()
            }
        elif isinstance(value, torch.Tensor):
            out[key] = value.to(dst)
    return out


def _chunk_batch(batch: dict, chunk: int) -> list[dict]:
    size = int(batch["rgb_features"].shape[0])
    chunks: list[dict] = []
    for start in range(0, size, chunk):
        end = min(start + chunk, size)
        part = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                part[key] = {k: (v[start:end] if isinstance(v, torch.Tensor) else v) for k, v in value.items()}
            elif isinstance(value, torch.Tensor):
                part[key] = value[start:end]
            else:
                part[key] = value
        chunks.append(part)
    return chunks


def run_p2_chain(
    cfg, *, root: Path, feature_root: Path, split: str, limit: int | None, device: str
) -> dict:
    """Phase-2 chain check: real cached features -> fusion -> decoder -> metrics.

    ``formal_result`` stays False (no trained weights / frozen config yet); this
    only proves the real video-feature pipeline is numerically sound.
    """
    dataset = load_part3_dataset(
        cfg.data.manifest,
        cfg.data.label_dir,
        split=split,
        root=root,
        synthetic=False,
        limit=limit,
    )
    vocab = build_vocab_from_dataset(dataset)
    split_root = feature_root / split
    first = load_real_features(split_root, dataset.sample_ids[0])
    feature_dim = {m: int(v.shape[1]) for m, v in first.items()}
    model = build_model_from_config(
        cfg, vocab, gloss_vocab_size=None, feature_dim=feature_dim
    )

    batches = build_real_eval_batch(dataset, split_root)
    records: list[dict] = []
    total_loss = 0.0
    token_total = 0
    sample_count = 0
    for batch in batches:
        model.eval()
        batch = _to_device(batch, device)
        with torch.no_grad():
            out = model(
                _features_from_batch(batch),
                batch["feature_masks"],
                target_texts=batch["target_texts"],
                generate=True,
            )
        loss = 0.0 if out.translation_loss is None else float(out.translation_loss.item())
        preds = out.generated_texts or []
        for sid, target, pred in zip(batch["sample_ids"], batch["target_texts"], preds):
            records.append(
                {
                    "sample_id": sid,
                    "reference": target,
                    "prediction": pred,
                    "bleu_1": round(bleu(target, pred, 1), 4),
                    "rouge_l": round(rouge_l(target, pred), 4),
                    "chrf": round(chrf(target, pred), 4),
                    "exact_match": float(target == pred),
                }
            )
            total_loss += loss
            sample_count += 1
        if out.visual_token_count is not None:
            token_total += int(out.visual_token_count.item())

    return {
        "status": "ok",
        "feature_source": "real",
        "modalities": list(cfg.fusion.modalities),
        "formal_result": False,
        "test_split_read": False,
        "split": split,
        "samples": len(records),
        "avg_ce_loss": round(total_loss / max(1, sample_count), 4),
        "visual_token_mean": round(token_total / max(1, len(batches)), 1),
        "bleu_1": round(_avg(records, "bleu_1"), 4),
        "rouge_l": round(_avg(records, "rouge_l"), 4),
        "chrf": round(_avg(records, "chrf"), 4),
        "device": device,
        "per_sample": records,
    }


def run_real_train(
    cfg,
    *,
    root: Path,
    feature_root: Path,
    train_split: str,
    eval_split: str,
    epochs: int,
    lr: float,
    batch_size: int,
    eval_limit: int,
    device: str,
) -> dict:
    """Teacher-forcing Chinese-sentence training on cached real features.

    Produces a baseline on the frozen evaluation split with BLEU/ROUGE/chrF.
    ``test`` is never touched on either split.
    """
    train_ds = load_part3_dataset(
        cfg.data.manifest, cfg.data.label_dir, split=train_split, root=root, synthetic=False
    )
    vocab = build_vocab_from_dataset(train_ds)
    train_root = feature_root / train_split
    first = load_real_features(train_root, train_ds.sample_ids[0])
    feature_dim = {m: int(v.shape[1]) for m, v in first.items()}
    model = build_model_from_config(cfg, vocab, gloss_vocab_size=None, feature_dim=feature_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    batches = build_real_training_batches(train_ds, train_root, batch_size=batch_size)
    loss_history: list[float] = []
    global_step = 0
    print(f"[train-real] train_examples={len(train_ds)} batches={len(batches)} epochs={epochs} lr={lr}", flush=True)
    for _epoch in range(epochs):
        model.train()
        for batch in batches:
            batch = _to_device(batch, device)
            optimizer.zero_grad()
            out = model(
                _features_from_batch(batch),
                batch["feature_masks"],
                target_texts=batch["target_texts"],
            )
            assert out.loss is not None
            out.loss.backward()
            optimizer.step()
            loss_history.append(float(out.loss.item()))
            global_step += 1
            if global_step % 25 == 0:
                print(f"[train-real] epoch={_epoch} step={global_step} loss={float(out.loss.item()):.4f}", flush=True)
    loss_start = loss_history[0] if loss_history else float("nan")
    print(f"[train-real] training done steps={global_step} loss_start={loss_start:.4f}", flush=True)

    # Frozen evaluation split
    eval_ds = load_part3_dataset(
        cfg.data.manifest, cfg.data.label_dir, split=eval_split, root=root,
        synthetic=False, limit=eval_limit,
    )
    eval_root = feature_root / eval_split
    eval_ds.records = [
        r for r in eval_ds.records
        if all((eval_root / f"{r.sample_id}.{m}.npy").exists() for m in ("rgb", "motion", "landmark"))
    ]
    print(f"[train-real] eval_rows={len(eval_ds)} limit={eval_limit}", flush=True)
    eval_batches = build_real_eval_batch(eval_ds, eval_root)
    records: list[dict] = []
    done_rows = 0
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    for eval_batch in eval_batches:
        for bi, batch in enumerate(_chunk_batch(eval_batch, chunk=8)):
            model.eval()
            batch = _to_device(batch, device)
            with torch.no_grad():
                out = model(
                    _features_from_batch(batch), batch["feature_masks"], generate=True
                )
            preds = out.generated_texts or []
            for sid, target, pred in zip(batch["sample_ids"], batch["target_texts"], preds):
                records.append(
                    {
                        "sample_id": sid,
                        "reference": target,
                        "prediction": pred,
                        "bleu_1": round(bleu(target, pred, 1), 4),
                        "bleu_2": round(bleu(target, pred, 2), 4),
                        "rouge_l": round(rouge_l(target, pred), 4),
                        "chrf": round(chrf(target, pred), 4),
                        "exact_match": float(target == pred),
                    }
                )
                done_rows += 1
            print(f"[train-real] eval chunk {bi} done_rows={done_rows}", flush=True)

    return {
        "status": "ok",
        "feature_source": "real",
        "formal_result": True,
        "test_split_read": False,
        "train_split": train_split,
        "eval_split": eval_split,
        "epochs": epochs,
        "global_steps": global_step,
        "train_examples": len(train_ds),
        "train_loss_start": round(loss_history[0], 4) if loss_history else None,
        "train_loss_end": round(loss_history[-1], 4) if loss_history else None,
        "eval_samples": len(records),
        "bleu_1": round(_avg(records, "bleu_1"), 4),
        "bleu_2": round(_avg(records, "bleu_2"), 4),
        "rouge_l": round(_avg(records, "rouge_l"), 4),
        "chrf": round(_avg(records, "chrf"), 4),
        "exact_match": round(_avg(records, "exact_match"), 4),
        "device": device,
        "per_sample": records,
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