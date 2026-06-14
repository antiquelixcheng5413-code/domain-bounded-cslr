from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from cslr.models import build_model


def export_checkpoint_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    sequence_length: int = 48,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    labels = [str(label) for label in checkpoint["labels"]]
    model_config = checkpoint["model_config"]
    model = build_model(model_config, len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, sequence_length, int(model_config["input_size"]))
    torch.onnx.export(
        model,
        example,
        output_path,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    metadata = {
        "labels": labels,
        "version": output_path.stem,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    metadata_path = output_path.with_suffix(".labels.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"model": str(output_path), "metadata": str(metadata_path)}
