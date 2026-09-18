"""Strict configuration loading and validation for Part 3.

The official ``test`` split is frozen for this phase: it is rejected *during*
config parsing, before any file or dataset is opened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALLOWED_SPLITS = frozenset({"train", "validation", "dev"})
# ``dev`` is accepted as an alias and normalised to ``validation``.
_SPLIT_ALIASES = {"train": "train", "dev": "validation", "validation": "validation"}


class Part3ConfigError(ValueError):
    """Raised when a Part 3 configuration is invalid or the test split is requested."""


@dataclass(frozen=True)
class DataConfig:
    manifest: Path
    label_dir: Path | None
    raw_labels: bool = True  # when True, ``label_dir`` must point at official CSL label CSVs


@dataclass(frozen=True)
class DecoderConfig:
    type: str
    hidden_dim: int
    num_layers: int
    num_heads: int
    feedforward_dim: int
    vocab_size: int = -1  # resolved automatically from the text index
    max_target_len: int = 128
    dropout: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int
    num_heads: int
    num_layers: int
    feedforward_dim: int
    dropout: float
    gloss_aux_weight: float
    decoder: DecoderConfig
    label_smoothing: float = 0.0
    beam_size: int = 1
    decoder_backbone: str = "tiny"  # "tiny" | "mt5"
    mt5_path: str | None = None  # local dir / HF id; required when backbone=="mt5"
    mt5_max_target_tokens: int = 64
    # mT5 trainable-parameter policy: "none" (all trainable, may degenerate to a
    # pure LM) | "cross_only" (freeze LM path: embeddings/self-attn/FF/LN; train
    # only visual_proj + decoder cross-attention, forcing visual conditioning).
    mt5_freeze: str = "none"
    # Auxiliary visual<->text alignment weight (backbone=="mt5" only): pulls the
    # pooled visual representation toward the mean target-token embedding via
    # cosine loss, giving the visual path a direct signal that bypasses the
    # decoder's LM dominance (counteracts degenerate pure-LM collapse).
    mt5_visual_aux_weight: float = 0.0


@dataclass(frozen=True)
class FusionConfig:
    modalities: tuple[str, ...]
    hidden_dim: int
    num_heads: int
    num_layers: int
    feedforward_dim: int
    dropout: float
    num_pool_tokens: int | None = None  # >0: SpaMo-style learnable-query pool; None/0: concat-as-is


@dataclass(frozen=True)
class SmokeConfig:
    train_steps: int
    dev_limit: int | None
    synthetic_features: bool
    seed: int
    report_dir: Path


@dataclass(frozen=True)
class Part3Config:
    split: str
    data: DataConfig
    model: ModelConfig
    fusion: FusionConfig
    smoke: SmokeConfig
    device: str = "cpu"
    cache_dir: Path | None = None
    source_config_path: Path | None = field(default=None, repr=False)

    def effective_split(self) -> str:
        return _SPLIT_ALIASES[self.split]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise Part3ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
    return raw


def _resolve_root(config: dict[str, Any]) -> Path:
    root = config.get("root")
    if root is None:
        raise Part3ConfigError("config missing required key 'root'")
    return Path(str(root))


def _load_split(config: dict[str, Any]) -> str:
    split = str(config.get("split", "")).strip().lower()
    if split not in ALLOWED_SPLITS:
        raise Part3ConfigError(
            "split 'test' is frozen for Part 3: the official 500-sample test split "
            "must not be read, inferred on, or tuned against during this phase"
            if split == "test"
            else f"invalid split {split!r}; allowed: {sorted(ALLOWED_SPLITS)}"
        )
    return _SPLIT_ALIASES[split]


def _require_positive_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise Part3ConfigError(f"{key} must be a positive int, got {value!r}")
    return value


def _require_nonneg_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Part3ConfigError(f"{key} must be a non-negative int, got {value!r}")
    return value


def _require_float(data: dict[str, Any], key: str, low: float, high: float) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise Part3ConfigError(f"{key} must be a float, got {value!r}")
    value = float(value)
    if not (low <= value <= high):
        raise Part3ConfigError(f"{key} must be in [{low}, {high}], got {value}")
    return value


def _load_decoder(block: dict[str, Any], hidden_dim: int) -> DecoderConfig:
    d = block.get("decoder", {})
    if not isinstance(d, dict):
        raise Part3ConfigError("decoder must be a mapping")
    return DecoderConfig(
        type=str(d.get("type", "transformer")),
        hidden_dim=d.get("hidden_dim", hidden_dim),
        num_layers=_require_positive_int(d, "num_layers") if "num_layers" in d else 2,
        num_heads=_require_positive_int(d, "num_heads") if "num_heads" in d else 4,
        feedforward_dim=d.get("feedforward_dim", hidden_dim * 4),
        max_target_len=_require_positive_int(d, "max_target_len") if "max_target_len" in d else 128,
        dropout=_require_float(d, "dropout", 0.0, 1.0) if "dropout" in d else 0.1,
    )


def load_config(path: str | Path) -> Part3Config:
    """Load and strictly validate a Part 3 YAML config.

    ``split=test`` is rejected here before any dataset construction.
    """
    cfg_path = Path(path)
    raw = _read_yaml(cfg_path)
    root = (_resolve_root(raw) / "data").expanduser()
    data_block = raw.get("data", {})
    if not isinstance(data_block, dict):
        raise Part3ConfigError("data must be a mapping")

    split_val = data_block.get("split")
    if split_val is None:
        split_val = raw.get("split")
    effective_split = _load_split({} if split_val is None else {"split": split_val})

    manifest = data_block.get("manifest")
    if manifest is None:
        raise Part3ConfigError("data.manifest is required")
    label_dir = data_block.get("label_dir")
    if label_dir is None:
        raise Part3ConfigError("data.label_dir is required (official CSL label directory)")

    model_block = raw.get("model", {})
    if not isinstance(model_block, dict):
        raise Part3ConfigError("model must be a mapping")
    hdim = _require_positive_int(model_block, "hidden_dim")
    model = ModelConfig(
        hidden_dim=hdim,
        num_heads=_require_positive_int(model_block, "num_heads"),
        num_layers=_require_positive_int(model_block, "num_layers"),
        feedforward_dim=_require_positive_int(model_block, "feedforward_dim")
        if "feedforward_dim" in model_block
        else hdim * 4,
        dropout=_require_float(model_block, "dropout", 0.0, 1.0),
        gloss_aux_weight=_require_float(model_block, "gloss_aux_weight", 0.0, 1.0),
        decoder=_load_decoder(model_block, hdim),
        label_smoothing=(
            _require_float(model_block, "label_smoothing", 0.0, 1.0)
            if "label_smoothing" in model_block else 0.0
        ),
        beam_size=(
            _require_positive_int(model_block, "beam_size")
            if "beam_size" in model_block else 1
        ),
        decoder_backbone=str(model_block.get("decoder_backbone", "tiny")),
        mt5_path=None if model_block.get("mt5_path") is None else str(model_block["mt5_path"]),
        mt5_max_target_tokens=(
            _require_positive_int(model_block, "mt5_max_target_tokens")
            if "mt5_max_target_tokens" in model_block else 64
        ),
        mt5_freeze=str(model_block.get("mt5_freeze", "none")),
        mt5_visual_aux_weight=(
            _require_float(model_block, "mt5_visual_aux_weight", 0.0, 10.0)
            if "mt5_visual_aux_weight" in model_block else 0.0
        ),
    )

    fusion_block = raw.get("fusion", {})
    if not isinstance(fusion_block, dict):
        raise Part3ConfigError("fusion must be a mapping")
    modalities = tuple(str(m) for m in fusion_block.get("modalities", ["rgb", "motion", "landmark"]))
    if not modalities or any(not m for m in modalities):
        raise Part3ConfigError("fusion.modalities must be a non-empty list")
    fusion = FusionConfig(
        modalities=modalities,
        hidden_dim=hdim,
        num_heads=_require_positive_int(fusion_block, "num_heads") if "num_heads" in fusion_block else model.num_heads,
        num_layers=_require_positive_int(fusion_block, "num_layers") if "num_layers" in fusion_block else model.num_layers,
        feedforward_dim=fusion_block.get("feedforward_dim", hdim * 4),
        dropout=_require_float(fusion_block, "dropout", 0.0, 1.0) if "dropout" in fusion_block else model.dropout,
        num_pool_tokens=(
            _require_nonneg_int(fusion_block, "num_pool_tokens")
            if "num_pool_tokens" in fusion_block else None
        ),
    )

    smoke_block = raw.get("smoke", {})
    if not isinstance(smoke_block, dict):
        raise Part3ConfigError("smoke must be a mapping")
    smoke = SmokeConfig(
        train_steps=_require_positive_int(smoke_block, "train_steps") if "train_steps" in smoke_block else 1,
        dev_limit=smoke_block.get("dev_limit"),
        synthetic_features=bool(smoke_block.get("synthetic_features", False)),
        seed=_require_nonneg_int(smoke_block, "seed") if "seed" in smoke_block else 0,
        report_dir=root / "artifacts" / "logs" / "part3-smoke",
    )

    cache_dir = raw.get("cache_dir")
    device = str(raw.get("device", "cpu"))

    return Part3Config(
        split=effective_split,
        data=DataConfig(
            manifest=root / str(manifest),
            label_dir=None if label_dir is None else root / str(label_dir),
        ),
        model=model,
        fusion=fusion,
        smoke=smoke,
        device=device,
        cache_dir=None if cache_dir is None else root / str(cache_dir),
        source_config_path=cfg_path,
    )