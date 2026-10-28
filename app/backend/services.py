from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _build_candidates():
    """候选识别引擎工厂（按优先级）。资源缺失者会被注册表自动跳过。"""
    from app.backend.tfnet_service import TFNetPublishedService, VacTrainedService

    return [TFNetPublishedService, VacTrainedService]


def build_registry() -> dict:
    reg: dict = {}
    for cls in _build_candidates():
        try:
            inst = cls()
        except Exception:  # noqa: BLE001 - 资源/导入问题不应拖垮整个应用
            continue
        if inst.ready:
            reg[inst.model_version] = inst
    return reg


# 模块加载时构建一次（仅做资源存在性检查，惰性加载权重）
REGISTRY = build_registry()


def default_model_name() -> str | None:
    # 优先发布 TFNet，其次任意可用模型
    if "tfnet" in REGISTRY:
        return "tfnet"
    return next(iter(REGISTRY)) if REGISTRY else None


def get_service(name: str | None):
    if name and name in REGISTRY:
        return REGISTRY[name]
    dn = default_model_name()
    return REGISTRY.get(dn) if dn else None


def available_models() -> list:
    return [s.info for s in REGISTRY.values()]


def create_recognition_service():
    """Backward-compatible 别名：返回默认模型服务。"""
    return get_service(None)
