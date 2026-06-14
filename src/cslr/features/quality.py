from __future__ import annotations

from collections.abc import Iterable

from cslr.contracts import QualityReport


def assess_frame_quality(
    frame_validity: Iterable[bool], minimum_valid_ratio: float = 0.80
) -> QualityReport:
    flags = list(frame_validity)
    total = len(flags)
    valid = sum(1 for flag in flags if flag)
    ratio = valid / total if total else 0.0
    warnings = []
    if total == 0:
        warnings.append("video contains no decodable frames")
    elif ratio < minimum_valid_ratio:
        warnings.append(
            f"landmark detection ratio {ratio:.1%} is below required {minimum_valid_ratio:.1%}"
        )
    return QualityReport(
        total_frames=total,
        valid_frames=valid,
        valid_ratio=ratio,
        accepted=total > 0 and ratio >= minimum_valid_ratio,
        warnings=warnings,
    )
