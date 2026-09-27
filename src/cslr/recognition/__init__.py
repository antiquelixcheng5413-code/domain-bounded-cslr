"""Ordered-gloss CTC recognition package (Part 4).

This package is additive: it does not modify the frozen Part 1/2 training
pipeline, manifest, feature configuration, or the existing set-based gloss
utilities in :mod:`cslr.data.gloss`.

The official CE-CSL ``test`` split (500 samples) stays frozen and is never
read, inferred on, or tuned against from this package.
"""

from __future__ import annotations

from cslr.recognition.gloss_sequence import (
    UNKNOWN_TOKEN,
    GlossSequenceConfig,
    GlossVocabulary,
    build_ordered_vocabulary,
    coverage_report,
    split_gloss_sequence,
)

__all__ = [
    "UNKNOWN_TOKEN",
    "GlossSequenceConfig",
    "GlossVocabulary",
    "build_ordered_vocabulary",
    "coverage_report",
    "split_gloss_sequence",
]
