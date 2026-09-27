import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cslr.contracts import SampleRecord
from cslr.recognition.dataset import filter_present
from cslr.recognition.gloss_sequence import build_ordered_vocabulary
from cslr.recognition.model import CTCConfig
from cslr.recognition.training import TrainingConfig, ctc_loss, learning_rate_at, train_ctc


def _reference_ctc_nll(log_probs: list[list[float]], target: list[int], blank: int = 0) -> float:
    """Independent CTC forward algorithm, used to pin down torch's input convention."""

    states = [blank]
    for token in target:
        states.append(token)
        states.append(blank)
    steps = len(log_probs)
    neg_inf = -math.inf
    alpha = [neg_inf] * len(states)
    alpha[0] = log_probs[0][states[0]]
    if len(states) > 1:
        alpha[1] = log_probs[0][states[1]]
    for step in range(1, steps):
        updated = [neg_inf] * len(states)
        for index in range(len(states)):
            candidates = [alpha[index]]
            if index - 1 >= 0:
                candidates.append(alpha[index - 1])
            if index - 2 >= 0 and states[index] != blank and states[index] != states[index - 2]:
                candidates.append(alpha[index - 2])
            best = max(candidates)
            if best == neg_inf:
                continue
            total = math.log(sum(math.exp(value - best) for value in candidates if value != neg_inf))
            updated[index] = total + best + log_probs[step][states[index]]
        alpha = updated
    ends = [alpha[-1], alpha[-2] if len(states) >= 2 else neg_inf]
    best = max(ends)
    return -(best + math.log(sum(math.exp(value - best) for value in ends if value != neg_inf)))


class LearningRateScheduleTests(unittest.TestCase):
    def test_warmup_is_linear_then_cosine_decays(self) -> None:
        base = 0.001
        self.assertAlmostEqual(learning_rate_at(1, 20, base, warmup_epochs=2), base / 2)
        self.assertAlmostEqual(learning_rate_at(2, 20, base, warmup_epochs=2), base)
        middle = learning_rate_at(11, 20, base, warmup_epochs=2)
        self.assertLess(middle, base)
        self.assertGreater(middle, base * 0.05)
        last = learning_rate_at(20, 20, base, warmup_epochs=2)
        self.assertAlmostEqual(last, base * 0.05, places=7)

    def test_learning_rate_never_collapses_below_the_floor(self) -> None:
        # the old ReduceLROnPlateau(patience=3) schedule dropped ~8x by epoch 22 and froze training
        rates = [learning_rate_at(epoch, 60, 0.0015, warmup_epochs=2) for epoch in range(1, 61)]
        self.assertGreaterEqual(min(rates), 0.0015 * 0.05 - 1e-12)
        self.assertGreater(rates[30], 0.0015 * 0.05)

    def test_constant_schedule_stays_flat(self) -> None:
        self.assertAlmostEqual(learning_rate_at(10, 10, 0.002, schedule="constant"), 0.002)

    def test_bad_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            learning_rate_at(1, 10, 0.001, schedule="nope")


class CtcLossConventionTests(unittest.TestCase):
    """The sign convention of ``nn.CTCLoss`` is a real trap: it silently flips training."""

    def test_matches_independent_ctc_forward_implementation(self) -> None:
        torch.manual_seed(0)
        steps, classes = 12, 4
        logits = torch.randn(1, steps, classes)
        targets = torch.tensor([[1, 2, 3]])
        target_lengths = torch.tensor([3])
        lengths = torch.tensor([steps])

        ours = ctc_loss(logits, lengths, targets, target_lengths, lengths, reduction="none")
        log_probs = torch.log_softmax(logits, dim=-1)[0].tolist()
        reference = _reference_ctc_nll(log_probs, [1, 2, 3])
        self.assertAlmostEqual(float(ours[0]), reference, places=3)

    def test_uniform_model_loss_is_positive_and_bounded(self) -> None:
        steps, classes = 32, 5
        logits = torch.zeros(1, steps, classes)
        targets = torch.tensor([[1, 2]])
        target_lengths = torch.tensor([2])
        lengths = torch.tensor([steps])
        value = float(ctc_loss(logits, lengths, targets, target_lengths, lengths))
        self.assertGreater(value, 0.0)
        # a uniform model cannot do better than the per-frame entropy bound
        self.assertLessEqual(value, steps * math.log(classes) + 1e-3)

    def test_padded_targets_and_lengths_are_honoured(self) -> None:
        torch.manual_seed(1)
        logits = torch.randn(2, 20, 6)
        targets = torch.tensor([[1, 2, 3], [4, 0, 0]])
        target_lengths = torch.tensor([3, 1])
        lengths = torch.tensor([20, 20])
        per_sample = ctc_loss(
            logits, lengths, targets, target_lengths, lengths, reduction="none"
        )
        self.assertEqual(per_sample.shape[0], 2)
        wide = torch.tensor([[1, 2, 3], [4, 5, 5]])
        wide_per_sample = ctc_loss(
            logits, lengths, wide, target_lengths, lengths, reduction="none"
        )
        # the padded cell of row 1 must be ignored
        self.assertAlmostEqual(float(per_sample[1]), float(wide_per_sample[1]), places=4)

    def test_impossible_alignment_is_rejected(self) -> None:
        logits = torch.zeros(1, 4, 5)
        targets = torch.tensor([[1, 2, 3, 4]])
        target_lengths = torch.tensor([4])
        lengths = torch.tensor([4])
        with self.assertRaises(ValueError):
            ctc_loss(logits, lengths, targets, target_lengths, lengths)

    def test_out_of_range_target_is_rejected(self) -> None:
        logits = torch.zeros(1, 10, 3)
        targets = torch.tensor([[99]])
        target_lengths = torch.tensor([1])
        lengths = torch.tensor([10])
        with self.assertRaises(ValueError):
            ctc_loss(logits, lengths, targets, target_lengths, lengths)


def _record(sample_id: str, label: str, split: str) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        video=Path(f"video/{split}/A/{sample_id}.mp4"),
        label=label,
        signer="A",
        session=split,
        split=split,
    )


def _make_synthetic_corpus(root: Path, count: int = 8, length: int = 24, width: int = 12) -> None:
    """Visually separable synthetic features: one token per frame block."""

    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for index in range(count):
        features = rng.normal(0, 0.02, size=(length, width)).astype(np.float32)
        # three distinct "signs", each dominating its own third of the timeline
        for token in range(3):
            start = token * (length // 3)
            stop = (token + 1) * (length // 3)
            features[start:stop, token] += 3.0
        np.save(root / f"train-{index:05d}.npy", features)


class FilterPresentTests(unittest.TestCase):
    def test_splits_records_by_feature_existence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "a.npy", np.zeros((2, 2), dtype=np.float32))
            records = [
                _record("a", "我/要", "train"),
                _record("b", "你/好", "train"),
            ]
            kept, missing = filter_present(records, root)
        self.assertEqual([record.sample_id for record in kept], ["a"])
        self.assertEqual(missing, ["b"])


class TrainingLoopTests(unittest.TestCase):
    def test_training_decreases_loss_and_recovers_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features = root / "features"
            _make_synthetic_corpus(features, count=6)
            # validation reuses the same synthetic files under its own ids
            for index in range(2):
                source = features / f"train-{index:05d}.npy"
                np.save(features / f"dev-{index:05d}.npy", np.load(source))
            labels = ["一/二/三", "一/二", "二/三", "一/三", "三/二/一", "二/一/三"]
            records = [
                _record(f"train-{index:05d}", labels[index], "train") for index in range(6)
            ] + [_record(f"dev-{index:05d}", labels[index], "validation") for index in range(2)]
            manifest = root / "manifest.csv"
            manifest.write_text(
                "sample_id,video,label,signer,session,split\n"
                + "\n".join(
                    f"{r.sample_id},video/{r.split}/A/{r.sample_id}.mp4,{r.label},{r.signer},{r.session},{r.split}"
                    for r in records
                )
                + "\n",
                encoding="utf-8",
            )

            vocabulary, _ = build_ordered_vocabulary(
                [record.label for record in records if record.split == "train"], min_frequency=1
            )
            model_config = CTCConfig(
                input_size=12,
                vocabulary_size=vocabulary.size,
                hidden_size=32,
                num_layers=1,
                dropout=0.0,
                projection_size=32,
            )
            training_config = TrainingConfig(
                epochs=25,
                batch_size=3,
                learning_rate=0.01,
                seed=0,
                device="cpu",
                amp=False,
                early_stopping_patience=25,
            )
            result = train_ctc(
                manifest_path=manifest,
                feature_root=features,
                vocabulary=vocabulary,
                model_config=model_config,
                training_config=training_config,
                output_path=root / "checkpoint.pt",
            )

            first_loss = result.history[0]["train_loss"]
            final_loss = result.history[-1]["train_loss"]
            checkpoint_exists = (root / "checkpoint.pt").exists()
            history_exists = (root / "checkpoint.history.json").exists()

            self.assertLess(final_loss, first_loss)
            self.assertGreater(result.vocabulary_size, 3)
            self.assertLessEqual(result.validation.metrics["wer"], 1.0)
            self.assertGreater(result.validation.metrics["blank_ratio"], 0.0)
            self.assertTrue(checkpoint_exists)
            self.assertTrue(history_exists)

    def test_present_only_reports_skipped_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features = root / "features"
            _make_synthetic_corpus(features, count=6)
            np.save(features / "dev-00000.npy", np.load(features / "train-00000.npy"))
            records = [
                _record(f"train-{index:05d}", "一/二/三", "train") for index in range(6)
            ] + [_record("dev-00000", "一/二/三", "validation")]
            # drop half of the train features on purpose
            for index in (1, 2, 3):
                (features / f"train-{index:05d}.npy").unlink()
            manifest = root / "manifest.csv"
            manifest.write_text(
                "sample_id,video,label,signer,session,split\n"
                + "\n".join(
                    f"{r.sample_id},video/{r.split}/A/{r.sample_id}.mp4,{r.label},{r.signer},{r.session},{r.split}"
                    for r in records
                )
                + "\n",
                encoding="utf-8",
            )
            vocabulary, _ = build_ordered_vocabulary(["一/二/三"], min_frequency=1)
            model_config = CTCConfig(
                input_size=12, vocabulary_size=vocabulary.size, hidden_size=16, num_layers=1, dropout=0.0
            )
            training_config = TrainingConfig(
                epochs=1, batch_size=2, device="cpu", amp=False, seed=0, early_stopping_patience=5
            )
            result = train_ctc(
                manifest_path=manifest,
                feature_root=features,
                vocabulary=vocabulary,
                model_config=model_config,
                training_config=training_config,
                output_path=root / "checkpoint.pt",
                present_only=True,
            )
        self.assertEqual(result.train_samples, 3)  # 6 minus 3 deleted
        self.assertEqual(result.validation_samples, 1)

    def test_present_only_without_any_features_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "sample_id,video,label,signer,session,split\n"
                "train-00001,video/train/A/train-00001.mp4,我/要,A,train,train\n"
                "dev-00001,video/dev/A/dev-00001.mp4,我/要,A,dev,validation\n",
                encoding="utf-8",
            )
            vocabulary, _ = build_ordered_vocabulary(["我/要"], min_frequency=1)
            model_config = CTCConfig(
                input_size=12, vocabulary_size=vocabulary.size, hidden_size=16, num_layers=1, dropout=0.0
            )
            training_config = TrainingConfig(epochs=1, batch_size=1, device="cpu", amp=False, seed=0)
            with self.assertRaises(ValueError):
                train_ctc(
                    manifest_path=manifest,
                    feature_root=root / "features",
                    vocabulary=vocabulary,
                    model_config=model_config,
                    training_config=training_config,
                    output_path=root / "checkpoint.pt",
                    present_only=True,
                )

    def test_missing_feature_file_raises_without_present_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "sample_id,video,label,signer,session,split\n"
                "train-00001,video/train/A/train-00001.mp4,我/要,A,train,train\n"
                "dev-00001,video/dev/A/dev-00001.mp4,我/要,A,dev,validation\n",
                encoding="utf-8",
            )
            vocabulary, _ = build_ordered_vocabulary(["我/要"], min_frequency=1)
            model_config = CTCConfig(
                input_size=12, vocabulary_size=vocabulary.size, hidden_size=16, num_layers=1, dropout=0.0
            )
            training_config = TrainingConfig(
                epochs=1,
                batch_size=1,
                device="cpu",
                amp=False,
                seed=0,
                # keep the strict path so a missing feature file surfaces as FileNotFoundError
                drop_unalignable_targets=False,
            )
            with self.assertRaises(FileNotFoundError):
                train_ctc(
                    manifest_path=manifest,
                    feature_root=root / "features",
                    vocabulary=vocabulary,
                    model_config=model_config,
                    training_config=training_config,
                    output_path=root / "checkpoint.pt",
                )


class UnalignableTargetTests(unittest.TestCase):
    def test_long_targets_are_dropped_for_a_short_frame_cache(self) -> None:
        from cslr.recognition.training import _drop_unalignable

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # a 4-frame cache cannot hold a 4-token target (needs >= 7 frames)
            np.save(root / "a.npy", np.zeros((4, 3), dtype=np.float32))
            np.save(root / "b.npy", np.zeros((16, 3), dtype=np.float32))
            vocabulary, _ = build_ordered_vocabulary(["一/二/三/四"], min_frequency=1)
            records = [
                _record("a", "一/二/三/四", "train"),
                _record("b", "一/二", "train"),
            ]
            kept, dropped = _drop_unalignable(records, vocabulary, root)
        self.assertEqual([r.sample_id for r in kept], ["b"])
        self.assertEqual(dropped, ["a"])

    def test_missing_file_counts_as_dropped(self) -> None:
        from cslr.recognition.training import _drop_unalignable

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vocabulary, _ = build_ordered_vocabulary(["一/二"], min_frequency=1)
            kept, dropped = _drop_unalignable([_record("missing", "一/二", "train")], vocabulary, root)
        self.assertEqual(kept, [])
        self.assertEqual(dropped, ["missing"])

    def test_alignment_rule_is_two_l_minus_one(self) -> None:
        from cslr.recognition.training import _drop_unalignable

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 3 tokens need >= 5 frames
            np.save(root / "ok.npy", np.zeros((5, 3), dtype=np.float32))
            np.save(root / "short.npy", np.zeros((4, 3), dtype=np.float32))
            vocabulary, _ = build_ordered_vocabulary(["一/二/三"], min_frequency=1)
            kept, dropped = _drop_unalignable(
                [_record("ok", "一/二/三", "train"), _record("short", "一/二/三", "train")],
                vocabulary,
                root,
            )
        self.assertEqual([r.sample_id for r in kept], ["ok"])
        self.assertEqual(dropped, ["short"])


if __name__ == "__main__":
    unittest.main()
