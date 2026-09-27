"""P0: zero-shot VL probe on CE-CSL dev samples (no training, no test split).

The probe answers one question only: does an off-the-shelf vision-language model
show any sign-language ability at all on CE-CSL? It reports BLEU, chrF,
ROUGE-L, exact match and distinct-1 over a deterministic dev sample, and writes
every prediction next to the reference so a human can judge the samples.

Two backends:

- ``api``: any OpenAI-compatible vision endpoint (Qwen-VL, GLM-4.5V, ...).
  Requires ``CSLR_VL_API_KEY`` (and optionally ``CSLR_VL_BASE_URL`` / model).
- ``local``: a local HuggingFace vision-language model (needs torch +
  transformers + accelerate, and enough VRAM for the chosen model/quantization).

Frames are extracted with OpenCV when available, otherwise with an ``ffmpeg``
binary found on PATH. The official test split is refused before any file is
opened.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from cslr.data.manifest import read_manifest, validate_manifest
from cslr.recognition.io import write_json
from cslr.recognition.text_metrics import bleu, chrf, distinct_ngrams, exact_match, rouge_l

DEFAULT_PROMPT = (
    "这是一段中国手语视频的连续帧。请根据手语动作判断说话内容，"
    "只输出一句最可能的中文句子，不要解释，不要加引号。"
)
FROZEN_SPLIT_ERROR = (
    "split 'test' is frozen for Part 4: the official 500-sample test split must not be "
    "read, inferred on, or tuned against during this phase"
)


def load_reference_sentences(data_root: Path, split_file: str) -> dict[str, str]:
    """Read ``Number -> Chinese Sentences`` from an official CE-CSL label CSV."""

    import csv

    path = data_root / "label" / split_file
    if not path.exists():
        raise FileNotFoundError(f"label file is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            (row["Number"] or "").strip(): (row["Chinese Sentences"] or "").strip()
            for row in csv.DictReader(handle)
        }


def extract_frames(video: Path, count: int, workdir: Path, size: int = 448) -> list[Path]:
    """Uniformly sample ``count`` frames from a video."""

    if count < 1:
        raise ValueError("frame count must be at least 1")
    if not video.exists():
        raise FileNotFoundError(f"video is missing: {video}")
    frames_dir = workdir / video.stem
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frames_dir.glob("*.jpg"))
    if len(existing) >= count:
        return existing[:count]

    try:
        import cv2  # type: ignore
    except ImportError:
        return _extract_frames_ffmpeg(video, count, frames_dir, size)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"unable to open video: {video}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            raise ValueError(f"video reports no frames: {video}")
        indices = [round(index * (total - 1) / max(count - 1, 1)) for index in range(count)]
        written: list[Path] = []
        for order, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            scale = size / max(height, width)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            path = frames_dir / f"{order:03d}.jpg"
            cv2.imwrite(str(path), frame)
            written.append(path)
        if not written:
            raise ValueError(f"no frame could be decoded: {video}")
        return written
    finally:
        capture.release()


def _extract_frames_ffmpeg(video: Path, count: int, frames_dir: Path, size: int) -> list[Path]:
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(
            "frame extraction needs either opencv-python (pip install opencv-python) or ffmpeg on PATH"
        )
    output = frames_dir / "%03d.jpg"
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(video),
        "-vf", f"fps={max(count, 1)},scale='min({size},iw)':-2",
        "-frames:v", str(count), str(output),
    ]
    subprocess.run(command, check=True)
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise ValueError(f"ffmpeg produced no frames for {video}")
    return frames


def _find_ffmpeg() -> str | None:
    from shutil import which

    return which("ffmpeg")


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def api_translate(
    frames: list[Path],
    prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout: float = 120.0,
    retries: int = 2,
) -> str:
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    for frame in frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(frame)}"},
            }
        )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 128,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return str(body["choices"][0]["message"]["content"]).strip()
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as error:
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"VL API call failed: {last_error}")


class LocalVlModel:
    """A loaded local vision-language model, reused across samples.

    Loading a 3B VL model takes minutes and a fixed amount of VRAM, so it is loaded once per
    probe rather than once per clip. ``cpu_offload`` moves the quantized weights partly to system
    RAM, which is what makes a 3B model usable at all on a 6 GB GPU (at a large speed cost).
    """

    def __init__(
        self,
        model_name: str,
        load_in_4bit: bool = True,
        cpu_offload: bool = False,
        max_gpu_memory: str | None = None,
    ) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # type: ignore
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "local backend needs torch + transformers with Qwen2.5-VL support installed"
            ) from error
        self.torch = torch
        quantization = None
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore

                quantization = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=cpu_offload,
                )
            except ImportError:  # pragma: no cover
                quantization = None
        max_memory = None
        if cpu_offload and max_gpu_memory:
            max_memory = {0: max_gpu_memory, "cpu": "32GiB"}
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization,
            torch_dtype=torch.bfloat16 if quantization is None else None,
            device_map="auto",
            max_memory=max_memory,
        )

    def translate(
        self, frames: list[Path], prompt: str, max_new_tokens: int = 64
    ) -> str:
        from qwen_vl_utils import process_vision_info  # type: ignore

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": str(frame)} for frame in frames]
                + [{"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
        ).to(self.model.device)
        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        trimmed = generated[:, inputs.input_ids.shape[1] :]
        decoded = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0].strip()


def local_translate(
    frames: list[Path],
    prompt: str,
    model_name: str,
    max_new_tokens: int = 64,
    load_in_4bit: bool = True,
    cpu_offload: bool = False,
) -> str:
    """Convenience wrapper that loads the model, translates one clip, then drops it."""

    return LocalVlModel(model_name, load_in_4bit, cpu_offload).translate(
        frames, prompt, max_new_tokens
    )


def run_probe(args: argparse.Namespace) -> int:
    if args.split == "test":
        raise SystemExit(FROZEN_SPLIT_ERROR)
    data_root = Path(args.data_root)
    records = read_manifest(args.manifest)
    validate_manifest(records)
    split_name = "validation" if args.split == "dev" else args.split
    split_records = [record for record in records if record.split == split_name]
    if not split_records:
        raise SystemExit(f"manifest has no records for split {args.split}")
    references = load_reference_sentences(data_root, f"{args.split}.csv")

    limit = args.limit if args.limit else len(split_records)
    selected = split_records[:limit]
    workdir = Path(args.frame_cache)
    workdir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("CSLR_VL_API_KEY", "")
    if args.backend == "api" and not api_key:
        raise SystemExit("CSLR_VL_API_KEY is not set; export it or use --backend local")

    predictions: list[str] = []
    expected: list[str] = []
    sample_ids: list[str] = []
    latencies: list[float] = []
    errors: list[dict[str, str]] = []
    started = time.perf_counter()

    local_model: LocalVlModel | None = None
    if args.backend == "local":
        print(
            json.dumps(
                {"loading_model": args.model, "cpu_offload": args.cpu_offload},
                ensure_ascii=False,
            ),
            flush=True,
        )
        local_model = LocalVlModel(
            args.model or "Qwen/Qwen2.5-VL-3B-Instruct",
            load_in_4bit=not args.no_4bit,
            cpu_offload=args.cpu_offload,
            max_gpu_memory=args.max_gpu_memory,
        )

    for record in selected:
        reference = references.get(record.sample_id, "")
        if not reference:
            print(f"skip {record.sample_id}: no reference sentence", file=sys.stderr)
            continue
        frames = extract_frames(data_root / record.video, args.frames, workdir)
        frame_started = time.perf_counter()
        try:
            if args.backend == "api":
                text = api_translate(
                    frames,
                    args.prompt,
                    args.model or "qwen-vl-max",
                    os.getenv(
                        "CSLR_VL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
                    ),
                    api_key,
                )
            else:
                assert local_model is not None
                text = local_model.translate(frames, args.prompt)
        except Exception as error:  # noqa: BLE001 - one bad sample must not lose the whole probe
            errors.append({"sample_id": record.sample_id, "error": f"{type(error).__name__}: {error}"})
            print(
                json.dumps(
                    {"sample_id": record.sample_id, "error": f"{type(error).__name__}: {error}"},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if not args.keep_going:
                raise
            continue
        latencies.append(time.perf_counter() - frame_started)
        predictions.append(text)
        expected.append(reference)
        sample_ids.append(record.sample_id)
        print(json.dumps({"sample_id": record.sample_id, "reference": reference, "prediction": text}, ensure_ascii=False), flush=True)

    if not predictions:
        raise SystemExit("no sample could be evaluated")

    prediction_chars = [list(text) for text in predictions]
    reference_chars = [list(text) for text in expected]
    report: dict[str, object] = {
        "stage": "P0",
        "backend": args.backend,
        "model": args.model,
        "split": args.split,
        "samples": len(predictions),
        "frames_per_sample": args.frames,
        "prompt": args.prompt,
        "formal_result": False,
        "test_split_read": False,
        "metrics": {
            **bleu(reference_chars, prediction_chars),
            "rouge_l": rouge_l(reference_chars, prediction_chars),
            "chrf": chrf(reference_chars, prediction_chars),
            "exact_match": exact_match(expected, predictions),
            "distinct_1": distinct_ngrams(prediction_chars, order=1)["distinct"],
            "distinct_2": distinct_ngrams(prediction_chars, order=2)["distinct"],
            "empty_predictions": sum(1 for text in predictions if not text.strip()),
            "latency_ms_mean": 1000 * sum(latencies) / len(latencies),
        },
        "samples_detail": [
            {
                "sample_id": sample_ids[index],
                "reference": expected[index],
                "prediction": predictions[index],
            }
            for index in range(len(predictions))
        ],
        "errors": errors,
        "requested_samples": len(selected),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    write_json(Path(args.report), report)
    print(json.dumps(report["metrics"], ensure_ascii=False))
    print(f"wrote {args.report}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cslr.recognition.p0_vl_probe")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/ce-csl.csv"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="dev", choices=["train", "dev", "validation", "test"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--backend", choices=["api", "local"], default="api")
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-4bit", dest="no_4bit", action="store_true", default=False)
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="allow quantized weights to spill into system RAM (needed on a 6 GB GPU)",
    )
    parser.add_argument(
        "--max-gpu-memory",
        default=None,
        help="cap VRAM for the local model, e.g. 4GiB",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="record a per-sample error and continue instead of aborting the whole probe",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--frame-cache", default="artifacts/logs/p0-frames")
    parser.add_argument("--report", default="artifacts/metrics/part4-p0-vl-zeroshot.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
