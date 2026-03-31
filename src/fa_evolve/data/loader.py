"""HuggingFace dataset loading for all 6 VQA skill clusters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


@dataclass
class VQASample:
    """A single VQA sample with unified schema."""
    question: str
    image: Any  # PIL Image
    answer: Any  # str for most, list[str] for VQAv2
    domain: str
    metric: str
    dataset_id: str


def _load_mathvista(split: str) -> list[VQASample]:
    ds = load_dataset("AI4Math/MathVista", split=split)
    samples = []
    for row in ds:
        answer = str(row.get("answer", ""))
        question = row.get("question", "")
        if row.get("choices"):
            choices_str = "\n".join(
                f"{chr(65+i)}. {c}" for i, c in enumerate(row["choices"])
            )
            question = f"{question}\n{choices_str}"
        samples.append(VQASample(
            question=question,
            image=row.get("decoded_image") or row.get("image"),
            answer=answer,
            domain="math",
            metric="mathvista",
            dataset_id="AI4Math/MathVista",
        ))
    return samples


def _load_docvqa(split: str) -> list[VQASample]:
    ds = load_dataset("lmms-lab/DocVQA", split=split)
    samples = []
    for row in ds:
        answers = row.get("answers", [])
        answer = answers[0] if answers else ""
        samples.append(VQASample(
            question=row.get("question", ""),
            image=row.get("image"),
            answer=answer,
            domain="documents",
            metric="anls",
            dataset_id="lmms-lab/DocVQA",
        ))
    return samples


def _load_chartqa(split: str) -> list[VQASample]:
    ds = load_dataset("HuggingFaceM4/ChartQA", split=split)
    samples = []
    for row in ds:
        samples.append(VQASample(
            question=row.get("question", ""),
            image=row.get("image"),
            answer=str(row.get("answer", "")),
            domain="charts",
            metric="chartqa",
            dataset_id="HuggingFaceM4/ChartQA",
        ))
    return samples


def _load_gqa(split: str) -> list[VQASample]:
    ds = load_dataset("lmms-lab/GQA", split=split)
    samples = []
    for row in ds:
        samples.append(VQASample(
            question=row.get("question", ""),
            image=row.get("image"),
            answer=str(row.get("answer", "")),
            domain="spatial",
            metric="exact_match",
            dataset_id="lmms-lab/GQA",
        ))
    return samples


def _load_scienceqa(split: str) -> list[VQASample]:
    ds = load_dataset("derek-thomas/ScienceQA", split=split)
    samples = []
    for row in ds:
        if not row.get("image"):
            continue
        choices = row.get("choices", [])
        answer_idx = row.get("answer", 0)
        answer_letter = chr(65 + answer_idx) if isinstance(answer_idx, int) else str(answer_idx)
        choices_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        question = f"{row.get('question', '')}\n{choices_str}"
        samples.append(VQASample(
            question=question,
            image=row["image"],
            answer=answer_letter,
            domain="science",
            metric="mc_match",
            dataset_id="derek-thomas/ScienceQA",
        ))
    return samples


def _load_vqav2(split: str) -> list[VQASample]:
    ds = load_dataset("HuggingFaceM4/VQAv2", split=split)
    samples = []
    for row in ds:
        raw_answers = row.get("answers", [])
        answer_list = []
        for a in raw_answers:
            if isinstance(a, str):
                answer_list.append(a)
            elif isinstance(a, dict):
                answer_list.append(str(a.get("answer", "")))
            else:
                answer_list.append(str(a))
        samples.append(VQASample(
            question=row.get("question", ""),
            image=row.get("image"),
            answer=answer_list,
            domain="natural",
            metric="vqa_soft",
            dataset_id="HuggingFaceM4/VQAv2",
        ))
    return samples


_LOADERS = {
    "math": _load_mathvista,
    "documents": _load_docvqa,
    "charts": _load_chartqa,
    "spatial": _load_gqa,
    "science": _load_scienceqa,
    "natural": _load_vqav2,
}


def load_cluster_data(cluster_name: str, split: str) -> list[VQASample]:
    """Load data for a specific skill cluster and split."""
    if cluster_name not in _LOADERS:
        raise ValueError(f"Unknown cluster: {cluster_name}. Available: {list(_LOADERS.keys())}")
    return _LOADERS[cluster_name](split)
