from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Iterable

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - notebook runtime may install it later
    load_dataset = None


@dataclass
class TokenizerRecipe:
    vocab_size: int = 32000
    model_type: str = "unigram"
    character_coverage: float = 1.0
    sample_max_lines: int = 800000
    sample_stride: int = 8


@dataclass
class FineWebEduRecipe:
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    split: str = "train"
    streaming: bool = True
    max_docs: int = 50000
    min_words: int = 40
    max_chars_per_doc: int = 12000
    text_keys: tuple[str, ...] = ("text", "content", "raw_content", "document")
    dedupe_on_text: bool = True


@dataclass
class PhaseRecipe:
    name: str
    broad_web_weight: int
    repo_weight: int
    reasoning_weight: int
    science_weight: int
    code_weight: int
    max_chars_per_doc: int = 12000
    chunk_words: int = 220


@dataclass
class EvalSlice:
    name: str
    description: str
    source_labels: list[str]


@dataclass
class CorpusRecipe:
    tokenizer: TokenizerRecipe = field(default_factory=TokenizerRecipe)
    fineweb_edu: FineWebEduRecipe = field(default_factory=FineWebEduRecipe)
    phases: list[PhaseRecipe] = field(
        default_factory=lambda: [
            PhaseRecipe("tokenizer", broad_web_weight=92, repo_weight=3, reasoning_weight=1, science_weight=2, code_weight=2),
            PhaseRecipe("base", broad_web_weight=96, repo_weight=1, reasoning_weight=1, science_weight=1, code_weight=1),
            PhaseRecipe("adapt", broad_web_weight=65, repo_weight=10, reasoning_weight=8, science_weight=10, code_weight=7),
            PhaseRecipe("sft", broad_web_weight=0, repo_weight=35, reasoning_weight=35, science_weight=20, code_weight=10),
        ]
    )
    eval_slices: list[EvalSlice] = field(
        default_factory=lambda: [
            EvalSlice("broad_edu_holdout", "Held-out educational web text from FineWeb Edu", ["broad_web"]),
            EvalSlice("repo_science", "README, AGENTS, and scientific notebook markdown", ["repo_markdown", "science_notebook"]),
            EvalSlice("systems_lm", "Transformer/distributed systems text", ["transformer_notebook", "repo_markdown"]),
            EvalSlice("instruction", "Curated instruction/reasoning examples", ["reasoning_seed"]),
        ]
    )


def _normalize_whitespace(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def _chunk_words(text: str, chunk_words: int) -> list[str]:
    words = text.split()
    chunks = []
    for start in range(0, len(words), chunk_words):
        chunk = " ".join(words[start : start + chunk_words]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _stable_hash(text: str) -> str:
    return sha1(text.encode("utf-8")).hexdigest()


def _extract_text_from_row(row: dict, text_keys: Iterable[str]) -> str | None:
    for key in text_keys:
        value = row.get(key)
        if value is None:
            continue
        text = _normalize_whitespace(str(value))
        if text:
            return text
    return None


def extract_notebook_cells(path: Path) -> list[dict[str, str]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    docs = []
    for idx, cell in enumerate(notebook.get("cells", [])):
        source = _normalize_whitespace("".join(cell.get("source", [])))
        if not source:
            continue
        label = "markdown" if cell.get("cell_type") == "markdown" else "code"
        docs.append({"id": f"{path.name}:{idx}", "label": label, "text": source})
    return docs


def build_repo_local_corpus(root: Path | str) -> list[dict[str, str]]:
    root = Path(root)
    documents: list[dict[str, str]] = []

    markdown_paths = [root / "README.md", root / "agents.md"]
    for path in markdown_paths:
        if path.exists():
            documents.append(
                {
                    "id": path.name,
                    "label": "repo_markdown",
                    "text": _normalize_whitespace(path.read_text(encoding="utf-8")),
                }
            )

    notebook_label_map = {
        "transformer.ipynb": "transformer_notebook",
        "scienceagentv1.ipynb": "science_notebook",
        "turbulence.ipynb": "science_notebook",
        "MHD_64.ipynb": "science_notebook",
    }
    for notebook_path in sorted(root.glob("*.ipynb")):
        label = notebook_label_map.get(notebook_path.name, "notebook")
        for cell in extract_notebook_cells(notebook_path):
            if cell["label"] == "markdown":
                documents.append({"id": cell["id"], "label": label, "text": cell["text"]})
            elif notebook_path.name == "transformer.ipynb":
                # Keep commented code and section headers from the transformer notebook as local systems text.
                if "#" in cell["text"]:
                    documents.append({"id": cell["id"], "label": label, "text": cell["text"]})

    reasoning_seed_prompts = [
        "Explain why grouped-query attention reduces KV-cache growth for a small decoder-only LM.",
        "Compare RMSNorm and LayerNorm for a student-scale autoregressive model.",
        "Describe how domain adaptation should differ from broad pretraining in this repository.",
        "Explain why a notebook-first training workflow can still support serious systems instrumentation.",
    ]
    for idx, prompt in enumerate(reasoning_seed_prompts):
        documents.append(
            {
                "id": f"reasoning_seed:{idx}",
                "label": "reasoning_seed",
                "text": prompt,
            }
        )

    deduped = {}
    for doc in documents:
        normalized = _normalize_whitespace(doc["text"])
        if len(normalized.split()) < 8:
            continue
        key = _stable_hash(normalized)
        deduped.setdefault(key, {"id": doc["id"], "label": doc["label"], "text": normalized})
    return list(deduped.values())


def stream_fineweb_edu(recipe: FineWebEduRecipe) -> list[dict[str, str]]:
    if load_dataset is None:
        raise ImportError(
            "The `datasets` library is required to load FineWeb Edu. "
            "Install it in the notebook environment with `%pip install datasets`."
        )

    dataset = load_dataset(recipe.dataset_name, split=recipe.split, streaming=recipe.streaming)
    documents: list[dict[str, str]] = []
    seen = set()
    kept = 0

    for idx, row in enumerate(dataset):
        text = _extract_text_from_row(row, recipe.text_keys)
        if text is None:
            continue
        text = text[: recipe.max_chars_per_doc].strip()
        if len(text.split()) < recipe.min_words:
            continue

        if recipe.dedupe_on_text:
            key = _stable_hash(text)
            if key in seen:
                continue
            seen.add(key)

        documents.append(
            {
                "id": f"fineweb_edu:{idx}",
                "label": "broad_web",
                "text": text,
                "source": recipe.dataset_name,
            }
        )
        kept += 1
        if kept >= recipe.max_docs:
            break

    return documents


def save_corpus_cache(documents: list[dict[str, str]], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(documents, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_corpus_cache(path: Path | str) -> list[dict[str, str]]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_or_build_fineweb_edu(recipe: FineWebEduRecipe, cache_path: Path | str | None = None, force_refresh: bool = False) -> list[dict[str, str]]:
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force_refresh:
            return load_corpus_cache(cache_path)

    documents = stream_fineweb_edu(recipe)
    if cache_path is not None:
        save_corpus_cache(documents, cache_path)
    return documents


def summarize_corpus_sources(documents: Iterable[dict[str, str]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for doc in documents:
        label = doc.get("label", "unknown")
        stats = summary.setdefault(label, {"documents": 0.0, "words": 0.0, "chars": 0.0})
        text = doc.get("text", "")
        stats["documents"] += 1.0
        stats["words"] += float(len(text.split()))
        stats["chars"] += float(len(text))
    return summary


def build_phase_documents(
    documents: Iterable[dict[str, str]],
    recipe: PhaseRecipe,
) -> dict[str, list[str]]:
    grouped = {
        "broad_web": [],
        "repo": [],
        "reasoning": [],
        "science": [],
        "code": [],
    }

    for doc in documents:
        text = doc["text"][: recipe.max_chars_per_doc]
        chunks = _chunk_words(text, recipe.chunk_words)
        label = doc["label"]
        if label == "broad_web":
            grouped["broad_web"].extend(chunks)
        elif label == "reasoning_seed":
            grouped["reasoning"].extend(chunks)
        elif label == "transformer_notebook":
            grouped["code"].extend(chunks)
        elif label == "science_notebook":
            grouped["science"].extend(chunks)
        elif label == "repo_markdown":
            grouped["repo"].extend(chunks)
        else:
            grouped["repo"].extend(chunks)
    return grouped


def weighted_phase_lines(phase_docs: dict[str, list[str]], recipe: PhaseRecipe) -> list[str]:
    def unique_lines(lines: list[str]) -> list[str]:
        out = []
        seen = set()
        for line in lines:
            normalized = _normalize_whitespace(line)
            if not normalized:
                continue
            key = _stable_hash(normalized)
            if key in seen:
                continue
            seen.add(key)
            out.append(normalized)
        return out

    broad_web = unique_lines(phase_docs["broad_web"])
    repo = unique_lines(phase_docs["repo"])
    reasoning = unique_lines(phase_docs["reasoning"])
    science = unique_lines(phase_docs["science"])
    code = unique_lines(phase_docs["code"])

    weighted = []
    weighted.extend(broad_web * max(recipe.broad_web_weight, 0))
    weighted.extend(repo * max(recipe.repo_weight, 0))
    weighted.extend(reasoning * max(recipe.reasoning_weight, 0))
    weighted.extend(science * max(recipe.science_weight, 0))
    weighted.extend(code * max(recipe.code_weight, 0))
    return weighted


def phase_summary(phase_name: str, phase_docs: dict[str, list[str]], phase_lines: list[str], recipe: PhaseRecipe) -> dict[str, object]:
    return {
        "phase_name": phase_name,
        "weights": asdict(recipe),
        "bucket_unique_counts": {
            "broad_web": len(set(phase_docs["broad_web"])),
            "repo": len(set(phase_docs["repo"])),
            "reasoning": len(set(phase_docs["reasoning"])),
            "science": len(set(phase_docs["science"])),
            "code": len(set(phase_docs["code"])),
        },
        "phase_line_count": len(phase_lines),
        "phase_word_count": sum(len(line.split()) for line in phase_lines),
    }


def build_corpus_manifest(
    broad_documents: list[dict[str, str]],
    repo_documents: list[dict[str, str]],
    recipe: CorpusRecipe,
    phase_line_map: dict[str, list[str]],
    phase_group_map: dict[str, dict[str, list[str]]],
    contamination: dict[str, dict[str, float]] | None = None,
) -> dict[str, object]:
    phase_meta = {}
    for phase in recipe.phases:
        if phase.name in phase_line_map and phase.name in phase_group_map:
            phase_meta[phase.name] = phase_summary(phase.name, phase_group_map[phase.name], phase_line_map[phase.name], phase)

    return {
        "fineweb_edu": asdict(recipe.fineweb_edu),
        "tokenizer": asdict(recipe.tokenizer),
        "source_summary": {
            "broad_web": summarize_corpus_sources(broad_documents),
            "repo_local": summarize_corpus_sources(repo_documents),
            "combined": summarize_corpus_sources([*broad_documents, *repo_documents]),
        },
        "phase_summary": phase_meta,
        "contamination": contamination or {},
    }


def contamination_report(train_lines: Iterable[str], eval_lines: Iterable[str], ngram_size: int = 8) -> dict[str, float]:
    def ngrams(text: str) -> set[str]:
        words = text.split()
        if len(words) < ngram_size:
            return {text}
        return {" ".join(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)}

    train_ngrams = set()
    for line in train_lines:
        train_ngrams.update(ngrams(line))

    eval_ngrams = set()
    for line in eval_lines:
        eval_ngrams.update(ngrams(line))

    overlap = train_ngrams & eval_ngrams
    overlap_rate = len(overlap) / max(len(eval_ngrams), 1)
    return {
        "train_ngram_count": float(len(train_ngrams)),
        "eval_ngram_count": float(len(eval_ngrams)),
        "overlap_ngram_count": float(len(overlap)),
        "overlap_rate": float(overlap_rate),
    }


def recipe_to_jsonable(recipe: CorpusRecipe) -> dict:
    return {
        "tokenizer": asdict(recipe.tokenizer),
        "fineweb_edu": asdict(recipe.fineweb_edu),
        "phases": [asdict(phase) for phase in recipe.phases],
        "eval_slices": [asdict(slc) for slc in recipe.eval_slices],
    }
