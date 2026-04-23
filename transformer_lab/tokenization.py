from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import TokenizerRecipe

try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover - installed in notebook runtime
    spm = None


def _require_sentencepiece():
    if spm is None:
        raise ImportError(
            "The `sentencepiece` library is required for tokenizer training. "
            "Install it in the notebook environment with `%pip install sentencepiece`."
        )


def sample_tokenizer_corpus(
    input_path: Path | str,
    output_path: Path | str,
    recipe: TokenizerRecipe,
    force_rebuild: bool = False,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force_rebuild:
        return output_path

    kept = 0
    seen = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            text = line.strip()
            if not text:
                continue
            if seen % max(recipe.sample_stride, 1) == 0:
                fout.write(text + "\n")
                kept += 1
                if kept >= recipe.sample_max_lines:
                    break
            seen += 1
    return output_path


def train_or_load_sentencepiece(
    training_text_path: Path | str,
    model_prefix: Path | str,
    recipe: TokenizerRecipe,
    force_retrain: bool = False,
) -> tuple[Path, Path]:
    _require_sentencepiece()
    model_prefix = Path(model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    model_path = model_prefix.with_suffix(".model")
    vocab_path = model_prefix.with_suffix(".vocab")

    if model_path.exists() and vocab_path.exists() and not force_retrain:
        return model_path, vocab_path

    spm.SentencePieceTrainer.Train(
        input=str(training_text_path),
        model_prefix=str(model_prefix),
        vocab_size=recipe.vocab_size,
        model_type=recipe.model_type,
        character_coverage=recipe.character_coverage,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3,
        shuffle_input_sentence=True,
        input_sentence_size=min(recipe.sample_max_lines, 1_000_000),
        train_extremely_large_corpus=False,
    )
    return model_path, vocab_path


def load_sentencepiece_processor(model_path: Path | str):
    _require_sentencepiece()
    model_path = Path(model_path)
    processor = spm.SentencePieceProcessor()
    processor.load(str(model_path))
    return processor


def encode_text_file_to_tokens(
    input_path: Path | str,
    output_path: Path | str,
    processor,
    eos_id: int,
    force_rebuild: bool = False,
) -> tuple[Path, int]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force_rebuild:
        arr = np.load(output_path, mmap_mode="r")
        return output_path, int(arr.shape[0])

    token_buffer: list[int] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            token_buffer.extend(processor.encode(text, out_type=int))
            token_buffer.append(int(eos_id))

    array = np.asarray(token_buffer, dtype=np.int32)
    np.save(output_path, array)
    return output_path, int(array.shape[0])


def split_token_array(
    token_path: Path | str,
    train_path: Path | str,
    val_path: Path | str,
    seq_len: int,
    val_fraction: float = 0.02,
    force_rebuild: bool = False,
) -> dict[str, int | str]:
    token_path = Path(token_path)
    train_path = Path(train_path)
    val_path = Path(val_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    if train_path.exists() and val_path.exists() and not force_rebuild:
        train = np.load(train_path, mmap_mode="r")
        val = np.load(val_path, mmap_mode="r")
        return {
            "token_path": str(token_path),
            "train_path": str(train_path),
            "val_path": str(val_path),
            "total_tokens": int(train.shape[0] + val.shape[0]),
            "train_tokens": int(train.shape[0]),
            "val_tokens": int(val.shape[0]),
        }

    tokens = np.load(token_path, mmap_mode="r")
    total = int(tokens.shape[0])
    if total <= seq_len + 1:
        raise ValueError(f"Token stream too short for seq_len={seq_len}: {total}")

    split_idx = int((1.0 - val_fraction) * total)
    split_idx = max(split_idx, seq_len + 1)
    split_idx = min(split_idx, total - (seq_len + 1))

    train_tokens = np.asarray(tokens[:split_idx], dtype=np.int32)
    val_tokens = np.asarray(tokens[split_idx:], dtype=np.int32)
    np.save(train_path, train_tokens)
    np.save(val_path, val_tokens)

    return {
        "token_path": str(token_path),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "total_tokens": int(total),
        "train_tokens": int(train_tokens.shape[0]),
        "val_tokens": int(val_tokens.shape[0]),
    }


def build_phase_token_artifacts(
    phase_file_map: dict[str, str | Path],
    tokenizer_dir: Path | str,
    token_dir: Path | str,
    recipe: TokenizerRecipe,
    seq_len: int,
    val_fraction: float = 0.02,
    force_retrain_tokenizer: bool = False,
    force_reencode: bool = False,
) -> dict[str, object]:
    tokenizer_dir = Path(tokenizer_dir)
    token_dir = Path(token_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_phase_path = Path(phase_file_map["tokenizer"])
    sampled_corpus_path = tokenizer_dir / "tokenizer_sampled.txt"
    sampled_corpus_path = sample_tokenizer_corpus(tokenizer_phase_path, sampled_corpus_path, recipe, force_rebuild=force_retrain_tokenizer)

    model_prefix = tokenizer_dir / f"spm_{recipe.vocab_size}"
    model_path, vocab_path = train_or_load_sentencepiece(
        sampled_corpus_path,
        model_prefix=model_prefix,
        recipe=recipe,
        force_retrain=force_retrain_tokenizer,
    )
    processor = load_sentencepiece_processor(model_path)

    phase_token_info: dict[str, object] = {}
    for phase_name, phase_path in phase_file_map.items():
        phase_path = Path(phase_path)
        token_path = token_dir / f"{phase_name}_tokens.npy"
        token_path, token_count = encode_text_file_to_tokens(
            phase_path,
            token_path,
            processor=processor,
            eos_id=processor.eos_id(),
            force_rebuild=force_reencode,
        )
        phase_record: dict[str, object] = {
            "phase_name": phase_name,
            "phase_text_path": str(phase_path),
            "token_path": str(token_path),
            "token_count": int(token_count),
        }

        if phase_name in {"base", "adapt"}:
            train_path = token_dir / f"{phase_name}_train_tokens.npy"
            val_path = token_dir / f"{phase_name}_val_tokens.npy"
            phase_record["split"] = split_token_array(
                token_path,
                train_path=train_path,
                val_path=val_path,
                seq_len=seq_len,
                val_fraction=val_fraction,
                force_rebuild=force_reencode,
            )
        phase_token_info[phase_name] = phase_record

    manifest = {
        "tokenizer_recipe": asdict(recipe),
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "sampled_corpus_path": str(sampled_corpus_path),
        "vocab_size": int(processor.get_piece_size()),
        "pad_id": int(processor.pad_id()),
        "bos_id": int(processor.bos_id()),
        "eos_id": int(processor.eos_id()),
        "unk_id": int(processor.unk_id()),
        "phase_token_info": phase_token_info,
    }
    return manifest


def save_token_manifest(manifest: dict[str, object], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
