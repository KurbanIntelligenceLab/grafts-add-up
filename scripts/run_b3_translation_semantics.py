"""Post-submission LaBSE screen of frozen B3 translations, without filtering.

Uses the complete official SentenceTransformer pipeline. Long texts retain a
standard 256-token prefix score plus a separately named full-text chunk-pooled
diagnostic. Neither score is a calibrated translation-quality probability.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL = "sentence-transformers/LaBSE"
REVISION = "836121a0533e5664b21c7aacc5d22951f2b8b25b"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def changed_number(text):
    # A sensitivity perturbation, not a human-labeled incorrect translation.
    m = re.search(r"(?<![\w.,])\d+(?:\.\d+)?(?![\w.,])", text)
    if m is None:
        return None
    replacement = format(Decimal(m.group()) + 1, "f")
    return text[:m.start()] + replacement + text[m.end():]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "results/b3/translation_audit_v1/paired_texts.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results/b3/translation_audit_v1")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("semantic_scores.jsonl", "semantic_summary.json", "embeddings.npz", "embedding_freeze.json"):
        if (args.output / name).exists():
            raise RuntimeError(f"Refusing to overwrite {name}")
    rows = [json.loads(s) for s in args.input.read_text(encoding="utf-8").splitlines() if s.strip()]
    assert len(rows) == 2000 and len({r["id"] for r in rows}) == 2000
    assert {r["language"] for r in rows} == {"fr", "zh"}
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["language"], r["probe"])].append(i)
    assert len(groups) == 4 and all(len(v) == 500 for v in groups.values())

    contract = {
        "model": MODEL, "revision": REVISION, "n_pairs": 2000,
        "input_sha256": digest(args.input), "script_sha256": digest(__file__),
        "dtype": "float32", "device": "cuda", "batch_size": args.batch_size,
        "max_tokens_including_specials": 256, "attention": "eager", "seed": 20261109,
        "quality_acceptance_threshold": None, "filter_or_retranslate": False,
        "primary_standard_score": "cosine of first-256-token normalized LaBSE embeddings",
        "long_text_diagnostic": "nonoverlapping token chunks; token-weighted mean of normalized chunk embeddings, L2-normalized before cosine",
        "coverage_diagnostic": "source-token-weighted mean of each source chunk's maximum cosine against any target chunk",
        "controls": {"wrong_target": "cyclic shift 137 within each language/probe group",
                     "source_number": "add 1 to first standalone Arabic numeric mention in English; unsupported cases have no control"},
        "control_limitation": "Controls assess sensitivity, not a human-calibrated fidelity threshold; numeric perturbations are not adjudicated error labels.",
        "timing": "protocol and source hashes recorded before embedding inference; post-submission, not externally preregistered",
    }
    write_json(args.output / "embedding_freeze.json", contract)
    print("Frozen protocol; importing inference libraries", flush=True)
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    # Avoid Windows DLL import-order problems in optional training dependencies.
    import pyarrow.dataset  # noqa: F401
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    assert torch.cuda.is_available(), "CUDA required by this declared run"
    torch.manual_seed(contract["seed"])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    snapshot = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    print("Loading pinned LaBSE on CUDA", flush=True)
    start = time.monotonic()
    model = SentenceTransformer(str(snapshot), device="cuda", local_files_only=True,
                                model_kwargs={"torch_dtype": torch.float32, "attn_implementation": "eager"})
    model.eval()
    model.float()
    assert model.max_seq_length == 256
    tokenizer = model.tokenizer
    capacity = 256 - tokenizer.num_special_tokens_to_add(pair=False)
    assert tokenizer.num_special_tokens_to_add(pair=False) == 2
    assert tokenizer.encode("", add_special_tokens=True) == [tokenizer.cls_token_id, tokenizer.sep_token_id]
    smoke_text = "A numerical example contains 12 pencils and 3 books."
    with torch.inference_mode():
        reference = model.encode([smoke_text], normalize_embeddings=True, convert_to_tensor=True,
                                 show_progress_bar=False)
        smoke_features = tokenizer([smoke_text], padding=True, return_tensors="pt")
        smoke_features = {k: v.to("cuda") for k, v in smoke_features.items()}
        direct = torch.nn.functional.normalize(model(smoke_features)["sentence_embedding"], p=2, dim=1)
        smoke_error = float((reference - direct).abs().max().item())
        assert smoke_error < 1e-6
    texts = {}
    text_meta = {}
    chunks = []
    chunk_index = {}

    def register(text):
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in texts:
            assert texts[key] == text
            return key
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        assert ids
        indices, weights = [], []
        for offset in range(0, len(ids), capacity):
            raw = ids[offset:offset + capacity]
            # Transformers 5 tokenizers removed build_inputs_with_special_tokens.
            # This pinned BERT tokenizer uses the verified [CLS] ... [SEP] form.
            token_ids = (tokenizer.cls_token_id, *raw, tokenizer.sep_token_id)
            if token_ids not in chunk_index:
                chunk_index[token_ids] = len(chunks)
                chunks.append(token_ids)
            indices.append(chunk_index[token_ids])
            weights.append(len(raw))
        texts[key] = text
        text_meta[key] = {"n_tokens": len(ids), "chunk_indices": indices, "chunk_weights": weights,
                          "exceeds_256_with_specials": len(ids) > capacity}
        return key

    keys = []
    for r in rows:
        mutant = changed_number(r["source_text"])
        keys.append((register(r["source_text"]), register(r["target_text"]),
                     register(mutant) if mutant is not None else None))
    input_path = args.output / "embedding_inputs.jsonl"
    with input_path.open("w", encoding="utf-8", newline="\n") as f:
        for key in sorted(texts):
            f.write(json.dumps({"sha256": key, "text": texts[key], **text_meta[key]}, ensure_ascii=False) + "\n")
    vectors = np.empty((len(chunks), 768), dtype=np.float32)
    print(f"Encoding {len(chunks)} unique chunks from {len(texts)} unique texts", flush=True)
    with torch.inference_mode():
        for offset in range(0, len(chunks), args.batch_size):
            batch = chunks[offset:offset + args.batch_size]
            features = tokenizer.pad({"input_ids": [list(x) for x in batch]}, padding=True, return_tensors="pt")
            features = {k: v.to("cuda") for k, v in features.items()}
            encoded = model(features)["sentence_embedding"]
            encoded = torch.nn.functional.normalize(encoded, p=2, dim=1)
            vectors[offset:offset + len(batch)] = encoded.cpu().numpy()
            if offset % (args.batch_size * 20) == 0:
                print(f"Encoded {offset + len(batch)}/{len(chunks)} chunks", flush=True)
    np.savez_compressed(args.output / "embeddings.npz", embeddings=vectors)
    with (args.output / "embedding_chunk_tokens.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for i, ids in enumerate(chunks):
            f.write(json.dumps({"index": i, "input_ids": ids}) + "\n")

    prefix, pooled = {}, {}
    for key, info in text_meta.items():
        v = vectors[info["chunk_indices"]]
        prefix[key] = v[0]
        mean = np.average(v, axis=0, weights=info["chunk_weights"])
        pooled[key] = (mean / np.linalg.norm(mean)).astype(np.float32)

    def cosine(a, b, kind):
        return float((prefix if kind == "prefix" else pooled)[a] @ (prefix if kind == "prefix" else pooled)[b])

    scores = [None] * len(rows)
    for group, indices in sorted(groups.items()):
        for j, i in enumerate(indices):
            a, b, c = keys[i]
            wrong = keys[indices[(j + 137) % 500]][1]
            source_chunks = vectors[text_meta[a]["chunk_indices"]]
            target_chunks = vectors[text_meta[b]["chunk_indices"]]
            coverage = np.average((source_chunks @ target_chunks.T).max(axis=1), weights=text_meta[a]["chunk_weights"])
            scores[i] = {
                "id": rows[i]["id"], "language": group[0], "probe": group[1],
                "source_text_sha256": a, "target_text_sha256": b,
                "source_tokens": text_meta[a]["n_tokens"], "target_tokens": text_meta[b]["n_tokens"],
                "source_chunk_count": len(text_meta[a]["chunk_indices"]), "target_chunk_count": len(text_meta[b]["chunk_indices"]),
                "source_prefix_truncated": text_meta[a]["exceeds_256_with_specials"],
                "target_prefix_truncated": text_meta[b]["exceeds_256_with_specials"],
                "prefix_cosine": cosine(a, b, "prefix"), "pooled_chunk_cosine": cosine(a, b, "pooled"),
                "source_chunk_coverage_cosine": float(coverage),
                "wrong_target_id": rows[indices[(j + 137) % 500]]["id"],
                "wrong_target_prefix_cosine": cosine(a, wrong, "prefix"),
                "wrong_target_pooled_cosine": cosine(a, wrong, "pooled"),
                "number_perturbed_source_sha256": c,
                "number_perturbed_prefix_cosine": cosine(c, b, "prefix") if c else None,
                "number_perturbed_pooled_cosine": cosine(c, b, "pooled") if c else None,
            }
    with (args.output / "semantic_scores.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in scores:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def stats(values):
        a = np.asarray(values, dtype=np.float64)
        return {"n": len(a), "mean": float(a.mean()), "min": float(a.min()),
                "p05": float(np.quantile(a, .05)), "median": float(np.median(a)),
                "p95": float(np.quantile(a, .95)), "max": float(a.max())}

    summary = {"protocol": contract, "groups": {}, "limitations": [
        "Post-submission automated diagnostics; no calibrated quality cutoff or independent human validation.",
        "Chunk pooling/coverage are descriptive extensions and do not establish semantic equivalence.",
        "Frozen probes and all earlier selection/evaluation results are unchanged."]}
    for group, indices in sorted(groups.items()):
        rs = [scores[i] for i in indices]
        mutated = [r for r in rs if r["number_perturbed_prefix_cosine"] is not None]
        summary["groups"]["/".join(group)] = {
            "n": len(rs),
            **{metric: stats([r[metric] for r in rs]) for metric in
               ("prefix_cosine", "pooled_chunk_cosine", "source_chunk_coverage_cosine", "wrong_target_prefix_cosine", "wrong_target_pooled_cosine")},
            "source_prefix_truncated": sum(r["source_prefix_truncated"] for r in rs),
            "target_prefix_truncated": sum(r["target_prefix_truncated"] for r in rs),
            "paired_exceeds_wrong_target_pooled": sum(r["pooled_chunk_cosine"] > r["wrong_target_pooled_cosine"] for r in rs),
            "numeric_perturbation_n": len(mutated),
            "numeric_prefix_score_drop": stats([r["prefix_cosine"] - r["number_perturbed_prefix_cosine"] for r in mutated]) if mutated else None,
            "lowest_pooled_ids": [r["id"] for r in sorted(rs, key=lambda r: r["pooled_chunk_cosine"])[:20]],
        }
    summary["execution"] = {"packages": {p: importlib.metadata.version(p) for p in
        ("torch", "sentence-transformers", "transformers", "huggingface-hub", "numpy", "pyarrow")},
        "gpu": torch.cuda.get_device_name(0), "elapsed_seconds": time.monotonic() - start,
        "direct_pipeline_vs_standard_encode_max_error": smoke_error,
        "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
        "weights_sha256": {str(p.relative_to(snapshot)).replace('\\', '/'): digest(p) for p in snapshot.rglob("*.safetensors")}}
    summary["artifacts"] = {p.name: digest(p) for p in args.output.iterdir() if p.is_file() and p.name in
        {"paired_texts.jsonl", "embedding_freeze.json", "semantic_scores.jsonl", "embedding_inputs.jsonl", "embeddings.npz", "embedding_chunk_tokens.jsonl"}}
    write_json(args.output / "semantic_summary.json", summary)
    print(json.dumps({k: {"mean": v["pooled_chunk_cosine"]["mean"], "p05": v["pooled_chunk_cosine"]["p05"]} for k, v in summary["groups"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
