"""Tokenize a Pile ``.jsonl.zst`` file into one flat uint16 token stream.

Documents are read in file order, encoded with the Pythia/GPT-NeoX tokenizer (no BOS),
and each is followed by ``<|endoftext|>`` (id 0). Reading stops at the end of the first
document that brings the stream to ``--max-tokens``, so the stream always ends on a whole
document and the ``<|endoftext|>`` count equals the document count.

Encoding runs in a worker pool, but ``Pool.imap`` returns chunks in submission order, so
the output is in document order. The split's entry in ``manifest.json`` records the counts,
the sha256 of the output and of the source file, and the tokenizer revision.

    uv run python data/tokenize_pile.py --split train --max-tokens 2.1e9 \
        --input $DATA_ROOT/raw/train/00.jsonl.zst --data-root $DATA_ROOT
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import zstandard
from tokenizers import Tokenizer

EOS_TOKEN = "<|endoftext|>"
_TOKENIZER: Tokenizer | None = None
_EOS_ID: int | None = None


def _init_worker(tokenizer_path: str) -> None:
    global _TOKENIZER, _EOS_ID
    _TOKENIZER = Tokenizer.from_file(tokenizer_path)
    # Text that happens to contain "<|endoftext|>" is encoded as plain text, so id 0 only
    # ever appears where we put it: after each document.
    _TOKENIZER.encode_special_tokens = True
    _EOS_ID = _TOKENIZER.token_to_id(EOS_TOKEN)


def encode_lines(lines: list[str]) -> tuple[np.ndarray, int, int]:
    """Encode raw jsonl lines -> (tokens with EOS after each doc, n_docs, n_empty_skipped)."""
    assert _TOKENIZER is not None and _EOS_ID is not None
    texts = [json.loads(line)["text"] for line in lines]
    kept = [t for t in texts if t]
    encodings = _TOKENIZER.encode_batch(kept, add_special_tokens=False)
    pieces = []
    for enc in encodings:
        pieces.append(np.asarray(enc.ids, dtype=np.uint32))
        pieces.append(np.asarray([_EOS_ID], dtype=np.uint32))
    tokens = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.uint32)
    if tokens.size and tokens.max() > np.iinfo(np.uint16).max:
        raise ValueError("token id does not fit in uint16")
    return tokens.astype(np.uint16), len(kept), len(texts) - len(kept)


def iter_line_chunks(path: Path, chunk_docs: int) -> Iterator[list[str]]:
    with open(path, "rb") as fh:
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh)
        chunk: list[str] = []
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            chunk.append(line)
            if len(chunk) == chunk_docs:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def sha256_file(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while data := fh.read(block):
            h.update(data)
    return h.hexdigest()


def read_source(path: Path) -> dict[str, str]:
    """``SOURCE`` files written by download.sh: repo on line 1, revision on line 2."""
    if not path.exists():
        return {}
    repo, revision = (path.read_text().splitlines() + ["", ""])[:2]
    return {"repo": repo, "revision": revision}


def tokenize_split(
    input_path: Path,
    output_path: Path,
    tokenizer_path: Path,
    max_tokens: int,
    workers: int,
    chunk_docs: int,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    digest = hashlib.sha256()
    n_tokens = n_docs = n_empty = n_eos = 0
    eos_id = Tokenizer.from_file(str(tokenizer_path)).token_to_id(EOS_TOKEN)
    t0 = time.time()
    next_report = 100_000_000

    with open(tmp_path, "wb") as out, mp.Pool(workers, _init_worker, (str(tokenizer_path),)) as pool:
        for tokens, docs, empty in pool.imap(encode_lines, iter_line_chunks(input_path, chunk_docs)):
            n_empty += empty
            if n_tokens + tokens.size >= max_tokens:
                # Keep documents up to and including the one that reaches max_tokens.
                eos_pos = np.flatnonzero(tokens == eos_id)
                cut = eos_pos[np.searchsorted(eos_pos + 1, max_tokens - n_tokens)] + 1
                docs = int(np.searchsorted(eos_pos, cut - 1)) + 1
                tokens = tokens[:cut]
            out.write(tokens.tobytes())
            digest.update(tokens.tobytes())
            n_tokens += tokens.size
            n_docs += docs
            n_eos += int(np.count_nonzero(tokens == eos_id))
            if n_tokens >= next_report:
                rate = n_tokens / (time.time() - t0)
                print(f"{n_tokens / 1e6:,.0f}M tokens, {n_docs:,} docs, {rate / 1e6:.2f}M tok/s", flush=True)
                next_report += 100_000_000
            if n_tokens >= max_tokens:
                pool.terminate()
                break
    if n_tokens < max_tokens:
        print(f"WARNING: input exhausted at {n_tokens:,} tokens (< {max_tokens:,})", flush=True)
    os.replace(tmp_path, output_path)
    return {
        "path": output_path.name,
        "dtype": "uint16",
        "n_tokens": n_tokens,
        "n_docs": n_docs,
        "n_eos": n_eos,
        "n_empty_docs_skipped": n_empty,
        "max_tokens_requested": max_tokens,
        "sha256": digest.hexdigest(),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, help="manifest key and output name: <split>.bin")
    p.add_argument("--input", required=True, type=Path, help=".jsonl.zst file")
    p.add_argument("--data-root", required=True, type=Path, help="holds tokenizer/, gets <split>.bin")
    p.add_argument("--max-tokens", required=True, type=float)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--chunk-docs", type=int, default=2000)
    p.add_argument("--no-source-hash", action="store_true", help="skip sha256 of the input file")
    args = p.parse_args()

    tokenizer_dir = args.data_root / "tokenizer"
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    output_path = args.data_root / f"{args.split}.bin"
    stats = tokenize_split(
        args.input, output_path, tokenizer_path, int(args.max_tokens), args.workers, args.chunk_docs
    )
    if stats["n_eos"] != stats["n_docs"]:
        raise RuntimeError(f"EOS count {stats['n_eos']} != document count {stats['n_docs']}")
    tok = Tokenizer.from_file(str(tokenizer_path))
    # download.sh writes raw/SOURCE; the train shard sits one level deeper (raw/train/).
    source = read_source(args.input.parent / "SOURCE") or read_source(args.input.parent.parent / "SOURCE")
    stats["source"] = {
        "file": str(args.input),
        "sha256": None if args.no_source_hash else sha256_file(args.input),
        **source,
    }
    stats["tokenizer"] = {
        **read_source(tokenizer_dir / "SOURCE"),
        "eos_id": tok.token_to_id(EOS_TOKEN),
        "vocab_size": tok.get_vocab_size(),
        "bos": None,
    }
    stats["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    manifest_path = args.data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[args.split] = stats
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
