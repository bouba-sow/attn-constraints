"""Check train.bin / val.bin against manifest.json (sha256 and token counts) before training on a
copied dataset, e.g. the Jean Zay data rsynced to another cluster. Exits non-zero on a mismatch.

    python tools/check_data.py $DATA_ROOT
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def main(root: Path) -> int:
    manifest = json.loads((root / "manifest.json").read_text())
    ok = True
    for split in ("train", "val"):
        entry = manifest[split]
        path = root / entry["path"]
        n_tokens = path.stat().st_size // 2  # uint16
        digest = sha256(path)
        good = digest == entry["sha256"] and n_tokens == entry["n_tokens"]
        ok &= good
        print(f"{split}: {'OK' if good else 'MISMATCH'} tokens {n_tokens:,} (manifest {entry['n_tokens']:,}) "
              f"sha256 {digest[:12]} (manifest {entry['sha256'][:12]})", flush=True)
    tok = manifest["train"].get("tokenizer", {})
    print(f"tokenizer: {tok.get('repo')}@{tok.get('revision')}; files in {root / 'tokenizer'}: "
          f"{sorted(p.name for p in (root / 'tokenizer').glob('*'))}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else ".")))
