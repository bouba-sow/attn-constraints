"""Save a model as a self-contained HF checkpoint (weights + config + code + tokenizer)."""

from __future__ import annotations

import shutil
from pathlib import Path

from .modeling import ConstrainedOPTForCausalLM

MODEL_DIR = Path(__file__).resolve().parent
CODE_FILES = ("configuration.py", "modeling.py")
AUTO_MAP = {
    "AutoConfig": "configuration.ConstrainedOPTConfig",
    "AutoModelForCausalLM": "modeling.ConstrainedOPTForCausalLM",
}


def save_hf(model: ConstrainedOPTForCausalLM, out_dir: str | Path, tokenizer_dir: str | Path | None) -> Path:
    """Write ``out_dir`` so it loads with
    ``AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True)`` and
    ``AutoTokenizer.from_pretrained(out_dir)``. Written to a temporary sibling first and
    renamed, so a killed job never leaves a half-written checkpoint under the final name.
    """
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.config.auto_map = dict(AUTO_MAP)
    model.save_pretrained(tmp, safe_serialization=True)
    for name in CODE_FILES:
        shutil.copy2(MODEL_DIR / name, tmp / name)
    if tokenizer_dir is not None:
        for f in Path(tokenizer_dir).iterdir():
            if f.is_file() and (f.name.startswith("tokenizer") or f.name == "special_tokens_map.json"):
                shutil.copy2(f, tmp / f.name)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return out_dir
