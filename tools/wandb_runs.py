"""Weights & Biases dashboard for the training runs, built from their train_log.csv files.

Training itself never touches W&B (compute nodes have no internet). This script, run on a
LOGIN node, pushes every run's train_log.csv to one W&B project, with the attention type,
size, seed and recipe from run_config.json as the run config. Use it through tools/wandb.sh,
which pins wandb; the API key comes from ~/.netrc (machine api.wandb.ai), never from the repo.

    tools/wandb.sh                         # sync all runs once
    tools/wandb.sh --watch 600             # ... and every 10 minutes

W&B run id = run name (e.g. window4_125m_s2), and <run_dir>/.wandb_sync.json records the last
step pushed, so each sync only appends new rows. If a run's train_log.csv ends before that step
(a crash made the run redo steps after its last resume point), the W&B run is replaced by a new
one (id {run}-v2, same display name) holding the whole CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

METRICS = ("tokens", "train_loss", "val_loss", "lr", "grad_norm", "tokens_per_s", "peak_mem_gb")
CONFIG_KEYS = ("attn", "attn_type", "window_size", "size", "seed", "recipe", "seq_len", "batch_seqs",
               "max_steps", "lr", "min_lr", "schedule", "warmup_steps", "micro_batch", "compile")


def default_runs_root() -> Path:
    return Path(os.environ.get("RUNS_ROOT") or f"{os.environ['SCRATCH']}/attn_constraints/runs")


def read_run(run_dir: Path):
    log = run_dir / "train_log.csv"
    if not log.exists():
        return None
    rows = []
    with open(log, newline="") as fh:
        for row in csv.DictReader(fh):
            m = {k: float(row[k]) for k in METRICS if row.get(k) not in (None, "")}
            if m:
                rows.append((int(row["step"]), m))
    if not rows:
        return None
    config: dict = {}
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        args = cfg.get("args", {})
        config.update({k: args[k] for k in CONFIG_KEYS if k in args})
        config.update(gpu=cfg.get("gpu"), git_commit=(cfg.get("git") or {}).get("commit"),
                      n_params=cfg.get("n_params"), slurm_job_id=cfg.get("slurm_job_id"))
    return rows, config, (run_dir / "DONE").exists()


STATE_FILE = ".wandb_sync.json"


def sync_runs(runs_root: Path, entity: str, project: str) -> None:
    """Append each run's new CSV rows to its W&B run.

    What was already pushed is tracked in <run_dir>/.wandb_sync.json ({"id", "step", "done"}),
    not read back from W&B: the W&B API lags right after a run is written. If train_log.csv now
    ends before the pushed step (a crash made the run redo steps after its last resume point),
    the old W&B run is deleted and the whole CSV goes to a new run id ({run}-v2, -v3, ...; W&B
    does not reuse deleted ids) with the same display name.
    """
    import wandb

    api = wandb.Api(timeout=60)
    wandb_dir = tempfile.mkdtemp(prefix="wandb_", dir=os.environ.get("TMPDIR"))
    n_new = 0
    try:
        for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
            data = read_run(run_dir)
            if data is None:
                continue
            rows, config, done = data
            name = run_dir.name
            state_path = run_dir / STATE_FILE
            state = json.loads(state_path.read_text()) if state_path.exists() else {"id": name, "step": -1, "done": False, "version": 1}
            if rows[-1][0] < state["step"]:
                print(f"  {name}: CSV ends at step {rows[-1][0]} < pushed step {state['step']} -> new W&B run", flush=True)
                try:
                    api.run(f"{entity}/{project}/{state['id']}").delete()
                except Exception as exc:
                    print(f"    could not delete {state['id']} ({type(exc).__name__}); delete it by hand", flush=True)
                version = state.get("version", 1) + 1
                state = {"id": f"{name}-v{version}", "step": -1, "done": False, "version": version}
            new = [(s, m) for s, m in rows if s > state["step"]]
            if not new and (state["done"] or not done):
                continue
            run = wandb.init(
                entity=entity, project=project, id=state["id"], name=name, resume="allow", dir=wandb_dir,
                group=str(config.get("attn", name.split("_")[0])),
                tags=[str(config.get("attn", "")), f"seed{config.get('seed', '')}", str(config.get("size", ""))],
                config=config,
                settings=wandb.Settings(silent=True, console="off", save_code=False, disable_git=True,
                                        x_disable_stats=True, x_disable_meta=True),
            )
            for step, metrics in new:
                run.log(metrics, step=step)
            run.summary["done"] = done
            run.finish()
            state.update(step=rows[-1][0], done=done)
            tmp = state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state) + "\n")
            os.replace(tmp, state_path)
            n_new += len(new)
            print(f"  {name:28s} +{len(new):5d} rows (to step {rows[-1][0]}){'  done' if done else ''}", flush=True)
    finally:
        shutil.rmtree(wandb_dir, ignore_errors=True)
    print(f"{time.strftime('%H:%M:%S')} synced {runs_root} -> {entity}/{project} ({n_new} new rows)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=Path, default=default_runs_root())
    p.add_argument("--entity", default=None, help="default: the logged-in user's default entity")
    p.add_argument("--project", default="attn-constraints-125m")
    p.add_argument("--watch", type=int, default=0, help="repeat every N seconds (0: once)")
    a = p.parse_args()
    import wandb

    entity = a.entity or wandb.Api().viewer.entity
    while True:
        try:
            sync_runs(a.runs_root, entity, a.project)
        except Exception as exc:  # network hiccup on the login node: keep watching
            if not a.watch:
                raise
            print(f"sync failed ({type(exc).__name__}: {exc}); retrying in {a.watch}s", flush=True)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
