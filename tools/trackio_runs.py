"""Trackio dashboard for the training runs, built from their train_log.csv files.

Training itself never touches trackio. This script, run on a LOGIN node, copies every run's
train_log.csv (and the attention type / size / seed / recipe from run_config.json) into one
local trackio project, and serves the dashboard. Use it through tools/trackio.sh, which pins
trackio and sets the storage location:

    tools/trackio.sh import                    # (re)build the project from all runs
    tools/trackio.sh import --watch 600        # ... and refresh every 10 minutes
    tools/trackio.sh show                      # dashboard on 127.0.0.1:7860 of this login node

Each import deletes and re-logs every run it finds, so it is safe to repeat and follows
resumed runs (a resume truncates train_log.csv back to the resume point). Runs whose log
has no rows yet are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

METRICS = ("tokens", "train_loss", "val_loss", "lr", "grad_norm", "tokens_per_s", "peak_mem_gb")
CONFIG_KEYS = ("attn", "attn_type", "window_size", "size", "seed", "recipe", "seq_len", "batch_seqs",
               "max_steps", "lr", "min_lr", "schedule", "warmup_steps", "micro_batch", "compile")


def default_runs_root() -> Path:
    scratch = os.environ.get("SCRATCH")
    return Path(os.environ.get("RUNS_ROOT") or f"{scratch}/attn_constraints/runs")


def read_run(run_dir: Path) -> tuple[list[dict], list[int], dict] | None:
    log = run_dir / "train_log.csv"
    if not log.exists():
        return None
    metrics_list, steps = [], []
    with open(log, newline="") as fh:
        for row in csv.DictReader(fh):
            m = {k: float(row[k]) for k in METRICS if row.get(k) not in (None, "")}
            if m:
                metrics_list.append(m)
                steps.append(int(row["step"]))
    if not metrics_list:
        return None
    config: dict = {"run_name": run_dir.name}
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        args = cfg.get("args", {})
        config.update({k: args[k] for k in CONFIG_KEYS if k in args})
        config.update(gpu=cfg.get("gpu"), git_commit=(cfg.get("git") or {}).get("commit"),
                      n_params=cfg.get("n_params"))
    config["done"] = (run_dir / "DONE").exists()
    return metrics_list, steps, config


def import_runs(runs_root: Path, project: str) -> None:
    from trackio.sqlite_storage import SQLiteStorage

    existing = set(SQLiteStorage.get_runs(project)) if SQLiteStorage.get_project_db_path(project).exists() else set()
    n = 0
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        data = read_run(run_dir)
        if data is None:
            continue
        metrics_list, steps, config = data
        if run_dir.name in existing:
            SQLiteStorage.delete_run(project, run_dir.name)
        SQLiteStorage.bulk_log(project=project, run=run_dir.name, metrics_list=metrics_list, steps=steps,
                               timestamps=[""] * len(steps), config=config)
        n += 1
        state = "done" if config["done"] else f"step {steps[-1]}"
        print(f"  {run_dir.name:28s} {len(steps):5d} rows  {state}", flush=True)
    print(f"{time.strftime('%H:%M:%S')} imported {n} runs from {runs_root} into project '{project}'", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("--runs-root", type=Path, default=default_runs_root())
    imp.add_argument("--project", default="attn-constraints-125m")
    imp.add_argument("--watch", type=int, default=0, help="repeat every N seconds (0: once)")
    show = sub.add_parser("show")
    show.add_argument("--project", default="attn-constraints-125m")
    show.add_argument("--port", type=int, default=7860)
    a = p.parse_args()

    if a.cmd == "import":
        while True:
            import_runs(a.runs_root, a.project)
            if not a.watch:
                break
            time.sleep(a.watch)
    else:
        import socket

        import trackio

        host = socket.gethostname()
        print(f"Dashboard on login node {host}, port {a.port} (localhost only). From your machine, tunnel\n"
              f"to this same login node, e.g.  ssh -N -L {a.port}:localhost:{a.port} <you>@{host}.idris.fr\n"
              f"(or through your usual Jean Zay ssh alias, if it reaches {host}), then open http://localhost:{a.port}",
              flush=True)
        trackio.show(project=a.project, open_browser=False, server_port=a.port, block_thread=True)


if __name__ == "__main__":
    main()
