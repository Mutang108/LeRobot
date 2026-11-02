"""Unified episode logging and evaluation outputs."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from starter import __version__

TERMINATION_REASONS = (
    "success",
    "timeout",
    "dropped",
    "missed_grasp",
    "out_of_bounds",
    "failed",
)


@dataclass
class RunManifest:
    seed: int
    env_name: str
    env_backend: str
    code_version: str
    python_version: str
    config_path: str | None
    checkpoint: str | None
    created_at: str
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        seed: int,
        env_name: str,
        env_backend: str,
        config_path: str | None = None,
        checkpoint: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> "RunManifest":
        return cls(
            seed=int(seed),
            env_name=env_name,
            env_backend=env_backend,
            code_version=__version__,
            python_version=platform.python_version(),
            config_path=config_path,
            checkpoint=checkpoint,
            created_at=datetime.now(timezone.utc).isoformat(),
            extra=extra or {},
        )


@dataclass
class EpisodeRecord:
    episode_id: int
    seed: int
    success: bool
    length: int
    return_sum: float
    termination_reason: str
    failure_class: str
    env_name: str
    checkpoint: str | None = None
    trajectory_path: str | None = None
    truncated: bool = False
    terminated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_episodes(records: list[EpisodeRecord]) -> dict[str, Any]:
    if not records:
        return {
            "num_episodes": 0,
            "success_rate": 0.0,
            "mean_length": 0.0,
            "median_length": 0.0,
            "mean_return": 0.0,
            "timeout_rate": 0.0,
            "truncated_rate": 0.0,
            "failure_counts": {},
        }
    # Importing scalar tracking must not require the evaluation dependencies.
    import numpy as np

    lengths = np.asarray([r.length for r in records], dtype=np.float64)
    returns = np.asarray([r.return_sum for r in records], dtype=np.float64)
    successes = np.asarray([r.success for r in records], dtype=np.float64)
    timeouts = np.asarray([r.termination_reason == "timeout" or r.truncated for r in records], dtype=np.float64)
    failure_counts: dict[str, int] = {}
    for rec in records:
        key = rec.failure_class or rec.termination_reason or "unknown"
        if rec.success:
            key = "success"
        failure_counts[key] = failure_counts.get(key, 0) + 1
    success_lengths = [r.length for r in records if r.success]
    return {
        "num_episodes": len(records),
        "success_rate": float(successes.mean()),
        "mean_length": float(lengths.mean()),
        "median_length": float(np.median(lengths)),
        "mean_success_length": float(np.mean(success_lengths)) if success_lengths else None,
        "median_success_length": float(np.median(success_lengths)) if success_lengths else None,
        "mean_return": float(returns.mean()),
        "timeout_rate": float(timeouts.mean()),
        "truncated_rate": float(np.mean([r.truncated for r in records])),
        "failure_counts": failure_counts,
    }


def write_eval_outputs(
    out_dir: str | Path,
    *,
    manifest: RunManifest,
    records: list[EpisodeRecord],
    config_copy: dict[str, Any] | None = None,
    write_json: bool = True,
    write_csv: bool = True,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize_episodes(records)
    payload = {
        "manifest": asdict(manifest),
        "summary": summary,
        "episodes": [r.to_dict() for r in records],
        "config": config_copy or {},
    }
    paths: dict[str, Path] = {}
    if write_json:
        json_path = out / "metrics.json"
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        paths["json"] = json_path
    if write_csv:
        csv_path = out / "episodes.csv"
        fieldnames = list(asdict(records[0]).keys()) if records else [
            "episode_id",
            "seed",
            "success",
            "length",
            "return_sum",
            "termination_reason",
            "failure_class",
            "env_name",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec.to_dict())
        paths["csv"] = csv_path
    if config_copy is not None:
        cfg_path = out / "config_copy.json"
        cfg_path.write_text(json.dumps(config_copy, indent=2, default=str), encoding="utf-8")
        paths["config"] = cfg_path
    return paths


def try_git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return None
    return None
