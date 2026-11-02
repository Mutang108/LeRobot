"""BC training entry: python -m starter.training.train_bc --config configs/bc_lift_cube.yaml"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starter.algorithms.bc import BCTrainer
from starter.config import PROJECT_ROOT, add_common_args, apply_common_overrides, get_device, resolve_config
from starter.data.hf_adapter import load_lerobot_dataset
from starter.data.loader import CanonicalDataset, episode_aware_split, make_dataloader
from starter.utils import build_policy, seed_all
from starter.training.checkpoint import load_training, save_training, training_identity, training_path
from starter.training.events import open_writer, log_row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a behavior-cloning policy on canonical LiftCube data.")
    add_common_args(parser, default_config=str(PROJECT_ROOT / "configs" / "bc_lift_cube.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, dest="learning_rate")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Dataset directory. Overrides data.path. Use any path visible to this process.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Trusted full training checkpoint, not an inference export.")
    return parser.parse_args(argv)


def load_training_data(cfg: dict) -> tuple:
    data_cfg = cfg.get("data", {})
    if data_cfg.get("allow_action_clipping", False):
        raise ValueError("Action clipping is diagnostic-only; BC requires verified unmodified labels")
    path = data_cfg.get("path")
    env_cfg = cfg.get("env", {})
    if not path:
        raise FileNotFoundError(
            "data.path is required. Pass --data-path or set data.path in the config. "
            "LiftCube does not ship a demonstration set in this repo."
        )
    root = Path(path)
    if not (root / "meta" / "info.json").exists():
        raise FileNotFoundError(
            f"{root} is not a LeRobot v2.1/v3 dataset (missing meta/info.json)"
        )
    return load_lerobot_dataset(
        root,
        camera=str(env_cfg.get("camera", "front")),
        image_height=int(env_cfg.get("image_height", 64)),
        image_width=int(env_cfg.get("image_width", 64)),
        include_ee=bool(env_cfg.get("include_ee_in_proprio", True)),
        max_episodes=data_cfg.get("max_episodes"),
        max_frames=data_cfg.get("max_frames"),
        action_units=data_cfg.get("action_units"),
        allow_action_clipping=bool(data_cfg.get("allow_action_clipping", False)),
    )


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    cfg = apply_common_overrides(resolve_config(args.config), args)
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("data", {})["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cfg.setdefault("train", {})["learning_rate"] = args.learning_rate
    if args.data_path is not None:
        cfg.setdefault("data", {})["path"] = args.data_path
    if args.max_episodes is not None:
        cfg.setdefault("data", {})["max_episodes"] = args.max_episodes
    if args.max_frames is not None:
        cfg.setdefault("data", {})["max_frames"] = args.max_frames

    seed = int(cfg.get("seed", 0))
    seed_all(seed)
    device = get_device(cfg.get("device"))
    loaded = load_training_data(cfg)
    payload, manifest, obs_spec, action_spec = loaded[:4]
    train_idx, val_idx = episode_aware_split(payload, val_ratio=float(cfg.get("data", {}).get("val_ratio", 0.25)), seed=seed)
    train_ds = CanonicalDataset(payload, obs_spec, action_spec, manifest, indices=train_idx)
    val_ds = CanonicalDataset(payload, obs_spec, action_spec, manifest, indices=val_idx, validate=False) if len(val_idx) else None
    batch_size = int(cfg.get("data", {}).get("batch_size", 16))
    workers = int(cfg.get("data", {}).get("num_workers", 0))
    if workers != 0:
        raise ValueError("Resumable BC currently requires data.num_workers=0")
    train_loader = make_dataloader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = make_dataloader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0) if val_ds is not None and len(val_ds) else None

    model = build_policy(cfg, obs_spec, action_spec, device)
    trainer = BCTrainer(model, log_every=int(cfg.get("train", {}).get("log_every", 1)))
    identity = training_identity("bc", cfg, payload)
    saved_path = training_path(cfg)
    if not args.resume and Path(saved_path).exists():
        raise FileExistsError(f"Existing training checkpoint: {saved_path}; use --resume or a new output path")
    progress = load_training(args.resume, model, identity) if args.resume else {"epoch": 0, "updates": 0, "history": []}
    history = progress["history"]
    writer = open_writer(cfg, "bc", start_step=int(progress["updates"]), resumed=bool(args.resume))

    def checkpoint_epoch(epoch, row):
        history.append(row)
        save_training(saved_path, model, identity, {"epoch": epoch, "updates": row["optimizer_steps"], "history": history})
        log_row(writer, "bc", row, row["optimizer_steps"])

    try:
        trainer.fit(
            train_loader,
            epochs=int(cfg.get("train", {}).get("epochs", 3)),
            max_batches_per_epoch=cfg.get("train", {}).get("max_batches_per_epoch"),
            val_loader=val_loader,
            start_epoch=int(progress["epoch"]),
            start_step=int(progress["updates"]),
            on_epoch_end=checkpoint_epoch,
        )
    finally:
        writer.close()
    ckpt = cfg.get("checkpoint", {}).get("path", "outputs/bc/checkpoint.pt")
    Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(ckpt)
    log_dir = Path(cfg.get("logging", {}).get("dir", "outputs/bc"))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "train_log.json").write_text(json.dumps({"history": history, "checkpoint": ckpt}, indent=2), encoding="utf-8")
    if cfg.get("logging", {}).get("save_config_copy", True):
        (log_dir / "config_copy.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    print(f"[bc] saved checkpoint -> {ckpt}")
    return {"checkpoint": ckpt, "history": history}


if __name__ == "__main__":
    main()
