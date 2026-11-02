"""RL training entry: python -m starter.training.train_rl --config configs/rl_lift_cube.yaml"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starter.algorithms.reference.ppo import PPOAlgorithm
from starter.config import PROJECT_ROOT, add_common_args, apply_common_overrides, get_device, resolve_config
from starter.envs.container import ensure_in_leisaac_container
from starter.envs.factory import make_env
from starter.envs.lift_cube import shutdown_sim_app
from starter.interfaces.model import SmallCNNPolicy
from starter.rewards.lift_cube import shape_reward
from starter.utils import build_policy, seed_all
from starter.envs.timing import decision_specs
from starter.training.checkpoint import load_training, save_training, training_identity, training_path
from starter.training.events import open_writer, log_row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online PPO training on the container LiftCube env.")
    add_common_args(parser, default_config=str(PROJECT_ROOT / "configs" / "rl_lift_cube.yaml"))
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--num-updates", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, dest="learning_rate")
    parser.add_argument("--bc-checkpoint", type=str, default=None,
                        help="Initialize RL policy from BC weights; overrides ppo.bc_checkpoint_path.")
    parser.add_argument("--resume", type=str, default=None, help="Trusted full PPO training checkpoint.")
    return parser.parse_args(argv)


def load_bc_checkpoint(model: SmallCNNPolicy, path: str, device: str) -> None:
    """Copy BC policy weights; PPO will create a fresh optimizer."""
    checkpoint = Path(path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"BC checkpoint not found: {checkpoint}")
    loaded = SmallCNNPolicy.load(str(checkpoint), device=device)
    bc_observation = loaded.observation_spec.to_dict()
    rl_observation = model.observation_spec.to_dict()
    bc_action = loaded.action_spec.to_dict()
    rl_action = model.action_spec.to_dict()
    # Notes are descriptive and do not affect the policy contract.
    bc_action.pop("notes", None)
    rl_action.pop("notes", None)
    if bc_observation != rl_observation or bc_action != rl_action:
        raise ValueError(
            f"BC checkpoint {checkpoint} specs differ from the RL environment; "
            "match camera, image size, proprio layout, action units and control frequency."
        )
    if loaded.feature_mode != model.feature_mode:
        raise ValueError("BC feature mode differs; legacy image-only weights cannot silently become fused weights")
    try:
        model.load_state_dict(loaded.state_dict(), strict=True)
    except RuntimeError as exc:
        raise ValueError(f"BC checkpoint {checkpoint} architecture differs from the RL policy") from exc


def main(argv: list[str] | None = None) -> dict:
    if argv is None:
        ensure_in_leisaac_container()
    args = parse_args(argv)
    cfg = apply_common_overrides(resolve_config(args.config), args)
    ppo_cfg = cfg.setdefault("ppo", {})
    if args.rollout_steps is not None:
        ppo_cfg["rollout_steps"] = args.rollout_steps
    if args.num_updates is not None:
        ppo_cfg["num_updates"] = args.num_updates
    if args.learning_rate is not None:
        ppo_cfg["learning_rate"] = args.learning_rate
    if args.bc_checkpoint is not None:
        ppo_cfg["bc_checkpoint_path"] = args.bc_checkpoint

    seed = int(cfg.get("seed", 0))
    seed_all(seed)
    device = get_device(cfg.get("device"))
    env = make_env(cfg)
    obs_spec, action_spec = decision_specs(env, float(ppo_cfg.get("policy_hz", 30.0)))
    model = build_policy(cfg, obs_spec, action_spec, device)
    bc_checkpoint = ppo_cfg.get("bc_checkpoint_path")
    if bc_checkpoint and not args.resume:
        load_bc_checkpoint(model, str(bc_checkpoint), device)
        print(f"[rl] initialized policy from BC checkpoint -> {bc_checkpoint}", flush=True)
    algo = PPOAlgorithm(
        model,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        value_coef=float(ppo_cfg.get("value_coef", 0.5)),
        entropy_coef=float(ppo_cfg.get("entropy_coef", 0.01)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        ppo_epochs=int(ppo_cfg.get("ppo_epochs", 2)),
        minibatch_size=int(ppo_cfg.get("minibatch_size", 32)),
        reward_shaper=shape_reward,
        bootstrap_timeouts=bool(ppo_cfg.get("bootstrap_timeouts", True)),
    )
    identity = training_identity("ppo", cfg)
    saved_path = training_path(cfg)
    if not args.resume and Path(saved_path).exists():
        raise FileExistsError(f"Existing training checkpoint: {saved_path}; use --resume or a new output path")
    progress = load_training(args.resume, model, identity) if args.resume else {
        "update": 0, "decision_steps": 0, "physics_ticks": 0, "history": []}
    history = progress["history"]
    num_updates = int(ppo_cfg.get("num_updates", 2))
    rollout_steps = int(ppo_cfg.get("rollout_steps", 640))
    if num_updates < progress["update"]:
        raise ValueError("num_updates is a total budget, smaller than restored progress")
    writer = open_writer(cfg, "ppo", start_step=int(progress["decision_steps"]), resumed=bool(args.resume))
    try:
        for update in range(progress["update"], num_updates):
            buf, roll_metrics = algo.collect(env, rollout_steps, seed=seed + update)
            stats = algo.update(buf, last_value=roll_metrics.get("last_value", 0.0))
            row = {"update": update, **roll_metrics, **stats}
            history.append(row)
            progress.update(update=update + 1,
                            decision_steps=progress["decision_steps"] + len(buf),
                            physics_ticks=progress["physics_ticks"] + int(roll_metrics["physics_ticks"]))
            row.update(decision_steps=progress["decision_steps"], total_physics_ticks=progress["physics_ticks"])
            save_training(saved_path, model, identity, progress)
            log_row(writer, "ppo", row, progress["decision_steps"])
            print(
                f"[rl] update={update} loss={stats.get('loss', float('nan')):.4f} "
                f"return={roll_metrics.get('mean_return', 0):.3f} steps={len(buf)}"
            )
    finally:
        env.close()
        writer.close()
    ckpt = cfg.get("checkpoint", {}).get("path", "outputs/rl/checkpoint.pt")
    Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
    algo.save(ckpt)
    log_dir = Path(cfg.get("logging", {}).get("dir", "outputs/rl"))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "train_log.json").write_text(
        json.dumps({"history": history, "checkpoint": ckpt, "bc_checkpoint_path": bc_checkpoint}, indent=2),
        encoding="utf-8",
    )
    if cfg.get("logging", {}).get("save_config_copy", True):
        (log_dir / "config_copy.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    print(f"[rl] saved checkpoint -> {ckpt}", flush=True)
    shutdown_sim_app()
    return {"checkpoint": ckpt, "history": history}


if __name__ == "__main__":
    main()
