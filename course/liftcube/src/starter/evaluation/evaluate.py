"""Unified evaluation: python -m starter.evaluation.evaluate --checkpoint PATH"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import numpy as np

from starter.config import PROJECT_ROOT, add_common_args, apply_common_overrides, get_device, resolve_config
from starter.envs.container import ensure_in_leisaac_container
from starter.envs.factory import make_env
from starter.envs.timing import policy_action_hold_steps
from starter.envs.lift_cube import shutdown_sim_app
from starter.interfaces.model import SmallCNNPolicy
from starter.interfaces.observation import Observation
from starter.interfaces.policy import PolicyModel
from starter.logging.metrics import EpisodeRecord, RunManifest, try_git_hash, write_eval_outputs
from starter.utils import seed_all


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved LiftCube policy and write JSON/CSV metrics.")
    add_common_args(parser, default_config=str(PROJECT_ROOT / "configs" / "eval_lift_cube.yaml"))
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true", default=None)
    parser.add_argument("--policy-class", type=str, default=None,
                        help="Custom policy loader as package.module:ClassName; defaults to SmallCNNPolicy.")
    parser.add_argument("--robot_platform", choices=("SO101",), default=None,
                        help="Select SO101 constants in the pinned OFT backend via its process CLI.")
    return parser.parse_args(argv)


def load_policy(
    path: str, device: str, policy_class: str | None = None
) -> PolicyModel:
    if not policy_class:
        return SmallCNNPolicy.load(path, device=device)
    try:
        module_name, class_name = policy_class.rsplit(":", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
    except (ValueError, ImportError, AttributeError) as exc:
        raise ValueError(
            "--policy-class must name an importable package.module:ClassName"
        ) from exc
    return cls.load(path, device=device)


def run_episode(
    env: Any,
    policy: PolicyModel,
    *,
    seed: int,
    episode_id: int,
    deterministic: bool,
    checkpoint: str | None,
    max_episode_steps: int = 640,
) -> EpisodeRecord:
    """Clear optional policy caches at both boundaries, including exceptions.

    Legacy stateless policies need no reset method. A reset must not reseed RNG.
    """
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    reset = getattr(policy, 'reset', None)
    if reset is not None and not callable(reset):
        raise TypeError('policy.reset must be callable')
    if reset is not None:
        reset()
    try:
        return _run_episode(env, policy, seed=seed, episode_id=episode_id,
                            deterministic=deterministic, checkpoint=checkpoint,
                            max_episode_steps=max_episode_steps)
    finally:
        if reset is not None:
            reset()


def _run_episode(
    env: Any,
    policy: PolicyModel,
    *,
    seed: int,
    episode_id: int,
    deterministic: bool,
    checkpoint: str | None,
    max_episode_steps: int = 640,
) -> EpisodeRecord:
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    obs, info = env.reset(seed=seed)
    if not isinstance(obs, Observation):
        raise TypeError("env.reset must return Observation")
    total = 0.0
    length = 0
    terminated = False
    truncated = False
    reason = ""
    success = False
    failure_class = "unknown"
    # Local adaptation: preserve official environment ticks/horizon while a
    # 30 Hz policy holds each target for two ticks in the actual 60 Hz task.
    hold_steps = policy_action_hold_steps(env, policy)
    action = None
    while True:
        if length % hold_steps == 0:
            action = np.asarray(policy.act(obs, deterministic=deterministic)).copy()
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        length += 1
        success = bool(info.get("success", False))
        reason = str(info.get("termination_reason") or "")
        failure_class = str(info.get("failure_class") or reason or "unknown")
        if length >= max_episode_steps and not (terminated or truncated):
            truncated = True
            reason = "timeout"
            failure_class = "timeout"
        if terminated or truncated:
            break
    if success:
        reason = "success"
        failure_class = "success"
    elif truncated and not reason:
        reason = "timeout"
        failure_class = "timeout"
    elif terminated and not reason:
        reason = "failed"
    return EpisodeRecord(
        episode_id=episode_id,
        seed=seed,
        success=success,
        length=length,
        return_sum=float(total),
        termination_reason=reason,
        failure_class=failure_class,
        env_name=getattr(env, "name", "unknown"),
        checkpoint=checkpoint,
        truncated=bool(truncated),
        terminated=bool(terminated),
    )


def evaluate(cfg: dict[str, Any]) -> dict[str, Any]:
    device = get_device(cfg.get("device"))
    ckpt = cfg.get("checkpoint", {}).get("path")
    if not ckpt:
        raise ValueError("--checkpoint / checkpoint.path is required")
    policy = load_policy(str(ckpt), device, cfg.get("eval", {}).get("policy_class"))
    return evaluate_policy(policy, cfg, checkpoint=str(ckpt))


def evaluate_policy(
    policy: PolicyModel, cfg: dict[str, Any], *, checkpoint: str | None = None,
    shutdown: bool = True,
) -> dict[str, Any]:
    """Evaluate an already constructed policy with the shared LiftCube protocol.

    The checkpoint loader is deliberately separate: student policy classes can
    use their own serialization, while this function keeps episode seeds,
    success criteria and JSON/CSV output identical across policies. Set
    shutdown=False when consuming the returned summary before closing Isaac.
    """
    seed = int(cfg.get("seed", 0))
    seed_all(seed)
    env = make_env(cfg)
    eval_cfg = cfg.get("eval", {})
    n_ep = int(eval_cfg.get("num_episodes", 4))
    deterministic = bool(eval_cfg.get("deterministic", True))
    records: list[EpisodeRecord] = []
    for i in range(n_ep):
        rec = run_episode(
            env,
            policy,
            seed=seed + i,
            episode_id=i,
            deterministic=deterministic,
            checkpoint=checkpoint,
            max_episode_steps=int(cfg.get("env", {}).get("max_episode_steps", 640)),
        )
        records.append(rec)
    env.close()
    log_dir = Path(cfg.get("logging", {}).get("dir", "outputs/eval"))
    manifest = RunManifest.build(
        seed=seed,
        env_name=getattr(env, "name", cfg.get("env", {}).get("id", "unknown")),
        env_backend="lift_cube",
        config_path=cfg.get("_config_path"),
        checkpoint=checkpoint,
        extra={"git": try_git_hash(), "deterministic": deterministic,
               "environment_hz": env.action_spec.frequency_hz,
               "policy_action_hold_steps": policy_action_hold_steps(env, policy)},
    )
    paths = write_eval_outputs(
        log_dir,
        manifest=manifest,
        records=records,
        config_copy=cfg if cfg.get("logging", {}).get("save_config_copy", True) else None,
        write_json=bool(cfg.get("logging", {}).get("write_json", True)),
        write_csv=bool(cfg.get("logging", {}).get("write_csv", True)),
    )
    summary = None
    json_path = paths.get("json")
    if json_path:
        import json

        summary = json.loads(json_path.read_text(encoding="utf-8"))["summary"]
        print(json.dumps({"summary": summary, "outputs": {k: str(v) for k, v in paths.items()}}, indent=2), flush=True)
    if shutdown:
        shutdown_sim_app()
    return {"paths": {k: str(v) for k, v in paths.items()}, "records": records, "summary": summary}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    if argv is None:
        ensure_in_leisaac_container()
    args = parse_args(argv)
    cfg = apply_common_overrides(resolve_config(args.config), args)
    if args.num_episodes is not None:
        cfg.setdefault("eval", {})["num_episodes"] = args.num_episodes
    if args.deterministic:
        cfg.setdefault("eval", {})["deterministic"] = True
    if args.policy_class:
        cfg.setdefault("eval", {})["policy_class"] = args.policy_class
    return evaluate(cfg)


if __name__ == "__main__":
    main()
