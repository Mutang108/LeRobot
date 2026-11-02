"""Calibrated visual rule baseline. --help/--check never create an environment."""
import argparse
import hashlib
import inspect
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from starter.scripted_liftcube import ScriptedLiftCube, SteadyBiasScriptedLiftCube, load_calibration, _rotation
from starter.interfaces.action import validate_action


def check_kinematics(policy):
    """Compare FK against independent calibration poses, then test IK round trips."""
    checks = policy.c.get("fk_checks", [])
    if len(checks) < 3:
        raise ValueError("At least three independent FK calibration checks are required")
    seen = set()
    for check in checks:
        q = np.asarray(check["motor"], dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Invalid calibration motor pose")
        validate_action(q, policy.action_spec)
        seen.add(tuple(q[:5]))
        expected = np.asarray(check["tcp_xyz"], dtype=float)
        if expected.shape != (3,) or not np.isfinite(expected).all():
            raise ValueError("Invalid independent TCP position")
        actual = policy._pose(q)
        if np.linalg.norm(actual[:3, 3] - expected) > .003:
            raise ValueError("Independent FK check exceeds 3mm")
        rotation = _rotation(check["tcp_rotation"], "independent TCP rotation")
        angle = np.arccos(np.clip((np.trace(actual[:3, :3].T @ rotation)-1)/2, -1, 1))
        if angle > .15:
            raise ValueError("Independent FK orientation exceeds 0.15rad")
        policy._solve(expected, q, q[5], rotation=rotation)
    if len(seen) < 3:
        raise ValueError("FK checks must cover three distinct arm poses")


class RecordedEnv:
    """Record public observations and execution diagnostics, never feed info to policy."""
    def __init__(self, env, stream, runtime_camera_calibration=False):
        self.env, self.stream = env, stream
        self.name, self.action_spec = env.name, env.action_spec
        self.frames = []
        self.policy = None
        self.runtime_camera_calibration = runtime_camera_calibration

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs.image.copy()]
        row = {"reset": kwargs, "q": obs.proprio[:6].tolist()}
        if self.runtime_camera_calibration:
            from starter.camera_calibration import read_camera_to_base
            transform = read_camera_to_base(self.env)
            self.policy.c['camera_to_base'] = transform
            row['calibration'] = dict(source='runtime-camera-base-api', camera_to_base=transform.tolist())
        self.stream.write(json.dumps(row) + "\n")
        self.stream.flush()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs.image.copy())
        row = {"tick": len(self.frames)-1, "policy": self.policy.telemetry,
               "sent_motor": np.asarray(action).tolist(), "measured_after": obs.proprio[:6].tolist(),
               "command": info.get("command"), "reward": float(reward),
               "success": bool(info.get("success", False)),
               "terminated": bool(terminated), "truncated": bool(truncated)}
        self.stream.write(json.dumps(row) + "\n")
        self.stream.flush()
        return obs, reward, terminated, truncated, info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Existing eval_lift_cube.yaml (required for simulation)")
    parser.add_argument("--output", type=Path, help="New output directory (never overwritten)")
    parser.add_argument("--runtime-camera-calibration", action="store_true",
        help="Opt-in matching simulator camera/base API, not a hardware capability claim")
    parser.add_argument("--privileged-teacher", action="store_true",
        help="Use explicit cube truth instead of vision; development data candidate, not visual baseline")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="CPU calibration and IK checks only")
    mode.add_argument("--run-simulation", action="store_true", help="Requires separate runtime authorization")
    args = parser.parse_args(argv)
    c = load_calibration(args.calibration)
    if c.get("kinematics_backend", "urdf") == "joint_chain":
        from starter.joint_chain import JointChainKinematics
        kin = JointChainKinematics(c["chain_path"], c["tcp_offset_m"], c["joint_names"])
    else:
        # Optional upstream dependency, not a reason to change the course runtime.
        from lerobot.model.kinematics import RobotKinematics
        kin = RobotKinematics(c["urdf_path"], target_frame_name=c["tcp_frame"], joint_names=c["joint_names"])
    policy_class = SteadyBiasScriptedLiftCube if c.get("gravity_compensation", False) else ScriptedLiftCube
    if c.get('grasp_presets'):
        from starter.scripted_geometry import GeometricLiftCube
        policy_class = GeometricLiftCube
    if args.privileged_teacher:
        from starter.privileged_liftcube import PrivilegedLiftCube
        policy_class = PrivilegedLiftCube
        if args.runtime_camera_calibration:
            parser.error('Privileged localization does not use runtime camera calibration')
    check_kinematics(policy_class(c, kin))
    if args.check:
        print(json.dumps({"status": "calibration_checks_passed", "simulation_run": False}))
        return
    if args.config is None or args.output is None:
        parser.error("--run-simulation requires --config and --output")
    from starter.config import resolve_config
    from starter.envs.factory import make_env
    from starter.envs.lift_cube import shutdown_sim_app
    from starter.evaluation.evaluate import run_episode
    from starter.logging.metrics import RunManifest, try_git_hash, write_eval_outputs
    from starter.logging.swanlab import ScalarWriter, log_episode

    cfg = resolve_config(args.config)
    e = cfg.get("env", {})
    if e.get("control_mode", "joint") != "joint":
        raise ValueError("The LiftCube grasp script emits joints, not planar EE actions; do not relabel its outputs")
    if (e.get("id") != "LeIsaac-SO101-LiftCube-v0" or e.get("camera") != "front"
            or [e.get("image_height"),e.get("image_width"),3] != c["image_shape"]
            or e.get("n_envs") != 1 or e.get("control_hz") != 30):
        raise ValueError("Rule baseline requires matching calibrated front RGB / scalar 30Hz LiftCube")
    count = int(cfg.get("eval", {}).get("num_episodes", 4))
    horizon = int(e.get("max_episode_steps", 640))
    if count < 1 or horizon < 1:
        raise ValueError("Positive episode/tick budgets are required")
    args.output.mkdir(parents=True, exist_ok=False)
    calibration_bytes = args.calibration.read_bytes()
    (args.output / "calibration.json").write_bytes(calibration_bytes)
    manifest = RunManifest.build(seed=int(cfg.get("seed", 0)), env_name=e["id"], env_backend="lift_cube",
        config_path=str(args.config), extra={"git": try_git_hash(),
        "policy": "privileged-liftcube-teacher-v1" if args.privileged_teacher else "front-rgb-rule-v1",
        "privileged_teacher": args.privileged_teacher, "automatic_bc_admission": False,
        "calibration_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
        "kinematics_backend": c.get("kinematics_backend", "urdf"),
        "kinematics_sha256": c.get("chain_sha256", c.get("urdf_sha256")),
        "kinematics_code_sha256": hashlib.sha256(Path(inspect.getfile(type(kin))).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256(Path(__file__).parents[1].joinpath("scripted_liftcube.py").read_bytes()).hexdigest(),
        "policy_class": f"{policy_class.__module__}:{policy_class.__name__}",
        "policy_class_sha256": hashlib.sha256(Path(inspect.getfile(policy_class)).read_bytes()).hexdigest(),
        "runtime_camera_calibration": args.runtime_camera_calibration,
        "plan_while_opening": c.get("plan_while_opening", False),
        "camera_calibration_code_sha256": (hashlib.sha256(Path(__file__).parents[1].joinpath("camera_calibration.py").read_bytes()).hexdigest()
                                           if args.runtime_camera_calibration else None),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (args.output / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    records, env = [], None
    cfg.setdefault("eval", {})["privileged_teacher"] = args.privileged_teacher
    cfg.setdefault("logging", {})["dir"] = str(args.output)
    writer = ScalarWriter(cfg, "teacher-eval" if args.privileged_teacher else "script-eval")
    status = {"status": "running", "complete_episodes": 0}
    (args.output / "run-status.json").write_text(json.dumps(status), encoding="utf-8")
    try:
        env = make_env(cfg)
        for episode in range(count):
            # A fresh state machine cannot leak target/stage across resets.
            policy = policy_class(c, kin)
            if args.privileged_teacher:
                from starter.privileged_liftcube import read_cube_in_base
                policy.read_cube = lambda: read_cube_in_base(env)
            with (args.output / f"episode-{episode:04d}.jsonl").open("x", encoding="utf-8") as stream:
                recorded = RecordedEnv(env, stream, runtime_camera_calibration=args.runtime_camera_calibration)
                recorded.policy = policy
                try:
                    record = run_episode(recorded, policy, seed=manifest.seed+episode,
                        episode_id=episode, deterministic=True, checkpoint=None, max_episode_steps=horizon)
                    records.append(record)
                    stream.write(json.dumps({"episode_result": record.to_dict(),
                                             "script_failure": policy.failure_reason}) + "\n")
                finally:
                    if recorded.frames:
                        np.savez_compressed(args.output / f"episode-{episode:04d}-rgb.npz",
                                            rgb=np.stack(recorded.frames))
                write_eval_outputs(args.output, manifest=manifest, records=records, config_copy=cfg)
                log_episode(writer, records, hz=env.action_spec.frequency_hz)
        status["status"] = "finished"
    except Exception as exc:
        status.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        status["complete_episodes"] = len(records)
        (args.output / "run-status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        if env is not None:
            env.close()
        writer.close()
        shutdown_sim_app()


if __name__ == "__main__":
    main()
