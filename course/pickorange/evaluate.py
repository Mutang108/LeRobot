"""PickOrange GR00T native-assisted development evaluation, separate from LiftCube.

This is a portable source port of a prior protocol, not a new measured result.
It preserves native success, physical stops and gravity, but explicitly uses
the upstream object-truth gripper-effort helper. No automatic BC admission.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import numpy as np

from common import (DEFAULT_PROTOCOL, load_protocol, publish, rng_state,
                    save_arrays, sha256, wait_file)


def check_imported_sources():
    reference = Path(__file__).resolve().parents[2] / 'third_party/leisaac/source/leisaac'
    files = ['leisaac.tasks.pick_orange.pick_orange_env_cfg',
             'leisaac.tasks.pick_orange.mdp.terminations',
             'leisaac.tasks.template.single_arm_env_cfg', 'leisaac.devices.action_process',
             'leisaac.utils.robot_utils', 'leisaac.utils.env_utils']
    result = {}
    for name in files:
        module = importlib.import_module(name)
        expected = reference / (name.replace('.', '/') + '.py')
        digest = sha256(module.__file__)
        if digest != sha256(expected):
            raise ValueError(f'Imported task source differs from pinned repository: {name}')
        result[name] = digest
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--episodes', type=int, help='Explicit prefix for smoke; recorded as a different budget')
    p.add_argument('--wall-seconds', type=float, default=3600)
    p.add_argument('--min-free-gib', type=float, default=20)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--run-simulation', action='store_true')
    # Parse --check before importing AppLauncher: no implicit GPU/application.
    preliminary, _ = p.parse_known_args(argv)
    cfg = load_protocol(preliminary.protocol)
    if preliminary.check:
        p.parse_args(argv)
        print(json.dumps(dict(status='protocol_checked', protocol=cfg, simulation_run=False)))
        return
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args(argv)
    if a.run_dir is None or a.wall_seconds <= 0 or a.min_free_gib < 0 or not np.isfinite([a.wall_seconds, a.min_free_gib]).all():
        p.error('A shared --run-dir and finite positive wall/nonnegative storage budgets are required')
    seeds = cfg['scene_seeds']
    if a.episodes is not None:
        if not 1 <= a.episodes <= len(seeds):
            p.error('--episodes must select a nonempty protocol prefix')
        seeds = seeds[:a.episodes]
    out = a.run_dir.resolve()
    wait_file(out / 'service-ready.json', min(a.wall_seconds, 900), out / 'service-error.json', out / 'service-summary.json')
    ready = json.loads((out / 'service-ready.json').read_text(encoding='utf-8'))
    if ready['protocol'] != cfg:
        raise ValueError('Evaluator/service protocol mismatch')
    # Exclusive role claim. A stale run must use a new directory, not overwrite.
    (out / 'evaluator.claim').open('x').close()
    app = env = None
    started = time.monotonic()
    results = []
    query = 0
    try:
        def budget():
            if time.monotonic()-started >= a.wall_seconds:
                raise TimeoutError('Wall budget reached; preserve the partial and completed prefix')
            if (out / 'STOP').exists() or (out / 'service-error.json').exists():
                raise RuntimeError('Run stopped or model service failed')
            if shutil.disk_usage(out).free < a.min_free_gib * 1024**3:
                raise RuntimeError('Output storage reserve reached')

        budget()
        app = AppLauncher(a).app
        import gymnasium as gym
        import h5py
        import torch
        import leisaac.tasks  # noqa: F401
        from isaaclab.managers import RecorderTermCfg, DatasetExportMode
        from isaaclab.managers.recorder_manager import RecorderTerm, RecorderManagerBaseCfg
        from isaaclab.utils import configclass
        from isaaclab_tasks.utils import parse_env_cfg
        from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
        from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot, convert_lerobot_action_to_leisaac
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from joint_contract import JOINTS, USD_LIMITS_DEG, rad_to_motor, motor_to_rad

        def snapshot():
            return dict(joint_pos=robot.data.joint_pos[0, ids].detach().cpu().numpy().copy(),
                objects=np.stack([env.scene[k].data.root_pos_w[0].detach().cpu().numpy().copy()
                    for k in ('Orange001', 'Orange002', 'Orange003', 'Plate')]),
                success=bool(env.cfg.terminations.success.func(env, **env.cfg.terminations.success.params)[0]))

        class FinalCapture(RecorderTerm):
            def record_pre_reset(self, env_ids):
                if hasattr(self._env, 'reset_buf') and bool(self._env.reset_buf.any()):
                    self._env.public_final = snapshot()
                    observations = self._env.observation_manager.compute(update_history=False)['policy']
                    self._env.public_final['rgb'] = {cam: observations[cam][0].detach().cpu().numpy().copy()
                                                    for cam in ('front', 'wrist')}
                return None, None

        @configclass
        class CaptureCfg(RecorderManagerBaseCfg):
            dataset_export_mode = DatasetExportMode.EXPORT_NONE
            export_in_record_pre_reset = False
            final_frame = RecorderTermCfg(class_type=FinalCapture)

        source_identity = check_imported_sources()
        env_cfg = parse_env_cfg(cfg['task_id'], device=a.device, num_envs=1)
        env_cfg.use_teleop_device('so101leader')
        env_cfg.seed = seeds[0]
        env_cfg.episode_length_s = cfg['episode_seconds']
        env_cfg.scene.robot.spawn.rigid_props.disable_gravity = False
        env_cfg.recorders = CaptureCfg()
        if env_cfg.dynamic_reset_gripper_effort_limit is not True:
            raise ValueError('Native-assisted protocol requires the explicit dynamic helper')
        env = gym.make(cfg['task_id'], cfg=env_cfg).unwrapped
        robot = env.scene['robot']
        ids, names = robot.find_joints(JOINTS, preserve_order=True)
        if names != JOINTS or env.action_manager.total_action_dim != 6 or abs(env.step_dt-1/60) > 1e-7 or env.max_episode_length != 3600:
            raise ValueError('Joint/action/timing contract differs from this protocol')
        terms = [env.action_manager.get_term(k) for k in env.action_manager.active_terms]
        if sum([list(t._joint_names) for t in terms], []) != JOINTS:
            raise ValueError('Action-manager joint order mismatch')
        offsets = torch.cat([torch.as_tensor(t._offset, device=env.device).expand(1, t.action_dim) for t in terms], dim=1)
        scales = torch.cat([torch.as_tensor(t._scale, device=env.device).expand(1, t.action_dim) for t in terms], dim=1)
        if not bool(torch.isfinite(offsets).all() & torch.isfinite(scales).all() & (scales != 0).all()):
            raise ValueError('Invalid action offsets/scales')
        publish(out / 'manifest.json', dict(protocol=cfg, scene_seeds=seeds,
            imported_source_sha256=source_identity, evaluator_sha256=sha256(__file__),
            action_offsets_rad=offsets[0].tolist(), action_scales=scales[0].tolist(),
            joint_names=JOINTS, object_truth_sent_to_model=False, oracle_actuator_assisted=True,
            formal_course_evaluation=False, automatic_bc_admission=False,
            terminal_rgb_verification='captured pre-reset; freshness requires independent runtime audit'))
        for episode, seed in enumerate(seeds):
            budget()
            directory = out / f'episode-{episode:04d}'
            directory.mkdir()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            obs, _ = env.reset(seed=seed)
            env.public_final = None
            torch.save(rng_state(torch, random), directory / 'simulation-rng-initial.pt')
            torch.save(env.scene.get_state(is_relative=True), directory / 'initial-scene.pt')
            episode_start, latencies, samples, query_ids = time.monotonic(), [], [], []
            with h5py.File(directory / 'trajectory.hdf5', 'x') as h5:
                h5.attrs.update(complete=False, env_seed=seed, oracle_actuator_assisted=True,
                                action_hz=60, physics_hz=60, automatic_bc_admission=False)
                streams = {key: h5.create_dataset(key, shape=(0, 6), maxshape=(None, 6), dtype='f4',
                    chunks=(64, 6), compression='lzf') for key in
                    ('raw_model_motor', 'target_rad', 'action_manager_raw', 'pre_joint_rad', 'post_joint_rad')}
                objects = h5.create_dataset('diagnostics/object_pos_w', shape=(0, 4, 3), maxshape=(None, 4, 3),
                                             dtype='f4', chunks=(64, 4, 3), compression='lzf')
                efforts = h5.create_dataset('diagnostics/gripper_effort_limit', shape=(0,), maxshape=(None,), dtype='f4')
                cameras = {cam: h5.create_dataset('policy_rgb/'+cam, shape=(0, 480, 640, 3),
                    maxshape=(None, 480, 640, 3), dtype='u1', chunks=(1, 480, 640, 3), compression='lzf')
                    for cam in ('front', 'wrist')}
                for step in range(3600):
                    if step % 16 == 0:
                        budget()
                        measured = robot.data.joint_pos[:, ids].detach().cpu().numpy().copy()
                        motor = convert_leisaac_action_to_lerobot(measured)
                        if not np.allclose(motor, rad_to_motor(measured), atol=1e-4):
                            raise ValueError('Upstream feedback mapping drift')
                        payload = {cam: obs['policy'][cam].detach().cpu().numpy().copy() for cam in cameras}
                        payload['joint_motor'] = motor
                        save_arrays(out / 'ipc' / f'request-{query:06d}.npz', **payload)
                        response = out / 'ipc' / f'response-{query:06d}.npz'
                        wait_file(response, min(120, max(.1, a.wall_seconds-(time.monotonic()-started))),
                                  out / 'service-error.json', out / 'service-summary.json')
                        with np.load(response, allow_pickle=False) as data:
                            chunk_motor = data['actions_motor'].copy()
                            latency = float(data['inference_sec'])
                            if int(data['query_id']) != query:
                                raise ValueError('Model query identity mismatch')
                            model_rng_digest = str(data['model_rng_sha256'])
                        if chunk_motor.shape != (16, 6) or not np.isfinite(chunk_motor).all() or not np.isfinite(latency) or latency < 0:
                            raise ValueError('Invalid model response')
                        chunk = convert_lerobot_action_to_leisaac(chunk_motor)
                        if not np.allclose(chunk, motor_to_rad(chunk_motor), atol=1e-4):
                            raise ValueError('Upstream target mapping drift')
                        latencies.append(latency)
                        samples.append(step)
                        query_ids.append(query)
                        for cam, dst in cameras.items():
                            dst.resize(len(samples), axis=0)
                            dst[-1] = payload[cam][0]
                        query += 1
                    target = chunk[step % 16]
                    command = (torch.as_tensor(target[None], device=env.device)-offsets) / scales
                    pre = robot.data.joint_pos[0, ids].detach().cpu().numpy().copy()
                    dynamic_reset_gripper_effort_limit_sim(env, 'so101leader')
                    effort = float(robot.data.joint_effort_limits[0, ids[-1]])
                    obs, _, terminated, truncated, _ = env.step(command)
                    done = bool(terminated[0] | truncated[0])
                    final = env.public_final if done else snapshot()
                    if final is None:
                        raise RuntimeError('Missing pre-reset terminal snapshot')
                    for key, value in dict(raw_model_motor=chunk_motor[step % 16], target_rad=target,
                        action_manager_raw=command[0].detach().cpu().numpy(),
                        pre_joint_rad=pre, post_joint_rad=final['joint_pos']).items():
                        if not np.isfinite(value).all():
                            raise ValueError('Non-finite execution record')
                        streams[key].resize(step+1, axis=0)
                        streams[key][-1] = value
                    objects.resize(step+1, axis=0)
                    objects[-1] = final['objects']
                    efforts.resize(step+1, axis=0)
                    efforts[-1] = effort
                    if step % 16 == 0 or done:
                        h5.flush()
                        publish(out / 'progress.json', dict(completed_episodes=len(results), episode=episode,
                            env_seed=seed, physics_steps=step+1, model_queries=query))
                    if done:
                        for cam in cameras:
                            h5.create_dataset('terminal_rgb/'+cam, data=final['rgb'][cam], compression='lzf')
                        break
                if not done:
                    raise RuntimeError('Budget ended without native termination')
                h5.create_dataset('sample_steps', data=np.asarray(samples))
                h5.create_dataset('query_ids', data=np.asarray(query_ids))
                h5.create_dataset('inference_seconds', data=np.asarray(latencies))
                h5.attrs.update(complete=True, success=final['success'], terminated=bool(terminated[0]),
                                truncated=bool(truncated[0]), physics_steps=step+1)
            # The model RNG saved before the last response is now an episode artifact.
            shutil.copyfile(out / 'model-rng-latest.pt', directory / 'model-rng-after-episode.pt')
            if sha256(directory / 'model-rng-after-episode.pt') != model_rng_digest:
                raise ValueError('Model end-of-episode RNG handshake mismatch')
            torch.save(rng_state(torch, random), directory / 'simulation-rng-after-episode.pt')
            row = dict(episode=episode, env_seed=seed, success=final['success'], physics_steps=step+1,
                simulated_seconds=(step+1)/60, wall_seconds=time.monotonic()-episode_start,
                model_queries=len(samples), mean_inference_seconds=float(np.mean(latencies)),
                terminated=bool(terminated[0]), truncated=bool(truncated[0]),
                trajectory=f'{directory.name}/trajectory.hdf5', trajectory_sha256=sha256(directory / 'trajectory.hdf5'),
                model_rng_sha256=model_rng_digest,
                oracle_actuator_assisted=True, terminal_rgb_verified=False, automatic_bc_admission=False)
            publish(directory / 'result.json', row)
            with (out / 'episodes.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            results.append(row)
            publish(out / 'report.json', dict(status='completed' if len(results) == len(seeds) else 'running',
                completed_episodes=len(results), target_episodes=len(seeds),
                successes=sum(r['success'] for r in results), results=results,
                success_rate_completed_prefix=sum(r['success'] for r in results)/len(results)))
        publish(out / 'summary.json', dict(status='completed', episodes=len(results),
            successes=sum(r['success'] for r in results), protocol_name=cfg['name'],
            formal_course_evaluation=False, automatic_bc_admission=False))
    except BaseException:
        publish(out / 'error.json', dict(status='failed_preserved', completed_episodes=len(results),
                                       traceback=traceback.format_exc()))
        raise
    finally:
        (out / 'STOP').touch()
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == '__main__':
    main()
