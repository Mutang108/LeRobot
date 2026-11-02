"""CPU audit of this directory's trajectories; does not approve BC admission."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from common import publish, sha256


def native_success(joints, objects):
    """Pinned PickOrange XYZ box and strict rest ranges, in canonical order."""
    joints, objects = np.asarray(joints), np.asarray(objects)
    if joints.shape != (6,) or objects.shape != (4, 3) or not np.isfinite(joints).all() or not np.isfinite(objects).all():
        raise ValueError('Invalid terminal joints/objects')
    delta = objects[:3] - objects[3]
    rest = np.array([[-30, 30], [-130, -70], [60, 120], [20, 80], [-30, 30], [-40, 20]])
    degrees = np.rad2deg(joints)
    return bool((np.abs(delta) < [.10, .10, .07]).all()
                and (degrees > rest[:, 0]).all() and (degrees < rest[:, 1]).all())


def audit_episode(run_dir, row):
    import h5py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from joint_contract import motor_to_rad, rad_to_motor
    run_dir = Path(run_dir).resolve()
    path = (run_dir / row['trajectory']).resolve()
    if not path.is_relative_to(run_dir):
        raise ValueError('Trajectory is outside run directory')
    if sha256(path) != row['trajectory_sha256']:
        raise ValueError('Trajectory SHA mismatch')
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    with h5py.File(path, 'r') as f:
        n = int(row['physics_steps'])
        if not f.attrs.get('complete') or n < 1 or n > 3600 or f.attrs['physics_steps'] != n or f.attrs['env_seed'] != row['env_seed']:
            raise ValueError('Incomplete or inconsistent episode identity')
        arrays = {key: f[key][:] for key in ('raw_model_motor', 'target_rad', 'action_manager_raw', 'pre_joint_rad', 'post_joint_rad')}
        if any(v.shape != (n, 6) or not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Execution stream shape/finiteness mismatch')
        if not np.allclose(motor_to_rad(arrays['raw_model_motor']), arrays['target_rad'], atol=1e-5):
            raise ValueError('Model-to-target mapping mismatch')
        sent = arrays['action_manager_raw'] * manifest['action_scales'] + manifest['action_offsets_rad']
        if not np.allclose(sent, arrays['target_rad'], atol=1e-5):
            raise ValueError('Action-manager conversion mismatch')
        if not np.allclose(arrays['pre_joint_rad'][1:], arrays['post_joint_rad'][:-1], atol=1e-6):
            raise ValueError('Physical continuity mismatch')
        objects = f['diagnostics/object_pos_w'][:]
        efforts = f['diagnostics/gripper_effort_limit'][:]
        if objects.shape != (n, 4, 3) or efforts.shape != (n,) or not np.isfinite(objects).all() or not np.isfinite(efforts).all():
            raise ValueError('Diagnostic stream mismatch')
        samples = f['sample_steps'][:]
        if not np.array_equal(samples, np.arange(0, n, 16)) or row['model_queries'] != len(samples):
            raise ValueError('Model-query alignment mismatch')
        query_ids = f['query_ids'][:]
        latencies = f['inference_seconds'][:]
        if query_ids.shape != samples.shape or not np.issubdtype(query_ids.dtype, np.integer) or not np.array_equal(query_ids, np.arange(query_ids[0], query_ids[0]+len(samples))):
            raise ValueError('Query identity mismatch')
        if latencies.shape != samples.shape or not np.isfinite(latencies).all() or (latencies < 0).any():
            raise ValueError('Inference latency mismatch')
        for cam in ('front', 'wrist'):
            if (f['policy_rgb/'+cam].shape != (len(samples), 480, 640, 3) or f['terminal_rgb/'+cam].shape != (480, 640, 3)
                    or f['policy_rgb/'+cam].dtype != np.uint8 or f['terminal_rgb/'+cam].dtype != np.uint8):
                raise ValueError('RGB sample/terminal shape mismatch')
        from common import validate_request
        for index, (query, start) in enumerate(zip(query_ids, samples)):
            with np.load(run_dir / 'ipc' / f'request-{query:06d}.npz', allow_pickle=False) as request:
                q = validate_request(request)
                if not np.allclose(q[0], rad_to_motor(arrays['pre_joint_rad'][start]), atol=1e-4):
                    raise ValueError('Query feedback differs from actual pre-action feedback')
                for cam in ('front', 'wrist'):
                    if not np.array_equal(request[cam][0], f['policy_rgb/'+cam][index]):
                        raise ValueError('Query RGB differs from trajectory')
            stop = min(n, int(start)+16)
            with np.load(run_dir / 'ipc' / f'response-{query:06d}.npz', allow_pickle=False) as response:
                actions = response['actions_motor']
                if actions.shape != (16, 6) or not np.isfinite(actions).all() or int(response['query_id']) != query:
                    raise ValueError('Invalid raw model response')
                if not np.array_equal(arrays['raw_model_motor'][start:stop], actions[:stop-start].astype(np.float32)):
                    raise ValueError('Executed model prefix differs from original response')
                if not np.isclose(float(response['inference_sec']), latencies[index], atol=1e-8):
                    raise ValueError('Query latency differs from response')
                last_rng_digest = str(response['model_rng_sha256'])
        rng_file = path.parent / 'model-rng-after-episode.pt'
        if row['model_rng_sha256'] != last_rng_digest or sha256(rng_file) != last_rng_digest:
            raise ValueError('Model RNG handshake mismatch')
        if not bool(row['terminated'] or row['truncated']) or (row['truncated'] and n != 3600):
            raise ValueError('Native terminal budget mismatch')
        if bool(f.attrs['terminated']) != row['terminated'] or bool(f.attrs['truncated']) != row['truncated']:
            raise ValueError('Native termination record mismatch')
        success = native_success(arrays['post_joint_rad'][-1], objects[-1])
        if success != row['success'] or success != bool(f.attrs['success']):
            raise ValueError('Native success recomputation mismatch')
    return dict(episode=row['episode'], env_seed=row['env_seed'], success=success,
                trajectory_audit_passed=True, terminal_rgb_freshness_verified=False,
                independent_cold_replay_verified=False, automatic_bc_admission=False,
                model_rng_handshake_verified=True, first_query=int(query_ids[0]), last_query=int(query_ids[-1]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError(a.output)
    manifest = json.loads((a.run_dir / 'manifest.json').read_text(encoding='utf-8'))
    rows = [json.loads(s) for s in (a.run_dir / 'episodes.jsonl').read_text(encoding='utf-8').splitlines() if s]
    if len(rows) > len(manifest['scene_seeds']) or [(r['episode'], r['env_seed']) for r in rows] != list(enumerate(manifest['scene_seeds'][:len(rows)])):
        raise ValueError('Completed episodes must be the unique ordered seed prefix')
    results = [audit_episode(a.run_dir, row) for row in rows]
    next_query = 0
    for result in results:
        if result['first_query'] != next_query:
            raise ValueError('Model queries are not unique and continuous across episodes')
        next_query = result['last_query']+1
    report = json.loads((a.run_dir / 'report.json').read_text(encoding='utf-8'))
    if report['results'] != rows or report['completed_episodes'] != len(rows) or report['successes'] != sum(r['success'] for r in rows):
        raise ValueError('Root report differs from complete episode records')
    publish(a.output, dict(status='trajectory_audit_passed', completed_episodes=len(rows),
        target_episodes=len(manifest['scene_seeds']), successes=sum(r['success'] for r in rows),
        results=results, automatic_bc_admission=False))


if __name__ == '__main__':
    main()
