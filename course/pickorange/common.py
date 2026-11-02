"""PickOrange's explicit development protocol and local, non-pickle IPC."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

DEFAULT_PROTOCOL = Path(__file__).with_name('protocols') / 'groot-native-assisted.json'
MODEL_REVISION = '10f20142f96f0b9ef139aa90855a73e28a83ca74'
BACKEND_REVISION = '4af2b622892f7dcb5aae5a3fb70bcb02dc217b96'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path=DEFAULT_PROTOCOL):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    required = dict(name='pickorange-groot-n15-native-assisted-v1', task_id='LeIsaac-SO101-PickOrange-v0',
        instruction='Pick three oranges and put them into the plate, then reset the arm to rest state.',
        model_repo='LightwheelAI/leisaac-pick-orange-v0', model_revision=MODEL_REVISION,
        backend_revision=BACKEND_REVISION, physics_hz=60, action_hz=60,
        chunk_length=16, execute_steps=16, denoising_steps=4, episode_seconds=60,
        dynamic_gripper_effort=True, software_action_clipping=False,
        robot_gravity_enabled=True, formal_course_evaluation=False, automatic_bc_admission=False)
    for key, value in required.items():
        if type(cfg.get(key)) is not type(value) or cfg[key] != value:
            raise ValueError(f'Unsupported protocol value: {key}')
    seeds = cfg.get('scene_seeds')
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Distinct integer scene seeds are required')
    if seeds != list(range(17, 117)):
        raise ValueError('This protocol freezes seeds 17..116; use --episodes only for a smoke prefix')
    if not isinstance(cfg.get('instruction'), str) or not cfg['instruction'].strip():
        raise ValueError('Task instruction is required')
    return cfg


def publish(path, value):
    path = Path(path)
    temporary = path.with_suffix('.new')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_arrays(path, **arrays):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix('.new')
    with temporary.open('xb') as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def wait_file(path, timeout, error=None, stopped=None):
    end = time.monotonic() + timeout
    while not Path(path).exists():
        if error is not None and Path(error).exists():
            raise RuntimeError(f'Peer failed; see {Path(error).name}')
        if stopped is not None and Path(stopped).exists():
            raise RuntimeError('Peer stopped before producing the requested file')
        if time.monotonic() >= end:
            raise TimeoutError(Path(path).name)
        time.sleep(.1)


def validate_request(data):
    if set(data) != {'front', 'wrist', 'joint_motor'}:
        raise ValueError('Model input whitelist is front, wrist and measured joints only')
    q = np.asarray(data['joint_motor'])
    if q.shape != (1, 6) or not np.isfinite(q).all():
        raise ValueError('Expected finite ordered [1,6] motor feedback')
    for cam in ('front', 'wrist'):
        image = np.asarray(data[cam])
        if image.shape != (1, 480, 640, 3) or image.dtype != np.uint8:
            raise ValueError('Expected real [1,480,640,3] uint8 RGB views')
    return q.astype(np.float64, copy=True)


def predict_groot(policy, data, instruction):
    q = validate_request(data)
    payload = {'state.single_arm': q[:, :5], 'state.gripper': q[:, 5:],
               'annotation.human.task_description': [instruction]}
    payload.update({'video.' + cam: np.asarray(data[cam]).copy() for cam in ('front', 'wrist')})
    result = policy.get_action(payload)
    arm, jaw = np.asarray(result['action.single_arm']), np.asarray(result['action.gripper'])
    if arm.shape != (16, 5) or jaw.shape != (16, 1):
        raise ValueError('Expected native GR00T action chunks [16,5] and [16,1]')
    actions = np.concatenate([arm, jaw], axis=1)
    if not np.isfinite(actions).all():
        raise ValueError('Non-finite model output')
    # Deliberately no software clamp: this protocol retains native physical stops.
    return actions


def rng_state(torch, random):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
