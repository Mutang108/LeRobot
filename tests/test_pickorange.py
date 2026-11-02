"""CPU contracts for the portable PickOrange native-assisted source port."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'course/pickorange'))
try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, 'NumPy required')
class PickOrangeTests(unittest.TestCase):
    def test_protocol_and_check_have_no_simulator_dependency(self):
        from common import load_protocol
        cfg = load_protocol()
        self.assertEqual(cfg['scene_seeds'], list(range(17, 117)))
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'course/pickorange/evaluate.py'), '--check'],
                                capture_output=True, text=True, check=True)
        self.assertFalse(json.loads(result.stdout)['simulation_run'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.json'
            cfg['dynamic_gripper_effort'] = False
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'dynamic_gripper_effort'):
                load_protocol(path)

    def test_model_whitelist_no_clipping_and_shape_guards(self):
        from common import predict_groot, validate_request
        data = dict(joint_motor=np.zeros((1, 6)),
                    front=np.zeros((1, 480, 640, 3), np.uint8),
                    wrist=np.ones((1, 480, 640, 3), np.uint8))
        class Policy:
            def get_action(self, payload):
                self.payload = payload
                return {'action.single_arm': np.full((16, 5), 101.), 'action.gripper': np.zeros((16, 1))}
        policy = Policy()
        actions = predict_groot(policy, data, 'task')
        self.assertEqual(actions[0, 0], 101)  # Explicitly native, not silently clamped.
        self.assertEqual(set(policy.payload), {'video.front', 'video.wrist', 'state.single_arm', 'state.gripper', 'annotation.human.task_description'})
        with self.assertRaisesRegex(ValueError, 'whitelist'):
            validate_request(dict(data, objects=np.zeros((3, 3))))
        with self.assertRaisesRegex(ValueError, 'RGB'):
            validate_request(dict(data, wrist=data['front'].astype(float)))

    def test_atomic_ipc_and_bounded_wait(self):
        from common import save_arrays, wait_file
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'request.npz'
            save_arrays(path, q=np.arange(6))
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['q'], np.arange(6))
            with self.assertRaises(FileExistsError):
                save_arrays(path, q=np.arange(6))
            with self.assertRaises(TimeoutError):
                wait_file(Path(tmp) / 'absent', 0)

    def test_native_success_strict_ranges_and_rest(self):
        from audit import native_success
        joints = np.deg2rad([0, -100, 90, 50, 0, -10])
        objects = np.zeros((4, 3))
        self.assertTrue(native_success(joints, objects))
        objects[0, 0] = .10
        self.assertFalse(native_success(joints, objects))
        objects[:] = 0
        self.assertFalse(native_success(np.zeros(6), objects))

    @unittest.skipUnless(importlib.util.find_spec('h5py'), 'h5py required for trajectory audit fixture')
    def test_closed_trajectory_audit_and_corruption(self):
        import h5py
        from audit import audit_episode
        from common import publish, save_arrays, sha256
        sys.path.insert(0, str(ROOT / 'course/scripts'))
        from joint_contract import motor_to_rad, rad_to_motor
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            publish(out / 'manifest.json', dict(action_offsets_rad=[0]*6, action_scales=[1]*6))
            path = out / 'trajectory.hdf5'
            joints = np.tile(np.deg2rad([0, -100, 90, 50, 0, -10]), (2, 1)).astype(np.float32)
            motor = rad_to_motor(joints)
            targets = motor_to_rad(motor)
            (out / 'ipc').mkdir()
            (out / 'model-rng-after-episode.pt').write_bytes(b'synthetic RNG artifact, not a model checkpoint')
            rng_digest = sha256(out / 'model-rng-after-episode.pt')
            save_arrays(out / 'ipc/request-000000.npz', joint_motor=motor[:1],
                        front=np.zeros((1, 480, 640, 3), np.uint8), wrist=np.zeros((1, 480, 640, 3), np.uint8))
            save_arrays(out / 'ipc/response-000000.npz', actions_motor=np.tile(motor[:1], (16, 1)),
                        inference_sec=np.asarray(.1), query_id=np.asarray(0), model_rng_sha256=np.asarray(rng_digest))
            with h5py.File(path, 'x') as f:
                f.attrs.update(complete=True, physics_steps=2, env_seed=17, success=True, terminated=True, truncated=False)
                for key, values in dict(raw_model_motor=motor, target_rad=targets, action_manager_raw=targets,
                                        pre_joint_rad=joints, post_joint_rad=joints).items():
                    f.create_dataset(key, data=values)
                f.create_dataset('diagnostics/object_pos_w', data=np.zeros((2, 4, 3)))
                f.create_dataset('diagnostics/gripper_effort_limit', data=np.ones(2))
                f.create_dataset('sample_steps', data=[0])
                f.create_dataset('query_ids', data=[0])
                f.create_dataset('inference_seconds', data=[.1])
                for cam in ('front', 'wrist'):
                    f.create_dataset('policy_rgb/'+cam, data=np.zeros((1, 480, 640, 3), np.uint8))
                    f.create_dataset('terminal_rgb/'+cam, data=np.zeros((480, 640, 3), np.uint8))
            row = dict(episode=0, env_seed=17, physics_steps=2, success=True, model_queries=1,
                       terminated=True, truncated=False, trajectory=path.name, trajectory_sha256=sha256(path))
            row['model_rng_sha256'] = rng_digest
            result = audit_episode(out, row)
            self.assertTrue(result['trajectory_audit_passed'])
            self.assertFalse(result['automatic_bc_admission'])
            row['success'] = False
            with self.assertRaisesRegex(ValueError, 'success'):
                audit_episode(out, row)


if __name__ == '__main__':
    unittest.main()
