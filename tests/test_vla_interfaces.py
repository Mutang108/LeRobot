"""CPU fixtures only: no downloaded model, optimizer, GPU or simulator."""
import ast
from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import random
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ('numpy', 'torch', 'yaml'))
ROOT = Path(__file__).resolve().parents[1]
if AVAILABLE:
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / 'course/liftcube/src'))
    from starter.models.joint_chunk import JointChunkPolicy, front_inputs, joint_chunk_to_motor
    from starter.models.smolvla import SmolVLAFrontPolicy, single_front_config, normalization_stats
    from starter.interfaces.observation import Observation, ObservationSpec
    from starter.interfaces.action import ActionSpec, motor_to_degrees
    from starter.evaluation.evaluate import run_episode


@unittest.skipUnless(AVAILABLE, 'Requires NumPy, CPU Torch and PyYAML')
class VLAInterfaceTests(unittest.TestCase):
    def obs(self):
        return Observation(np.full((64, 64, 3), 127, np.uint8), np.r_[np.zeros(6), np.full(15, 999)],
                           {'object_pos': 'must not enter payload', 'task': 'untrusted'})

    def config(self):
        features = {'observation.state': SimpleNamespace(shape=(6,))}
        features.update({f'observation.images.camera{i}': SimpleNamespace(shape=(3, 256, 256)) for i in (1, 2, 3)})
        class Config(SimpleNamespace):
            @property
            def image_features(self):
                return {k: v for k, v in self.input_features.items() if '.images.' in k}
        return Config(input_features=features, output_features={'action': SimpleNamespace(shape=(6,))},
                      empty_cameras=0, n_obs_steps=1, adapt_to_pi_aloha=False,
                      use_delta_joint_actions_aloha=False, rtc_config=None)

    def test_front_payload_excludes_padding_metadata_and_extra_views(self):
        payload = front_inputs(self.obs(), ObservationSpec(), 'lift cube', 'observation.images.camera1')
        self.assertEqual(set(payload), {'observation.images.camera1', 'observation.state', 'task'})
        np.testing.assert_array_equal(payload['observation.state'], np.zeros(6))
        self.assertEqual(payload['observation.images.camera1'].shape, (3, 64, 64))
        np.testing.assert_allclose(payload['observation.images.camera1'], 127 / 255)
        self.assertEqual(payload['task'], 'lift cube')

    def test_single_front_config_does_not_mutate_checkpoint(self):
        cfg = self.config()
        new = single_front_config(cfg, 'observation.images.camera1')
        self.assertEqual(len(new.image_features), 1)
        self.assertEqual(new.empty_cameras, 2)
        self.assertEqual((len(cfg.image_features), cfg.empty_cameras), (3, 0))
        with self.assertRaises(ValueError):
            single_front_config(cfg, 'observation.images.camera2')
        cfg.adapt_to_pi_aloha = True
        with self.assertRaises(ValueError):
            single_front_config(cfg, 'observation.images.camera1')

    def test_real_upstream_mask_method_with_single_front(self):
        # Execute only the pinned, reviewed image method, not the VLM/model class.
        source = ROOT / 'third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SmolVLAPolicy')
        method = deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_images'))
        scope = {'torch': torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), 'exec'), scope)
        cfg = single_front_config(self.config(), 'observation.images.camera1')
        # SmolVLAConfig.validate_features() adds the requested empty slot features.
        cfg.input_features.update({f'observation.images.empty_camera_{i}': SimpleNamespace() for i in range(cfg.empty_cameras)})
        cfg.resize_imgs_with_padding = None
        images, masks = scope['prepare_images'](SimpleNamespace(config=cfg),
            {'observation.images.camera1': torch.full((1, 3, 64, 64), 0.5)})
        self.assertEqual(len(images), 3)
        self.assertEqual([mask.item() for mask in masks], [True, False, False])
        self.assertTrue(torch.all(images[1] == -1))
        self.assertTrue(torch.all(images[2] == -1))
        self.assertTrue(torch.all(images[0] == 0))

    def test_stats_require_real_source_and_positive_std(self):
        doc = {'source': 'synthetic test fixture, not deployment stats', 'stats': {
            k: {'mean': [0]*6, 'std': [1]*6} for k in ('observation.state', 'action')}}
        self.assertEqual(normalization_stats(doc)['action']['std'].shape, (6,))
        for key, value in (('source', ''),):
            with self.assertRaises(ValueError):
                normalization_stats({**doc, key: value})
        for invalid in (0, -1, float('nan')):
            bad = deepcopy(doc)
            bad['stats']['action']['std'][0] = invalid
            with self.assertRaises(ValueError):
                normalization_stats(bad)

    def test_chunk_execution_replans_and_returns_copies(self):
        calls = []
        def predict(obs, task):
            calls.append(task)
            return np.arange(30, dtype=np.float32).reshape(5, 6)
        policy = JointChunkPolicy(predict, task='lift', execute_steps=2)
        first = policy.act(self.obs()); first[:] = 99
        np.testing.assert_array_equal(policy.act(self.obs()), np.arange(6, 12))
        np.testing.assert_array_equal(policy.act(self.obs()), np.arange(6))
        self.assertEqual(len(calls), 2)

    def test_task_change_reset_and_fault_clear_queues(self):
        resets = []
        policy = JointChunkPolicy(lambda o, t: np.zeros((3, 6)), task='old', execute_steps=2,
                                  reset_backend=lambda: resets.append(1))
        policy.act(self.obs())
        policy.set_task('new')
        self.assertEqual(len(policy._actions), 0)
        policy.act(self.obs())
        bad = self.obs(); bad.proprio[0] = np.nan
        with self.assertRaises(ValueError):
            policy.act(bad)
        self.assertEqual(len(policy._actions), 0)
        self.assertEqual(len(resets), 2)

    def test_reset_preserves_rng_and_pi05_reuses_front_contract(self):
        policy = JointChunkPolicy(lambda o, t: np.zeros((2, 6)), task='lift', execute_steps=2)
        py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        policy.reset()
        self.assertEqual(py, random.getstate())
        np.testing.assert_array_equal(np_state[1], np.random.get_state()[1])
        self.assertEqual(np_state[2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        payload = front_inputs(self.obs(), ObservationSpec(), 'lift', 'observation.images.base_0_rgb')
        self.assertNotIn('observation.images.left_wrist_0_rgb', payload)
        self.assertEqual(payload['observation.state'].shape, (6,))
        self.assertFalse(torch.cuda.is_initialized())

    def test_runner_success_and_reset_failure_clear_queue(self):
        class Env:
            action_spec = ActionSpec(frequency_hz=60)
            name = 'fixture'
            def reset(env, seed=None): return self.obs(), {}
            def step(env, action): return self.obs(), 1, True, False, {'success': True}
        policy = JointChunkPolicy(lambda o, t: np.zeros((3, 6)), task='lift', execute_steps=3)
        result = run_episode(Env(), policy, seed=1, episode_id=0, deterministic=True, checkpoint=None)
        self.assertTrue(result.success)
        self.assertEqual(result.length, 1)
        self.assertFalse(policy._actions)
        policy.act(self.obs())
        env = Env()
        def fail(seed=None): raise RuntimeError('fixture reset failure')
        env.reset = fail
        with self.assertRaisesRegex(RuntimeError, 'fixture reset'):
            run_episode(env, policy, seed=1, episode_id=0, deterministic=True, checkpoint=None)
        self.assertFalse(policy._actions)

    def test_bad_chunks_never_reach_execution(self):
        for chunk in (np.zeros((0, 6)), np.zeros((2, 7)), np.zeros((1, 6)),
                      np.full((2, 6), np.nan), np.full((2, 6), 101)):
            policy = JointChunkPolicy(lambda o, t: chunk, task='lift', execute_steps=2)
            with self.assertRaises(ValueError):
                policy.act(self.obs())
            self.assertFalse(policy._actions)

    def test_execution_contract_rejected_at_construction(self):
        for options in ({'execute_steps': True}, {'execute_steps': 0}, {'task': ' '},
                        {'frequency_hz': True}, {'frequency_hz': '30'}, {'frequency_hz': float('nan')},
                        {'output_units': 'eef'}, {'observation_spec': replace(ObservationSpec(), control_hz=60)},
                        {'observation_spec': replace(ObservationSpec(), camera='wrist')},
                        {'observation_spec': replace(ObservationSpec(), image_shape=(0, 64, 3))}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                JointChunkPolicy(lambda o, t: np.zeros((2, 6)), **{'task': 'lift', **options})
        with self.assertRaises(TypeError):
            JointChunkPolicy(None, task='lift')
        with self.assertRaises(TypeError):
            JointChunkPolicy(lambda o, t: None, task='lift', reset_backend=True)

    def test_degree_bridge_does_not_treat_gripper_as_angle(self):
        motor = np.array([[10, -20, 30, -40, 50, 60]], dtype=np.float32)
        degrees = motor_to_degrees(motor)
        degrees[:, 5] = motor[:, 5]
        np.testing.assert_allclose(joint_chunk_to_motor(degrees, 'arm_degrees_gripper_motor'), motor, atol=3e-5)
        with self.assertRaises(ValueError):
            joint_chunk_to_motor(np.zeros((2, 7)), 'eef')

    def test_smol_backend_injection_pre_post_and_reset(self):
        calls = []
        class Backend:
            config = SimpleNamespace(n_action_steps=3)
            def reset(self): calls.append('reset')
            def predict_action_chunk(self, batch):
                self.batch = batch
                return torch.full((1, 3, 6), .5, dtype=torch.bfloat16)
        backend = Backend()
        policy = SmolVLAFrontPolicy(backend, lambda b: b, lambda a: a * 10, task='lift',
                                   image_key='observation.images.camera1', execute_steps=2)
        np.testing.assert_array_equal(policy.act(self.obs()), np.full(6, 5))
        self.assertEqual(len(backend.batch), 3)
        policy.reset()
        self.assertEqual(calls, ['reset'])
        backend.config.n_action_steps = 50
        with self.assertRaisesRegex(ValueError, 'horizon'):
            policy.act(self.obs())
        self.assertFalse(policy._actions)

    def test_runner_boundaries_timeouts_errors_hold_and_legacy(self):
        class Env:
            action_spec = ActionSpec(frequency_hz=60)
            name = 'fixture'
            def reset(env, seed=None):
                env.actions = []
                return self.obs(), {}
            def step(env, action):
                env.actions.append(action.copy())
                return self.obs(), 0, False, False, {}
        queries = []
        policy = JointChunkPolicy(lambda o, t: queries.append(1) or np.zeros((4, 6)), task='lift', execute_steps=4)
        env = Env()
        for i in range(2):
            result = run_episode(env, policy, seed=i, episode_id=i, deterministic=True, checkpoint=None, max_episode_steps=3)
            self.assertTrue(result.truncated)
            self.assertFalse(policy._actions)
            np.testing.assert_array_equal(env.actions[0], env.actions[1])
        self.assertEqual(len(queries), 2)
        def fail(action): raise RuntimeError('fixture execution failure')
        env.step = fail
        with self.assertRaisesRegex(RuntimeError, 'fixture execution'):
            run_episode(env, policy, seed=1, episode_id=1, deterministic=True, checkpoint=None)
        self.assertFalse(policy._actions)
        legacy = SimpleNamespace(action_spec=ActionSpec(frequency_hz=30), act=lambda o, deterministic: np.zeros(6))
        run_episode(Env(), legacy, seed=1, episode_id=1, deterministic=True, checkpoint=None, max_episode_steps=1)


if __name__ == '__main__':
    unittest.main()
