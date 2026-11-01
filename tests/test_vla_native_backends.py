"""Native backend boundary fixtures. No real models, GPU, network or optimizer."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(k) for k in ('numpy', 'torch', 'yaml'))
if AVAILABLE:
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'course/liftcube/src'))
    from starter.interfaces.action import ActionSpec, JOINT_NAMES, motor_to_degrees
    from starter.interfaces.observation import Observation, ObservationSpec
    from starter.models.smolvla import SmolVLAFrontPolicy
    from starter.models.pi05 import Pi05FrontPolicy, single_front_config, strict_author_state
    from starter.models.gr00t import Gr00tFrontPolicy, load_groot_weights
    from starter.models.openvla_oft import OpenVLAOFTFrontPolicy, check_statistics
    from starter.models.backend_io import require_complete_load, tensor_chunk
    from starter.evaluation.evaluate import parse_args, run_episode


@unittest.skipUnless(AVAILABLE, 'Requires CPU Torch, NumPy and PyYAML')
class NativeBackendTests(unittest.TestCase):
    def obs(self):
        return Observation(np.full((64, 64, 3), 120, np.uint8), np.r_[np.zeros(6), np.full(15, 999)],
                           {'object_pose': [1, 2, 3], 'task': 'ignore metadata'})

    def pi_backend(self):
        class Backend:
            config = SimpleNamespace(device='cpu', n_action_steps=4)
            resets = 0
            calls = 0
            def reset(b): b.resets += 1
            def predict_action_chunk(b, batch):
                b.calls += 1; b.batch = batch
                return torch.full((1, 4, 6), 12.)
        return Backend()

    def groot_backend(self, extra_camera=False):
        class Backend:
            calls = 0
            def get_modality_config(b):
                names = {'video': ['video.webcam'] + (['video.wrist'] if extra_camera else []),
                         'state': ['state.single_arm', 'state.gripper'],
                         'action': ['action.single_arm', 'action.gripper'],
                         'language': ['annotation.human.task_description']}
                return {key: SimpleNamespace(modality_keys=value, delta_indices=list(range(4)) if key == 'action' else [0])
                        for key, value in names.items()}
            def get_action(b, payload):
                b.calls += 1; b.batch = payload
                return {'action.single_arm': np.full((4, 5), 11.), 'action.gripper': np.full((4, 1), 12.)}
        return Backend()

    def stats(self):
        return {key: {'min': [-100]*5+[0], 'max': [100]*6, 'mask': [True]*6}
                for key in ('action', 'proprio')}

    def oft(self, joint_units='motor', **kwargs):
        self.oft_calls = []
        def predict(*args, **kw):
            self.oft_calls.append((args, kw))
            return np.full((4, 6), 13.)
        return OpenVLAOFTFrontPolicy(SimpleNamespace(norm_stats={'so101': self.stats()}), object(), object(), object(),
                                    predict, unnorm_key='so101', task='lift cube', joint_units=joint_units,
                                    center_crop=False, chunk_size=4, execute_steps=2, **kwargs)

    def test_pi05_native_batch_and_embedded_outputs(self):
        backend = self.pi_backend()
        policy = Pi05FrontPolicy(backend, task='lift cube', execute_steps=2)
        np.testing.assert_array_equal(policy.act(self.obs()), np.full(6, 12))
        self.assertEqual(set(backend.batch), {'task', 'observation.state', 'observation.images.base_0_rgb'})
        self.assertEqual(backend.batch['task'], ['lift cube'])
        self.assertEqual(backend.batch['observation.state'].shape, (1, 6))
        self.assertEqual(backend.batch['observation.images.base_0_rgb'].shape, (1, 3, 64, 64))
        self.assertTrue(torch.all(backend.batch['observation.state'] == 0))
        policy.act(self.obs()); self.assertEqual(backend.calls, 1)
        policy.set_task('place cube'); policy.act(self.obs())
        self.assertEqual(backend.batch['task'], ['place cube'])
        self.assertEqual(backend.resets, 1)

    def test_bfloat16_tensor_output_uses_float32_without_clipping(self):
        output = torch.full((1, 4, 6), 12., dtype=torch.bfloat16)
        array = tensor_chunk(output)
        self.assertEqual(array.dtype, np.float32)
        np.testing.assert_array_equal(array, np.full((4, 6), 12))
        backend = self.pi_backend()
        backend.predict_action_chunk = lambda batch: output
        policy = Pi05FrontPolicy(backend, task='lift', execute_steps=2)
        np.testing.assert_array_equal(policy.act(self.obs()), np.full(6, 12))
        output[:] = 101
        policy.reset()
        with self.assertRaises(ValueError): policy.act(self.obs())
        self.assertFalse(policy._actions)

    def test_pi05_config_copy_and_wrong_state_condition(self):
        class Config(SimpleNamespace):
            @property
            def image_features(self): return {k: v for k, v in self.input_features.items() if '.images.' in k}
        cfg = Config(state_cond=True, n_obs_steps=1, compile_model=False, fuse_qkv=False, fuse_gate_up=False,
                     input_features={'observation.state': SimpleNamespace(shape=(6,)), **{
                         f'observation.images.{key}': SimpleNamespace(shape=(3, 480, 640))
                         for key in ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')}},
                     output_features={'action': SimpleNamespace(shape=(6,))})
        new = single_front_config(cfg)
        self.assertEqual(len(cfg.image_features), 3)
        self.assertEqual(len(new.image_features), 1)
        self.assertEqual(new.empty_cameras, 2)
        cfg.state_cond = False
        with self.assertRaises(ValueError): single_front_config(cfg)

    def test_pi05_strict_alias_recovery_no_same_shape_guess(self):
        class Graph(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.model = torch.nn.Module()
                s.model.suffix_embedder = torch.nn.Module()
                s.model.suffix_embedder.state_proj = torch.nn.Linear(2, 2, bias=False)
                s.alias = s.model.suffix_embedder.state_proj
                s.other = torch.nn.Linear(2, 2, bias=False)
        graph = Graph()
        good = {'state_proj.weight': torch.ones(2, 2), 'other.weight': torch.full((2, 2), 2.)}
        strict_author_state(graph, good)
        self.assertTrue(torch.all(graph.other.weight == 2))
        with self.assertRaises(ValueError): strict_author_state(graph, {'state_proj.weight': torch.ones(2, 2)})
        with self.assertRaises(ValueError):
            strict_author_state(graph, {**good, 'alias.weight': torch.zeros(2, 2)})

    def test_groot_single_real_view_five_plus_one(self):
        backend = self.groot_backend()
        policy = Gr00tFrontPolicy(backend, task='lift cube', execute_steps=2)
        np.testing.assert_array_equal(policy.act(self.obs()), [11]*5+[12])
        self.assertEqual(set(backend.batch), {'video.webcam', 'state.single_arm', 'state.gripper',
                                              'annotation.human.task_description'})
        self.assertEqual(backend.batch['video.webcam'].shape, (1, 64, 64, 3))
        self.assertEqual(backend.batch['video.webcam'].dtype, np.uint8)
        self.assertEqual(backend.batch['state.single_arm'].shape, (1, 5))
        self.assertEqual(backend.batch['state.gripper'].shape, (1, 1))
        with self.assertRaises(ValueError): Gr00tFrontPolicy(self.groot_backend(True), task='lift', execute_steps=2)

    def test_groot_bad_batch_not_squeezed_or_truncated(self):
        backend = self.groot_backend()
        backend.get_action = lambda p: {'action.single_arm': np.zeros((1, 4, 5)), 'action.gripper': np.zeros((4, 1))}
        policy = Gr00tFrontPolicy(backend, task='lift', execute_steps=2)
        with self.assertRaises(ValueError): policy.act(self.obs())
        self.assertFalse(policy._actions)

    def test_groot_checked_loader_bypasses_model_only_override(self):
        calls = []
        class HFBase:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls.append((cls, path, kwargs))
                return torch.nn.Linear(2, 2), {}
        class Native(HFBase):
            @classmethod
            def from_pretrained(cls, *a, **kw):
                raise AssertionError('Native wrapper cannot return loading info safely')
        with patch.dict(sys.modules, {'transformers': SimpleNamespace(PreTrainedModel=HFBase)}):
            model = load_groot_weights(Native, '/trusted/model', 'cpu')
        self.assertIs(calls[0][0], Native)
        self.assertTrue(calls[0][2]['output_loading_info'])
        self.assertTrue(calls[0][2]['local_files_only'])
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
            info = {key: ['fixture']}
            with self.assertRaises(ValueError) as caught: require_complete_load(info)
            self.assertIs(caught.exception.loading_info, info)

    def test_oft_native_function_arguments_and_no_double_unnormalization(self):
        policy = self.oft()
        np.testing.assert_array_equal(policy.act(self.obs()), np.full(6, 13))
        args, kw = self.oft_calls[0]
        cfg, model, processor, payload, task = args
        self.assertEqual(cfg.num_images_in_input, 1)
        self.assertEqual(set(payload), {'full_image', 'additional_images', 'state'})
        self.assertEqual(payload['additional_images'], [])
        self.assertEqual(payload['full_image'].dtype, np.uint8)
        self.assertEqual(task, 'lift cube')
        self.assertIs(kw['action_head'], policy.action_head)
        self.assertIs(kw['proprio_projector'], policy.proprio_projector)
        policy.set_task('place cube'); policy.act(self.obs())
        self.assertEqual(self.oft_calls[-1][0][-1], 'place cube')

    def test_oft_refuses_seven_dimensional_or_discrete_fallback(self):
        policy = self.oft()
        policy.predict_action = lambda *a, **kw: np.zeros((4, 7))
        with self.assertRaises(ValueError): policy.act(self.obs())
        with self.assertRaises(ValueError):
            OpenVLAOFTFrontPolicy(SimpleNamespace(norm_stats={}), None, None, None, None,
                unnorm_key='so101', task='lift', joint_units='motor', center_crop=False, chunk_size=4)

    def test_native_horizons_must_match_before_any_action(self):
        backend = self.pi_backend()
        backend.config = SimpleNamespace(device='cpu', n_action_steps=50)
        pi = Pi05FrontPolicy(backend, task='lift', execute_steps=2)
        with self.assertRaisesRegex(ValueError, 'horizon'): pi.act(self.obs())
        self.assertFalse(pi._actions)
        oft = self.oft()
        oft.predict_action = lambda *a, **k: np.zeros((3, 6))
        with self.assertRaisesRegex(ValueError, 'horizon'): oft.act(self.obs())
        self.assertFalse(oft._actions)

    def test_oft_stats_validation(self):
        for bad in ('dimension', 'mask', 'equal', 'nan'):
            stats = self.stats()
            if bad == 'dimension': stats['proprio']['min'] = [0]*7
            if bad == 'mask': stats['action']['mask'][5] = False
            if bad == 'equal': stats['action']['max'][0] = -100
            if bad == 'nan': stats['action']['min'][0] = float('nan')
            with self.assertRaises(ValueError): check_statistics(stats)

    def test_native_mixed_unit_roundtrip_preserves_jaw(self):
        obs = self.obs()
        obs.proprio[:6] = [10, -20, 30, -40, 50, 60]
        backend = self.groot_backend()
        def echo(payload):
            return {'action.single_arm': np.repeat(payload['state.single_arm'], 4, axis=0),
                    'action.gripper': np.repeat(payload['state.gripper'], 4, axis=0)}
        backend.get_action = echo
        groot = Gr00tFrontPolicy(backend, task='lift', joint_units='arm_degrees_gripper_motor', execute_steps=2)
        oft = self.oft(joint_units='arm_degrees_gripper_motor')
        oft.predict_action = lambda cfg, m, p, payload, task, **kw: np.repeat(payload['state'][None], 4, axis=0)
        for policy in (groot, oft):
            np.testing.assert_allclose(policy.act(obs), obs.proprio[:6], atol=3e-5)
            self.assertEqual(policy.last_raw_chunk[0, 5], 60)

    def test_all_backends_runner_boundary_and_rng(self):
        owner = self
        class Env:
            action_spec = ActionSpec(frequency_hz=60)
            name = 'fixture'
            def reset(s, seed=None): return owner.obs(), {}
            def step(s, action): return owner.obs(), 0, True, False, {'success': True}
        for policy in (Pi05FrontPolicy(self.pi_backend(), task='lift', execute_steps=2),
                       Gr00tFrontPolicy(self.groot_backend(), task='lift', execute_steps=2), self.oft()):
            rng = torch.get_rng_state().clone()
            for i in range(2):
                result = run_episode(Env(), policy, seed=i, episode_id=i, deterministic=True, checkpoint=None)
                self.assertTrue(result.success)
                self.assertFalse(policy._actions)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_loaders_reject_wrong_identity_before_optional_imports(self):
        for cls, backend in ((SmolVLAFrontPolicy, 'smolvla'), (Pi05FrontPolicy, 'pi05-vlash'), (Gr00tFrontPolicy, 'gr00t-n1.5'),
                             (OpenVLAOFTFrontPolicy, 'openvla-oft-so101')):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'adapter.json'
                path.write_text(json.dumps({'backend': backend, 'joint_names': list(reversed(JOINT_NAMES)),
                                            'joint_units': 'motor'}))
                with self.assertRaisesRegex(ValueError, 'joint order'): cls.load(path)

    def test_all_descriptors_validate_execution_before_optional_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'adapter.json'
            good = dict(joint_names=list(JOINT_NAMES), joint_units='motor', model_path='.',
                        model_revision='fixture', backend_revision='fixture', calibration_source='fixture',
                        task='lift', observation_spec=ObservationSpec().to_dict(), execute_steps=2, frequency_hz=30)
            for cls, backend in ((SmolVLAFrontPolicy, 'smolvla'), (Pi05FrontPolicy, 'pi05-vlash'),
                                 (Gr00tFrontPolicy, 'gr00t-n1.5'), (OpenVLAOFTFrontPolicy, 'openvla-oft-so101')):
                for invalid, message in (({'task': ' '}, 'task'), ({'execute_steps': False}, 'execute_steps'),
                                         ({'frequency_hz': 0}, 'frequency_hz'),
                                         ({'observation_spec': {**good['observation_spec'], 'control_hz': 60}}, 'clocks'),
                                         ({'observation_spec': {**good['observation_spec'], 'camera': 'wrist'}}, 'front')):
                    with self.subTest(cls=cls.__name__, invalid=invalid):
                        path.write_text(json.dumps({**good, 'backend': backend, **invalid}))
                        with self.assertRaisesRegex(ValueError, message): cls.load(path)

    def test_oft_platform_flag_parses_without_mutating_process_or_importing_backend(self):
        before = sys.argv.copy()
        args = parse_args(['--robot_platform', 'SO101'])
        self.assertEqual(args.robot_platform, 'SO101')
        self.assertEqual(sys.argv, before)
        self.assertNotIn('prismatic.vla.constants', sys.modules)


if __name__ == '__main__':
    unittest.main()
