"""CPU-only synthetic checks. No Isaac, public demonstrations or formal training."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch", "yaml"))
if AVAILABLE:
    import numpy as np
    import torch
    import yaml
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "course/liftcube/src"))
    from starter.algorithms.bc import BCTrainer
    from starter.algorithms.reference.ppo import PPOAlgorithm, compute_gae
    from starter.data.loader import CanonicalDataset, make_dataloader
    from starter.envs.timing import decision_specs, policy_action_hold_steps
    from starter.interfaces.action import ActionSpec
    from starter.interfaces.model import SmallCNNPolicy, _GridAveragePool
    from starter.interfaces.observation import Observation, ObservationSpec
    from starter.training.checkpoint import capture_rng, load_training, save_training
    from starter.utils import seed_all
    from starter.training import train_bc, train_rl
    from starter.training.events import open_writer, log_row


@unittest.skipUnless(AVAILABLE, "Optional CPU application checks require NumPy, Torch and PyYAML")
class LiftCubeTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_all(41)
        self.obs_spec = ObservationSpec()
        self.action_spec = ActionSpec()

    def policy(self):
        return SmallCNNPolicy(self.obs_spec, self.action_spec, hidden_dim=8)

    def batch(self, n=4):
        return {"image": np.zeros((n, 64, 64, 3), np.uint8),
                "proprio": np.zeros((n, 21), np.float32),
                "action": np.zeros((n, 6), np.float32)}

    def assert_tree_equal(self, left, right):
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_proprio_fusion_padding_and_legacy_export(self):
        model = self.policy()
        batch = self.batch()
        before = model(batch).detach().clone()
        batch["proprio"][:, :6] = 75
        self.assertFalse(torch.equal(before, model(batch)))
        reference = model(batch).detach().clone()
        batch["proprio"][:, 6:12] = 998
        batch["proprio"][:, 18:] = -998
        self.assertTrue(torch.equal(reference, model(batch)))
        model.update(batch)
        self.assertGreater(model.body[0].weight.grad.abs().sum().item(), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "policy.pt")
            model.save(path)
            self.assertTrue(torch.equal(model(batch), SmallCNNPolicy.load(path)(batch)))
            legacy = SmallCNNPolicy(self.obs_spec, self.action_spec, hidden_dim=8, feature_mode="image_only")
            payload = legacy.checkpoint_payload()
            payload.pop("feature_mode")
            restored = SmallCNNPolicy.from_payload(payload)
            self.assertTrue(torch.equal(legacy(batch), restored(batch)))

    def test_uniform_pool_preserves_adaptive_values_and_gradients(self):
        for shape in [(8, 8), (12, 12), (11, 9)]:
            a = torch.randn(2, 3, *shape, dtype=torch.float64, requires_grad=True)
            b = a.detach().clone().requires_grad_()
            actual = _GridAveragePool((4, 4))(a)
            expected = torch.nn.functional.adaptive_avg_pool2d(b, (4, 4))
            torch.testing.assert_close(actual, expected)
            actual.square().sum().backward()
            expected.square().sum().backward()
            torch.testing.assert_close(a.grad, b.grad)

    def test_timeout_bootstrap_does_not_leak_across_reset(self):
        adv, _ = compute_gae(torch.tensor([1., 100.]), torch.zeros(2), torch.ones(2),
                             torch.tensor(0.), .9, .95, next_values=torch.tensor([2., 0.]))
        torch.testing.assert_close(adv, torch.tensor([2.8, 100.]))

    def test_holding_partial_terminal_and_native_success(self):
        class Env:
            action_spec = ActionSpec(frequency_hz=60)
            observation_spec = ObservationSpec(control_hz=60)

            def reset(self, seed=None):
                self.ticks = 0
                return self.obs(), {}

            def obs(self):
                return Observation(np.zeros((64, 64, 3), np.uint8), np.zeros(21, np.float32))

            def step(self, action):
                self.ticks += 1
                return self.obs(), 1., False, self.ticks == 3, {"success": False}

            def get_privileged_state(self):
                return None

        env = Env()
        obs, action = decision_specs(env, 30)
        policy = SmallCNNPolicy(obs, action, hidden_dim=8)
        algo = PPOAlgorithm(policy)
        algo.reward_shaper = lambda **kw: kw["public_reward"]
        with torch.no_grad():
            policy.critic.weight.zero_()
            policy.critic.bias.fill_(3.)
        buf, metrics = algo.collect(env, 2, seed=17)
        self.assertEqual(metrics["physics_ticks"], 3)
        self.assertEqual(buf.rewards, [2., 1.])
        self.assertEqual(buf.dones, [0., 1.])
        self.assertEqual(buf.next_values[-1], 3.)
        self.assertEqual(metrics["success_count"], 0)
        self.assertTrue(np.isfinite(algo.update(buf)["loss"]))
        algo.bootstrap_timeouts = False
        self.assertEqual(algo.collect(env, 2, seed=17)[0].next_values[-1], 0.)
        policy.action_spec = ActionSpec(frequency_hz=40)
        with self.assertRaises(ValueError):
            policy_action_hold_steps(env, policy)

    def test_bc_epoch_resume_next_update_optimizer_and_rng(self):
        batch = self.batch(8)
        payload = {**batch, "episode_id": np.repeat([0, 1], 4), "step_id": np.tile(np.arange(4), 2),
                   "terminated": np.tile([False, False, False, True], 2), "truncated": np.zeros(8, bool)}
        dataset = CanonicalDataset(payload, self.obs_spec, self.action_spec, validate=False)
        model = self.policy()
        loader = make_dataloader(dataset, batch_size=2, shuffle=True)
        trainer = BCTrainer(model)
        trainer.fit(loader, epochs=1)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "training.pt")
            save_training(path, model, {"kind": "bc"}, {"epoch": 1})
            expected = trainer.fit(loader, epochs=2, start_epoch=1, start_step=4)
            expected_state = model.state_dict()
            expected_opt = model.opt.state_dict()
            expected_rng = capture_rng()
            other = self.policy()
            new_loader = make_dataloader(dataset, batch_size=2, shuffle=True)
            progress = load_training(path, other, {"kind": "bc"})
            actual = BCTrainer(other).fit(new_loader, epochs=2, start_epoch=progress["epoch"], start_step=4)
            self.assertEqual(actual, expected)
            self.assert_tree_equal(expected_state, other.state_dict())
            self.assert_tree_equal(expected_opt, other.opt.state_dict())
            self.assert_tree_equal(expected_rng, capture_rng())
            with self.assertRaises(ValueError):
                load_training(path, other, {"kind": "changed"})
            Path(path).write_bytes(b"incomplete")
            with self.assertRaises(ValueError):
                load_training(path, other, {"kind": "bc"})

    def test_ppo_minibatch_resume_next_update(self):
        policy = self.policy()
        algo = PPOAlgorithm(policy, minibatch_size=2)
        batch = self.batch()
        image = policy.processor(batch["image"])
        proprio = torch.from_numpy(batch["proprio"])
        norm = torch.zeros((4, 6))
        with torch.no_grad():
            logp, _, values = policy.evaluate_actions({"image": image, "proprio": proprio}, norm)
        data = {"image": image, "proprio": proprio, "norm_action": norm, "log_prob": logp,
                "value": values, "reward": torch.arange(4).float(), "done": torch.tensor([0., 1., 0., 1.]),
                "next_value": torch.zeros(4)}
        algo.update(data)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "ppo.pt")
            save_training(path, policy, {"kind": "ppo"}, {"update": 1})
            expected = algo.update(data)
            state, opt, rng = policy.state_dict(), policy.opt.state_dict(), capture_rng()
            restored = self.policy()
            next_algo = PPOAlgorithm(restored, minibatch_size=2)
            load_training(path, restored, {"kind": "ppo"})
            self.assertEqual(expected, next_algo.update(data))
            self.assert_tree_equal(state, restored.state_dict())
            self.assert_tree_equal(opt, restored.opt.state_dict())
            self.assert_tree_equal(rng, capture_rng())

    def test_bc_entrypoint_resume_matches_continuous_run(self):
        payload = {**self.batch(8), "timestamp": np.tile(np.arange(4) / 30, 2),
                   "episode_id": np.repeat([0, 1], 4), "step_id": np.tile(np.arange(4), 2),
                   "terminated": np.tile([False, False, False, True], 2), "truncated": np.zeros(8, bool)}
        loaded = (payload, None, self.obs_spec, self.action_spec)
        with tempfile.TemporaryDirectory() as directory, patch.object(train_bc, "load_training_data", return_value=loaded):
            cfg = Path(directory) / "config.yaml"
            cfg.write_text(yaml.safe_dump({"seed": 7, "device": "cpu", "model": {"hidden_dim": 8},
                "data": {"batch_size": 2}, "logging": {"dir": directory, "swanlab": {"mode": "disabled"}}}))
            one, two = str(Path(directory) / "continuous.pt"), str(Path(directory) / "resumed.pt")
            train_bc.main(["--config", str(cfg), "--epochs", "2", "--checkpoint", one])
            train_bc.main(["--config", str(cfg), "--epochs", "1", "--checkpoint", two])
            train_bc.main(["--config", str(cfg), "--epochs", "2", "--checkpoint", two,
                           "--resume", two + ".training.pt"])
            a = torch.load(one + ".training.pt", weights_only=False)
            b = torch.load(two + ".training.pt", weights_only=False)
            self.assert_tree_equal(a, b)

    def test_training_scalar_hook(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = open_writer({"logging": {"dir": directory, "swanlab": {"mode": "disabled"}}})
            with patch.object(writer, "log") as logged:
                log_row(writer, "bc", {"loss": .25}, 4)
                logged.assert_called_once_with({"loss": .25}, 4, prefix="bc")
            writer.close()

    def test_training_rejects_diagnostic_clipping(self):
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            train_bc.load_training_data({"data": {"allow_action_clipping": True}})

    def test_ppo_entrypoint_boundary_resume_matches_continuous_run(self):
        class Env:
            action_spec = ActionSpec(frequency_hz=60)
            observation_spec = ObservationSpec(control_hz=60)

            def reset(self, seed=None):
                self.ticks = 0
                return self.obs(), {}

            def obs(self):
                return Observation(np.zeros((64, 64, 3), np.uint8), np.zeros(21, np.float32))

            def step(self, action):
                self.ticks += 1
                return self.obs(), float(action[0]) / 100, self.ticks == 3, False, {"success": False}

            def get_privileged_state(self):
                return None

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory, patch.object(train_rl, "make_env", side_effect=lambda _: Env()):
            cfg = Path(directory) / "config.yaml"
            cfg.write_text(yaml.safe_dump({"seed": 7, "device": "cpu", "model": {"hidden_dim": 8},
                "ppo": {"policy_hz": 30, "minibatch_size": 2, "rollout_steps": 2},
                "logging": {"dir": directory, "swanlab": {"mode": "disabled"}}}))
            one, two = str(Path(directory) / "continuous.pt"), str(Path(directory) / "resumed.pt")
            train_rl.main(["--config", str(cfg), "--num-updates", "2", "--checkpoint", one])
            train_rl.main(["--config", str(cfg), "--num-updates", "1", "--checkpoint", two])
            train_rl.main(["--config", str(cfg), "--num-updates", "2", "--checkpoint", two,
                           "--resume", two + ".training.pt"])
            a = torch.load(one + ".training.pt", weights_only=False)
            b = torch.load(two + ".training.pt", weights_only=False)
            self.assert_tree_equal(a, b)


if __name__ == "__main__":
    unittest.main()
