"""Scalar tracking checks: synthetic CPU fixtures, no cloud login or simulator."""
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "course/liftcube/src"))
from starter.logging.swanlab import ScalarWriter, safe_config, scalar_metrics
from starter.scripts.train_lerobot import run_with_swanlab


class FakeSDK:
    def __init__(self):
        self.calls, self.points, self.finishes = [], [], []
        self.active, self.fail_log = False, False

    def Settings(self, **kwargs):
        return kwargs

    def has_run(self):
        return self.active

    def get_run(self):
        return self.run

    def init(self, **kwargs):
        self.calls.append(kwargs)
        self.active = True
        random.random()
        if "numpy" in sys.modules:
            sys.modules["numpy"].random.rand()
        if "torch" in sys.modules:
            sys.modules["torch"].rand(1)
        sys.excepthook = lambda *args: None
        signal.signal(signal.SIGINT, lambda *args: None)
        self.run = SimpleNamespace(id=f"fake{len(self.calls)}", log=self.log, finish=self.finish)
        return self.run

    def log(self, values, *, step):
        random.random()
        if self.fail_log:
            raise OSError("Synthetic tracking outage")
        self.points.append((step, values))

    def finish(self, **kwargs):
        random.random()
        self.finishes.append(kwargs)
        self.active = False


class SwanLabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.sdk = FakeSDK()
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {k: v for k, v in os.environ.items()
                               if not k.startswith("LEROBOT_SWANLAB_")}, clear=True).start()
        patch("starter.logging.swanlab._load_sdk", return_value=self.sdk).start()
        self.cfg = dict(seed=17, logging=dict(dir=self.temp.name))

    def test_allowlist_and_numeric_only(self):
        config = dict(seed=17, env=dict(id="LiftCube", camera="front"),
                      data=dict(path="/private/dataset", batch_size=8),
                      model=dict(type="/private/model"), api_key="secret")
        self.assertEqual(safe_config(config, "bc"), dict(stage="bc", scalar_only=True,
            seed=17, **{"env.id": "LiftCube", "env.camera": "front", "data.batch_size": 8}))
        self.assertEqual(scalar_metrics(dict(loss=.1, bad=float("nan"), image=[1, 2],
            checkpoint_path=3, api_key=123, nested=dict(rate=.5))), {"loss": .1, "nested/rate": .5})

    def test_disabled_is_import_and_file_free(self):
        self.cfg["logging"]["swanlab"] = dict(mode="disabled")
        writer = ScalarWriter(self.cfg, "check")
        writer.log(dict(loss=.1), 1)
        writer.close()
        self.assertFalse(self.sdk.calls)
        self.assertFalse(list(Path(self.temp.name).iterdir()))

    def test_online_requires_project_and_never_replaces_active_run(self):
        with patch.dict(os.environ, LEROBOT_SWANLAB_MODE="online"):
            with self.assertRaisesRegex(ValueError, "explicit private"):
                ScalarWriter(self.cfg, "test")
        self.sdk.active = True
        with self.assertRaisesRegex(RuntimeError, "already active"):
            ScalarWriter(self.cfg, "test")
        self.assertFalse(self.sdk.finishes)

    def test_settings_steps_segments_and_hooks(self):
        hook, interrupt = sys.excepthook, signal.getsignal(signal.SIGINT)
        self.cfg["logging"]["swanlab"] = dict(name="训练/ACT")
        writer = ScalarWriter(self.cfg, "bc")
        self.assertIs(sys.excepthook, hook)
        self.assertIs(signal.getsignal(signal.SIGINT), interrupt)
        kwargs = self.sdk.calls[0]
        self.assertFalse(kwargs["public"])
        self.assertFalse(kwargs["reinit"])
        self.assertEqual(kwargs["name"], "训练-ACT")
        settings = kwargs["settings"]
        self.assertFalse(any(settings["probe"].values()))
        self.assertEqual(settings["terminal"]["proxy_type"], "none")
        self.assertEqual(settings["run"]["resume"], "never")
        self.assertEqual(settings["integration"]["webhook"]["url"], "")
        writer.log(dict(loss=.25), 4, prefix="bc")
        writer.log(dict(success=True), 1, prefix="eval")
        with self.assertRaises(ValueError):
            writer.log(dict(loss=.3), 4, prefix="bc")
        writer.close()
        writer.close()
        resumed = ScalarWriter(self.cfg, "bc", start_step=4, resumed=True)
        with self.assertRaisesRegex(ValueError, "predates"):
            resumed.log(dict(loss=.1), 3, prefix="bc")
        resumed.log(dict(loss=.1), 5, prefix="bc")
        resumed.close()
        self.assertEqual(self.sdk.calls[1]["name"], "训练-ACT-segment2")
        receipt = json.loads((resumed.directory / "tracking.json").read_text(encoding="utf-8"))
        self.assertEqual((receipt["start_step"], receipt["resumed"]), (4, True))
        points = [json.loads(line) for line in (writer.directory / "scalars.jsonl").read_text().splitlines()]
        self.assertEqual(points[0], dict(step=4, metrics={"bc/loss": .25}))

    def test_outage_preserves_journal_and_interrupted_status(self):
        writer = ScalarWriter(self.cfg, "bc")
        self.sdk.fail_log = True
        with self.assertWarns(RuntimeWarning):
            writer.log(dict(loss=.25), 4)
        try:
            raise RuntimeError("Private exception/path must not reach tracking")
        except RuntimeError:
            writer.close()
        self.assertEqual(self.sdk.finishes[-1]["state"], "crashed")
        self.assertNotIn("Private", self.sdk.finishes[-1]["error"])
        self.assertEqual(json.loads((writer.directory / "scalars.jsonl").read_text())["step"], 4)
        receipt = json.loads((writer.directory / "tracking.json").read_text())
        self.assertEqual(receipt["status"], "closed_degraded")

    def test_logging_does_not_advance_loaded_rngs(self):
        numpy = __import__("numpy") if importlib.util.find_spec("numpy") else None
        torch = __import__("torch") if importlib.util.find_spec("torch") else None
        python = random.getstate()
        np_state = numpy.random.get_state() if numpy is not None else None
        cpu = torch.get_rng_state().clone() if torch is not None else None
        writer = ScalarWriter(self.cfg, "rng")
        writer.log(dict(loss=.5), 1)
        writer.close()
        self.assertEqual(python, random.getstate())
        if numpy is not None:
            after = numpy.random.get_state()
            self.assertEqual(np_state[0], after[0])
            numpy.testing.assert_array_equal(np_state[1], after[1])
            self.assertEqual(np_state[2:], after[2:])
        if torch is not None:
            self.assertTrue(torch.equal(cpu, torch.get_rng_state()))

    def test_native_training_hook_routes_scalars_and_restores_on_error(self):
        cfg = SimpleNamespace(seed=17, policy=SimpleNamespace(type="act", chunk_size=50),
            batch_size=8, job_name="训练-ACT", output_dir=self.temp.name, resume=True,
            wandb=SimpleNamespace(enable=False, project=None))
        original = object()

        def train(config):
            self.assertTrue(config.wandb.enable)
            logger = trainer.WandBLogger(config)
            logger.log_dict(dict(loss=.1, image=[1], path="private"), 100)
            logger.log_dict(dict(success_rate=.5), 100, mode="eval")
            logger.log_video("private.mp4", 100)
            logger.log_policy("private-model")
            raise RuntimeError("Synthetic stop")

        trainer = SimpleNamespace(WandBLogger=original, train=train)
        with self.assertRaisesRegex(RuntimeError, "Synthetic stop"):
            run_with_swanlab(trainer, cfg)
        self.assertIs(trainer.WandBLogger, original)
        self.assertFalse(cfg.wandb.enable)
        self.assertIsNone(cfg.wandb.project)
        self.assertEqual(self.sdk.points, [(100, {"train/loss": .1}), (100, {"eval/success_rate": .5})])
        self.assertEqual(self.sdk.finishes[-1]["state"], "crashed")

    @unittest.skipUnless(all(importlib.util.find_spec(p) for p in ("numpy", "torch", "yaml")),
                         "Evaluation fixture requires CPU application dependencies")
    def test_liftcube_evaluator_logs_completed_prefix_only(self):
        import numpy as np
        from starter.evaluation import evaluate
        from starter.interfaces.action import ActionSpec
        from starter.interfaces.observation import Observation

        class Env:
            name = "synthetic-liftcube"
            action_spec = ActionSpec(frequency_hz=60)

            def reset(self, seed):
                self.tick = 0
                return self.obs(), {}

            def obs(self):
                return Observation(np.zeros((64, 64, 3), np.uint8), np.zeros(21, np.float32))

            def step(self, action):
                self.tick += 1
                return self.obs(), 1., self.tick == 2, False, dict(success=True)

            def close(self):
                pass

        policy = SimpleNamespace(action_spec=ActionSpec(frequency_hz=30), act=lambda *a, **kw: np.zeros(6))
        config = dict(self.cfg, eval=dict(num_episodes=2), env=dict(max_episode_steps=4))
        with patch.object(evaluate, "make_env", return_value=Env()):
            result = evaluate.evaluate_policy(policy, config, shutdown=False)
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual([p[0] for p in self.sdk.points], [1, 2])
        self.assertEqual(self.sdk.points[-1][1]["eval/completed_episodes"], 2)
        self.assertEqual(self.sdk.points[-1][1]["eval/success_rate_completed_prefix"], 1.)
        self.assertAlmostEqual(self.sdk.points[-1][1]["eval/simulated_seconds"], 2/60)
        self.assertEqual(len(self.sdk.finishes), 1)


class RealSDKTests(unittest.TestCase):
    @unittest.skipUnless(all(importlib.util.find_spec(p) for p in ("swanlab", "numpy", "torch")),
                         "Optional SwanLab/CPU application dependencies not installed")
    def test_real_offline_sdk_roundtrip(self):
        # Isolated SDK process: no login, no project creation, no media or simulation.
        script = '''
import json, pathlib, random, signal, sys
import numpy as np
import torch
from starter.logging.swanlab import ScalarWriter
hook, interrupt = sys.excepthook, signal.getsignal(signal.SIGINT)
python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
w = ScalarWriter({'seed':17, 'logging':{'dir':sys.argv[1]}}, 'cpu-check')
assert w.run.mode == 'offline'
assert w.run.config['stage'] == 'cpu-check'
assert sys.excepthook is hook and signal.getsignal(signal.SIGINT) is interrupt
w.log({'loss':.25, 'image':[1,2]}, 4, prefix='bc')
w.log({'success':True}, 1, prefix='eval')
directory = w.run.dir
w.close()
assert not w.degraded
assert list(directory.rglob('*'))
from swanlab.sdk.internal.core_python.store import DataStoreReader
from swanlab.proto.swanlab.record.v1.record_pb2 import Record
reader = DataStoreReader()
reader.open(next(directory.glob('*.swanlab')))
records = [Record.FromString(blob) for blob in reader]
reader.close()
points = [(r.scalar.key, r.scalar.step) for r in records if r.HasField('scalar')]
assert points == [('bc/loss',4), ('eval/success',1)], points
assert not any(r.HasField('media') or r.HasField('log') for r in records)
start = next(r.start for r in records if r.HasField('start'))
assert not start.public and not start.description and not start.tags and not start.group
receipt = json.loads((w.directory/'tracking.json').read_text())
assert receipt['status'] == 'closed'
assert receipt['last_steps'] == {'bc/loss':4, 'eval/success':1}
assert not any(p.name in ('requirements.txt','metadata.json','conda.yml') for p in directory.rglob('*'))
assert random.getstate() == python
after = np.random.get_state()
assert numpy[0] == after[0] and numpy[2:] == after[2:]
np.testing.assert_array_equal(numpy[1], after[1])
assert torch.equal(cpu, torch.get_rng_state())
print('REAL_OFFLINE_OK')
'''
        env = {k: v for k, v in os.environ.items() if not k.startswith(("LEROBOT_SWANLAB_", "SWANLAB_"))}
        env.update(PYTHONPATH=str(ROOT / "course/liftcube/src"), PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="")
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-B", "-c", script, directory], env=env,
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("REAL_OFFLINE_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
