"""Scalar-only SwanLab tracking; local evidence and model state stay independent."""
from contextlib import contextmanager
import importlib
from importlib.metadata import version
import json
import math
from numbers import Real
import os
from pathlib import Path
import random
import re
import signal
import sys
import threading
import warnings


def _load_sdk():
    sdk = importlib.import_module("swanlab")
    if version("swanlab") != "0.10.3":
        raise RuntimeError("Scalar tracking privacy settings require the tested swanlab==0.10.3")
    return sdk


@contextmanager
def preserve_rng():
    """Logging must not advance the already-loaded scientific RNGs or initialize CUDA."""
    python = random.getstate()
    numpy, torch = sys.modules.get("numpy"), sys.modules.get("torch")
    np_state = numpy.random.get_state() if numpy is not None else None
    cpu = torch.get_rng_state() if torch is not None else None
    cuda = torch.cuda.get_rng_state_all() if torch is not None and torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python)
        if np_state is not None:
            numpy.random.set_state(np_state)
        if cpu is not None:
            torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def safe_config(config, stage):
    """Allowlist rather than redact a full config (which may contain private paths)."""
    result = {"stage": stage, "scalar_only": True}
    sections = {
        "": ("seed",),
        "env": ("id", "camera", "image_height", "image_width", "control_hz", "max_episode_steps"),
        "model": ("type", "hidden_dim"),
        "train": ("epochs", "learning_rate"),
        "data": ("batch_size",),
        "ppo": ("policy_hz", "rollout_steps", "num_updates", "learning_rate"),
        "eval": ("num_episodes", "deterministic", "privileged_teacher", "oracle_actuator_assisted"),
        "policy": ("type", "chunk_size", "n_action_steps", "num_inference_steps"),
    }
    for section, keys in sections.items():
        source = config.get(section, {}) if section else config
        for key in keys:
            value = source.get(key)
            name = f"{section}.{key}" if section else key
            if isinstance(value, Real) and math.isfinite(value):
                result[name] = value.item() if hasattr(value, "item") else value
            elif isinstance(value, str) and re.fullmatch(r"[\w.:-]{1,100}", value):
                result[name] = value
    return result


def scalar_metrics(values, prefix=""):
    """Flatten numeric metrics only. Paths, media and arbitrary objects never pass."""
    result = {}
    for key, value in values.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_/-]*", key):
            continue
        if any(word in key.lower() for word in ("token", "password", "secret", "api_key", "hostname", "path")):
            continue
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(scalar_metrics(value, name))
        elif isinstance(value, Real) and math.isfinite(value):
            result[name] = value.item() if hasattr(value, "item") else value
    return result


class ScalarWriter:
    def __init__(self, config, stage, *, start_step=0, resumed=False):
        options = config.get("logging", {}).get("swanlab", {})
        self.mode = os.environ.get("LEROBOT_SWANLAB_MODE", options.get("mode", "offline"))
        if self.mode not in ("offline", "online", "disabled"):
            raise ValueError("SwanLab mode must be offline, online or disabled")
        self.run = None
        self.directory = None
        self.last_steps = {}
        self.start_step = start_step
        self.closed = False
        self.degraded = False
        if self.mode == "disabled":
            return
        project = os.environ.get("LEROBOT_SWANLAB_PROJECT", options.get("project"))
        workspace = os.environ.get("LEROBOT_SWANLAB_WORKSPACE", options.get("workspace"))
        if self.mode == "online" and not project:
            raise ValueError("Online tracking requires an explicit private SwanLab project")
        name = os.environ.get("LEROBOT_SWANLAB_NAME", options.get("name", f"liftcube-{stage}"))
        if not isinstance(name, str) or not name.strip() or len(name) > 128 or any(c in name for c in "\r\n\\"):
            raise ValueError("Invalid SwanLab display name")
        name = name.replace("/", "-", 1)
        with preserve_rng():
            try:
                sdk = _load_sdk()
            except ImportError as exc:
                raise RuntimeError("Install the logging extra (swanlab==0.10.3) before starting this run") from exc
            if sdk.has_run():
                raise RuntimeError("A SwanLab run is already active; do not replace another writer")
            # Explicitly disable every probe, terminal capture and inherited webhook.
            settings = sdk.Settings(mode=self.mode, interactive=False, terminal={"proxy_type": "none"},
                project={"workspace": workspace, "public": False},
                probe={key: False for key in ("hardware", "runtime", "requirements", "conda", "git", "swanlab", "monitor")},
                core={"skip_store": False},
                run={"id": None, "resume": "never", "parallel": "none", "history": None, "config": None},
                experiment={"description": None, "tags": [], "group": None, "job_type": None},
                integration={"webhook": {"url": "", "value": ""}})
            root = Path(config.get("logging", {}).get("dir", "outputs")) / "swanlab"
            root.mkdir(parents=True, exist_ok=True)
            segment = 1
            while True:
                self.directory = root / f"segment-{segment:04d}"
                try:
                    self.directory.mkdir()
                    break
                except FileExistsError:
                    segment += 1
            display_name = name if segment == 1 else f"{name}-segment{segment}"
            self.receipt = dict(mode=self.mode, name=display_name, segment=segment,
                start_step=int(start_step) if start_step is not None else None,
                resumed=bool(resumed), config=safe_config(config, stage),
                status="initializing", scalar_only=True)
            self._receipt()
            hook, interrupt = sys.excepthook, signal.getsignal(signal.SIGINT)
            try:
                self.run = sdk.init(mode=self.mode, project=project or "LeRobot-offline", workspace=workspace,
                    name=display_name, log_dir=str(self.directory / "sdk"), public=False, reinit=False,
                    config=self.receipt["config"], settings=settings)
            except Exception:
                if sdk.has_run():
                    try:
                        sdk.get_run().finish(state="crashed", error="Tracking initialization failed.")
                    except Exception:
                        pass
                self.receipt["status"] = "initialization_failed"
                self._receipt()
                raise RuntimeError("SwanLab initialization failed; inspect local SDK diagnostics before starting work") from None
            finally:
                # SDK signal/excepthook handlers upload raw tracebacks, even with
                # terminal capture disabled. Our callers close with a generic status.
                sys.excepthook = hook
                if threading.current_thread() is threading.main_thread():
                    signal.signal(signal.SIGINT, interrupt)
            self.receipt.update(status="running", run_id=self.run.id)
            try:
                self._receipt()
            except OSError:
                self._warn()

    def _receipt(self):
        target = self.directory / "tracking.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    def _warn(self):
        if not self.degraded:
            warnings.warn("SwanLab recording degraded; preserve local experiment evidence and inspect tracking.json", RuntimeWarning)
        self.degraded = True

    def log(self, values, step, *, prefix=""):
        if self.closed:
            raise RuntimeError("Tracking writer is closed")
        if self.mode == "disabled":
            return
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Metric step must be a nonnegative integer")
        if self.start_step is not None and step < self.start_step:
            raise ValueError("Metric step predates the restored checkpoint")
        metrics = scalar_metrics(values, prefix)
        if not metrics:
            return
        if any(step <= self.last_steps.get(key, -1) for key in metrics):
            raise ValueError("Metric steps must strictly increase per key within a segment")
        self.last_steps.update({key: step for key in metrics})
        # Journal first. SDK/cloud failures cannot interrupt a healthy worker.
        try:
            with (self.directory / "scalars.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(step=step, metrics=metrics), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            self._warn()
        with preserve_rng():
            try:
                self.run.log(metrics, step=step)
            except Exception:
                self._warn()

    def close(self):
        failed = sys.exc_info()[0] is not None
        if self.closed or self.mode == "disabled":
            self.closed = True
            return
        self.closed = True
        with preserve_rng():
            try:
                if failed:
                    self.run.finish(state="crashed", error="Execution interrupted; inspect local experiment evidence.")
                else:
                    self.run.finish()
            except Exception:
                self._warn()
        self.receipt.update(status="closed_degraded" if self.degraded else "interrupted" if failed else "closed",
                            last_steps=self.last_steps)
        try:
            self._receipt()
        except OSError:
            self._warn()


def log_episode(writer, records, *, hz):
    """Publish only a completed prefix, never count a partial as a failed episode."""
    record = records[-1]
    writer.log(dict(success=record.success, success_rate_completed_prefix=sum(r.success for r in records)/len(records),
        completed_episodes=len(records), env_seed=record.seed, length=record.length,
        return_sum=record.return_sum, simulated_seconds=record.length/hz,
        terminated=record.terminated, truncated=record.truncated), len(records), prefix="eval")
