"""Shared single-front inputs and bounded joint-chunk execution, without GT."""
from collections import deque
from dataclasses import replace
from numbers import Real

import numpy as np

from starter.interfaces.action import ActionSpec, degrees_to_motor, validate_action
from starter.interfaces.observation import ObservationSpec


def validate_front_spec(spec):
    if (spec.camera != 'front' or spec.proprio_dim not in (18, 21)
            or spec.image_layout != 'HWC' or spec.image_dtype != 'uint8'
            or len(spec.image_shape) != 3 or spec.image_shape[-1] != 3
            or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in spec.image_shape)):
        raise ValueError('Expected single front and the canonical 18/21-D proprio layout')
    if spec.proprio_layout != ObservationSpec().proprio_layout:
        raise ValueError('Unknown proprio ordering')


def execution_spec(*, task, execute_steps, frequency_hz, observation_spec=None, output_units='motor'):
    """Validate cheap execution contracts before any heavyweight model loading."""
    if isinstance(execute_steps, bool) or not isinstance(execute_steps, int) or execute_steps < 1:
        raise ValueError('execute_steps must be a positive integer')
    if not isinstance(task, str) or not task.strip():
        raise ValueError('A nonempty task is required')
    if (isinstance(frequency_hz, bool) or not isinstance(frequency_hz, Real)
            or not np.isfinite(frequency_hz) or frequency_hz <= 0):
        raise ValueError('frequency_hz must be finite and positive')
    if output_units not in ('motor', 'arm_degrees_gripper_motor'):
        raise ValueError('Unsupported action units')
    spec = observation_spec or replace(ObservationSpec(), control_hz=float(frequency_hz))
    validate_front_spec(spec)
    if spec.control_hz != float(frequency_hz):
        raise ValueError('Observation and action decision clocks must match')
    return spec


def front_inputs(observation, spec, task, image_key):
    """Return only RGB, measured q and caller-supplied language, never metadata."""
    validate_front_spec(spec)
    observation.validate(spec)
    if not isinstance(task, str) or not task.strip():
        raise ValueError('A nonempty task is required')
    return {
        image_key: np.ascontiguousarray(observation.image.transpose(2, 0, 1), dtype=np.float32) / 255,
        'observation.state': observation.proprio[:6].copy(),
        'task': task,
    }


def joint_chunk_to_motor(chunk, units='motor'):
    """Decode explicit SO101 units; never truncate dimensions or clip targets."""
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] != 6:
        raise ValueError('Expected a nonempty [time, 6] action chunk')
    if not np.isfinite(chunk).all():
        raise ValueError('Action chunk contains nonfinite values')
    if units == 'arm_degrees_gripper_motor':
        converted = degrees_to_motor(chunk)
        converted[:, 5] = chunk[:, 5]  # Gripper is not an angle in this contract.
        chunk = converted
    elif units != 'motor':
        raise ValueError('Unsupported action units; end-effector commands require another adapter')
    return validate_action(chunk, ActionSpec()).copy()


class JointChunkPolicy:
    """Inference-only queue shared by explicit joint-space VLA backends.

    predict(observation, task) returns raw, *unnormalized* [time, 6] targets.
    reset_backend must clear backend caches, not reseed random streams.
    """

    def __init__(self, predict, *, task, observation_spec=None, execute_steps=20,
                 frequency_hz=30, output_units='motor', reset_backend=None):
        self.observation_spec = execution_spec(task=task, execute_steps=execute_steps,
            frequency_hz=frequency_hz, observation_spec=observation_spec, output_units=output_units)
        if not callable(predict) or (reset_backend is not None and not callable(reset_backend)):
            raise TypeError('predict and optional reset_backend must be callable')
        self.predict = predict
        self.task = task
        self.action_spec = replace(ActionSpec(), frequency_hz=float(frequency_hz))
        self.execute_steps = execute_steps
        self.output_units = output_units
        self.reset_backend = reset_backend
        self._actions = deque()
        self.last_raw_chunk = None
        self.last_motor_chunk = None

    def reset(self):
        self._actions.clear()
        self.last_raw_chunk = self.last_motor_chunk = None
        if self.reset_backend is not None:
            self.reset_backend()

    def set_task(self, task):
        if not isinstance(task, str) or not task.strip():
            raise ValueError('A nonempty task is required')
        if task != self.task:
            self.reset()
            self.task = task

    def act(self, observation, deterministic=False):
        # deterministic is a common runner argument, not permission to reseed a VLA.
        try:
            observation.validate(self.observation_spec)
            if not self._actions:
                raw = np.asarray(self.predict(observation, self.task), dtype=np.float32)
                motor = joint_chunk_to_motor(raw, self.output_units)
                if len(motor) < self.execute_steps:
                    raise ValueError('Predicted chunk is shorter than execute_steps')
                self.last_raw_chunk, self.last_motor_chunk = raw.copy(), motor.copy()
                self._actions.extend(row.copy() for row in motor[:self.execute_steps])
            return self._actions.popleft().copy()
        except Exception:
            self.reset()
            raise
