"""Local backend descriptors and explicit native joint representations."""
import json
from pathlib import Path

import numpy as np

from starter.interfaces.action import JOINT_NAMES, motor_to_degrees
from starter.interfaces.observation import ObservationSpec
from .joint_chunk import execution_spec, front_inputs


def read_adapter(path, backend):
    path = Path(path).resolve()
    options = json.loads(path.read_text(encoding='utf-8'))
    if options['backend'] != backend or options['joint_names'] != list(JOINT_NAMES):
        raise ValueError('Wrong backend or SO101 joint order')
    for key in ('model_revision', 'backend_revision', 'calibration_source'):
        if not isinstance(options.get(key), str) or not options[key].strip():
            raise ValueError(f'Record {key}')
    model = (path.parent / options['model_path']).resolve()
    if not model.is_dir():
        raise ValueError('model_path must be an existing local directory')
    if options['joint_units'] not in ('motor', 'arm_degrees_gripper_motor'):
        raise ValueError('Unknown joint representation; EEF policies are not SO101 joint policies')
    execution_spec(task=options['task'], execute_steps=options['execute_steps'],
        frequency_hz=options['frequency_hz'], output_units=options['joint_units'],
        observation_spec=ObservationSpec.from_dict(options['observation_spec']))
    return path, options, model


def native_front(observation, spec, task, image_key, units):
    batch = front_inputs(observation, spec, task, image_key)
    if units == 'arm_degrees_gripper_motor':
        motors = batch['observation.state']
        state = motor_to_degrees(motors)
        state[5] = motors[5]
        batch['observation.state'] = state
    elif units != 'motor':
        raise ValueError('Unsupported input joint units')
    return batch


def tensor_chunk(output):
    """Unbatch a native tensor; never trim padded joints at our boundary."""
    array = output.detach().float().cpu().numpy()  # NumPy has no native bfloat16 dtype.
    if array.ndim != 3 or array.shape[0] != 1 or array.shape[-1] != 6:
        raise ValueError('Expected native [1, time, 6] output')
    return array[0]


def require_complete_load(info):
    if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        error = ValueError('Incomplete backend weight load; inspect exception.loading_info')
        error.loading_info = info
        raise error
