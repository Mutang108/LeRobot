"""Single-front SmolVLA inference adapter for the pinned LeRobot v0.4.2 API."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from starter.interfaces.action import JOINT_NAMES
from starter.interfaces.observation import ObservationSpec
from .joint_chunk import JointChunkPolicy, execution_spec, front_inputs
from .backend_io import tensor_chunk


def single_front_config(config, image_key):
    """Keep one real image and request backend-generated masked empty slots."""
    config = deepcopy(config)
    if config.input_features['observation.state'].shape != (6,) or config.output_features['action'].shape != (6,):
        raise ValueError('SmolVLA checkpoint must use six state/action channels')
    image_keys = list(config.image_features)
    if not image_keys or image_key != image_keys[0]:
        raise ValueError('Front must bind to the first trained camera slot')
    if config.n_obs_steps != 1 or config.adapt_to_pi_aloha or config.use_delta_joint_actions_aloha:
        raise ValueError('Only single-observation, absolute SO101 joint control is supported')
    if getattr(config, 'rtc_config', None) is not None:
        raise ValueError('RTC is not supported by this synchronous adapter')
    config.input_features = {key: feature for key, feature in config.input_features.items()
                             if key not in image_keys or key == image_key}
    config.empty_cameras = len(image_keys) - 1
    return config


def normalization_stats(document):
    """Require explicit six-channel checkpoint/training statistics, not identity fallback."""
    if not isinstance(document.get('source'), str) or not document['source'].strip():
        raise ValueError('Normalization statistics need a recorded source')
    result = {}
    for key in ('observation.state', 'action'):
        result[key] = {}
        for field in ('mean', 'std'):
            value = np.asarray(document['stats'][key][field], dtype=np.float32)
            if value.shape != (6,) or not np.isfinite(value).all():
                raise ValueError('Expected finite six-channel mean/std statistics')
            if field == 'std' and np.any(value <= 0):
                raise ValueError('Standard deviation must be positive')
            result[key][field] = value
    return result


class SmolVLAFrontPolicy(JointChunkPolicy):
    """Backend-injected constructor supports CPU fixtures without model downloads."""

    def __init__(self, policy, preprocessor, postprocessor, *, task, image_key,
                 observation_spec=None, execute_steps=20, frequency_hz=30):
        self.policy, self.preprocessor, self.postprocessor = policy, preprocessor, postprocessor
        self.image_key = image_key
        super().__init__(self._predict, task=task, observation_spec=observation_spec,
                         execute_steps=execute_steps, frequency_hz=frequency_hz,
                         reset_backend=policy.reset)

    def _predict(self, observation, task):
        import torch
        batch = front_inputs(observation, self.observation_spec, task, self.image_key)
        batch = {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                 for key, value in batch.items()}
        with torch.inference_mode():
            output = self.postprocessor(self.policy.predict_action_chunk(self.preprocessor(batch)))
        actions = tensor_chunk(output)
        if len(actions) != self.policy.config.n_action_steps:
            raise ValueError('SmolVLA output horizon differs from its loaded configuration')
        return actions

    @classmethod
    def load(cls, path, device='cpu'):
        """Load a local adapter JSON and trusted local weights; never download implicitly."""
        path = Path(path).resolve()
        options = json.loads(path.read_text(encoding='utf-8'))
        if options['joint_names'] != list(JOINT_NAMES) or options['joint_units'] != 'motor':
            raise ValueError('Adapter must explicitly match SO101 joint order and motor units')
        for key in ('model_revision', 'calibration_source'):
            if not isinstance(options.get(key), str) or not options[key].strip():
                raise ValueError(f'Record {key}')
        spec = execution_spec(task=options['task'], execute_steps=options['execute_steps'],
            frequency_hz=options['frequency_hz'],
            observation_spec=ObservationSpec.from_dict(options['observation_spec']))
        model = (path.parent / options['model_path']).resolve()
        vlm = (path.parent / options['vlm_path']).resolve()
        if not model.is_dir() or not vlm.is_dir():
            raise ValueError('model_path and vlm_path must be existing local directories')
        stats = normalization_stats(options['normalization'])
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
        cfg = PreTrainedConfig.from_pretrained(str(model), local_files_only=True)
        if not isinstance(cfg, SmolVLAConfig):
            raise ValueError('Expected a SmolVLA checkpoint')
        cfg = single_front_config(cfg, options['image_key'])
        for key, expected in (('STATE', 'MEAN_STD'), ('ACTION', 'MEAN_STD'), ('VISUAL', 'IDENTITY')):
            mode = cfg.normalization_mapping[key]
            if getattr(mode, 'value', mode) != expected:
                raise ValueError('Expected state/action MEAN_STD and visual IDENTITY normalization')
        cfg.device, cfg.load_vlm_weights, cfg.vlm_model_name = device, False, str(vlm)
        policy = SmolVLAPolicy.from_pretrained(str(model), config=cfg, local_files_only=True, strict=True)
        policy.eval()
        pre, post = make_smolvla_pre_post_processors(cfg, dataset_stats={
            key: {name: torch.from_numpy(value) for name, value in fields.items()}
            for key, fields in stats.items()})
        return cls(policy, pre, post, task=options['task'], image_key=options['image_key'],
                   observation_spec=spec,
                   execute_steps=options['execute_steps'], frequency_hz=options['frequency_hz'])
