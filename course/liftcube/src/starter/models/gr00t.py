"""Single-front GR00T N1.5 adapter using native modality transforms/statistics."""
import json

import numpy as np

from starter.interfaces.observation import ObservationSpec
from .joint_chunk import JointChunkPolicy
from .backend_io import native_front, read_adapter, require_complete_load


def load_groot_weights(model_class, model_path, device):
    import torch
    from transformers import PreTrainedModel
    # N1.5 overrides from_pretrained and assumes a model-only return. Call the
    # HF base loader to obtain loading_info without downloading or retuning.
    loaded, info = PreTrainedModel.from_pretrained.__func__(
        model_class, model_path, local_model_path=model_path, torch_dtype=torch.bfloat16,
        local_files_only=True, output_loading_info=True)
    require_complete_load(info)
    return loaded.eval().requires_grad_(False).to(device)


class Gr00tFrontPolicy(JointChunkPolicy):
    def __init__(self, policy, *, task, image_key='video.webcam', joint_units='motor',
                 observation_spec=None, execute_steps=16, frequency_hz=30):
        self.policy, self.image_key, self.joint_units = policy, image_key, joint_units
        modality = policy.get_modality_config()
        expected = {'video': [image_key], 'state': ['state.single_arm', 'state.gripper'],
                    'action': ['action.single_arm', 'action.gripper'],
                    'language': ['annotation.human.task_description']}
        for key, names in expected.items():
            if list(modality[key].modality_keys) != names:
                raise ValueError('GR00T modality keys must match the trained single-front SO101 contract')
        if any(list(modality[key].delta_indices) != [0] for key in ('video', 'state', 'language')):
            raise ValueError('Historical observations are not available in this adapter')
        self.horizon = len(modality['action'].delta_indices)
        if list(modality['action'].delta_indices) != list(range(self.horizon)):
            raise ValueError('GR00T actions must use consecutive decision steps')
        super().__init__(self._predict, task=task, observation_spec=observation_spec, execute_steps=execute_steps,
                         frequency_hz=frequency_hz, output_units=joint_units,
                         reset_backend=getattr(policy, 'reset', None))

    def _predict(self, observation, task):
        import torch
        data = native_front(observation, self.observation_spec, task, self.image_key, self.joint_units)
        q = data['observation.state']
        payload = {self.image_key: observation.image.copy()[None],
                   'state.single_arm': q[None, :5], 'state.gripper': q[None, 5:],
                   'annotation.human.task_description': [task]}
        with torch.inference_mode():
            output = self.policy.get_action(payload)  # Native transforms unnormalize outputs.
        arm, jaw = np.asarray(output['action.single_arm']), np.asarray(output['action.gripper'])
        if arm.shape != (self.horizon, 5) or jaw.shape != (self.horizon, 1):
            raise ValueError('Expected unbatched [horizon,5] arm and [horizon,1] jaw')
        return np.concatenate((arm, jaw), axis=-1)

    @classmethod
    def load(cls, path, device='cpu'):
        path, options, model = read_adapter(path, 'gr00t-n1.5')
        metadata = json.loads((model / 'experiment_cfg/metadata.json').read_text(encoding='utf-8'))
        tag = options['embodiment_tag']
        if tag not in metadata:
            raise ValueError('Checkpoint lacks trained SO101 embodiment metadata')
        from transformers.utils.hub import is_offline_mode
        if not is_offline_mode():
            raise ValueError('Set HF_HUB_OFFLINE=1 before importing the backend; pre-cache backbone assets')
        from gr00t.experiment.data_config import load_data_config
        from gr00t.model.policy import Gr00tPolicy
        from gr00t.model.gr00t_n1 import GR00T_N1_5
        cfg = load_data_config(options['data_config'])
        if len(cfg.video_keys) != 1:
            raise ValueError('Do not use a dual-camera data config for single-front inference')
        raw = json.loads((model / 'config.json').read_text(encoding='utf-8'))
        if raw['action_head_cfg']['action_horizon'] != len(cfg.action_indices):
            raise ValueError('Action horizon mismatch; do not recreate a random action head')
        class CheckedPolicy(Gr00tPolicy):
            def _load_model(self, model_path):
                self.model = load_groot_weights(GR00T_N1_5, model_path, self.device)
        policy = CheckedPolicy(model_path=str(model), embodiment_tag=tag, modality_config=cfg.modality_config(),
                               modality_transform=cfg.transform(), device=device)
        return cls(policy, task=options['task'], image_key=cfg.video_keys[0], joint_units=options['joint_units'],
                   observation_spec=ObservationSpec.from_dict(options['observation_spec']),
                   execute_steps=options['execute_steps'], frequency_hz=options['frequency_hz'])
