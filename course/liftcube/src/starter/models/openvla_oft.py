"""SO101 continuous OFT adapter; never use vanilla 7-D EEF decoding."""
import json
from types import SimpleNamespace

import numpy as np

from starter.interfaces.observation import ObservationSpec
from .joint_chunk import JointChunkPolicy
from .backend_io import native_front, read_adapter, require_complete_load


def check_statistics(stats):
    for modality in ('action', 'proprio'):
        low, high = (np.asarray(stats[modality][name], dtype=np.float32) for name in ('min', 'max'))
        mask = np.asarray(stats[modality].get('mask', [True]*6))
        if (low.shape != (6,) or high.shape != (6,) or not np.isfinite([low, high]).all()
                or np.any(high <= low) or mask.shape != (6,) or mask.dtype != np.bool_ or not mask.all()):
            raise ValueError('OFT BOUNDS requires finite ordered six-channel ranges and all-channel masks')


def load_local_processor(model_path):
    """Use pinned implementation classes, not checkpoint AutoImageProcessor code."""
    from transformers import AutoTokenizer
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    image = PrismaticImageProcessor.from_pretrained(str(model_path), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    return PrismaticProcessor(image_processor=image, tokenizer=tokenizer)


class OpenVLAOFTFrontPolicy(JointChunkPolicy):
    def __init__(self, model, processor, action_head, proprio_projector, predict_action, *,
                 unnorm_key, task, joint_units, center_crop, chunk_size, observation_spec=None,
                 execute_steps=8, frequency_hz=30):
        if action_head is None or proprio_projector is None:
            raise ValueError('A trained continuous head and proprio projector are required')
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')
        self.chunk_size = chunk_size
        if not isinstance(center_crop, bool):
            raise ValueError('center_crop must be a boolean')
        check_statistics(model.norm_stats[unnorm_key])
        self.model, self.processor = model, processor
        self.action_head, self.proprio_projector = action_head, proprio_projector
        self.predict_action = predict_action
        self.cfg = SimpleNamespace(num_images_in_input=1, use_proprio=True,
                                   unnorm_key=unnorm_key, center_crop=center_crop, use_film=False)
        self.joint_units = joint_units
        super().__init__(self._predict, task=task, observation_spec=observation_spec, execute_steps=execute_steps,
                         frequency_hz=frequency_hz, output_units=joint_units)

    def _predict(self, observation, task):
        data = native_front(observation, self.observation_spec, task, 'front', self.joint_units)
        payload = {'full_image': observation.image.copy(), 'additional_images': [],
                   'state': data['observation.state']}
        output = self.predict_action(self.cfg, self.model, self.processor, payload, task,
                                     action_head=self.action_head, proprio_projector=self.proprio_projector,
                                     use_film=False)
        chunk = np.asarray(output, dtype=np.float32)  # Upstream already unnormalizes actions.
        if chunk.shape != (self.chunk_size, 6):
            raise ValueError('OFT output horizon/dimensions differ from the loaded SO101 contract')
        return chunk

    @classmethod
    def load(cls, path, device='cpu'):
        path, options, model_path = read_adapter(path, 'openvla-oft-so101')
        if not isinstance(options['center_crop'], bool):
            raise ValueError('center_crop must be a boolean')
        if options['trained_camera_keys'] != ['front']:
            raise ValueError('A three-camera checkpoint cannot be relabeled as single-front')
        files = [(path.parent / options[key]).resolve() for key in ('action_head_path', 'proprio_projector_path')]
        if not all(file.is_file() for file in files):
            raise ValueError('Missing trained action head or proprio projector')
        # These constants are selected before importing dependent OFT model modules.
        from prismatic.vla.constants import ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK, ACTION_PROPRIO_NORMALIZATION_TYPE
        if (ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK) != (6, 6, options['chunk_size']):
            raise ValueError('Initialize the pinned OFT backend with SO101 constants before loading')
        if getattr(ACTION_PROPRIO_NORMALIZATION_TYPE, 'value', ACTION_PROPRIO_NORMALIZATION_TYPE) != 'bounds':
            raise ValueError('This adapter supports the SO101 BOUNDS recipe only')
        import torch
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.models.action_heads import L1RegressionActionHead
        from prismatic.models.projectors import ProprioProjector
        from experiments.robot.openvla_utils import get_vla_action
        # Fail before allocating weights on a mismatched device.
        from experiments.robot import openvla_utils
        if str(openvla_utils.DEVICE) != str(torch.device(device)):
            raise ValueError('OFT helper device differs from requested device; use a matching isolated backend')
        model, info = OpenVLAForActionPrediction.from_pretrained(str(model_path), local_files_only=True,
                            torch_dtype=torch.bfloat16, output_loading_info=True)
        require_complete_load(info)
        model.eval().requires_grad_(False).to(device)
        model.vision_backbone.set_num_images_in_input(1)
        stats_file = model_path / 'dataset_statistics.json'
        if stats_file.is_file():
            model.norm_stats = json.loads(stats_file.read_text(encoding='utf-8'))
        check_statistics(model.norm_stats[options['unnorm_key']])
        processor = load_local_processor(model_path)
        head = L1RegressionActionHead(input_dim=model.llm_dim, hidden_dim=model.llm_dim, action_dim=6)
        projector = ProprioProjector(llm_dim=model.llm_dim, proprio_dim=6)
        for component, file in zip((head, projector), files):
            if file.suffix == '.safetensors':
                from safetensors.torch import load_file
                state = load_file(str(file), device='cpu')
            else:
                state = torch.load(file, map_location='cpu', weights_only=True)
            component.load_state_dict(state, strict=True)
            component.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
        return cls(model, processor, head, projector, get_vla_action,
                   unnorm_key=options['unnorm_key'], task=options['task'], joint_units=options['joint_units'],
                   center_crop=options['center_crop'], chunk_size=options['chunk_size'],
                   observation_spec=ObservationSpec.from_dict(options['observation_spec']),
                   execute_steps=options['execute_steps'], frequency_hz=options['frequency_hz'])
