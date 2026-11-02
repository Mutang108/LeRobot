"""SO101 VLASH PI05 with embedded statistics; not the stock HF PI05 graph."""
from copy import deepcopy
import json

from starter.interfaces.observation import ObservationSpec
from .joint_chunk import JointChunkPolicy
from .backend_io import native_front, read_adapter, tensor_chunk

FRONT_KEY = 'observation.images.base_0_rgb'


def single_front_config(config):
    cfg = deepcopy(config)
    if not cfg.state_cond or cfg.input_features['observation.state'].shape != (6,) or cfg.output_features['action'].shape != (6,):
        raise ValueError('Expected state-conditioned, six-channel SO101 PI05')
    keys = list(cfg.image_features)
    if len(keys) != 3 or keys[0] != FRONT_KEY:
        raise ValueError('Expected the original three-slot PI05 camera order')
    cfg.input_features = {k: v for k, v in cfg.input_features.items() if k not in keys or k == FRONT_KEY}
    cfg.empty_cameras = 2
    if cfg.n_obs_steps != 1 or cfg.compile_model or cfg.fuse_qkv or cfg.fuse_gate_up:
        raise ValueError('Only the unfused single-observation inference graph is supported')
    return cfg


def strict_author_state(policy, weights):
    """Map author names and restore only exact aliases already sharing model storage."""
    import torch
    rules = (('action_in_proj.', 'model.suffix_embedder.action_in_proj.'),
             ('action_out_proj.', 'model.action_out_proj.'),
             ('time_mlp_in.', 'model.suffix_embedder.time_mlp_in.'),
             ('time_mlp_out.', 'model.suffix_embedder.time_mlp_out.'),
             ('state_proj.', 'model.suffix_embedder.state_proj.'),
             ('state_mlp_in.', 'model.suffix_embedder.state_mlp_in.'),
             ('state_mlp_out.', 'model.suffix_embedder.state_mlp_out.'),
             ('paligemma_with_expert.gemma_expert.', 'model.action_expert.'),
             ('paligemma_with_expert.paligemma.', 'model.vlm.'))
    mapped = {}
    for key, tensor in weights.items():
        new = next((after + key[len(before):] for before, after in rules if key.startswith(before)), key)
        if new in mapped:
            raise ValueError('Duplicate remapped PI05 tensor')
        mapped[new] = tensor
    expected = policy.state_dict()
    def storage(t):
        return (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()), t.dtype)
    groups = {}
    for key in mapped.keys() & expected.keys():
        if mapped[key].shape != expected[key].shape:
            raise ValueError('PI05 tensor shape mismatch')
        group = groups.setdefault(storage(expected[key]), [])
        if group and not torch.equal(mapped[group[0]], mapped[key]):
            raise ValueError('Conflicting checkpoint values for a model storage alias')
        group.append(key)
    for key in expected.keys() - mapped.keys():
        aliases = groups.get(storage(expected[key]), [])
        if aliases:
            mapped[key] = mapped[aliases[0]]
    if mapped.keys() != expected.keys():
        raise ValueError('Missing or unexpected PI05 tensors')
    for key, value in mapped.items():
        if 'buffer_' in key or 'normalize_' in key or 'unnormalize_' in key:
            if not torch.isfinite(value).all() or ('std' in key and torch.any(value <= 0)):
                raise ValueError('Invalid embedded normalization buffer')
    policy.load_state_dict(mapped, strict=True)


class Pi05FrontPolicy(JointChunkPolicy):
    def __init__(self, policy, *, task, observation_spec=None, execute_steps=20, frequency_hz=30):
        self.policy = policy
        super().__init__(self._predict, task=task, observation_spec=observation_spec,
                         execute_steps=execute_steps, frequency_hz=frequency_hz, reset_backend=policy.reset)

    def _predict(self, observation, task):
        import torch
        batch = native_front(observation, self.observation_spec, task, FRONT_KEY, 'motor')
        device = self.policy.config.device
        batch = {key: torch.from_numpy(value).unsqueeze(0).to(device) if key != 'task' else [value]
                 for key, value in batch.items()}
        with torch.inference_mode():
            output = self.policy.predict_action_chunk(batch)  # Own embedded normalization both ways.
        chunk = tensor_chunk(output)
        if len(chunk) != self.policy.config.n_action_steps:
            raise ValueError('PI05 output horizon differs from its loaded configuration')
        return chunk

    @classmethod
    def load(cls, path, device='cpu'):
        path, options, model = read_adapter(path, 'pi05-vlash')
        if options['joint_units'] != 'motor':
            raise ValueError('This SO101 PI05 contract requires motor units')
        # VLASH hardcodes its tokenizer name: require an explicitly offline backend/cache.
        from transformers.utils.hub import is_offline_mode
        if not is_offline_mode():
            raise ValueError('Set HF_HUB_OFFLINE=1 before importing the backend; pre-cache the authorized tokenizer')
        import draccus
        import torch
        from safetensors.torch import load_file
        from vlash.policies.pi05.configuration_pi05 import PI05Config
        from vlash.policies.pi05.modeling_pi05 import PI05Policy
        config = json.loads((model / 'config.json').read_text(encoding='utf-8'))
        if config.pop('type') != 'pi05':
            raise ValueError('Not a PI05 checkpoint')
        cfg = single_front_config(draccus.decode(PI05Config, config))
        cfg.device = 'cpu'
        policy = PI05Policy(cfg)
        strict_author_state(policy, load_file(str(model / 'model.safetensors'), device='cpu'))
        policy.eval().requires_grad_(False).to(device)
        policy.config.device = device
        return cls(policy, task=options['task'], observation_spec=ObservationSpec.from_dict(options['observation_spec']),
                   execute_steps=options['execute_steps'], frequency_hz=options['frequency_hz'])
