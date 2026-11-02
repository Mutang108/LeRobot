"""Pinned LeRobot offline trainer with scalar-only SwanLab instead of W&B.

Run with the native lerobot-train arguments. No upstream source is modified;
the existing main-process logger hook is replaced only for this invocation.
"""
from starter.logging.swanlab import ScalarWriter


class SwanLabLogger:
    def __init__(self, cfg):
        self.writer = ScalarWriter(dict(seed=cfg.seed,
            policy={key: getattr(cfg.policy, key, None) for key in
                    ("type", "chunk_size", "n_action_steps", "num_inference_steps")},
            data=dict(batch_size=cfg.batch_size),
            logging=dict(dir=str(cfg.output_dir), swanlab=dict(name=cfg.job_name))),
            "lerobot-train", start_step=None, resumed=cfg.resume)

    def log_dict(self, d, step=None, mode="train", custom_step_key=None):
        if custom_step_key is not None:
            raise ValueError("This pinned trainer uses explicit optimizer steps, not custom step keys")
        self.writer.log(d, step, prefix=mode)

    def log_policy(self, checkpoint_dir):
        """Keep checkpoints local; never publish artifacts through tracking."""

    def log_video(self, video_path, step, mode="train"):
        """Videos remain local even when the native trainer requests an upload."""


def run_with_swanlab(trainer, cfg):
    """Keep the hook scoped and close only writers created by this invocation."""
    original = trainer.WandBLogger
    writers = []

    def create_logger(config):
        logger = SwanLabLogger(config)
        writers.append(logger.writer)
        return logger

    # The native trainer gates its logger with these fields. No W&B SDK is used.
    enabled, project = cfg.wandb.enable, cfg.wandb.project
    cfg.wandb.enable, cfg.wandb.project = True, "swanlab-routing-hook"
    trainer.WandBLogger = create_logger
    try:
        return trainer.train(cfg)
    finally:
        trainer.WandBLogger = original
        cfg.wandb.enable, cfg.wandb.project = enabled, project
        for writer in writers:
            writer.close()


def main():
    from lerobot.configs import parser
    from lerobot.scripts import lerobot_train

    @parser.wrap()
    def train(cfg: lerobot_train.TrainPipelineConfig):
        return run_with_swanlab(lerobot_train, cfg)

    train()


if __name__ == "__main__":
    main()
