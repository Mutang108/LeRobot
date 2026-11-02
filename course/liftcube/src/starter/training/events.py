"""Training curves; independent of inference exports and resume checkpoints."""
from starter.logging.swanlab import ScalarWriter


def open_writer(config, stage="train", *, start_step=0, resumed=False):
    return ScalarWriter(config, stage, start_step=start_step, resumed=resumed)


def log_row(writer, stage, row, step):
    if writer is not None:
        writer.log(row, step, prefix=stage)
