from typing import Tuple


class DataConfig():
    def __init__(self,
                 data_path: str,
                 image_type: str,
                 image_shape: Tuple[int, int, int],
                 model_name: str,
                 flip_lr: bool,
                 load_n_percent: bool):
        self.data_path = data_path
        self.image_type = image_type
        self.image_shape = image_shape
        self.model_name = model_name
        self.flip_lr = flip_lr
        self.load_n_percent = load_n_percent


class GanTrainingConfig():
    def __init__(self,
                 plot: bool,
                 disc_labels: Tuple[float, float],
                 gen_label: float,
                 gen_batch_size: int,
                 disc_batch_size: int,
                 preview_rows: int,
                 preview_cols: int,
                 preview_margin: int):
        self.plot = plot
        self.disc_labels = disc_labels
        self.gen_label = gen_label
        self.gen_batch_size = gen_batch_size
        self.disc_batch_size = disc_batch_size
        self.preview_rows = preview_rows
        self.preview_cols = preview_cols
        self.preview_margin = preview_margin
        assert gen_batch_size % disc_batch_size == 0
