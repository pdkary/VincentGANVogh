from typing import Callable, Tuple, List
from tensorflow.keras.metrics import Metric

class DataConfig():
    def __init__(self,
                 data_path: str,
                 image_type: str,
                 image_shape: Tuple[int, int, int],
                 model_name: str,
                 flip_lr: bool,
                 load_n_percent: bool,
                 load_scale_function: Callable,
                 save_scale_function: Callable):
        self.data_path = data_path
        self.image_type = image_type
        self.image_shape = image_shape
        self.model_name = model_name
        self.flip_lr = flip_lr
        self.load_n_percent = load_n_percent
        self.load_scale_function = load_scale_function
        self.save_scale_function = save_scale_function

class GanTrainingConfig():
    def __init__(self,
                 plot: bool,
                 disc_labels: Tuple[float, float],
                 gen_label: float,
                 batch_size: int,
                 disc_batches_per_epoch: int,
                 gen_batches_per_epoch: int,
                 metrics: List[Metric],
                 preview_rows: int,
                 preview_cols: int,
                 preview_margin: int):
        self.plot = plot
        self.disc_labels = disc_labels
        self.gen_label = gen_label
        self.batch_size = batch_size
        self.disc_batches_per_epoch = disc_batches_per_epoch
        self.gen_batches_per_epoch = gen_batches_per_epoch
        self.metrics = metrics
        self.preview_rows = preview_rows
        self.preview_cols = preview_cols
        self.preview_margin = preview_margin
