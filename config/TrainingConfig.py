from typing import Callable, Tuple, List
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from dataclasses import dataclass, field

@dataclass
class DataConfig:
    data_path: str
    image_type: str
    image_shape: Tuple[int, int, int]
    model_name: str
    flip_lr: bool
    load_n_percent: bool
    load_scale_function: Callable
    save_scale_function: Callable

@dataclass
class GanTrainingConfig:
    plot: bool
    disc_labels: Tuple[float, float]
    gen_label: float
    batch_size: int
    gen_loss_function: Loss = None
    disc_loss_function: Loss = None
    style_loss_function: Loss = None
    gen_optimizer: Optimizer = None
    disc_optimizer: Optimizer = None
    metrics: List[Metric] = field(default_factory=list)
    style_loss_coeff: float = 0.25,
    disc_batches_per_epoch: int = 1
    gen_batches_per_epoch: int = 1
    preview_rows: int = 4
    preview_cols: int = 6
    preview_margin: int = 16
    
@dataclass
class GanTrainingResult:
    loss: float
    metrics: List[float]
    stds: List[float] = field(default_factory=list)
    means: List[float] = field(default_factory=list)

