from trainers.GanOptimizer import GanOptimizer
from config.TrainingConfig import GanTrainingConfig, DataConfig

gan_training_config = GanTrainingConfig(
    plot=False,
    disc_labels=[1.0,0.0],
    gen_label=0.5
)

data_config = DataConfig(
    data_path='test_images',    
    image_type=".png",
    image_shape=(64,64,3),
    batch_size=32,
    model_name='/simplegan_generator_model_',
    flip_lr=True,
    load_n_percent=10,
    preview_rows=4,
    preview_cols=6,
    preview_margin=16
)

hypergan = GanOptimizer(gan_training_config,data_config)
hypergan.tune_n_batches(100,20,10)