from GanConfig import GanShapeConfig,GanBuildingConfig,GanTrainingConfig
from GanTrainer import GanTrainer
 
gan_shape_config = GanShapeConfig(
    img_shape=(256,256,3),
    latent_size=50,
    style_size=100,
    kernel_size=3,
    style_layer_size=64,
    style_layers=8,
    gen_layer_shapes=[(1024,3),(512,3),(256,3),(128,2),(64,2),(16,2)],
    disc_layer_shapes=[(64,2),(128,2),(256,3),(512,3),(512,3)],
    disc_dense_sizes=[4096,4096],
    minibatch_size=256,
)
 
gan_building_config = GanBuildingConfig(
    relu_alpha=0.1,
    dropout_rate=0.5,
    batch_norm_momentum=0.8
)
 
gan_training_config = GanTrainingConfig(
    learning_rate=1e-4,
    disc_loss_function="binary_crossentropy",
    gen_loss_function="binary_crossentropy",
    gauss_factor=0.02,
    batch_size=8,
    preview_rows=3,
    preview_cols=4,
    data_path='/content/drive/MyDrive/Colab/VanGogh',
    image_type=".jpg",
    model_name='/GANVogh_generator_model_'
)
 
#TRAINING
ERAS = 100
EPOCHS = 2500
BATCHES_PER_EPOCH = 1
PRINT_EVERY = 10

gan_trainer = GanTrainer(gan_shape_config,gan_building_config,gan_training_config)
gan_trainer.train_n_eras(ERAS,EPOCHS,BATCHES_PER_EPOCH,PRINT_EVERY)