from trainers.GenTapeTrainer import GenTapeTrainer
from helpers.GanValidator import GanValidator
from config.TrainingConfig import DataConfig, GanTrainingConfig
from models.InputModel import GenLatentSpaceInput
from config.GeneratorConfig import *
from config.CallableConfig import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Activation

from models.VGG19 import vgg19_config

leakyRELU_conv = ActivationConfig(LeakyReLU,dict(alpha=0.05))
sigmoid = ActivationConfig(Activation,dict(activation="sigmoid"))
batch_norm = NormalizationConfig(BatchNormalization,dict(momentum=0.8))
 
gen_model_config = GeneratorModelConfig(
    img_shape = (256,256,3),
    input_model = GenLatentSpaceInput(100,(4,4,512),128,2),
    gen_layers = [
        GenLayerConfig(512,  2, 3, leakyRELU_conv, upsampling=True ),
        GenLayerConfig(512,  2, 3, leakyRELU_conv, upsampling=True ),
        GenLayerConfig(256,  2, 3, leakyRELU_conv, upsampling=True ),
        GenLayerConfig(128,  2, 3, leakyRELU_conv, upsampling=True ),
        GenLayerConfig(64,   2, 3, leakyRELU_conv, upsampling=True ),
        GenLayerConfig(32,   2, 3, leakyRELU_conv, upsampling=True ),
        GenLayerConfig(3,    1, 1, sigmoid,        upsampling=False)],
    normalization = batch_norm,
    gen_optimizer = Adam(learning_rate=5e-4,beta_1=0.55)
)
 
gan_training_config = GanTrainingConfig(
    plot=False
)

data_config = DataConfig(
    data_path='test_images',
    image_type=".png",
    image_shape=(256,256,3),
    batch_size=4,
    model_name='/GANVogh_generator_model_',
    flip_lr=True,
    load_n_percent=10,
    preview_rows=3,
    preview_cols=4,
    preview_margin=16
)

validator = GanValidator()
size_pass = validator.validate_gen_sizes(gen_model_config)
style_pass = validator.validate_style(gen_model_config)
noise_pass = validator.validate_noise(gen_model_config)

all_pass = size_pass and style_pass and noise_pass

if all_pass:
    VGV = GenTapeTrainer(gen_model_config,vgg19_config,gan_training_config,[data_config])
if all_pass:
    VGV.train_n_eras(eras=1,epochs=10,batches_per_epoch=1,printerval=2,ma_size=1)