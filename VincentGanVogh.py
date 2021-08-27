from models.GanInput import GenConstantInput, GenLatentSpaceInput
from tensorflow.keras.optimizers import Adam
from third_party_layers.InstanceNormalization import InstanceNormalization
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Activation
from config.TrainingConfig import DataConfig, GanTrainingConfig
from trainers.GenTapeTrainer import GenTapeTrainer
from config.CallableConfig import *
from config.GeneratorConfig import *

from models.VGG19 import get_vgg19
 
from google.colab import drive
drive.mount('/content/drive')
 
leakyRELU_style = ActivationConfig(LeakyReLU,dict(alpha=0.1))
leakyRELU_conv = ActivationConfig(LeakyReLU,dict(alpha=0.05))
sigmoid = ActivationConfig(Activation,dict(activation="sigmoid"))
instance_norm = NormalizationConfig(InstanceNormalization)
batch_norm = NormalizationConfig(BatchNormalization,dict(momentum=0.8))
 
gen_model_config = GeneratorModelConfig(
    img_shape = (256,256,3),
    input_model = GenLatentSpaceInput(100,(4,4,512),100,0,leakyRELU_style),
    
    style_model_config = StyleModelConfig(
        style_latent_size = 100,
        style_layer_size = 512,
        style_layers = 3,
        style_activation = leakyRELU_style),
    
    noise_model_config = NoiseModelConfig(
        noise_image_size = (256,256,3),
        kernel_size = 1,
        gauss_factor = 1),
    
    gen_layers = [
        GenLayerConfig(512,  4, 3, leakyRELU_conv, upsampling=False, style=False, noise=False),
        GenLayerConfig(512,  4, 3, leakyRELU_conv, upsampling=True,  style=False, noise=False),
        GenLayerConfig(256,  4, 3, leakyRELU_conv, upsampling=True,  style=False, noise=False),
        GenLayerConfig(128,  3, 3, leakyRELU_conv, upsampling=True,  style=True, noise=True),
        GenLayerConfig(64,   3, 3, leakyRELU_conv, upsampling=True,  style=True, noise=True),
        GenLayerConfig(32,   3, 3, leakyRELU_conv, upsampling=True,  style=True, noise=True),
        GenLayerConfig(16,   2, 3, leakyRELU_conv, upsampling=True,  style=True, noise=False),
        GenLayerConfig(3,    1, 1, sigmoid,        upsampling=False, style=True, noise=False)],
    normalization = batch_norm,
    gen_optimizer = Adam(learning_rate=5e-4,beta_1=0.1,beta_2=0.9,epsilon=1e-7)
)
 
disc_model_config = get_vgg19(leakyRELU_conv,leakyRELU_style,sigmoid,instance_norm)
 
gan_training_config = GanTrainingConfig(
    plot=True,
    #[real_image_label, not_image_label]
    disc_labels=[1.0,0.0],
    #desired label
    gen_label=1.0
)

data_config = DataConfig(
    data_path='/content/drive/MyDrive/Colab/Handwritten/English',    
    image_type=".png",
    image_shape=(256,256,3),
    batch_size=4,
    model_name='/simplegan_generator_model_',
    flip_lr=False,
    load_n_percent=10,
    preview_rows=4,
    preview_cols=6,
    preview_margin=16
)
 
VGV = GenTapeTrainer(gen_model_config,disc_model_config,gan_training_config,[data_config])