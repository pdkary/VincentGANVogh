from config.TrainingConfig import DataConfig, GanTrainingConfig
from config.DiscriminatorConfig import *
from models.GanInput import RealImageInput
from config.GeneratorConfig import *
from config.CallableConfig import *
from tensorflow.keras.optimizers import Adam
from third_party_layers.InstanceNormalization import InstanceNormalization
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Activation
from trainers.GenTapeTrainer import GenTapeTrainer

from google.colab import drive
drive.mount('/content/drive')

leakyRELU_style = ActivationConfig(LeakyReLU,dict(alpha=0.1))
leakyRELU_conv = ActivationConfig(LeakyReLU,dict(alpha=0.05))
sigmoid = ActivationConfig(Activation,dict(activation="sigmoid"))
instance_norm = NormalizationConfig(InstanceNormalization)
batch_norm = NormalizationConfig(BatchNormalization,dict(momentum=0.8))

style_data_config = DataConfig(
    data_path='/content/drive/MyDrive/Colab/VanGogh',    
    image_type=".jpg",
    image_shape=(256,256,3),
    batch_size=4,
    model_name='/GANVogh_generator_model_',
    flip_lr=True,
    load_n_percent=8,
    preview_rows=4,
    preview_cols=6,
    preview_margin=16
)

image_data_config = DataConfig(
    data_path='/content/drive/MyDrive/Colab/AnimalFaces/dog',    
    image_type=".jpg",
    image_shape=(256,256,3),
    batch_size=4,
    model_name='/GANVogh_dog_generator_model_',
    flip_lr=True,
    load_n_percent=1,
    preview_rows=4,
    preview_cols=6,
    preview_margin=16
)
 
gen_model_config = GeneratorModelConfig(
    img_shape = (256,256,3),
    input_model = RealImageInput(image_data_config),
    
    style_model_config = StyleModelConfig(
        style_latent_size = 100,
        style_layer_size = 128,
        style_layers = 3,
        style_activation = leakyRELU_style),
    
    noise_model_config = NoiseModelConfig(
        noise_image_size = (256,256,1),
        kernel_size = 1,
        max_std_dev = 1),
    
    gen_layers = [
        GenLayerConfig(128,  3, 3, leakyRELU_conv, upsampling=False,  style=True, noise=False),
        GenLayerConfig(64,   3, 3, leakyRELU_conv, upsampling=False,  style=True, noise=False),
        GenLayerConfig(32,   3, 3, leakyRELU_conv, upsampling=False,  style=True, noise=False),
        GenLayerConfig(16,   2, 3, leakyRELU_conv, upsampling=False,  style=True, noise=False),
        GenLayerConfig(3,    1, 1, sigmoid,        upsampling=False,  style=True, noise=False)],
    normalization = batch_norm,
    gen_optimizer = Adam(learning_rate=5e-4,beta_1=0.1,beta_2=0.9,epsilon=1e-7)
)
 
 
##VGG-19
disc_model_config = DiscriminatorModelConfig(
    img_shape = (256,256,3),
    disc_conv_layers=[
        DiscConvLayerConfig(64,  2, 3, leakyRELU_conv, instance_norm),
        DiscConvLayerConfig(64,  2, 3, leakyRELU_conv, instance_norm),
        DiscConvLayerConfig(128, 2, 3, leakyRELU_conv, instance_norm),
        DiscConvLayerConfig(256, 4, 3, leakyRELU_conv, instance_norm),
        DiscConvLayerConfig(512, 4, 3, leakyRELU_conv, instance_norm),
        DiscConvLayerConfig(512, 4, 3, leakyRELU_conv, instance_norm)],
    disc_dense_layers=[
        DiscDenseLayerConfig(4096, leakyRELU_conv, 0.5),
        DiscDenseLayerConfig(4096, leakyRELU_conv, 0.5),
        DiscDenseLayerConfig(1000, leakyRELU_conv, 0.5)],
    minibatch=True,
    minibatch_size=64,
    disc_optimizer = Adam(learning_rate=0.002,beta_1=0.1,beta_2=0.9,epsilon=1e-7)
)
 
gan_training_config = GanTrainingConfig(
    plot=True
)

VGV = GenTapeTrainer(gen_model_config,disc_model_config,gan_training_config,style_data_config)