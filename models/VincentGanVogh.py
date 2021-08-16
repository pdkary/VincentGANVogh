from keras.layers.core import Activation
from third_party_layers.InstanceNormalization import InstanceNormalization
from keras.layers import BatchNormalization, LeakyReLU
from GanConfig import DiscConvLayerConfig, DiscDenseLayerConfig, StyleModelConfig,NoiseModelConfig,GenLayerConfig,GeneratorModelConfig,DiscriminatorModelConfig,GanTrainingConfig
from trainers.GanTrainer import GanTrainer
from keras.optimizers import Adam

from google.colab import drive
drive.mount('/content/drive')
 
style_model_config = StyleModelConfig(
    style_latent_size = 100,
    style_layer_size = 512,
    style_layers = 3,
    style_activation = LeakyReLU(0.1)
)
 
noise_model_config = NoiseModelConfig(
    noise_image_size = (256,256,3),
    noise_kernel_size = 1,
    gauss_factor = 1
)

gen_model_config = GeneratorModelConfig(
    img_shape = (256,256,3),
    gen_constant_shape = (4,4,512),
    gen_layers = [
        GenLayerConfig(512, 3, 3, LeakyReLU,    {'alpha':0.05},          False, True, True ),
        GenLayerConfig(256, 3, 3, LeakyReLU,    {'alpha':0.05},          True,  True, True ),
        GenLayerConfig(128, 3, 3, LeakyReLU,    {'alpha':0.05},          True,  True, True ),
        GenLayerConfig(64,  3, 3, LeakyReLU,    {'alpha':0.05},          True,  True, True ),
        GenLayerConfig(32,  2, 3, LeakyReLU,    {'alpha':0.05},          True,  True, True ),
        GenLayerConfig(16,  2, 3, LeakyReLU,    {'alpha':0.05},          True,  True, True ),
        GenLayerConfig(8,   2, 3, LeakyReLU,    {'alpha':0.05},          True,  True, True ),
        GenLayerConfig(3,   1, 3, Activation,   {'activation':'sigmoid'},False, True, False)],
    non_style_normalization_layer=BatchNormalization(momentum=0.8),
    gen_loss_function="binary_crossentropy",
    gen_optimizer = Adam(learning_rate=5e-3,beta_1=0.0,beta_2=0.99,epsilon=1e-7)
)
 
##VGG-19
disc_model_config = DiscriminatorModelConfig(
    img_shape = (256,256,3),
    disc_conv_layers=[
        DiscConvLayerConfig(64,  2, 3, LeakyReLU(0.05), InstanceNormalization()),
        DiscConvLayerConfig(128, 2, 3, LeakyReLU(0.05), InstanceNormalization()),
        DiscConvLayerConfig(256, 4, 3, LeakyReLU(0.05), InstanceNormalization()),
        DiscConvLayerConfig(512, 4, 3, LeakyReLU(0.05), InstanceNormalization()),
        DiscConvLayerConfig(512, 4, 3, LeakyReLU(0.05), InstanceNormalization())],
    disc_dense_layers=[
        DiscDenseLayerConfig(4096, LeakyReLU(0.05), 0.5),
        DiscDenseLayerConfig(4096, LeakyReLU(0.05), 0.5),
        DiscDenseLayerConfig(1000, LeakyReLU(0.05), 0.5)],
    minibatch=True,
    minibatch_size=4,
    disc_loss_function="binary_crossentropy",
    disc_optimizer = Adam(learning_rate=5e-3,beta_1=0.0,beta_2=0.99,epsilon=1e-7)
)
 
gan_training_config = GanTrainingConfig(
    batch_size=4,
    preview_rows=4,
    preview_cols=6,
    data_path='/content/drive/MyDrive/Colab/Handwritten/English',
    image_type=".png",
    model_name='/GANVogh_char_generator_model_',
    flip_lr=False,
    load_n_percent=10
)
 
VGV = GanTrainer(gen_model_config,noise_model_config,style_model_config,disc_model_config,gan_training_config)