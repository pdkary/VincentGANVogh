from third_party_layers.InstanceNormalization import InstanceNormalization
from keras.layers import BatchNormalization, LeakyReLU, Activation
from GanConfig import *
from trainers.GanTrainer import GanTrainer
from keras.optimizer_v2.adam import Adam
 
from google.colab import drive
drive.mount('/content/drive')
 
leakyRELU_style = ActivationConfig(LeakyReLU,dict(alpha=0.1))
leakyRELU_conv = ActivationConfig(LeakyReLU,dict(alpha=0.1819))
sigmoid = ActivationConfig(Activation,dict(activation="sigmoid"))
softmax = ActivationConfig(Activation,dict(activation="softmax"))
linear = ActivationConfig(Activation,dict(activation="linear"))
instance_norm = NormalizationConfig(InstanceNormalization)
batch_norm = NormalizationConfig(BatchNormalization,dict(momentum=0.2))
 
style_model_config = StyleModelConfig(
    style_latent_size = 100,
    style_layer_size = 128,
    style_layers = 3,
    style_activation = leakyRELU_style
)
 
noise_model_config = NoiseModelConfig(
    noise_image_size = (256,256,1),
    kernel_size = 1,
    gauss_factor = 1
)
 
gen_model_config = GeneratorModelConfig(
    img_shape = (256,256,3),
    gen_constant_shape = (4,4,512),
    gen_layers = [
        GenLayerConfig(512,  3, 3, leakyRELU_conv, upsampling=False, style=True, noise=True),
        GenLayerConfig(512,  3, 3, leakyRELU_conv, upsampling=True,  style=True, noise=True),
        GenLayerConfig(256,  3, 3, leakyRELU_conv, upsampling=True,  style=True, noise=True),
        GenLayerConfig(128,  3, 3, leakyRELU_conv, upsampling=True,  style=True, noise=True),
        GenLayerConfig(64,   3, 3, leakyRELU_conv, upsampling=True,  style=True, noise=True),
        GenLayerConfig(32,   3, 3, leakyRELU_conv, upsampling=True,  style=True, noise=True),
        GenLayerConfig(16,   3, 3, leakyRELU_conv, upsampling=True,  style=True, noise=True),
        GenLayerConfig(8,    3, 3, leakyRELU_conv, upsampling=False, style=True, noise=True),
        GenLayerConfig(3,    1, 1, sigmoid,        upsampling=False, style=True, noise=True)],
    non_style_normalization = instance_norm,
    gen_loss_function="binary_crossentropy",
    gen_optimizer = Adam(learning_rate=0.002,beta_1=0.0,beta_2=0.99,epsilon=1e-7)
)
 
##VGG-19
disc_model_config = DiscriminatorModelConfig(
    img_shape = (256,256,3),
    disc_conv_layers=[
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
    disc_loss_function="binary_crossentropy",
    disc_optimizer = Adam(learning_rate=0.002,beta_1=0.0,beta_2=0.99,epsilon=1e-7)
)
 
gan_training_config = GanTrainingConfig(
    batch_size=4,
    preview_rows=4,
    preview_cols=6,
    data_path='/content/drive/MyDrive/Colab/Handwritten/English',
    image_type=".png",
    model_name='/GANVogh_char_generator_model_',
    flip_lr=False,
    load_n_percent=40
)
 
VGV = GanTrainer(gen_model_config,noise_model_config,style_model_config,disc_model_config,gan_training_config)