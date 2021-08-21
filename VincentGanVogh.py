from layers.AdaptiveInstanceNormalization import AdaptiveInstanceNormalization
from models.GeneratorInput import GenConstantInput, GenLatentSpaceInput
from models.GanConfig import *
from keras.optimizers import Adam
from third_party_layers.InstanceNormalization import InstanceNormalization
from keras.layers import BatchNormalization, LeakyReLU, Activation, LayerNormalization
from trainers.GanTrainer import GanTrainer

leakyRELU_style = ActivationConfig(LeakyReLU,dict(alpha=0.1))
leakyRELU_conv = ActivationConfig(LeakyReLU,dict(alpha=0.05))
sigmoid = ActivationConfig(Activation,dict(activation="sigmoid"))
instance_norm = NormalizationConfig(InstanceNormalization)
layer_norm = NormalizationConfig(LayerNormalization)
batch_norm = NormalizationConfig(BatchNormalization,dict(momentum=0.8))

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
    input_model = GenLatentSpaceInput(100,(4,4,512),128,2),
    gen_layers = [
        GenLayerConfig(512,  4, 3, leakyRELU_conv, upsampling=False, style=True, noise=False),
        GenLayerConfig(512,  4, 3, leakyRELU_conv, upsampling=True,  style=True, noise=False),
        GenLayerConfig(256,  4, 3, leakyRELU_conv, upsampling=True,  style=True, noise=False),
        GenLayerConfig(128,  3, 3, leakyRELU_conv, upsampling=True,  style=True, noise=False),
        GenLayerConfig(64,   3, 3, leakyRELU_conv, upsampling=True,  style=True, noise=False),
        GenLayerConfig(32,   3, 3, leakyRELU_conv, upsampling=True,  style=True, noise=False),
        GenLayerConfig(16,   2, 3, leakyRELU_conv, upsampling=True,  style=True, noise=False),
        GenLayerConfig(3,    1, 1, sigmoid,        upsampling=False, style=True, noise=False)],
    non_style_normalization = instance_norm,
    gen_loss_function="binary_crossentropy",
    gen_optimizer = Adam(learning_rate=5e-4,beta_1=0.1,beta_2=0.9,epsilon=1e-7)
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
    disc_optimizer = Adam(learning_rate=0.002,beta_1=0.1,beta_2=0.9,epsilon=1e-7)
)
 
gan_training_config = GanTrainingConfig(
    batch_size=4,
    preview_rows=4,
    preview_cols=6,
    data_path='test_images',
    image_type=".png",
    model_name='/GANVogh_generator_model_',
    flip_lr=True,
    load_n_percent=10,
    plot=False,
)
 
VGV = GanTrainer(gen_model_config,noise_model_config,style_model_config,disc_model_config,gan_training_config)
VGV.train_n_eras(eras=1,epochs=10,batches_per_epoch=1,printerval=2,ma_size=1)