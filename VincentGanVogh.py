from models.GanInput import GenConstantInput, GenLatentSpaceInput
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from third_party_layers.InstanceNormalization import InstanceNormalization
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Activation
from config.TrainingConfig import DataConfig, GanTrainingConfig
from trainers.GenTapeTrainer import GenTapeTrainer
from config.CallableConfig import *
from config.GeneratorConfig import *
 
from models.VGG19 import get_vgg19

##activations
leakyRELU_style = ActivationConfig(LeakyReLU,dict(alpha=0.1))
leakyRELU_conv = ActivationConfig(LeakyReLU,dict(alpha=0.08))
sigmoid = ActivationConfig(Activation,dict(activation="sigmoid"))
linear = ActivationConfig(Activation,dict(activation="linear"))
##normalizations
instance_norm = NormalizationConfig(InstanceNormalization)
batch_norm = NormalizationConfig(BatchNormalization,dict(momentum=0.5))
##optimizers
adam = CallableConfig(Adam,dict(learning_rate=0.002))
img_shape = (256,256,3)
 
gen_model_config = GeneratorModelConfig(
    img_shape = img_shape,
    input_model = GenConstantInput((4,4,512)),
    # input_model = GenLatentSpaceInput(100,(4,4,512),512,2,leakyRELU_style),
    
    style_model_config = StyleModelConfig(
        style_latent_size = 100,
        style_layer_size = 512,
        style_layers = 4,
        style_activation = leakyRELU_style),
    
    noise_model_config = NoiseModelConfig(
        noise_image_size = img_shape,
        kernel_size = 1,
        max_std_dev = 1),
 
    gen_layers = [
        GenLayerConfig(512,           3, 3, leakyRELU_conv, style=True, transpose=True, strides=2, noise=True),
        GenLayerConfig(512,           3, 3, leakyRELU_conv, style=True, transpose=True, strides=2, noise=True),
        GenLayerConfig(256,           3, 3, leakyRELU_conv, style=True, transpose=True, strides=2, noise=True),
        GenLayerConfig(128,           3, 3, leakyRELU_conv, style=True, transpose=True, strides=2, noise=True),
        GenLayerConfig(64,            2, 3, leakyRELU_conv, style=True, upsampling=True, noise=True),
        GenLayerConfig(32,            2, 3, leakyRELU_conv, style=True, upsampling=True, noise=True),
        GenLayerConfig(img_shape[-1], 1, 1, sigmoid,        style=True)],
    normalization = instance_norm,
    gen_optimizer = adam.get(),
    loss_function= BinaryCrossentropy()
)
 
disc_model_config = get_vgg19(img_shape[-1],leakyRELU_conv,leakyRELU_style,sigmoid,instance_norm,adam.get(),64,0.4)
 
gan_training_config = GanTrainingConfig(
    plot=True,
    #[real_image_label, not_image_label]
    disc_labels=[1.0,0.0],
    #desired label
    gen_label=0.5,
    gen_batch_size=16,
    disc_batch_size=4,
    preview_rows=4,
    preview_cols=6,
    preview_margin=16
)
 
data_config = DataConfig(
    data_path='test_images',   
    image_type=".png",
    image_shape=img_shape,
    model_name='/simplegan_generator_model_',
    flip_lr=False,
    load_n_percent=10
)

VGV = GenTapeTrainer(gen_model_config,disc_model_config,gan_training_config,[data_config])
 
#TRAINING
ERAS = 1
EPOCHS = 1
BATCHES_PER_EPOCH = 1
PRINT_EVERY = 10
MOVING_AVERAGE_SIZE = 20
 
VGV.train_n_eras(ERAS,EPOCHS,BATCHES_PER_EPOCH,PRINT_EVERY,MOVING_AVERAGE_SIZE)