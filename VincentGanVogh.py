import numpy as np
from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU
from tensorflow.keras.losses import binary_crossentropy,categorical_crossentropy, kl_divergence, mean_squared_error
from tensorflow.keras.metrics import Accuracy, Mean, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2

from config.GanConfig import GenLayerConfig
from config.TrainingConfig import DataConfig, GanTrainingConfig
from layers.CallableConfig import ActivationConfig, NormalizationConfig,RegularizationConfig
from layers.GanInput import GenConstantInput, GenLatentSpaceInput, RealImageInput
from models.Discriminator import Discriminator
from models.Generator import Generator
from models.NoiseModel import ConstantNoiseModel, LatentNoiseModel
from models.StyleModel import ImageStyleModel, LatentStyleModel
from models.VGG19 import get_vgg19
from third_party_layers.InstanceNormalization import InstanceNormalization
from trainers.CombinedTrainer import CombinedTrainer
from trainers.GenTapeTrainer import GenTapeTrainer

# from google.colab import drive
# drive.mount('/content/drive')
 
##activations
gen_lr_dense = ActivationConfig(LeakyReLU,"gen_lr_dense",dict(alpha=0.1))
gen_lr_conv = ActivationConfig(LeakyReLU,"gen_lr_conv",dict(alpha=0.08))

disc_lr_dense = ActivationConfig(LeakyReLU,"disc_lr_dense",dict(alpha=0.1))
disc_lr_conv = ActivationConfig(LeakyReLU,"disc_lr_conv",dict(alpha=0.08))

sigmoid = ActivationConfig(Activation,"sigmoid",dict(activation="sigmoid"))
softmax = ActivationConfig(Activation,"softmax",dict(activation="softmax"))
tanh = ActivationConfig(Activation,"tanh",dict(activation="tanh"))
relu = ActivationConfig(Activation,"relu",dict(activation="relu"))
linear = ActivationConfig(Activation,"linear",dict(activation="linear"))

##normalizations
instance_norm = NormalizationConfig(InstanceNormalization)
batch_norm = NormalizationConfig(BatchNormalization,dict(momentum=0.8))
##regularizers
l2 = RegularizationConfig(L2)
##desired image shape
img_shape = (256,256,3)

def map_to_range(input_arr,new_max,new_min):
  img_max = float(np.max(input_arr))
  img_min = float(np.min(input_arr))
  old_range = float(img_max - img_min + 1e-6)
  new_range = (new_max - new_min)
  return new_range*(input_arr - img_min)/old_range + float(new_min)

#data input
data_config = DataConfig(
    data_path='test_images',   
    image_type=".png",
    image_shape=img_shape,
    model_name='/simplegan_generator_model_',
    flip_lr=False,
    load_n_percent=10,
    load_scale_function = lambda x: map_to_range(x,1.0,-1.0),
    save_scale_function = lambda x: map_to_range(x,255.0,0.0)
)

image_source = RealImageInput(data_config)

##style models
# style_model = ImageStyleModel(image_source,32,2,3,3,100,gen_lr_dense,gen_lr_conv,l2,downsample_factor=4)
# style_model = LatentStyleModel(100,3,512,gen_lr_dense)
style_model = None

##noise model
# noise_model = LatentNoiseModel(img_shape,gen_lr_conv,l2)
noise_model = None

##input models
# input_model = GenConstantInput((4,4,512))
input_model = GenLatentSpaceInput(100,(2,2,1024),512,3,gen_lr_dense)

## layer shorthands 
gl = lambda f,c: GenLayerConfig(f,c,3,gen_lr_conv,upsampling=True,style=True,noise=True)
glt = lambda f,c: GenLayerConfig(f,c,3,gen_lr_conv,upsampling=True,transpose=True,style=True,noise=True)
output_layer = GenLayerConfig(img_shape[-1],1,1,tanh,style=True)

#Generator model
generator = Generator(
    img_shape = img_shape,
    input_model = input_model,
    gen_layers = [gl(512,3),gl(512,3),gl(256,3),gl(256,3),gl(128,3),gl(128,3),gl(64,3),output_layer],
    gen_optimizer = Adam(learning_rate=2e-3,beta_1=0.0),
    loss_function = binary_crossentropy,
    normalization = batch_norm
)

#Discriminator Model
discriminator = get_vgg19(
    input_channels = img_shape[-1],
    conv_activation = disc_lr_conv,
    dense_activation = disc_lr_dense,
    final_activation = softmax,
    normalization = batch_norm,
    optimizer = Adam(learning_rate=2e-3,beta_1=0.0),
    loss_function = binary_crossentropy,
    lite = True
)

#Training config
gan_training_config = GanTrainingConfig(
    plot=True,
    #[real_image_label, not_image_label]
    disc_labels=[1.0,0.0],
    #desired label
    gen_label=1.0,
    batch_size=4,
    disc_batches_per_epoch = 1,
    gen_batches_per_epoch = 1,
    metrics = [Accuracy,MeanSquaredError],
    preview_rows=4,
    preview_cols=6,
    preview_margin=16
)

#Trainer
VGV = GenTapeTrainer(generator,discriminator,gan_training_config,[image_source])


#TRAINING
ERAS = 1
EPOCHS = 1
BATCHES_PER_EPOCH = 1
PRINT_EVERY = 10
MOVING_AVERAGE_SIZE = 20
 
VGV.train_n_eras(ERAS,EPOCHS,BATCHES_PER_EPOCH,PRINT_EVERY,MOVING_AVERAGE_SIZE)
