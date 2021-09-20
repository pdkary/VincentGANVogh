from trainers.MatchedGanStyleTrainer import MatchedGanStyleTrainer
from models.MatchedGan import get_matched_gan
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
from models.NoiseModel import ConstantNoiseModel, LatentNoiseModel
from models.StyleModel import ImageStyleModel, LatentStyleModel
from models.VGG19 import get_vgg19
from third_party_layers.InstanceNormalization import InstanceNormalization

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
style_model = LatentStyleModel(100,3,512,gen_lr_dense)
# style_model = None

##noise model
noise_model = LatentNoiseModel(img_shape,gen_lr_conv,l2)
# noise_model = None

##input models
input_model = GenConstantInput((8,8,512))
# input_model = GenLatentSpaceInput(100,(8,8,512),512,3,gen_lr_dense)

discriminator,generator = get_matched_gan(
    img_shape = img_shape,
    gen_input_model = input_model,
    layer_sizes = [(64,2),(128,2),(256,3),(512,3),(512,3)],
    disc_optimizer = Adam(learning_rate=2e-3,beta_1=0.0),
    disc_loss_func = binary_crossentropy,
    disc_conv_activation = disc_lr_conv,
    disc_dense_activation = disc_lr_dense,
    disc_final_activation = sigmoid,
    gen_optimizer = Adam(learning_rate=2e-3,beta_1=0.0),
    gen_loss_func = binary_crossentropy,
    gen_conv_activation = gen_lr_conv,
    gen_final_activation = sigmoid,
    normalization = instance_norm,
    style_model = style_model,
    noise_model = noise_model,
    disc_output_dim = 1
)

style_loss_func = mean_squared_error
#Training config
gan_training_config = GanTrainingConfig(
    plot=True,
    #[real_image_label, not_image_label]
    disc_labels=[1.0,0.0],
    #desired label
    gen_label=1.0,
    batch_size=4,
    metrics = [Accuracy,MeanSquaredError],
)

#Trainer
# VGV = GenTapeTrainer(generator,discriminator,gan_training_config,[image_source])
VGV = MatchedGanStyleTrainer(generator,discriminator,gan_training_config,style_loss_func,[image_source])


#TRAINING
ERAS = 1
EPOCHS = 1
PRINT_EVERY = 10
MOVING_AVERAGE_SIZE = 20
 
VGV.train_n_eras(ERAS,EPOCHS,PRINT_EVERY,MOVING_AVERAGE_SIZE)
