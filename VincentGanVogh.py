from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU
from tensorflow.python.keras.losses import BinaryCrossentropy, MeanSquaredError, losses_utils
from tensorflow.keras.metrics import Accuracy, Mean
from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2

from config.GanConfig import GenLayerConfig, DiscConvLayerConfig, DiscDenseLayerConfig
from config.TrainingConfig import DataConfig, GanTrainingConfig

from layers.GanInput import GenConstantInput, GenLatentSpaceInput,RealImageInput
from layers.CallableConfig import ActivationConfig, NormalizationConfig, RegularizationConfig
from helpers.DataHelper import map_to_range, map_to_std_mean

from models.Discriminator import Discriminator
from models.Generator import Generator
from models.NoiseModel import ConstantNoiseModel, LatentNoiseModel
from models.StyleModel import LatentStyleModel

from third_party_layers.InstanceNormalization import InstanceNormalization

# from trainers.CombinedTrainer import CombinedTrainer
# from trainers.GenTapeTrainer import GenTapeTrainer
from trainers.MatchedGanStyleTrainer import MatchedGanStyleTrainer

# from google.colab import drive
# drive.mount('/content/drive')
 
##activations
gen_lr = ActivationConfig(LeakyReLU,dict(alpha=0.08))
disc_lr = ActivationConfig(LeakyReLU,dict(alpha=0.08))
untracked_lr = ActivationConfig(LeakyReLU,dict(alpha=0.08))
dense_lr = ActivationConfig(LeakyReLU,dict(alpha=0.1))

sigmoid = ActivationConfig(Activation,dict(activation="sigmoid"))
softmax = ActivationConfig(Activation,dict(activation="softmax"))
tanh = ActivationConfig(Activation,dict(activation="tanh"))
relu = ActivationConfig(Activation,dict(activation="relu"))
linear = ActivationConfig(Activation,dict(activation="linear"))

##normalizations
instance_norm = NormalizationConfig(InstanceNormalization)
batch_norm = NormalizationConfig(BatchNormalization,dict(momentum=0.8))
##regularizers
l2 = RegularizationConfig(L2)

##desired image shape
img_shape = (256,256,3)

#data input
data_config = DataConfig(
    data_path='test_images',    
    image_type=".png",
    image_shape=img_shape,
    model_name='/test_generator_model_',
    flip_lr=False,
    load_n_percent=3,
    load_scale_function=lambda x: map_to_range(x,1.0,-1.0),
    save_scale_function=lambda x: map_to_range(x,255.0,0.0)
)

image_source = RealImageInput(data_config)

##style models // noise_models // input models
# input_model = GenLatentSpaceInput(100,(2,2,512),512,8,sigmoid)
# noise_model = None

# style_model = None
style_model = LatentStyleModel(100,3,512,dense_lr)
noise_model = LatentNoiseModel(img_shape,dense_lr,max_std_dev=1.0)
input_model = GenConstantInput((4,4,512))

## layer shorthands 
gl_tracked = lambda f,c: GenLayerConfig(f,c,3,gen_lr,upsampling=True,style=True)
gl_untracked = lambda f,c: GenLayerConfig(f,c,3,untracked_lr,upsampling=True,noise=True,style=True)
gl_out = GenLayerConfig(img_shape[-1],1,1,tanh)

dcl_nodown = lambda f,c: DiscConvLayerConfig(f,c,3,disc_lr,dropout_rate=0.4,downsampling=False,normalization=instance_norm)
dcl = lambda f,c: DiscConvLayerConfig(f,c,3,   disc_lr,dropout_rate=0.4,normalization=instance_norm)
ddl = lambda s : DiscDenseLayerConfig(s,dense_lr,0.4)
d_out = DiscDenseLayerConfig(1, sigmoid, 0.0)

#Generator model
generator = Generator(
    img_shape = img_shape,
    input_model = input_model,
    gen_layers = [gl_untracked(512,4),gl_untracked(512,4),gl_tracked(256,4),gl_tracked(128,2),gl_untracked(64,2),gl_tracked(64,2),gl_out],
    gen_optimizer = Adam(learning_rate=2e-3),
    loss_function = BinaryCrossentropy(reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE),
    metrics = [Mean, Accuracy],
    style_model = style_model,
    noise_model = noise_model,
    normalization=instance_norm
)
    
#Discriminator Model
discriminator = Discriminator(
    img_shape = img_shape,
    disc_conv_layers = [dcl(64,2),dcl(128,2),dcl(256,4),dcl(512,4),dcl(512,4)],
    disc_dense_layers = [ddl(4096),ddl(4096),ddl(1000),d_out],
    minibatch_size = 4,
    disc_optimizer = Adam(learning_rate=2e-3),
    loss_function = BinaryCrossentropy(reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE),
    metrics = [Mean,Accuracy],
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
    style_loss_function = MeanSquaredError(reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE),
    style_loss_mean_std_axis = [1,2],
    style_loss_coeff = 1.0
)
#Trainer
VGV = MatchedGanStyleTrainer(generator,discriminator,gan_training_config,[image_source])

#TRAINING
ERAS = 100
EPOCHS = 5000
PRINT_EVERY = 10
MOVING_AVERAGE_SIZE = 100

VGV.train_n_eras(ERAS,EPOCHS,PRINT_EVERY,MOVING_AVERAGE_SIZE)
