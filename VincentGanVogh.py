from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU
from tensorflow.python.keras.losses import BinaryCrossentropy, MeanSquaredError, losses_utils
from tensorflow.keras.metrics import Accuracy, Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2

from config.GanConfig import GenLayerConfig, DiscConvLayerConfig, DiscDenseLayerConfig
from config.TrainingConfig import DataConfig, GanTrainingConfig
from helpers.DataHelper import map_to_range
from inputs.GanInput import ConstantInput, LatentSpaceInput, RealImageInput

from layers.CallableConfig import ActivationConfig, NormalizationConfig, RegularizationConfig

from models.Discriminator import Discriminator
from models.Generator import Generator

from third_party_layers.InstanceNormalization import InstanceNormalization

from trainers.GradTapeStyleTrainer import GradTapeStyleTrainer

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
    load_n_percent=10,
    load_scale_function=lambda x: map_to_range(x,1.0,-1.0),
    save_scale_function=lambda x: map_to_range(x,255.0,0.0)
)

image_source = RealImageInput(data_config)

##style models // noise_models // input models
# style_model = None
# noise_model = None
# input_model = GenLatentSpaceInput(100,(2,2,1024),1024,2,dense_lr)

latent_input = LatentSpaceInput([100])
constant_input = ConstantInput((2,2,1024))

## layer shorthands

# matched gen layers
mgl_un = lambda f,c,id: GenLayerConfig(f,c,3,gen_lr,upsampling=True,noise=True,style=True,track_id=id)
mgl_u = lambda f,c,id: GenLayerConfig(f,c,3,gen_lr,upsampling=True,style=True,track_id=id)
mgl = lambda f,c,id: GenLayerConfig(f,c,3,gen_lr,style=True,track_id=id)
# unmatched gen layers
gl_un = lambda f,c: GenLayerConfig(f,c,3,gen_lr,upsampling=True,noise=True,normalization=instance_norm)
gl_u = lambda f,c: GenLayerConfig(f,c,3,gen_lr,upsampling=True,normalization=instance_norm)
gl = lambda f,c: GenLayerConfig(f,c,3,gen_lr,normalization=instance_norm)

#gen out
g_out = GenLayerConfig(img_shape[-1],1,3,sigmoid,normalization=instance_norm)

## NOTE: ALL discriminator layers will be tracked, generator must have equivalent tracked layers
dcl = lambda f,c,id: DiscConvLayerConfig(f,c,3,   disc_lr,dropout_rate=0.4,normalization=instance_norm,track_id=id)
ddl = lambda s : DiscDenseLayerConfig(s,dense_lr,0.4)

#Generator model
generator = Generator(
    gan_input = latent_input,
    dense_layers=[1000,4096],
    conv_input_shape=(4,4,512),
    conv_layers = [mgl_u(512,2,"1"),
                  mgl_u(512,2,"2"),
                  mgl_u(512,2,"3"),
                  mgl_u(256,2,"4"),
                  mgl_u(128,2,"5"),
                  mgl_u(64,2,"6"),
                  g_out],
    style_input=latent_input,
    style_layers=[100,100,100,100],
    dense_activation=sigmoid
)
    
#Discriminator Model
discriminator = Discriminator.from_generator(generator,image_source,sigmoid)
# discriminator = Discriminator(
#     real_image_input = image_source,
#     conv_layers = [dcl(64,2,"6"),
#                    dcl(128,2,"5"),
#                    dcl(256,2,"4"),
#                    dcl(512,2,"3"),
#                    dcl(512,2,"2"),
#                    dcl(512,2,"1")],
#     dense_layers = [4096,4096,1000,1],
#     minibatch_size = 8,
#     dropout_rate = 0.1,
#     activation = sigmoid
# )

#Training config
gan_training_config = GanTrainingConfig(
    plot=True,
    #[real_image_label, not_image_label]
    disc_labels=[1.0,0.0],
    #desired label
    gen_label=1.0,
    batch_size=8,
    gen_loss_function = BinaryCrossentropy(reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE),
    disc_loss_function = BinaryCrossentropy(reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE),
    style_loss_function = MeanSquaredError(reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE),
    gen_optimizer = Adam(learning_rate=2e-3,beta_1=0.0,beta_2=0.9,epsilon=1e-7),
    disc_optimizer = Adam(learning_rate=2e-3,beta_1=0.0,beta_2=0.9,epsilon=1e-7),
    metrics = [Mean,Accuracy],
    style_loss_coeff = 0.0,
    disc_batches_per_epoch = 1,
)
#Trainer
VGV = GradTapeStyleTrainer(generator,discriminator,gan_training_config)
VGV.compile()
# # TRAINING
# ERAS = 100
# EPOCHS = 5000
# PRINT_EVERY = 10
# MOVING_AVERAGE_SIZE = 100

# VGV.train_n_eras(ERAS,EPOCHS,PRINT_EVERY,MOVING_AVERAGE_SIZE)
