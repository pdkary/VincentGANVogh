from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU
from tensorflow.keras.metrics import Accuracy, Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.python.keras.losses import (BinaryCrossentropy,
                                            MeanSquaredError, losses_utils)

from config.GanConfig import DiscConvLayerConfig, GenLayerConfig
from config.TrainingConfig import DataConfig, GanTrainingConfig
from helpers.DataHelper import map_to_range
from inputs.GanInput import ConstantInput, LatentSpaceInput, RealImageInput
from layers.CallableConfig import (ActivationConfig, NoneCallable,
                                   NormalizationConfig, RegularizationConfig)
from models.Discriminator import Discriminator
from models.Generator import Generator
from third_party_layers.InstanceNormalization import InstanceNormalization
from trainers.SimpleTrainer import SimpleTrainer
from trainers.StyleTrainer import StyleTrainer

# from google.colab import drive
# drive.mount('/content/drive')
 
##activations
conv_lr = ActivationConfig(LeakyReLU,dict(alpha=0.08))
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
channels = img_shape[-1]
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
## inputs
image_source = RealImageInput(data_config)
latent_input = LatentSpaceInput([100])
constant_input = ConstantInput([100])

## layer shorthands
def gen_layer(f,c,k,act=conv_lr,u=True,t=False,n=0.1,id="",norm=NoneCallable):
  return GenLayerConfig(f,c,k,act,upsampling=u,transpose=t,noise=n,track_id=id,normalization=norm)

def disc_layer(f,c,k,act=conv_lr,d=True,t=False,dropout=0.0,id="",norm=NoneCallable):
  return DiscConvLayerConfig(f,c,k,act,dropout,d,track_id=id,normalization=norm)

#Generator model
gen_out = gen_layer(channels,1,1,sigmoid,u=False,n=0.0)

generator = Generator(
    gan_input = latent_input,
    dense_layers=[1000,4096,4096],
    conv_input_shape=(4,4,512),
    conv_layers = [gen_layer(512,4,3),
                   gen_layer(512,4,3),
                   gen_layer(256,3,3),
                   gen_layer(128,3,3),
                   gen_layer(64, 2,3),
                   gen_layer(32, 2,3),
                   gen_layer(3,  4,3,u=False),
                   gen_out],
    dense_activation=dense_lr,
)   
#Discriminator Model
discriminator = Discriminator(
    real_image_input = image_source,
    conv_layers = [ disc_layer(64, 3,3),
                    disc_layer(128,3,3),
                    disc_layer(256,3,3),
                    disc_layer(512,4,3),
                    disc_layer(512,4,3)],
    dense_layers = [4096,4096,1000,1],
    minibatch_size = 32,
    dense_activation=dense_lr,
    final_activation=sigmoid,
)

#Training config
gan_training_config = GanTrainingConfig(
    plot=False,
    #[real_image_label, not_image_label]
    disc_labels=[1.0,0.0],
    #desired label
    gen_label=0.5,
    batch_size=6,
    gen_loss_function = BinaryCrossentropy(reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE),
    disc_loss_function = BinaryCrossentropy(reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE),
    gen_optimizer = Adam(learning_rate=5e-6),#beta_1=0.0,beta_2=0.99,epsilon=1e-7),
    disc_optimizer = Adam(learning_rate=5e-6),#beta_1=0.0,beta_2=0.99,epsilon=1e-7),
    metrics = [Mean],
    preview_rows=3,
    preview_cols=4,
    preview_margin=6
)
#Trainer
gDNA = generator.toDNA()
print(gDNA)
print(gDNA.output_shape)
# VGV = SimpleTrainer(generator,discriminator,gan_training_config)
# VGV.compile()

# #TRAINING
# ERAS = 1
# EPOCHS = 10
# PRINT_EVERY = 1
# MOVING_AVERAGE_SIZE = 5

# VGV.train_n_eras(ERAS,EPOCHS,PRINT_EVERY,MOVING_AVERAGE_SIZE)