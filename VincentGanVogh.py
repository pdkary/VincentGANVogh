from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.losses import BinaryCrossentropy, losses_utils

from config.CallableConfig import ActivationConfig, NoneCallable
from config.GanConfig import DenseLayerConfig, DiscConvLayerConfig, GenLayerConfig, SimpleActivations, SimpleNormalizations
from config.TrainingConfig import DataConfig, GanTrainingConfig
from helpers.DataHelper import map_to_range
from inputs.GanInput import ConstantInput, DenseModelInput, LatentSpaceInput, RealImageInput
from models.Discriminator import Discriminator
from models.GanConverter import GanConverter
from models.Generator import Generator
from models.UNET import generate_UNET
from trainers.SimpleTrainer import SimpleTrainer

# from google.colab import drive
# drive.mount('/content/drive')

##desired image shape
img_shape = (256,256,3)
channels = img_shape[-1]
#data input
data_config = DataConfig(
    data_path='test_images',    
    image_type=".png",
    image_shape=img_shape,
    model_name='/test_generator_model_',
    flip_lr=True,
    load_n_percent=100,
    load_scale_function=lambda x: map_to_range(x,1.0,0.0),
    save_scale_function=lambda x: map_to_range(x,255.0,0.0)
)
## inputs
image_source = RealImageInput(data_config)
latent_input = LatentSpaceInput([100])
constant_input = ConstantInput([100])
dense_input = DenseModelInput([512],512,8)

##activations
conv_lr = SimpleActivations.leakyRelu_p08.value
dense_lr = SimpleActivations.leakyRelu_p1.value

## layer shorthands
def dense_layer(s,a=SimpleActivations.leakyRelu_p1.value,dr=0.0,ms=0,md=4):
      return DenseLayerConfig(s,a,dr,ms,md)

def gen_layer(f,c,k,act=conv_lr,u=True,t=False,n=0.1,id="",norm=NoneCallable):
  return GenLayerConfig(f,c,k,act,upsampling=u,transpose=t,noise=n,track_id=id,normalization=norm)

def disc_layer(f,c,k,act=conv_lr,d=True,t=False,dr=0.0,id="",norm=NoneCallable):
  return DiscConvLayerConfig(f,c,k,act,downsampling=d,dropout_rate=dr,track_id=id,normalization=norm)

#Generator model
##activations
conv_lr = SimpleActivations.leakyRelu_p08.value
dense_lr = SimpleActivations.leakyRelu_p1.value
linear = SimpleActivations.linear.value
sigmoid = SimpleActivations.sigmoid.value
tanh = SimpleActivations.tanh.value

##norms
batch_norm = SimpleNormalizations.batch_norm.value
instance_norm = SimpleNormalizations.instance_norm.value
from layers.AdaptiveInstanceNormalization import adain_config
adain = adain_config

## layer shorthands
def dense_layer(s,a=SimpleActivations.leakyRelu_p1.value,dr=0.0,ms=0,md=4,features=False):
  return DenseLayerConfig(s,a,dr,ms,md,train_features=features)

def gen_layer(f,c,k,act=conv_lr,u=True,t=True,n=0.0,id="",norm=NoneCallable):
  return GenLayerConfig(f,c,k,act,upsampling=u,transpose=t,noise=n,track_id=id,normalization=norm)

def disc_layer(f,c,k,act=conv_lr,d=True,t=False,dr=0.00,n=0.0,id="",norm=NoneCallable):
  return DiscConvLayerConfig(f,c,k,act,downsampling=d,dropout_rate=dr,track_id=id,normalization=norm,noise=n)

def gen_upsample_only():
  return GenLayerConfig(0,0,0,linear,upsampling=True)

def disc_downsample_only():
  return DiscConvLayerConfig(0,0,0,linear,downsampling=True)

##UNET MODEL

#Generator model
# generator = generate_UNET(image_source)
generator = Generator(
    gan_input = dense_input,
    dense_layers=[
      dense_layer(1000,a=linear),
      dense_layer(4096),
      dense_layer(4096)
    ],
    conv_input_shape=(4,4,512),
    conv_layers = [
      gen_layer(512,4,3,norm=adain),
      gen_layer(512,4,3,norm=adain),
      gen_layer(256,3,3,norm=adain),
      gen_layer(128,3,3,norm=adain),
      gen_layer(64, 2,3,norm=adain),
      gen_layer(channels,1,1,sigmoid)
    ]
)   

# #Discriminator Model
discriminator = Discriminator(
    gan_input = image_source,
    conv_input_shape = image_source.input_shape,
    conv_layers = [ 
      disc_downsample_only(),
      disc_layer(64, 2,3,norm=instance_norm),
      disc_layer(128,3,3,norm=instance_norm),
      disc_layer(256,3,3,norm=instance_norm),
      disc_layer(512,4,3,norm=instance_norm),
      disc_layer(512,4,3,norm=instance_norm),
    ],
    dense_layers = [
      dense_layer(4096,ms=128,md=3),
      dense_layer(4096,ms=64,md=3),
      dense_layer(1000,ms=32,md=3,features=True),
      dense_layer(1,a=sigmoid)
    ]
)

#Training config
gan_training_config = GanTrainingConfig(
    plot=True,
    #[real_image_label, not_image_label]
    disc_labels=[1.0,0.0],
    #desired label
    gen_label=1.0,
    batch_size=6,
    gen_loss_function = BinaryCrossentropy(reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE),
    disc_loss_function = BinaryCrossentropy(reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE),
    gen_optimizer = Adam(learning_rate=4e-5),#beta_1=0.0,beta_2=0.99,epsilon=1e-7),
    disc_optimizer = Adam(learning_rate=4e-5),#beta_1=0.0,beta_2=0.99,epsilon=1e-7),
    metrics = [Mean],
    preview_rows=3,
    preview_cols=4,
    preview_margin=6
)
#Trainer
VGV = SimpleTrainer(generator,discriminator,gan_training_config)
VGV.compile()

#TRAINING
ERAS = 1
EPOCHS = 10
PRINT_EVERY = 1
MOVING_AVERAGE_SIZE = 5

VGV.train_n_eras(ERAS,EPOCHS,PRINT_EVERY,MOVING_AVERAGE_SIZE)
