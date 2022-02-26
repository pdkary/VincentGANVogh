from config.CallableConfig import NoneCallable
from config.GanConfig import DiscConvLayerConfig, GenLayerConfig, SimpleActivations
from inputs.GanInput import GanInput
from models import Generator

conv_lr = SimpleActivations.leakyRelu_p08.value
sigmoid = SimpleActivations.sigmoid.value

def up_layer(f,c,k,act=conv_lr,u=True,t=True,n=0.1,id="",norm=NoneCallable,concat_with=""):
  return GenLayerConfig(f,c,k,act,upsampling=u,transpose=t,noise=n,track_id=id,normalization=norm,concat_with=concat_with)

def down_layer(f,c,k,act=conv_lr,d=True,t=False,dr=0.0,n=0.0,id="",norm=NoneCallable,concat_with=""):
  return DiscConvLayerConfig(f,c,k,act,downsampling=d,transpose=t,dropout_rate=dr,track_id=id,normalization=norm,noise=n,concat_with=concat_with)

def generate_UNET(input: GanInput):
    return Generator(
        gan_input = input,
        conv_layers = [
            down_layer(3,1,3,concat_with="0"),
            down_layer(3,2,3,concat_with="1"),
            down_layer(3,4,3,concat_with="2"),
            down_layer(3,8,3,concat_with="3"),
            down_layer(3,16,3),
            up_layer(3,8,3,concat_with="3"),
            up_layer(3,4,3,concat_with="2"),
            up_layer(3,2,3,concat_with="1"),
            up_layer(3,1,3,concat_with="0"),
            up_layer(3,1,1,sigmoid,n=0.0,u=False)
        ]
    )
        