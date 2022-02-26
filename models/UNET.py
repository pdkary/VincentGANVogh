from config.CallableConfig import NoneCallable
from config.GanConfig import DiscConvLayerConfig, GenLayerConfig, SimpleActivations, SimpleNormalizations
from inputs.GanInput import GanInput
from models.Generator import Generator

conv_lr = SimpleActivations.leakyRelu_p08.value
sigmoid = SimpleActivations.sigmoid.value

i_norm = SimpleNormalizations.instance_norm.value

def up_layer(f,c,k,act=conv_lr,u=True,t=True,n=0.1,id="",norm=i_norm,concat_with=""):
  return GenLayerConfig(f,c,k,act,upsampling=u,transpose=t,noise=n,track_id=id,normalization=norm,concat_with=concat_with)

def down_layer(f,c,k,act=conv_lr,d=True,t=False,dr=0.0,n=0.25,id="",norm=i_norm,concat_with=""):
  return DiscConvLayerConfig(f,c,k,act,downsampling=d,transpose=t,dropout_rate=dr,track_id=id,normalization=norm,noise=n,concat_with=concat_with)

def generate_UNET(input: GanInput,F: int, C: int, depth = 5, up_noise = 0.1, down_noise = 0.1):
    conv_up_layers = []
    conv_down_layers = []
    for i in range(depth-1):
        key = str(i)
        conv_down_layers.append(down_layer((2**i)*F,C,3,concat_with=key,n=down_noise))
        conv_up_layers = [up_layer(F,C,3,concat_with=key,n=up_noise)] + conv_up_layers
    
    conv_down_layers.append(down_layer((2**(depth - 1))*F,C,3,n=down_noise))
    conv_down_layers.extend(conv_up_layers)
    conv_down_layers.append(up_layer(input.input_shape[-1],1,3,act=sigmoid,n=0))
    return Generator(
        gan_input = input,
        conv_layers = conv_down_layers
    )
        