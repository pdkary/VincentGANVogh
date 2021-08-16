from Generator import Generator
from Discriminator import Discriminator
from keras_visualizer import visualizer
from keras.optimizers import Adam
from GanConfig import StyleModelConfig,NoiseModelConfig,GeneratorModelConfig,DiscriminatorModelConfig

style_model_config = StyleModelConfig(
    style_latent_size = 100,
    style_layer_size = 512,
    style_layers = 3,
    style_relu_alpha = 0.1
)
 
noise_model_config = NoiseModelConfig(
    noise_image_size = (256,256,4),
    noise_kernel_size = 1,
    gauss_factor = 1
)
 
gen_model_config = GeneratorModelConfig(
    img_shape = (256,256,4),
    gen_constant_shape = (4,4,512),
    gen_kernel_size = 3,
    gen_layer_shapes=     [ (512,2),(256,2),(128,2),(64,2),(32,2),(16,2),(8,2) ],
    gen_layer_upsampling= [  False,   True,  True,   True,  True,  True,  True ],
    gen_layer_using_style=[  True,    True,  True,   True,  True,  True,  True ],
    gen_layer_noise=      [  True,    True,  True,   True,  True,  True,  True ],
    gen_relu_alpha = 0.05,
    batch_norm_momentum = 0.8,
    gen_loss_function="binary_crossentropy",
    gen_optimizer = Adam(learning_rate=5e-4,beta_1=0.5)
)
 
##VGG-19
disc_model_config = DiscriminatorModelConfig(
    img_shape = (256,256,4),
    disc_kernel_size = 3,
    disc_layer_shapes=[(64,2),(128,2),(256,4),(512,4),(512,4)],
    disc_dense_sizes=   [4096,4096,1000],
    disc_layer_dropout= [True,True,True],
    disc_relu_alpha=0.05,
    dropout_rate=0.5,
    minibatch=True,
    minibatch_size=4,
    disc_loss_function="binary_crossentropy",
    disc_optimizer = Adam(learning_rate=5e-4,beta_1=0.5)
)

G = Generator(gen_model_config,noise_model_config,style_model_config).build()
D = Discriminator(disc_model_config).build()
# visualizer(G,"Generator")
visualizer(D,"Discriminator")