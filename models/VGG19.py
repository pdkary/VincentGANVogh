from config.DiscriminatorConfig import *
from keras.layers import LeakyReLU
from third_party_layers.InstanceNormalization import InstanceNormalization
from keras.optimizer_v2.adam import Adam

instance_norm = NormalizationConfig(InstanceNormalization)
leakyRELU_dense = ActivationConfig(LeakyReLU,dict(alpha=0.1))
leakyRELU_conv = ActivationConfig(LeakyReLU,dict(alpha=0.08))

vgg19_config = DiscriminatorModelConfig(
    img_shape = (256,256,3),
    disc_conv_layers=[
        DiscConvLayerConfig(64,  2, 3, leakyRELU_conv, instance_norm),
        DiscConvLayerConfig(128, 2, 3, leakyRELU_conv, instance_norm),
        DiscConvLayerConfig(256, 3, 3, leakyRELU_conv, instance_norm),
        DiscConvLayerConfig(512, 3, 3, leakyRELU_conv, instance_norm),
        DiscConvLayerConfig(512, 3, 3, leakyRELU_conv, instance_norm)],
    disc_dense_layers=[
        DiscDenseLayerConfig(4096, leakyRELU_conv, 0.4),
        DiscDenseLayerConfig(4096, leakyRELU_conv, 0.4),
        DiscDenseLayerConfig(1000, leakyRELU_conv, 0.4)],
    minibatch=True,
    minibatch_size=32,
    disc_optimizer = Adam(learning_rate=0.002,beta_1=0.5)
)