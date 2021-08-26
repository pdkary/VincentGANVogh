from config.DiscriminatorConfig import DiscriminatorModelConfig, DiscConvLayerConfig,DiscDenseLayerConfig, ActivationConfig, NormalizationConfig
from tensorflow.keras.optimizers import Adam

def get_vgg19(conv_activation:ActivationConfig,
              dense_activation:ActivationConfig,
              final_activation:ActivationConfig,
              normalization:NormalizationConfig):
    return DiscriminatorModelConfig(
        img_shape = (256,256,3),
        disc_conv_layers=[
            DiscConvLayerConfig(64,  2, 3, conv_activation, normalization),
            DiscConvLayerConfig(128, 2, 3, conv_activation, normalization),
            DiscConvLayerConfig(256, 3, 3, conv_activation, normalization),
            DiscConvLayerConfig(512, 3, 3, conv_activation, normalization),
            DiscConvLayerConfig(512, 3, 3, conv_activation, normalization)],
        disc_dense_layers=[
            DiscDenseLayerConfig(4096, dense_activation, 0.5),
            DiscDenseLayerConfig(4096, dense_activation, 0.5),
            DiscDenseLayerConfig(1000, dense_activation, 0.5),
            DiscDenseLayerConfig(1,    final_activation, 0.5)],
        minibatch=True,
        minibatch_size=32,
        disc_optimizer = Adam(learning_rate=0.002,beta_1=0.0,beta_2=0.99,epsilon=1e-8))
    