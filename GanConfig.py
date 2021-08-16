class StyleModelConfig():
  def __init__(self,
               style_latent_size,
               style_layer_size,
               style_layers,
               style_activation):
    self.style_latent_size = style_latent_size
    self.style_layer_size = style_layer_size
    self.style_layers = style_layers
    self.style_activation = style_activation

class NoiseModelConfig():
  def __init__(self,
               noise_image_size,
               noise_kernel_size,
               gauss_factor):
    self.noise_image_size = noise_image_size
    self.noise_kernel_size = noise_kernel_size
    self.gauss_factor = gauss_factor

class GeneratorModelConfig():
  def __init__(self,
               img_shape,
               gen_constant_shape,
               gen_kernel_size,
               gen_layer_shapes,
               gen_layer_upsampling,
               gen_layer_using_style,
               gen_layer_noise,
               convolution_activation,
               non_style_normalization_layer,
               gen_loss_function,
               gen_optimizer):
    self.img_shape = img_shape
    self.gen_constant_shape = gen_constant_shape
    self.gen_kernel_size = gen_kernel_size
    self.gen_layer_shapes = gen_layer_shapes
    self.gen_layer_upsampling = gen_layer_upsampling
    self.gen_layer_using_style = gen_layer_using_style
    self.gen_layer_noise = gen_layer_noise
    self.convolution_activation = convolution_activation
    self.non_style_normalization_layer = non_style_normalization_layer
    self.gen_loss_function = gen_loss_function
    self.gen_optimizer = gen_optimizer

class DiscriminatorModelConfig():
  def __init__(self,
               img_shape,
               disc_kernel_size,
               disc_layer_shapes,
               disc_dense_sizes,
               disc_layer_dropout,
               convolution_activation,
               normalization_layer,
               dropout_rate,
               minibatch,
               minibatch_size,
               disc_loss_function,
               disc_optimizer):
    self.img_shape = img_shape
    self.disc_kernel_size = disc_kernel_size
    self.disc_layer_shapes = disc_layer_shapes
    self.disc_dense_sizes = disc_dense_sizes
    self.disc_layer_dropout = disc_layer_dropout
    self.convolution_activation = convolution_activation
    self.normalization_layer = normalization_layer
    self.dropout_rate = dropout_rate
    self.minibatch = minibatch
    self.minibatch_size = minibatch_size
    self.disc_loss_function = disc_loss_function
    self.disc_optimizer = disc_optimizer

class GanTrainingConfig():
  def __init__(self,
               batch_size,
               preview_rows,
               preview_cols,
               data_path,
               image_type,
               model_name,
               flip_lr,
               load_n_percent):
    self.batch_size = batch_size
    self.preview_rows = preview_rows
    self.preview_cols = preview_cols
    self.data_path = data_path
    self.image_type = image_type
    self.model_name = model_name
    self.flip_lr = flip_lr
    self.load_n_percent = load_n_percent
