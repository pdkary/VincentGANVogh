class StyleModelConfig():
  def __init__(self,
               style_latent_size,
               style_layer_size,
               style_layers,
               relu_alpha):
    self.style_latent_size = style_latent_size
    self.style_layer_size = style_layer_size
    self.style_layers = style_layers
    self.style_relu_alpha = relu_alpha

class NoiseModelConfig():
  def __init__(self,
               noise_latent_size,
               noise_layer_size,
               noise_model_layers,
               noise_kernel_size):
    self.noise_latent_size = noise_latent_size
    self.noise_layer_size = noise_layer_size
    self.noise_model_layers = noise_model_layers
    self.noise_kernel_size = noise_kernel_size

class GeneratorModelConfig():
  def __init__(self,
               img_shape,
               gen_constant_shape,
               gen_kernel_size,
               gen_layer_shapes,
               gen_layer_upsampling,
               gen_layer_using_style,
               gen_layer_noise,
               relu_alpha,
               batch_norm_momentum):
    self.img_shape = img_shape
    self.gen_constant_shape = gen_constant_shape
    self.gen_kernel_size = gen_kernel_size
    self.gen_layer_shapes = gen_layer_shapes
    self.gen_layer_upsampling = gen_layer_upsampling
    self.gen_layer_using_style = gen_layer_using_style
    self.gen_layer_noise = gen_layer_noise
    self.gen_relu_alpha = relu_alpha
    self.batch_norm_momentum = batch_norm_momentum

class DiscriminatorModelConfig():
  def __init__(self,
               img_shape,
               disc_kernel_size,
               disc_layer_shapes,
               disc_dense_sizes,
               disc_layer_dropout,
               relu_alpha,
               dropout_rate,
               minibatch,
               minibatch_size):
    self.img_shape = img_shape
    self.disc_kernel_size = disc_kernel_size
    self.disc_layer_shapes = disc_layer_shapes
    self.disc_dense_sizes = disc_dense_sizes
    self.disc_layer_dropout = disc_layer_dropout
    self.relu_alpha = relu_alpha
    self.dropout_rate = dropout_rate
    self.minibatch = minibatch
    self.minibatch_size = minibatch_size

class GanTrainingConfig():
  def __init__(self,
               gen_optimizer,
               disc_optimizer,
               disc_loss_function,
               gen_loss_function,
               gauss_factor,
               batch_size,
               preview_rows,
               preview_cols,
               data_path,
               image_type,
               model_name,
               flip_lr,
               load_n_percent):
    self.gen_optimizer = gen_optimizer
    self.disc_optimizer = disc_optimizer
    self.disc_loss_function = disc_loss_function
    self.gen_loss_function = gen_loss_function
    self.gauss_factor = gauss_factor
    self.batch_size = batch_size
    self.preview_rows = preview_rows
    self.preview_cols = preview_cols
    self.data_path = data_path
    self.image_type = image_type
    self.model_name = model_name
    self.flip_lr = flip_lr
    self.load_n_percent = load_n_percent
