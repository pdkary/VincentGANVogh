class GanShapeConfig():
  def __init__(self,img_shape,latent_size,style_size,kernel_size,
               style_layer_size,style_layers,gen_layer_shapes,
               gen_layer_upsampling,disc_layer_shapes,
               disc_dense_sizes,minibatch_size):
    self.img_shape = img_shape
    self.latent_size = latent_size
    self.style_size = style_size
    self.kernel_size = kernel_size
    self.style_layer_size = style_layer_size
    self.style_layers = style_layers
    self.gen_layer_shapes = gen_layer_shapes
    self.gen_layer_upsampling = gen_layer_upsampling
    self.disc_layer_shapes = disc_layer_shapes
    self.disc_dense_sizes = disc_dense_sizes
    self.minibatch_size = minibatch_size

class GanBuildingConfig():
  def __init__(self,relu_alpha,dropout_rate,batch_norm_momentum):
    self.relu_alpha = relu_alpha
    self.dropout_rate = dropout_rate
    self.batch_norm_momentum = batch_norm_momentum

class GanTrainingConfig():
  def __init__(self,learning_rate,disc_loss_function,gen_loss_function,use_latent_noise,gauss_factor,batch_size,preview_rows,
               preview_cols,data_path,image_type,model_name):
    self.learning_rate = learning_rate
    self.disc_loss_function = disc_loss_function
    self.gen_loss_function = gen_loss_function
    self.use_latent_noise = use_latent_noise
    self.gauss_factor = gauss_factor
    self.batch_size = batch_size
    self.preview_rows = preview_rows
    self.preview_cols = preview_cols
    self.data_path = data_path
    self.image_type = image_type
    self.model_name = model_name

class GanConfig(GanShapeConfig,GanBuildingConfig,GanTrainingConfig):
  def __init__(self,gan_shape_config,gan_building_config,gan_training_config):
    GanShapeConfig.__init__(self,**gan_shape_config.__dict__)
    GanBuildingConfig.__init__(self,**gan_building_config.__dict__)
    GanTrainingConfig.__init__(self,**gan_training_config.__dict__)
    self.shape_config = gan_shape_config
    self.build_config = gan_shape_config
    self.train_config = gan_shape_config
    self.img_size = self.img_shape[1]
    self.channels = self.img_shape[-1]
