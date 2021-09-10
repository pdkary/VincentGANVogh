from config.TrainingConfig import DataConfig
from tensorflow.keras.regularizers import L2
from config.GeneratorConfig import NoiseModelConfig
from models.GanInput import RealImageInput
from layers.AdaptiveAdd import AdaptiveAdd
from tensorflow.keras.layers import Cropping2D,Cropping2D, Conv2D

class ImageNoiseModel(NoiseModelConfig,RealImageInput):
    def __init__(self,noise_config: NoiseModelConfig,data_config: DataConfig):
        NoiseModelConfig.__init__(self,**noise_config.__dict__)
        RealImageInput.__init__(self,data_config)

    def add(self,input_tensor):
        n_size = self.model.shape[1]
        i_size = input_tensor.shape[1]
        noise = Cropping2D((n_size-i_size)//2)(self.model)
        noise = Conv2D(input_tensor.shape[-1],self.kernel_size,padding='same', kernel_regularizer=L2(),kernel_initializer='he_normal')(noise)
        return AdaptiveAdd()([input_tensor,noise])