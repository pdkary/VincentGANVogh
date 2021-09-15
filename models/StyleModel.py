from matplotlib import image
from layers.GanInput import RealImageInput
from config.GeneratorConfig import StyleModelConfig
from tensorflow.keras.layers import Input
    
class StyleModel(StyleModelConfig):
    def __init__(self,style_config,image_input: RealImageInput):
        StyleModelConfig.__init__(self,**style_config.__dict__)
        self.image_input = image_input
        self.input = image_input.input
        self.model = self.style_activation.get()(self.input)
    
    def get_batch(self,batch_size):
        return self.image_input.get_batch(batch_size)