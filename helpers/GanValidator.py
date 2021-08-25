from keras.engine import input_layer
import numpy as np
from config.GeneratorConfig import GeneratorModelConfig

class GanValidator():
    def validate_gen_sizes(self,gen_config: GeneratorModelConfig) -> bool:
        input_shape = gen_config.input_model.model.shape[1:]
        expected_output_shape = gen_config.img_shape
        
        upsample_factor = 2**sum([x.upsampling for x in gen_config.gen_layers[0]])
        output_channels = gen_config.gen_layers[0][-1].filters
        print(input_shape,expected_output_shape,upsample_factor, output_channels)
        xm = expected_output_shape[0] == input_shape[0]*upsample_factor
        ym = expected_output_shape[1] == input_shape[1]*upsample_factor
        zm = expected_output_shape[2] == output_channels
        return xm and ym and zm

    def validate_style(self,gen_config: GeneratorModelConfig) -> bool:
        has_model  = gen_config.style_model_config is not None
        using_style = np.any([l.style for l in gen_config.gen_layers[0]])
        return has_model == using_style
    
    def validate_noise(self,gen_config: GeneratorModelConfig) -> bool:
        has_model = gen_config.noise_model_config is not None
        using_noise = np.any([l.style for l in gen_config.gen_layers[0]])
        return has_model == using_noise
    
    def validate_normalization(self,gen_config: GeneratorModelConfig) -> bool:
        all_style = np.all([l.style for l in gen_config.gen_layers[0]])
        has_non_style_norm = gen_config.normalization is not None
        return all_style or has_non_style_norm
        
        
