from typing import Callable, Dict, Tuple


class CallableConfig():
    def __init__(self,
                 callable: Callable,
                 args: Dict = {},
                 kwargs: Dict = {}):
        self.callable = callable
        self.args = args
        self.kwargs = kwargs

    def get(self):
        if self.callable is None:
            return None
        else:
            return self.callable(**self.args, **self.kwargs)

class NamedCallableConfig(CallableConfig):
    def __init__(self, callable: Callable, name: str, args: Dict = {}, kwargs: Dict = {}):
        super().__init__(callable, args=args, kwargs=kwargs)
        self.layer_dict = {}
        self.shape_count = {}
        self.name = name

    def get(self,input_shape):
        shape_name = "_".join(input_shape)
        name = self.name + shape_name
        if self.shape_count[shape_name] is not None:
            self.shape_count[shape_name] += 1
        else:
            self.shape_count[shape_name] = 1
        
        self.kwargs["name"] = name + "_" + str(self.shape_count[shape_name])
        self.layer_dict[name] = self.callable(**self.args, **self.kwargs)
        return self.layer_dict[name]

class ActivationConfig(NamedCallableConfig):
    def __init__(self, callable: Callable, name: str, args: Dict, kwargs: Dict):
        super().__init__(callable, name, args=args, kwargs=kwargs)

class NormalizationConfig(CallableConfig):
    pass


class RegularizationConfig(CallableConfig):
    pass

NoneCallable = CallableConfig(None)

class DiscConvLayerConfig():
    def __init__(self,
                 filters: int,
                 convolutions: int,
                 kernel_size: int,
                 dropout_rate: float,
                 activation_config: ActivationConfig,
                 normalization: NormalizationConfig):
        self.filters = filters
        self.convolutions = convolutions
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.activation_config = activation_config
        self.normalization = normalization


class DiscDenseLayerConfig():
    def __init__(self,
                 size: int,
                 activation_config: ActivationConfig,
                 dropout_rate: int):
        self.size = size
        self.activation_config = activation_config
        self.dropout_rate = dropout_rate


class GenLayerConfig():
    def __init__(self,
                 filters: int,
                 convolutions: int,
                 kernel_size: int,
                 activation: ActivationConfig,
                 strides: Tuple[int, int] = (1, 1),
                 transpose: bool = False,
                 upsampling: bool = False,
                 style: bool = False,
                 noise: bool = False):
        self.filters = filters
        self.convolutions = convolutions
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides
        self.transpose = transpose
        self.upsampling = upsampling
        self.style = style
        self.noise = noise
