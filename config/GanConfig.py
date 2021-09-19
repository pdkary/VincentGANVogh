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
        print(input_shape)
        print(type(input_shape))
        if type(input_shape) == int or type(input_shape) ==  float:
            shape_name = "_" + str(input_shape)
        else:
            shape_name = "_".join(input_shape)
        name = self.name + shape_name
        if shape_name in self.shape_count.keys():
            self.shape_count[shape_name] += 1
        else:
            self.shape_count[shape_name] = 1
        
        self.kwargs["name"] = name + "_" + str(self.shape_count[shape_name])
        self.layer_dict[name] = self.callable(**self.args, **self.kwargs)
        return self.layer_dict[name]

class ActivationConfig(NamedCallableConfig):
    def __init__(self, callable: Callable, name: str, args: Dict = {}, kwargs: Dict = {}):
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
                 activation: ActivationConfig,
                 normalization: NormalizationConfig):
        self.filters = filters
        self.convolutions = convolutions
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.normalization = normalization


class DiscDenseLayerConfig():
    def __init__(self,
                 size: int,
                 activation: ActivationConfig,
                 dropout_rate: int):
        self.size = size
        self.activation = activation
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
