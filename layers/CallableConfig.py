from helpers.DataHelper import shape_to_key
from typing import Callable, Dict

class CallableConfig():
    def __init__(self,
                 callable: Callable,
                 args: Dict = {},
                 kwargs: Dict = {}):
        self.callable = callable
        self.args = args
        self.kwargs = kwargs

    def get(self,input_shape=None):
        if self.callable is None:
            return lambda x: x
        else:
            return self.callable(**self.args, **self.kwargs)
        
class NormalizationConfig(CallableConfig):
    pass


class RegularizationConfig(CallableConfig):
    pass

NoneCallable = CallableConfig(None)

class TrackedCallableConfig(CallableConfig):
    def __init__(self, callable: Callable, name: str, args: Dict = {}, kwargs: Dict = {}):
        super().__init__(callable, args=args, kwargs=kwargs)
        self.layer_dict = {}
        self.name = name

    def get(self,input_shape):
        shape_key = shape_to_key(input_shape)
            
        if shape_key in self.layer_dict.keys():
            num_layers = len(self.layer_dict[shape_key])
            self.kwargs["name"] = self.name + "_" + shape_key + "_" + str(num_layers)
            layer = self.callable(**self.args,**self.kwargs)
            self.layer_dict[shape_key].append(layer)
        else:
            self.kwargs["name"] = self.name + "_" + shape_key + "_0"
            layer = self.callable(**self.args,**self.kwargs)
            self.layer_dict[shape_key] = [layer]
        return layer
        
class ActivationConfig(TrackedCallableConfig):
    def __init__(self, callable: Callable, name: str, args: Dict = {}, kwargs: Dict = {}):
        super().__init__(callable, name, args=args, kwargs=kwargs)

