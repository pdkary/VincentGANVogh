from typing import Callable, Dict
import numbers

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
        
class NormalizationConfig(CallableConfig):
    pass


class RegularizationConfig(CallableConfig):
    pass

NoneCallable = CallableConfig(None)

class NamedCallableConfig(CallableConfig):
    def __init__(self, callable: Callable, name: str, args: Dict = {}, kwargs: Dict = {}):
        super().__init__(callable, args=args, kwargs=kwargs)
        self.layer_dict = {}
        self.shape_count = {}
        self.name = name

    def get(self,input_shape):
        if isinstance(input_shape,numbers.Number):
            shape_name = "_" + str(input_shape)
        else:
            input_shape = list(filter(None,input_shape))
            input_shape = [str(x) for x in input_shape]
            shape_name = "_" + "_".join(input_shape)
            
        if shape_name in self.shape_count.keys():
            self.shape_count[shape_name] += 1
        else:
            self.shape_count[shape_name] = 1
        
        name = self.name + shape_name + "_" + str(self.shape_count[shape_name])
        self.kwargs["name"] = name
        self.layer_dict[name] = self.callable(**self.args, **self.kwargs)
        return self.layer_dict[name]
    
    def find_by_size(self,input_shape):
        if isinstance(input_shape,numbers.Number):
            shape_name = "_" + str(input_shape)
        else:
            input_shape = list(filter(None,input_shape))
            input_shape = [str(x) for x in input_shape]
            shape_name = "_" + "_".join(input_shape)
        return [self.layer_dict[n] for n in self.layer_dict.keys() if shape_name in n]
        
        
        
        

class ActivationConfig(NamedCallableConfig):
    def __init__(self, callable: Callable, name: str, args: Dict = {}, kwargs: Dict = {}):
        super().__init__(callable, name, args=args, kwargs=kwargs)

