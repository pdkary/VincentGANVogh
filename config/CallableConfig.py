from typing import Callable, Dict

from tensorflow.python.keras.layers.advanced_activations import LeakyReLU

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
            return lambda x: x
        else:
            return self.callable(*self.args, **self.kwargs)
        
class NormalizationConfig(CallableConfig):
    pass


class RegularizationConfig(CallableConfig):
    pass

class ActivationConfig(CallableConfig):
    def __init__(self, callable: Callable, args: Dict = {}, kwargs: Dict = {}):
        super().__init__(callable, args=args, kwargs=kwargs)
        if "activation" in kwargs.keys():
            self.name = kwargs["activation"]
        elif callable.__class__ == LeakyReLU.__class__:
            self.name = "leakyRelu_p" + str(kwargs["alpha"])[2:]

NoneCallable = CallableConfig(None)
        

