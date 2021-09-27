from typing import Callable, Dict

class CallableConfig():
    def __init__(self,
                 callable: Callable,
                 args: Dict = {},
                 kwargs: Dict = {}):
        self.callable = callable
        self.args = args
        self.kwargs = kwargs

    def get(self,name=None):
        if self.callable is None:
            return lambda x: x
        else:
            if name is not None:
                self.kwargs["name"] = self.callable.__name__
            return self.callable(**self.args, **self.kwargs)
        
class NormalizationConfig(CallableConfig):
    pass


class RegularizationConfig(CallableConfig):
    pass

class ActivationConfig(CallableConfig):
    pass

NoneCallable = CallableConfig(None)
        

