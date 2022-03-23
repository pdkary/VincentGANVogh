from typing import Tuple
from tensorflow.keras.layers import Dense, Dropout, Reshape
from config.CallableConfig import ActivationConfig
from config.GanConfig import DenseLayerConfig
from models.builders.BuilderBase import BuilderBase
from third_party_layers.MinibatchDiscrimination import MinibatchDiscrimination
import numpy as np

class DenseModelBuilder(BuilderBase):

    def reshape(self,shape: Tuple[int,int,int]):
        self.out = Dense(np.prod(shape))(self.out)
        self.out = Reshape(shape)(self.out)
        return self

    def layer(self,size):
        self.out = Dense(size)(self.out)
        return self.out
    
    def track(self):
        pass

    def block(self,config: DenseLayerConfig):
        name = str(config.size) + "_" + str(self.layer_count)
        if config.minibatch_size > 0 and config.minibatch_dim > 0:
            self.out = MinibatchDiscrimination(config.minibatch_size,config.minibatch_dim)(self.out)
        self.out = self.layer(config.size)
        if config.dropout_rate > 0.0:
            self.out = Dropout(config.dropout_rate,name="dense_dropout_"+name)(self.out)
        self.out = config.activation.get()(self.out)
        if config.train_features:
            self.feature_layers.append(self.out)
        self.layer_count += 1
        return self
