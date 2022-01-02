from inputs.GanInput import GanInput

class BatchedInputModel():
    def __init__(self,gan_input:GanInput):
        self.gan_input: GanInput = gan_input
        self.input = self.gan_input.input_layer
        
    def get_training_batch(self,batch_size):
        b = [self.gan_input.get_training_batch(batch_size)]
        return b

    def get_validation_batch(self,batch_size):
        b = [self.gan_input.get_validation_batch(batch_size)]
        return b