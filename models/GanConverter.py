from inputs.GanInput import GanInput
from models.Discriminator import Discriminator
from models.Generator import Generator

class GanConverter:
    @staticmethod
    def to_generator(gan_input: GanInput,discriminator: Discriminator):
        dDNA = discriminator.to_DNA()
        print("--------------DISC DNA----------------")
        dDNA.summary()
        
        gDNA = dDNA.to_gen_DNA()
        gDNA.dense_layers = gDNA.dense_layers[1:]
        print("--------------GEN DNA----------------")
        gDNA.summary()
        return Generator.from_DNA(gan_input,gDNA)
    
    @staticmethod
    def to_discriminator(gan_input: GanInput,generator: Generator, output_dim: int = 1):
        gDNA = generator.to_DNA()
        dDNA = gDNA.to_disc_DNA()
        dDNA.dense_layers.append(output_dim)
        return Discriminator.from_DNA(gan_input, dDNA)