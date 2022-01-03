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
        print("--------------GEN DNA----------------")
        gDNA.summary()
        return Generator.from_DNA(gan_input,gDNA)
    
    @staticmethod
    def to_discriminator(gan_input: GanInput,generator: Generator):
        gDNA = generator.to_DNA()
        print("--------------GEN DNA----------------")
        gDNA.summary()
        dDNA = gDNA.to_disc_DNA()
        print("--------------DISC DNA----------------")
        dDNA.summary()
        return Discriminator.from_DNA(gan_input, dDNA)