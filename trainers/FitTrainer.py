from trainers.AbstractTrainer import AbstractTrainer


class FitTrainer(AbstractTrainer):
    def train_generator(self, source_input, gen_input):
        loss,avg = self.generator.train_on_batch(gen_input,source_input)
        return loss,avg
    
    def train_discriminator(self, disc_input, gen_input):
        generated_images = self.generator(gen_input, training=False)
        real_loss,real_avg = self.discriminator.train_on_batch(disc_input,self.real_label)
        fake_loss,fake_avg = self.discriminator.train_on_batch(generated_images,self.fake_label)
        loss = (real_loss + fake_loss)/2
        avg = (real_avg + fake_avg)/2
        return loss,avg
