from models.Generator import Generator
from models.Discriminator import Discriminator
from trainers.AbstractTrainer import AbstractTrainer
from config.TrainingConfig import GanTrainingConfig
import numpy as np

class ViewableTrainer(AbstractTrainer):
    def __init__(self,
                 generator: Generator,
                 discriminator: Discriminator,
                 gan_training_config: GanTrainingConfig):
        super().__init__(generator,discriminator,gan_training_config)
        self.gview_index = 1
        self.dview_index = 1
        
    def save_views(self,name):
        data_helper = self.D.gan_input
        gvi,dvi = self.gview_index, self.dview_index

        preview_seed = self.G.get_validation_batch(self.preview_size)
        gen_output = self.generator.predict(preview_seed)
        gen_images,gen_views = np.array(gen_output[0]),gen_output[gvi:]

        if self.D.view_layers:
            image_batch = data_helper.get_validation_batch(self.preview_size)
            disc_real_out = self.discriminator.predict(image_batch)
            disc_real_preds,disc_real_views = disc_real_out[0],list(reversed(disc_real_out[dvi:]))
            disc_fake_out = self.discriminator.predict(gen_images)
            disc_fake_preds,disc_fake_views = disc_fake_out[0],list(reversed(disc_fake_out[dvi:]))
            data_helper.save_viewed("disc/real/"+name,disc_real_preds,disc_real_views,self.preview_margin)
            data_helper.save_viewed("disc/fake/"+name,disc_fake_preds,disc_fake_views,self.preview_margin)

        if self.G.view_layers:
            data_helper.save_viewed(name,gen_images,gen_views,self.preview_margin)
