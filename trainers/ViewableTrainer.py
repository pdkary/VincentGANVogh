from helpers.DataHelper import DataHelper
from inputs.GanInput import RealImageInput
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
        data_source: RealImageInput = self.D.gan_input
        data_helper: DataHelper = self.D.gan_input.data_helper
        gvi,dvi = self.gview_index, self.dview_index

        preview_seed = self.G.get_validation_batch(self.preview_size)
        gen_output = self.generator.predict(preview_seed)
        gen_images,gen_views = np.array(gen_output[0]),gen_output[gvi:]

        if self.D.view_layers:
            image_batch = data_source.get_validation_batch(self.preview_size)
            disc_real_out = self.discriminator.predict(image_batch)
            disc_real_preds,disc_real_views = disc_real_out[0],list(reversed(disc_real_out[dvi:]))
            disc_fake_out = self.discriminator.predict(gen_images)
            disc_fake_preds,disc_fake_views = disc_fake_out[0],list(reversed(disc_fake_out[dvi:]))
            
            print([x.shape for x in disc_real_views])
            print([x.shape for x in disc_fake_views])
            total_preds = np.append(disc_real_preds, disc_fake_preds)
            total_views = [np.append(disc_real_views[i],disc_fake_views[i]) for i in range(len(disc_real_views))]
            print([x.shape for x in total_views])
            data_helper.save_viewed_predictions("disc/"+name,total_preds,total_views,self.preview_margin)

        if self.G.view_layers:
            data_helper.save_viewed_images(name,gen_images,gen_views,self.preview_margin)
