import numpy as np
import tensorflow as tf
from models.GanInput import RealImageInput
from keras_tuner import BayesianOptimization
from numpy.lib.function_base import median
from config.TrainingConfig import DataConfig, GanTrainingConfig
from models.Generator.HyperGAN import HyperGAN

class GanOptimizer(GanTrainingConfig):
    def __init__(self,
                 gan_training_config: GanTrainingConfig,
                 data_config:         DataConfig):
        super().__init__(**gan_training_config.__dict__)
        self.batch_size = data_config.batch_size
        self.preview_size = data_config.preview_cols*data_config.preview_rows
        self.hyper_gan = HyperGAN("hyper_generator",True,self.batch_size,self.preview_size)
        self.image_source: RealImageInput = RealImageInput(data_config)
        self.image_source.load()
        self.model_output_path = data_config.data_path + "/models"

        self.gen_tuner = BayesianOptimization(
            self.hyper_gan,
            objective="accuracy",
            max_trials=5)

    def tune_on_batch(self,batch_id,epochs=20,save_images=False):
        gen_input = self.hyper_gan.G.get_input()
        real_images = self.image_source.get_batch()
        
        gen_images = self.hyper_gan.generator(gen_input,training=False)
        
        print("="*100 + "     GENERATOR TRAINING      " + "="*100)
        self.gen_tuner.search(gen_input,tf.ones(shape=(self.batch_size)),epochs=epochs)
        print("="*100 + "     DISCRIMINATOR TRAINING      " + "="*100)
        self.hyper_gan.discriminator.fit(real_images,tf.ones(shape=(self.batch_size)),epochs=epochs)
        self.hyper_gan.discriminator.fit(gen_images,tf.zeros(shape=(self.batch_size)),epochs=epochs)
        
        if save_images:
            print("="*100 + "     SAVING IMAGES      " + "="*100)
            preview_seed = self.hyper_gan.G.get_input(training=False)
            generated_images = np.array(self.hyper_gan.generator.predict(preview_seed))
            self.image_source.save(batch_id,generated_images)
    
    def tune_n_batches(self,n,epochs,printerval=10):
        for i in range(n):
            self.tune_on_batch(i,epochs,i%printerval==0)
        filename = self.image_source.data_helper.model_name + "%d"%((i+1)*n)
        self.hyper_gan.generator.save(self.model_output_path + filename)