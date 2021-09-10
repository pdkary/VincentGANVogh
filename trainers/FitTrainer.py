from typing import List
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from models.Generator import Generator
from models.Discriminator import Discriminator
from models.GanInput import RealImageInput
from numpy.lib.function_base import median
from config.DiscriminatorConfig import DiscriminatorModelConfig
from config.GeneratorConfig import GeneratorModelConfig
from config.TrainingConfig import DataConfig, GanTrainingConfig

cross_entropy = tf.keras.losses.BinaryCrossentropy()


class FitTrainer(GanTrainingConfig):
    def __init__(self,
               gen_model_config:    GeneratorModelConfig,
               disc_model_config:   DiscriminatorModelConfig,
               gan_training_config: GanTrainingConfig,
               data_config: DataConfig):
        GanTrainingConfig.__init__(self,**gan_training_config.__dict__)
        self.batch_size = data_config.batch_size
        self.preview_size = data_config.preview_cols*data_config.preview_rows
        self.G: Generator = Generator(gen_model_config,self.batch_size,self.preview_size)
        self.D: Discriminator = Discriminator(disc_model_config)
        self.image_source: RealImageInput = RealImageInput(data_config)
        self.model_output_path = data_config.data_path + "/models"
        
        self.generator = self.G.build()
        self.discriminator = self.D.build()
        D = self.discriminator
        D.trainable = False
        adversarial_model = D(self.G.functional_model)
        self.A = Model(inputs=self.G.input,outputs=adversarial_model,name="Adversarial_Model")
        self.A.compile(optimizer=self.G.gen_optimizer,
                           loss="binary_crossentropy",
                           metrics=['accuracy'])
        self.A.summary()
        self.image_source.load()
    
    def train_on_batch(self,batch_id,epochs=20,save_images=False):
        gen_input = self.G.get_input(self.batch_size)
        real_images = self.image_source.get_batch()
        gen_images = self.generator(gen_input,training=False)
        
        print("="*100 + "     GENERATOR TRAINING      " + "="*100)
        self.generator.fit(gen_input,tf.ones(shape=(self.batch_size)),epochs=epochs)
        print("="*100 + "     DISCRIMINATOR TRAINING      " + "="*100)
        self.discriminator.fit(real_images,tf.ones(shape=(self.batch_size)),epochs=epochs)
        self.discriminator.fit(gen_images,tf.zeros(shape=(self.batch_size)),epochs=epochs)
        
        if save_images:
            print("="*100 + "     SAVING IMAGES      " + "="*100)
            preview_seed = self.G.get_input(self.preview_size)
            generated_images = np.array(self.generator.predict(preview_seed))
            self.image_source.save(batch_id,generated_images)
            
    def train_n_batches(self,batches,epochs,printerval):
        for i in range(batches):
            self.train_on_batch(i,epochs,printerval)
            filename = self.image_source.data_helper.model_name + "%d"%((i+1)*epochs)
            self.generator.save(self.model_output_path + filename)