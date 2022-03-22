import tensorflow as tf
import tensorflow.keras.backend as K

def simple_log_loss_disc(real_results,gen_results):
    log_real = K.log(real_results)
    log_gen = K.log(tf.ones_like(gen_results) - gen_results)
    return K.mean(log_real + log_gen,axis=0)

def simple_log_loss_gen(gen_results):
    log_res = K.log(tf.ones_like(gen_results) - gen_results)
    return K.mean(log_res,axis=0)