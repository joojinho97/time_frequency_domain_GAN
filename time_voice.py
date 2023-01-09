import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Conv2D,Dense,Conv2DTranspose
from tensorflow.python.keras.models import Model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
import time
import tensorflow.keras as tk
import tensorflow_addons as tfa
from biosppy.signals import tools as tools
import neural_structured_learning as nsl
from scipy import stats
from biosppy.signals import tools as tools
import neural_structured_learning as nsl
from scipy import stats
import sklearn.preprocessing as skp
import neural_structured_learning as nsl
from sklearn.utils import shuffle
from scipy import signal
import json
import librosa
time_len = 512
import tensorflow_io as tfio
from pydub import AudioSegment
from loss import *

def filter_ecg(signal, sampling_rate):
    
    signal = np.array(signal)
    order = int(0.3 * sampling_rate)
    filtered, _, _ = tools.filter_signal(signal=signal,
                                  ftype='FIR',
                                  band='bandpass',
                                  order=order,
                                  frequency=[3, 45],
                                  sampling_rate=sampling_rate)
    
    return filtered
    
    
def preprocess(path):
    
    file_list = os.listdir(path)
    
    file_json = sorted([file for file in file_list if file.endswith(".json")])
    file_mp3 = sorted([file for file in file_list if file.endswith(".wav")])
    
    print(len(file_mp3))

    lens=24000
    
    data_clone = np.zeros((1,131072))
    data = np.zeros((lens,131072))
    
    q=0
    data_label = np.zeros_like(data)
    for s,i in enumerate(range(len(file_json[:lens]))):



      a = AudioSegment.from_file("{}{}".format(path,file_mp3[i]),format = 'wav')
      y = np.array(a.get_array_of_samples())

      if len(y)<131072:
        continue
      elif q==24000:
        break
      else:
        print('q',q)
        q+=1 
      y =y[:131072]

      
      print(s)

      data[i]=y
      data_label[i] = y+np.max(y)*0.5
      

    data = np.expand_dims(data,axis=-1)  
    data_label =  np.expand_dims(data_label,axis=-1)



    print(data.shape)
    return data, data_label
    
    
    
def train_step(input_image, target, epoch,generator,discriminator,generator_optimizer,discriminator_optimizer,s,p):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      gen_output = generator(input_image, training=True)
    
      disc_real_output = discriminator([input_image,target], training=True)
      disc_generated_output = discriminator([input_image,gen_output], training=True)
    
      gen_total_loss, gen_gan_loss, gen_l1_loss  = generator_loss(disc_generated_output, gen_output, target,p)
      disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
      
    
    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))
    
    
    print('epoch {} gen_total_loss {} gen_gan_loss {} gen_l1_loss {}'.format(s,gen_total_loss,gen_gan_loss,gen_l1_loss))

kernel = 16
stride= 4
def onelead_model():


  
    initializer = tf.random_normal_initializer(0., 0.02)
    time_len = 131072

    
    encoder_inputs=keras.Input(shape=(time_len,1),name='data')
    
    x=tf.keras.layers.Conv1D(64, (kernel), strides=(stride), padding='same',kernel_initializer=initializer, use_bias=False)(encoder_inputs)#8
    x_0=tf.keras.layers.Activation('LeakyReLU')(x)
    
    x=tf.keras.layers.Conv1D(128, (kernel), strides=(stride), padding='same',kernel_initializer=initializer, use_bias=False)(x_0)#4
    x=tf.keras.layers.BatchNormalization()(x)
    x_1=tf.keras.layers.Activation('LeakyReLU')(x)
    
    x=tf.keras.layers.Conv1D(256, (kernel), strides=(stride), padding='same',kernel_initializer=initializer, use_bias=False)(x_1)#2
    x=tf.keras.layers.BatchNormalization()(x)
    x_2 =tf.keras.layers.Activation('LeakyReLU')(x)
   
    
    x=tf.keras.layers.Conv1D(512, (kernel), strides=(stride), padding='same',kernel_initializer=initializer, use_bias=False)(x_2)#2
    x=tf.keras.layers.BatchNormalization()(x)
    x_3=tf.keras.layers.Activation('LeakyReLU')(x)
    
    
    x=tf.keras.layers.Conv1D(1024, (kernel), strides=(stride), padding='same',kernel_initializer=initializer, use_bias=False)(x_3)#2
    x=tf.keras.layers.BatchNormalization()(x)
    x_4=tf.keras.layers.Activation('LeakyReLU')(x)
    

    
    x=tf.keras.layers.Conv1DTranspose(512, (kernel), strides=(stride),padding='same',kernel_initializer=initializer,use_bias=False)(x_4)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation('relu')(x)
    
    
    x=tf.keras.layers.Concatenate()([x,x_3])
    x=tf.keras.layers.Conv1DTranspose(256, (kernel), strides=(stride),padding='same',kernel_initializer=initializer,use_bias=False)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation('relu')(x)
    
    
    x=tf.keras.layers.Concatenate()([x,x_2])
    x=tf.keras.layers.Conv1DTranspose(128, (kernel), strides=(stride),padding='same',kernel_initializer=initializer,use_bias=False)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation('relu')(x)
    
    x=tf.keras.layers.Concatenate()([x,x_1])
    x=tf.keras.layers.Conv1DTranspose(64, (kernel), strides=(stride),padding='same',kernel_initializer=initializer,use_bias=False)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation('relu')(x)
    
    x=tf.keras.layers.Concatenate()([x,x_0])
    encoder_outputs = layers.Conv1DTranspose(1, (kernel), strides=(stride), padding="same",kernel_initializer=initializer, use_bias=False)(x)

    return keras.Model(encoder_inputs,encoder_outputs)
    
    
    
    
    
    



def train_(train,train_label,p,path):
    
    epochs=10
    time_len = 131072
    batch_size = 32
    initializer = tf.random_normal_initializer(0., 0.02)
    

    encoder_inputs=keras.Input(shape=(time_len,1),name='data')
    
    x=tf.keras.layers.Conv1D(64, (kernel), strides=(stride), padding='same',kernel_initializer=initializer, use_bias=False)(encoder_inputs)#8
    x_0=tf.keras.layers.Activation('LeakyReLU')(x)
    
    x=tf.keras.layers.Conv1D(128, (kernel), strides=(stride), padding='same',kernel_initializer=initializer, use_bias=False)(x_0)#4
    x=tf.keras.layers.BatchNormalization()(x)
    x_1=tf.keras.layers.Activation('LeakyReLU')(x)
    
    x=tf.keras.layers.Conv1D(256, (kernel), strides=(stride), padding='same',kernel_initializer=initializer, use_bias=False)(x_1)#2
    x=tf.keras.layers.BatchNormalization()(x)
    x_2 =tf.keras.layers.Activation('LeakyReLU')(x)
   
    
    x=tf.keras.layers.Conv1D(512, (kernel), strides=(stride), padding='same',kernel_initializer=initializer, use_bias=False)(x_2)#2
    x=tf.keras.layers.BatchNormalization()(x)
    x_3=tf.keras.layers.Activation('LeakyReLU')(x)
    
    
    x=tf.keras.layers.Conv1D(1024, (kernel), strides=(stride), padding='same',kernel_initializer=initializer, use_bias=False)(x_3)#2
    x=tf.keras.layers.BatchNormalization()(x)
    x_4=tf.keras.layers.Activation('LeakyReLU')(x)
    

    
    x=tf.keras.layers.Conv1DTranspose(512, (kernel), strides=(stride),padding='same',kernel_initializer=initializer,use_bias=False)(x_4)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation('relu')(x)
    
    
    x=tf.keras.layers.Concatenate()([x,x_3])
    x=tf.keras.layers.Conv1DTranspose(256, (kernel), strides=(stride),padding='same',kernel_initializer=initializer,use_bias=False)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation('relu')(x)
    
    
    x=tf.keras.layers.Concatenate()([x,x_2])
    x=tf.keras.layers.Conv1DTranspose(128, (kernel), strides=(stride),padding='same',kernel_initializer=initializer,use_bias=False)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation('relu')(x)
    
    x=tf.keras.layers.Concatenate()([x,x_1])
    x=tf.keras.layers.Conv1DTranspose(64, (kernel), strides=(stride),padding='same',kernel_initializer=initializer,use_bias=False)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation('relu')(x)
    
    x=tf.keras.layers.Concatenate()([x,x_0])
    encoder_outputs = layers.Conv1DTranspose(1, (kernel), strides=(stride), padding="same",kernel_initializer=initializer, use_bias=False)(x)

    
    
    generator=keras.Model(encoder_inputs,encoder_outputs)
    
    
    
    generator.summary()
    

    
    inp = tf.keras.layers.Input(shape=[time_len,1], name='input_image')
    tar = tf.keras.layers.Input(shape=[time_len,1], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])
    x = tf.keras.layers.Conv1D(64, (4), strides=(4),kernel_initializer=initializer,padding='same', use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x=tf.keras.layers.Conv1D(128, (4), strides=(4),kernel_initializer=initializer,padding='same', use_bias=False)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x=tf.keras.layers.Conv1D(256, (4), strides=(4),kernel_initializer=initializer,padding='same', use_bias=False)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.ZeroPadding1D()(x)
    conv = tf.keras.layers.Conv1D(512, (4), strides=(4),kernel_initializer=initializer,padding='same',use_bias=False)(x)
    batchnorm1=tf.keras.layers.BatchNormalization()(conv)
    
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding1D()(leaky_relu)
    last = tf.keras.layers.Conv1D(1, (4), strides=(4),kernel_initializer=initializer,activation='sigmoid')(zero_pad2)
    
    
    discriminator=tf.keras.Model(inputs=[inp,tar], outputs=last)
    discriminator.summary()
    


    


    
    generator_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(1e-4), beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(1e-4), beta_1=0.5)

    for epoch in range(epochs):
      
      print("Epoch: ", epoch)

      
      for i in range(int(len(train)/32)-1):
        tr=train[batch_size*i:batch_size*(i+1)]
        tr_label=train_label[batch_size*i:batch_size*(i+1)]
        train_step(tr, tr_label, epoch,generator,discriminator,generator_optimizer,discriminator_optimizer,epoch,p)

      if (epoch + 1) % 1 == 0:
        generator.save_weights(f'/home/jhjoo/voice/tf_hightone/generator{epoch+1}.h5')

    

    

if __name__=='__main__':
    
    path = "/home/jhjoo/voice_data/train_data/cc/60~69/Male/"
    p = 50
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    data, data_label = preprocess(path)
    train_(data,data_label,p,path)    
    

    
    



