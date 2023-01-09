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
from scipy import linalg
from time_voice import onelead_model
import librosa.display, librosa
import matplotlib as mpl

def filter_ecg(signal, sampling_rate):
    
    signal = np.array(signal)
    order = int(0.3 * sampling_rate)
    filtered, _, _ = tools.filter_signal(signal=signal,
                                  ftype='FIR',
                                  band='bandpass',
                                  order=order,
                                  frequency=[5000, 8000],
                                  sampling_rate=sampling_rate)
    
    return filtered
    
    
def preprocess(path):
    
    file_list = os.listdir(path)
    file_json = sorted([file for file in file_list if file.endswith(".json")])
    file_mp3 = sorted([file for file in file_list if file.endswith(".wav")])
    
    print('list',file_list[3])

    lens=40000
    lens_1=3
    q=0
    data_clone = np.zeros((1,131072))
    data = np.zeros((lens_1,131072))
    tdata_label = np.zeros_like(data)
    data_label = np.zeros_like(data)
    
    for s,i in enumerate(range(len(file_json[lens:lens+lens_1]))):
      a = AudioSegment.from_file("{}{}".format(path,file_mp3[i]),format = 'wav')
      y = np.array(a.get_array_of_samples())
      if len(y)<131072:
        continue
        
      else:
        print('q',q)
        q+=1
        
      y =y[:131072]
      data[i]=y
      y = np.fft.fft(y) 
      y_ = filter_ecg(y,48000)
      #plt.show(y_)

    data = np.expand_dims(data,axis=-1)
    data_label = np.expand_dims(data_label,axis=-1)    
    return data, data_label
    
    
def preprocess_fourier(path):
    
    file_list = os.listdir(path)
    file_json = sorted([file for file in file_list if file.endswith(".json")])
    file_mp3 = sorted([file for file in file_list if file.endswith(".wav")])
    
    print(len(file_mp3))

    lens=40000
    lens_1=10
    data = np.zeros((lens_1,65536))
    
    for s,i in enumerate(range(len(file_json[lens:lens+lens_1]))):
      with open("{}{}".format(path,file_json[i])) as f:
        json_data = json.load(f)

      a = AudioSegment.from_file("{}{}".format(path,file_mp3[i]),format = 'wav')
      y = np.array(a.get_array_of_samples())
      if len(y)<70000:
        continue  
      y =y[1000:66536]
      print(s)
      data[i]=y

    data = np.expand_dims(data,axis=-1)
    data_label = np.zeros_like(data)
    tdata_label = np.zeros_like(data)

    for i in range(data_label.shape[0]):
      data_clone = max(data[i,:,0])*0.1+data[i,:,0]
      data_clone = filter_ecg(data_clone, 16000)
      data_label[i,:,0] = np.fft.fft(data_clone)/ len(data_clone)

    return data, data_label
    

def preprocess_spectrogram(path):
    
    file_list = os.listdir(path)
    file_json = sorted([file for file in file_list if file.endswith(".json")])
    file_mp3 = sorted([file for file in file_list if file.endswith(".wav")])
    for i in range(len(file_mp3)):
      if str(file_mp3[i]) == 'script1_s_0038-17291-02-02-KNG-M-08-A.wav':
        print('i : ',i)

    lens=40000
    lens_1= 10
    
    data_clone = np.zeros((1,65536))
    data = np.zeros((lens_1,65536))
    data_label = np.zeros((lens_1,1025,129))
    print(data.shape) 
    for s,i in enumerate(range(len(file_json[lens:lens+lens_1]))):
      with open("{}{}".format(path,file_json[i])) as f:
        json_data = json.load(f)
      a = AudioSegment.from_file("{}{}".format(path,file_mp3[i]),format = 'wav')
      y = np.array(a.get_array_of_samples(),dtype=np.float)
      if len(y)<70000:
        print('no')
        continue  
      y =y[1000:66536]

      data[i]=y
      y = skp.minmax_scale(y, (-1, 1))
      y = filter_ecg(y, 48000)
      y = librosa.stft(y, n_fft=2048, hop_length=512, win_length=2048)
      magnitude = np.abs(y)
      log_spectrogram = librosa.amplitude_to_db(magnitude)
      data_label[i] = log_spectrogram
    data = np.expand_dims(data,axis=-1)
    
    return data, data_label
    

def mae(y_true, y_pred):
    return np.abs(np.mean(np.subtract(y_pred, y_true),axis=-1))
    
    
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    
def prd(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred)**2) / np.sum(y_true**2)*100)
    

def MAE_(y_true, y_pred): 
	return np.mean(np.abs(y_true)*np.abs(y_true - y_pred))
    

def calculate_metric(y_true,y_pred,data_len,generator,generator2=None):
    print(data_len)
    
    result=0
    for j in range(data_len):
      RMSE=tf.keras.metrics.MeanSquaredError()
      RMSE.update_state(y_true[j][:,:],y_pred[j][:,:])
      result+=np.sqrt(RMSE.result().numpy())
    print('RMSE keras :',result/data_len)
    
    result =0
    for j in range(data_len):
      result+=rmse(y_true[j][:,:],y_pred[j][:,:])
    print('rmse :', result/data_len)
    result=0
    for j in range(data_len):
      MAE=tf.keras.metrics.MeanAbsoluteError()
      MAE.update_state(y_true[j][:,:],y_pred[j][:,:])
      result+=MAE.result().numpy()
    print('MAE :',result/data_len) 
    result=0
    
    for j in range(data_len):
      result+=prd(y_true[j][:,:],y_pred[j][:,:])
    print('prd :', result/data_len)

    
def test(tdata,tdata_label):

    pix2pix_generator= onelead_model() 
    pix2pix_generator.load_weights('/home/jhjoo/voice/tf/generator7.h5')   
    td=np.ones_like(tdata)

    for i in range(tdata.shape[0]):
      tt=np.expand_dims(tdata[i],axis=0)     
      predict_pix2pix = pix2pix_generator(tt, training=False)
      predict_pix2pix = np.fft.fft(predict_pix2pix[:,:,0])/len(predict_pix2pix[0,:,0])
      #predict_pix2pix = np.fft.fft(predict_pix2pix[0,:,0])/len(predict_pix2pix[0,:,0])
      #predict_pix2pix = filter_ecg(predict_pix2pix,48000)
      td[i,:,0]=predict_pix2pix

    predict_pix2pix= td
    data_label = np.ones_like(tdata_label)
    
    for i in range(tdata_label.shape[0]):
      #p = filter_ecg(tdata_label[i,:,0],48000)
      p = np.fft.fft(tdata_label[i,:,0])/len(tdata_label[i,:,0])
      data_label[i,:,0] = p

    tdata_label = data_label
           
    k=0
    for i in range(tdata_label.shape[0]):
      t=np.arange(0.,131072.,1)

      k=0
      fog,ax=plt.subplots(1,2,figsize=(20,10))
      fog.tight_layout()
      ax[0].set_title("predict")
      ax[1].set_title("label")
      ax[1].set_ylim([-50,50])
      ax[0].plot(t,predict_pix2pix[i][:,0])
      ax[1].plot(t,tdata_label[i][:,0])
    plt.show()   
    
    
def test_fourier(tdata,tdata_label):

    
    pix2pix_generator= onelead_model() 
    pix2pix_generator.load_weights('/home/jhjoo/voice/tf/generator7.h5')   
    
    td=np.zeros_like(tdata)
    for i in range(tdata.shape[0]):
      tt=np.expand_dims(tdata[i],axis=0)
      predict_pix2pix = pix2pix_generator(tt, training=False)
      predict_pix2pix = np.fft(predict_pix2pix[3,:,0])
      td[i,:,:]=predict_pix2pix[0,:,:]
    predict_pix2pix= td

    print('pix2pix ======')
    #calculate_metric(tdata_label,predict_pix2pix,tdata.shape[0],pix2pix_generator)
    
    data_label = np.zeros_like(tdata_label)
    print('u',tdata_label.shape)
    for i in range(data_label.shape[0]):  
      #data_label[i,:,0] = np.fft.fft(predict_pix2pix[i,:,0])/ len(predict_pix2pix[i,:,0]) 
      y = filter_ecg(predict_pix2pix[i,:,0], 16000)
      data_label[i,:,0] = np.fft.fft(y)/ len(y) 
      
    t=np.arange(0.,data_label.shape[1],1)

    k=0
    for i in range(10):
      fog,ax=plt.subplots(1,2,figsize=(20,10))
      fog.tight_layout()
      ax[0].set_title("v1")
      ax[1].set_title("label v1")
      ax[0].plot(t,data_label[i][:,:])
      ax[1].plot(t,tdata_label[i][:,:])
    plt.show()
    
    
def test_spectrogram(tdata,tdata_label):

    pix2pix_generator= onelead_model() 
    pix2pix_generator.load_weights('/home/jhjoo/voice/tf/generator7.h5')    
    td=np.zeros_like(tdata)
    for i in range(tdata.shape[0]):
      tt=np.expand_dims(tdata[i],axis=0)
      predict_pix2pix = pix2pix_generator(tt, training=False)
      td[i,:,:]=predict_pix2pix[0,:,:]
    predict_pix2pix= td
    data_label = np.zeros_like((tdata_label))
    
    for i in range(data_label.shape[0]):
      y = skp.minmax_scale(predict_pix2pix[i,:,0], (-1, 1))
      y = filter_ecg(y, 48000)
      y = librosa.stft(y,n_fft=2048, hop_length=512, win_length=2048)
      magnitude = np.abs(y)
      log_spectrogram = librosa.amplitude_to_db(magnitude)
      print(log_spectrogram.shape)
      data_label[i] = log_spectrogram

    print('pix2pix ======')
    
    calculate_metric(tdata_label,data_label,tdata.shape[0],pix2pix_generator)

    plt.figure(figsize=(20,10))
    librosa.display.specshow(data_label[3],sr=16000,hop_length=512)
    plt.show()
    
    librosa.display.specshow(tdata_label[3],sr=16000,hop_length=512)
    plt.show()


if __name__=='__main__':

  path = "/home/jhjoo/voice_data/train_data/cc/60~69/Male/"
  os.environ["CUDA_VISIBLE_DEVICES"]='-1'
  data, data_label = preprocess(path)
  test(data,data_label)
  

  

    

    
    



