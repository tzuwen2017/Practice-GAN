# coding: utf-8
from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LSTM, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
import requests
import pandas as pd
from io import StringIO 
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt

import sys

import numpy as np

class GANFD():
    def __init__(self):
             
        # Optimizer
        optimizer_g = Adam(0.002, 0.5)
        optimizer_d = Adam(0.00002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer_d,
            metrics=['accuracy'])
       
        # Build the generator
        self.generator = self.build_generator()
        
        # Pretrain the generator
        self.lstm = self.build_generator()
        self.lstm.compile(loss=self.Lp, optimizer=optimizer_g)
        self.train_lstm(self.lstm,epochs=50,batch_size=64)
        lstm_weight = self.lstm.get_weights()
        #print(self.lstm.get_weights()[0])
        #print("=========================================")
        
        # The generator takes noise as input and generates imgs        
        X = Input(shape=(60,5))
        Y_false_t1 = self.generator(X)
        
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
                
        # The discriminator takes generated images as input and determines validity
        true_Y_T = Input(shape=(60,1))       
        valid = self.discriminator([true_Y_T,Y_false_t1])
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([X,true_Y_T], [valid,Y_false_t1])
        #print(self.combined.get_weights()[0])
        #print("=========================================")
        
        self.combined.set_weights(lstm_weight)
        self.combined.compile(loss=['mse',self.Lp], optimizer=optimizer_g)
        #self.combined.set_weights(lstm_weight)
        
        #print(self.combined.get_weights()[0])
        #self.combined.load_weights('lstm_model_weights.h5',by_name=True)
        # show model
        # self.generator.summary()
        # self.discriminator.summary()
    
    def RMSRE(y_true, y_pred):
        return np.sqrt(np.mean((((y_pred - y_true) / y_true) ** 2)))
    
    def DPA(y_true, y_pred):
        i = np.zeros((y_true.shape[0]))
        for t in range(len(i)):
            if (y_true[t+1]-y_true[t])(y_pred[t+1]-y_true[t]) > 0:
                i[t] = 1
        return np.mean(i)
    
    def Lp(y_true, y_pred):
        x = y_true - y_pred
        t = K.sum(x**2) ** (1./2)
        #return tf.linalg.norm(y_true - y_pred, ord=2)
        return t
          
    def build_generator(self):
        
        inputs = Input(shape=(60,5), name="g_input")
        x = LSTM(5, return_sequences=False, name="g_LSTM")(inputs)
        x = Dense(1,activation='linear', name="g_outputs")(x)
        outputs = Reshape((1,1))(x)

        return Model(inputs, outputs)

    def build_discriminator(self):
        
        inputs_T = Input(shape=(60,1), name="d_input_1")
        inputs_t1 = Input(shape=(1,1), name="d_input_2")  
        merged = Concatenate(axis=1)([inputs_T, inputs_t1])
        #inputs = Input(shape=(61,1), name="d_input_1")
        x = Conv1D(32, 4, strides=2, name="d_1d_1")(merged)
        x = LeakyReLU(alpha=0.01)(x)
        x = Conv1D(64, 4, strides=2, name="d_1d_2")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)
        '''
        x = Convolution1D(128, 4, strides=2, name="d_1d_3")(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        '''
        x = Dense(128, name="d_4")(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Flatten()(x)
        validity = Dense(1, activation='sigmoid', name="d_outputs")(x)

        return Model([inputs_T, inputs_t1], validity)
    def train_lstm(self, model, epochs, batch_size):
    
        model.fit(train_x,train_y[:,-1].reshape(train_y.shape[0],1,1),epochs=epochs,batch_size=batch_size)
        
        epoch = -1
        self.predict(model,epoch)

    def train(self, epochs,sample_interval=10):
        
        # Adversarial ground truths
        valid = np.ones((train_y.shape[0], 1))
        fake = np.zeros((train_y.shape[0], 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise and generate a batch of new images
            gen_y_t1 = self.generator.predict(train_x)
                      
            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch([train_y[:,:-1],train_y[:,-1].reshape(train_y.shape[0],1,1)], valid)
            d_loss_fake = self.discriminator.train_on_batch([train_y[:,:-1],gen_y_t1], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch([train_x,train_y[:,:-1]], [valid,train_y[:,-1].reshape(train_y.shape[0],1,1)])
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] [G lp: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.predict(self.generator,epoch)
                   
    def predict(self,model, epoch):
        #predict
        pre_test = model.predict(test_x)
        pre_train = model.predict(train_x)
        
        #還原
        pre_test = Re_normalize(pre_test,y_mean, y_maxmin)
        pre_train = Re_normalize(pre_train,y_mean, y_maxmin)
        
        tru_train = Re_normalize(train_y[:,-1].reshape(train_y.shape[0],1,1),y_mean, y_maxmin)
        tru_test = Re_normalize(test_y[:,-1].reshape(test_y.shape[0],1,1),y_mean, y_maxmin)
        tru_test_y1t = Re_normalize(test_y[:,-2].reshape(test_y.shape[0],1,1),y_mean, y_maxmin)
            
        print(np.mean(abs(pre_test - tru_test)))
        print(np.mean(abs(pre_train - tru_train)))
        print(np.mean(abs(tru_test_y1t - tru_test)))
        
        plt.plot(tru_test.ravel(), label='tru',color='green')
        plt.plot(pre_test.ravel(), label='pre',color='red')        
        plt.legend(loc="lower left")
        plt.savefig("images1/test_%d.png" % epoch)
        plt.close()
        
        plt.plot(tru_train.ravel(), label='tru',color='green')
        plt.plot(pre_train.ravel(), label='pre',color='red')        
        plt.legend(loc="lower left")
        plt.show()
        
def normalize(train):
    mean = train.apply(lambda x: np.mean(x))
    maxmin = train.apply(lambda x: np.max(x) - np.min(x))
    train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))    
    return train_norm, mean, maxmin
    
def Re_normalize(x,mean, maxmin):    
    return x * maxmin + mean

#從存起來的地方載入
txdata = pd.read_pickle("./tx170101_170301.pkl")
    
tx_x_data, x_mean, x_maxmin = normalize(txdata.loc[:,['Open','High','Low','Close','Volume']])
tx_y_data, y_mean, y_maxmin = normalize(txdata.loc[:,['Close']])
tx_x_data = np.asarray(tx_x_data)
tx_y_data = np.asarray(tx_y_data)
x_mean = np.asarray(x_mean)
x_maxmin = np.asarray(x_maxmin)
y_mean = np.asarray(y_mean)
y_maxmin = np.asarray(y_maxmin)
  
#以天為單位把每天資料切下來處理(每天18000筆)
m=60 #取前60秒
n=1 #預測後1秒
    
x=[]
y=[]
y_combined=[]
y_t=[]
for i in range(int(len(tx_y_data)/18000)):
    x_data = tx_x_data[i*18000:(i+1)*18000].astype(float).reshape(18000,5)
    y_data = tx_y_data[i*18000:(i+1)*18000].astype(float).reshape(18000,1)
    #每70筆一組(前60為x,後10為y)
    for j in range(18000-m-n+1):
        x.append(x_data[j:j+m])            
        y.append(y_data[j:j+m+n])
        y_combined.append(y_data[j:j+m])
        y_t.append(y_data[j+m-1:j+m])
        #y.append(x_data[j+m:j+m+n])
        
x = np.asarray(x)
y = np.asarray(y)
y_combined = np.asarray(y_combined)
y_t = np.asarray(y_t)
    
train_x = x[0:(18000-m-n+1)*20,:,:]  #取前20天當訓練集
train_y = y[0:(18000-m-n+1)*20,:]
train_y_combined = y_combined[0:(18000-m-n+1)*20,:]
    
test_x = x[(18000-m-n+1)*20:(18000-m-n+1)*21,:,:]  #21天
test_y = y[(18000-m-n+1)*20:(18000-m-n+1)*21,:]
test_y_combined = y_combined[(18000-m-n+1)*20:(18000-m-n+1)*21,:]

ganfd = GANFD()
ganfd.train(epochs=500)
