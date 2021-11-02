'''
MILCore contains functions for the training and validation of the 
MIL model used for disease classification

This code was adapted from the following repositories: 
https://github.com/utayao/Atten_Deep_MIL
https://github.com/AMLab-Amsterdam/AttentionDeepMIL

We thank the authors of these reposistories for their work 

Copyright (C) 2021, Rajaram Lab - UTSouthwestern 

This file is part of anc-2021-dl-wm-tauopathy.

anc-2021-dl-wm-tauopathy is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

anc-2021-dl-wm-tauopathy is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with anc-2021-dl-wm-tauopathy.  If not, see <http://www.gnu.org/licenses/>.

Anthony Vega, 2021
'''

import os as os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import numpy as np
import time

import glob
from collections import Counter 

import tensorflow as tf

import tensorflow as tf 
if tf.__version__[0]=='1':
    from tensorflow.python.keras.utils import Sequence
    from tensorflow.python.keras.utils import to_categorical
    from tensorflow.python.keras import backend as K
    from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
    from tensorflow.python.keras import activations, initializers, regularizers
    from tensorflow.python.keras.initializers import glorot_uniform
    from tensorflow.python.keras.utils import multi_gpu_model
    from tensorflow.python.keras.models import Model, load_model
    from tensorflow.python.keras.optimizers import SGD,Adam
    from tensorflow.python.keras.regularizers import l2
    from tensorflow.python.keras.layers import Input, Dense, Layer, Dropout, Conv2D, MaxPooling2D, Flatten, multiply
    # from .metrics import bag_accuracy, bag_loss
    # from .custom_layers import Mil_Attention, Last_Sigmoid
    from tensorflow.python.keras.applications import InceptionV3, VGG16,VGG19
    from tensorflow.python.keras.utils import CustomObjectScope
else:
    from tensorflow.keras.utils import Sequence
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
    from tensorflow.keras import activations, initializers, regularizers
    from tensorflow.keras.initializers import glorot_uniform
    from tensorflow.keras.utils import multi_gpu_model
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.optimizers import SGD,Adam
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.layers import Input, Dense, Layer, Dropout, Conv2D, MaxPooling2D, Flatten, multiply
    # from .metrics import bag_accuracy, bag_loss
    # from .custom_layers import Mil_Attention, Last_Sigmoid
    from tensorflow.keras.applications import InceptionV3, VGG16,VGG19
    from tensorflow.keras.utils import CustomObjectScope
from functools import partial

# %%

class MILGenerator(Sequence):
  'Generates data for Keras'
  def __init__(self, data, labels, sampleNumbers,
               numberOfClasses, batch_size=64,
                shuffle=True,preproc_fn= lambda x:np.float32(x)/255, augmentations = None):
      'Initialization'
      self.data=data
      self.numberOfPatches=data.shape[0]
      self.labels = labels       
      self.numberOfClasses=numberOfClasses
      self.sampleNumbers=sampleNumbers

      self.batch_size = batch_size
      self.shuffle = shuffle
      self.preproc=preproc_fn
      self.augmentations = augmentations
      self.sampleIdx={}
      self.uniqueSampleNumbers=np.unique(self.sampleNumbers)
      for sampleNumber in self.uniqueSampleNumbers:
          self.sampleIdx[sampleNumber]=np.where(self.sampleNumbers==sampleNumber)[0]      
      self.indexList=[]
      self.on_epoch_end()


  def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(self.numberOfPatches / self.batch_size))

  def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      
      indexes = self.indexList[index]


      dataOut=np.zeros((len(indexes),self.data.shape[1],self.data.shape[2],self.data.shape[3]),dtype=np.float32)
          
      for i,idx in enumerate(indexes):
           # squeeze image to eliminate extra dimension
          currPatch = np.squeeze(self.data[idx, :, :, :])
          if self.augmentations is not None:
              currPatch = self.augmentations.augment_image(currPatch)
          #dataOut[i,:,:,:]=self.preproc(np.squeeze(self.data[idx,:,:,:]))
          dataOut[i,:,:,:]=self.preproc(currPatch)
      return dataOut, to_categorical(self.labels[indexes], num_classes=self.numberOfClasses)#, indexes

  def on_epoch_end(self):
      'Updates indexes after each epoch'
      
      nBatch=int(np.floor(self.numberOfPatches / self.batch_size))
      pSample=np.array([len(self.sampleIdx[sampleNumber]) for sampleNumber in self.uniqueSampleNumbers])/self.numberOfPatches

      self.indexList=[]
      for i in range(nBatch):
          sampleNumber=np.random.choice(self.uniqueSampleNumbers,p=pSample)
          self.indexList.append(np.random.choice(self.sampleIdx[sampleNumber],size=self.batch_size))
          



              
              
        
# %% Custom Layers

class Last_Softmax(Layer):
    """
    Attention Activation

    This layer contains a FC layer which only has one neural with sigmoid actiavtion
    and MIL pooling. The input of this layer is instance features. Then we obtain
    instance scores via this FC layer. And use MIL pooling to aggregate instance scores
    into bag score that is the output of Score pooling layer.
    This layer is used in mi-Net.

    # Arguments
        output_dim: Positive integer, dimensionality of the output space
        kernel_initializer: Initializer of the `kernel` weights matrix
        bias_initializer: Initializer of the `bias` weights
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the `bias` weights
        use_bias: Boolean, whether use bias or not
        pooling_mode: A string,
                      the mode of MIL pooling method, like 'max' (max pooling),
                      'ave' (average pooling), 'lse' (log-sum-exp pooling)

    # Input shape
        2D tensor with shape: (batch_size, input_dim)
    # Output shape
        2D tensor with shape: (1, units)
    """
    # def __init__(self, output_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #                 kernel_regularizer=None, bias_regularizer=None,
    #                 use_bias=True, **kwargs):
    def __init__(self, output_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',name='FC1_Softmax',
                    kernel_regularizer=None, bias_regularizer=None,
                    use_bias=True, **kwargs):

        self.output_dim = output_dim

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        super(Last_Softmax, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = int(input_shape[1])

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                        initializer=self.kernel_initializer,
                                        name='kernel',
                                        regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None

        self.input_built = True

    def call(self, x, mask=None):
        n, d = x.shape
        x = K.sum(x, axis=0, keepdims=True)
        # compute instance-level score
        x = K.dot(x, self.kernel)
        if self.use_bias:
            x = K.bias_add(x, self.bias)

        # sigmoid
        out = K.softmax(x)


        return out

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = int(self.output_dim)
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'use_bias': self.use_bias
        }
        base_config = super(Last_Softmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class Last_Sigmoid(Layer):
    """
    Attention Activation

    This layer contains a FC layer which only has one neural with sigmoid actiavtion
    and MIL pooling. The input of this layer is instance features. Then we obtain
    instance scores via this FC layer. And use MIL pooling to aggregate instance scores
    into bag score that is the output of Score pooling layer.
    This layer is used in mi-Net.

    # Arguments
        output_dim: Positive integer, dimensionality of the output space
        kernel_initializer: Initializer of the `kernel` weights matrix
        bias_initializer: Initializer of the `bias` weights
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the `bias` weights
        use_bias: Boolean, whether use bias or not
        pooling_mode: A string,
                      the mode of MIL pooling method, like 'max' (max pooling),
                      'ave' (average pooling), 'lse' (log-sum-exp pooling)

    # Input shape
        2D tensor with shape: (batch_size, input_dim)
    # Output shape
        2D tensor with shape: (1, units)
    """
    # def __init__(self, output_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #                 kernel_regularizer=None, bias_regularizer=None,
    #                 use_bias=True, **kwargs):
    def __init__(self, output_dim=1, kernel_initializer='glorot_uniform', bias_initializer='zeros',name='FC1_sigmoid',
                    kernel_regularizer=None, bias_regularizer=None,
                    use_bias=True, **kwargs):

        self.output_dim = output_dim

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        super(Last_Sigmoid, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = int(input_shape[1])

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                        initializer=self.kernel_initializer,
                                        name='kernel',
                                        regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None

        self.input_built = True

    def call(self, x, mask=None):
        n, d = x.shape
        x = K.sum(x, axis=0, keepdims=True)
        # compute instance-level score
        x = K.dot(x, self.kernel)
        if self.use_bias:
            x = K.bias_add(x, self.bias)

        # sigmoid
        out = K.sigmoid(x)


        return out

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = int(self.output_dim)
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'use_bias': self.use_bias
        }
        base_config = super(Last_Sigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Normalize_Layer(Layer):

    # def __init__(self, output_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #                 kernel_regularizer=None, bias_regularizer=None,
    #                 use_bias=True, **kwargs):
    def __init__(self,  **kwargs):


        super(Normalize_Layer, self).__init__(**kwargs)



    def call(self, x):
        mag=1/(K.sum(x,axis=0)+1E-6)
        out = x*mag
        # compute instance-level score

        return out

    def compute_output_shape(self, input_shape):

        return input_shape

    def get_config(self):
        config = {
        }
        base_config = super(Normalize_Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# %% Losses and Metrics
def bag_accuracy(y_true, y_pred):
    """Compute accuracy of one bag.
    Parameters
    ---------------------
    y_true : Tensor (N x 1)
        GroundTruth of bag.
    y_pred : Tensor (1 X 1)
        Prediction score of bag.
    Return
    ---------------------
    acc : Tensor (1 x 1)
        Accuracy of bag label prediction.
    """
    y_true = K.mean(y_true, axis=0, keepdims=False)
    y_pred = K.mean(y_pred, axis=0, keepdims=False)
    acc = K.mean(K.equal(y_true, K.round(y_pred)))
    return acc


def bag_loss(y_true, y_pred):
    """Compute binary crossentropy loss of predicting bag loss.
    Parameters
    ---------------------
    y_true : Tensor (N x 1)
        GroundTruth of bag.
    y_pred : Tensor (1 X 1)
        Prediction score of bag.
    Return
    ---------------------
    loss : Tensor (1 x 1)
        Binary Crossentropy loss of predicting bag label.
    """
    y_true = K.mean(y_true, axis=0, keepdims=False)
    y_pred = K.mean(y_pred, axis=0, keepdims=False)
    loss = K.mean(K.binary_crossentropy(y_true, tf.squeeze(y_pred)), axis=-1)
    return loss


def weighted_bag_loss(weightsList):
    def lossFunc(true, pred):
        y_true = K.cast(K.mean(true, axis=0, keepdims=False), dtype='int32')
        loss = bag_loss(true,pred)
        loss = loss*K.gather(weightsList,y_true)
        return loss
    
    return lossFunc

def MilNetwork(input_dim, args, class_weights,numberOfClasses=2,activationType='softmax',useMulGpu=False):

    try:
        lr = args.init_lr
        weight_decay = args.init_lr
        momentum = args.momentum
    except:
        lr = args['lr']
        weight_decay = args['lr']
        momentum = args['momentum']

    model =VGG19(weights = "imagenet", include_top=False, input_shape = input_dim)
    

    # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
    for layer in model.layers[:5]:
        layer.trainable = False
    
    #Adding custom Layers 
    x = model.output
    # x = Conv2D(1024,kernel_size=(7,7),kernel_regularizer=l2(weight_decay),activation='relu')(x)
    x = Conv2D(1024,kernel_size=(7,7),kernel_regularizer=l2(weight_decay),activation='relu')(x)

    x = Dropout(0.5)(x)
    # x = Conv2D(1024,kernel_size=(1,1),kernel_regularizer=l2(weight_decay),activation='relu')(x)
    x = Conv2D(1024,kernel_size=(1,1),kernel_regularizer=l2(weight_decay),activation='relu',name='profile')(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)

    y = model.output
    # x = Conv2D(1024,kernel_size=(7,7),kernel_regularizer=l2(weight_decay),activation='relu')(x)
    y = Conv2D(1024,kernel_size=(7,7),activation='relu')(y)

    # x = Conv2D(1024,kernel_size=(1,1),kernel_regularizer=l2(weight_decay),activation='relu')(x)
    y = Conv2D(1,kernel_size=(1,1),activation='sigmoid',name='attention')(y)

    y = Flatten()(y)



    alpha = Normalize_Layer()(y)
    x_mul = multiply([alpha,x])
    
    if activationType =='softmax':
        out = Last_Softmax(numberOfClasses)(x_mul)
    elif activationType =='sigmoid':
        if numberOfClasses==2:
            out = Last_Sigmoid(output_dim=1)(x_mul)
        else:
            raise SystemExit('sigmoid activation only support two output classes')
    else:
        raise SystemExit(activationType+' is not valid activation type. Must be sigmoid or softmax')
    
    model = Model(inputs= model.input, outputs = out)

    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss=bag_loss, metrics=[bag_accuracy])
    else:
        # model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss=bag_loss, metrics=[bag_accuracy])
        model.compile(optimizer=SGD(lr=lr,momentum=momentum), loss=weighted_bag_loss(class_weights), metrics=[bag_accuracy])
        parallel_model = model

    return parallel_model
