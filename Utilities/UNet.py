'''
UNet contains functions for the tranining and inference for the
UNet model used for aggregate classification

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
import openslide as oSlide
import DLUtils as dlutil
import ImageUtils as iu
from skimage.morphology import remove_small_objects
import progressbar
import re
from PIL import Image,ImageDraw
import pickle
import numpy as np

import tensorflow as tf
if tf.__version__[0]=='1':
    from tensorflow.python.keras.layers import Conv2D, MaxPooling2D,Input,UpSampling2D,Lambda
    from tensorflow.python.keras.utils import Sequence
    from tensorflow.python.keras.utils import to_categorical
    from tensorflow.python.keras import backend as K
    from tensorflow.python.keras.models import Model
    from tensorflow.python.keras.models import load_model
else:
    from tensorflow.keras.layers import Conv2D, MaxPooling2D,Input,UpSampling2D,Lambda
    from tensorflow.keras.utils import Sequence
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Model
    from tensorflow.keras.models import load_model

from functools import partial

import time
import imgaug as ia
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
import cv2

# %%
def image_softmax(input):
  label_dim = -1
  d = K.exp(input - K.max(input, axis=label_dim, keepdims=True))
  return d / K.sum(d, axis=label_dim, keepdims=True)
image_softmax.__name__='image_softmax'

def crop_and_concatenate(comb):
    x1=comb[0]
    x2=comb[1]
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [x1_shape[0], x2_shape[1], x2_shape[2], x1_shape[3]]
    #x1_crop = tf.slice(x1, offsets, size)
    x1_crop=x1[:,offsets[1]:(offsets[1]+size[1]),offsets[2]:(offsets[2]+size[2]),:]
    return K.concatenate([x1_crop, x2], -1)
def concat_output_shape(comb):
    shape=list(tf.shape(comb[0]))
    shape[3]=shape[3]+tf.shape(comb[1])
    return tuple(shape)

def soft_dice(y_true, y_pred):
    # y_pred is softmax output of shape (num_samples, num_classes)
    # y_true is one hot encoding of target (shape= (num_samples, num_classes))
    intersect = K.sum(y_pred * y_true, axis=[0,1,2])
    denominator = K.sum(y_pred, axis=[0,1,2]) + K.sum(y_true, axis=[0,1,2])
    dice_scores = K.constant(2) * intersect / (denominator + K.constant(1e-6))
    return 1-K.mean(dice_scores)

def hard_dice(y_true, y_pred):
    # y_pred is softmax output of shape (num_samples, num_classes)
    # y_true is one hot encoding of target (shape= (num_samples, num_classes))
    y_predH=K.one_hot(K.argmax(y_pred,axis=-1),tf.shape(y_pred)[-1])
    intersect = K.sum(y_predH * y_true, axis=[0,1,2])
    denominator = K.sum(y_predH, axis=[0,1,2]) + K.sum(y_true, axis=[0,1,2])
    dice_scores = K.constant(2) * intersect / (denominator + K.constant(1e-6))
    return 1-K.mean(dice_scores)

def soft_jaccard(y_true, y_pred):
    # y_pred is softmax output of shape (num_samples, num_classes)
    # y_true is one hot encoding of target (shape= (num_samples, num_classes))
    intersect = K.sum(y_pred * y_true, axis=[0,1,2])
    denominator = (K.sum(y_pred, axis=[0,1,2]) + K.sum(y_true, axis=[0,1,2]))-intersect
    jacc_scores =  intersect / (denominator + K.constant(1e-6))
    return 1-K.mean(jacc_scores)

def generalized_tversky(y_true,y_pred,a,b):
    intersect = K.sum(y_pred * y_true, axis=[0,1,2])
    t1=K.sum(y_pred * (1-y_true), axis=[0,1,2])
    t2=K.sum((1-y_pred) * y_true, axis=[0,1,2])
    tversky_scores=intersect/(intersect+a*t1+b*t2+ K.constant(1e-6))
    return 1-K.mean(tversky_scores)

def zeropad(comb):
    inputTensor=comb[0]
    targetTensor=comb[1]
    inputshape = tf.shape(inputTensor)
    targetshape = tf.shape(targetTensor)
    #paddings=[[0,m-inputshape[i]] for (i,m) in enumerate (targetshape)]
    paddings=[[0,0],[0,targetshape[1]-inputshape[1]],[0,targetshape[2]-inputshape[2]],[0,0]]
    # offsets for the top left corner of the crop
    
    return tf.pad(inputTensor,paddings,'CONSTANT',constant_values=0)



__EPS = 1e-5

def weighted_image_categorical_crossentropy(y_true, y_pred,weights):
    y_pred = K.clip(y_pred, __EPS, 1 - __EPS)
    score=-K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred),axis=-1)
    w1=K.reshape(K.gather(weights,K.argmax(y_true)),K.shape(score))
    #w2=K.reshape(K.gather(weights,K.argmax(y_pred)),K.shape(score))
    #return K.mean(score*(w1+w2)/2)

    return K.mean(score*w1)

def spatial_weighted_image_categorical_crossentropy(y_trueIn, y_pred,weights,spatialWeightMax,spatialWeightExp):
    y_true=y_trueIn[:,:,:,:-1]
    dists=y_trueIn[:,:,:,-1]
    y_pred = K.clip(y_pred, __EPS, 1 - __EPS)
    score=-K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred),axis=-1)
    w1=K.reshape(K.gather(weights,K.argmax(y_true)),K.shape(score))
    w2=spatialWeightMax*K.exp(-spatialWeightExp*dists)
    mask=tf.equal(dists,0)
    #w2=tf.multiply(tf.boolean_mask(w2,mask),K.cast(mask,'float32'))
    zeros = tf.zeros_like(w2)
    w2=tf.where(mask,zeros,w2)
    #w2=K.reshape(K.gather(weights,K.argmax(y_pred)),K.shape(score))
    #return K.mean(score*(w1+w2)/2)
    w=w1+w2

    return K.mean(score*w)

def spatial_weighted_image_categorical_crossentropy_alt(y_trueIn, y_pred,weights):
    y_true=y_trueIn[:,:,:,:-1]
    spatialWeights=y_trueIn[:,:,:,-1]
    y_pred = K.clip(y_pred, __EPS, 1 - __EPS)
    score=-K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred),axis=-1)
    w1=K.reshape(K.gather(weights,K.argmax(y_true)),K.shape(score))
    w2=spatialWeights

    #w2=K.reshape(K.gather(weights,K.argmax(y_pred)),K.shape(score))
    #return K.mean(score*(w1+w2)/2)
    w=w1+w2

    return K.mean(score*w)

def image_categorical_crossentropy(y_true, y_pred):
    y_pred = K.clip(y_pred, __EPS, 1 - __EPS)
    score=-K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred),axis=-1)

    return K.mean(score)

def weighted_composition_loss(y_true, y_pred,weights):
    y_pred = K.clip(y_pred, __EPS, 1 - __EPS)
    score=-K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred),axis=-1)
    w1=K.reshape(K.gather(weights,K.argmax(y_true)),K.shape(score))
    pixelLevelLoss=K.mean(score*w1)

    nClasses=K.shape(y_true)[-1]
    nPix=K.shape(y_true)[1]*K.shape(y_true)[2]


    # outShape=(K.shape(y_true)[0],nPix)
    # groundTruthClasses=K.one_hot(K.reshape(K.argmax(y_true),outShape),nClasses)
    # groundTruthComposition=K.sum(groundTruthClasses,axis=1)/K.cast(nPix,'float32')
    # predictedClasses=K.one_hot(K.reshape(K.argmax(y_pred),outShape),nClasses)
    # predictedComposition=K.sum(predictedClasses,axis=1)/K.cast(nPix,'float32')

    groundTruthComposition=K.sum(y_true,axis=[1,2])/K.cast(nPix,'float32')
    predictedComposition=K.sum(y_pred,axis=[1,2])/K.cast(nPix,'float32')

    compositionLoss=K.mean(K.categorical_crossentropy(groundTruthComposition,predictedComposition))


    #w2=K.reshape(K.gather(weights,K.argmax(y_pred)),K.shape(score))
    #return K.mean(score*(w1+w2)/2)  
    return pixelLevelLoss+compositionLoss


def weighted_image_categorical_crossentropy_symmetrized(y_true, y_pred,weights):
    y_pred = K.clip(y_pred, __EPS, 1 - __EPS)
    score=-K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred),axis=-1)
    w1=K.reshape(K.gather(weights,K.argmax(y_true)),K.shape(score))
    w2=K.reshape(K.gather(weights,K.argmax(y_pred)),K.shape(score))
    return K.mean(score*(w1+w2)/2)  

def composition_loss(y_true,y_pred):
    nPix=K.shape(y_true)[1]*K.shape(y_true)[2]
    groundTruthComposition=K.sum(K.cast(y_true,'float32'),axis=[1,2])/K.cast(nPix,'float32')
    predictedComposition=K.sum(K.cast(y_pred,'float32'),axis=[1,2])/K.cast(nPix,'float32')

    compositionLoss=K.mean(K.categorical_crossentropy(groundTruthComposition,predictedComposition))
    return compositionLoss
# %%  


class UNet():
    def __init__(self):
        print ('build UNet ...')

    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)
    

    def create_model(self, img_shape, num_class):

        
        inputs = Input(shape = img_shape)

        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up_conv5 = UpSampling2D(size=(2, 2))(conv5)
        #ch, cw = self.get_crop_shape(conv4, up_conv5)
        #crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
        #up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        up6=Lambda(crop_and_concatenate,output_shape=concat_output_shape)([conv4,up_conv5])
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up_conv6 = UpSampling2D(size=(2, 2))(conv6)
        #ch, cw = self.get_crop_shape(conv3, up_conv6)
        #crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
        #up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
        up7=Lambda(crop_and_concatenate)([conv3,up_conv6])
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up_conv7 = UpSampling2D(size=(2, 2))(conv7)
        #ch, cw = self.get_crop_shape(conv2, up_conv7)
        #crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
        #up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        up8=Lambda(crop_and_concatenate)([conv2,up_conv7])
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up_conv8 = UpSampling2D(size=(2, 2))(conv8)
        #ch, cw = self.get_crop_shape(conv1, up_conv8)
        #crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
        #up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        up9=Lambda(crop_and_concatenate)([conv1,up_conv8])
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        #ch, cw = self.get_crop_shape(inputs, conv9)
        #conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        conv9=Lambda(zeropad)([conv9,inputs])
        conv10 = Conv2D(num_class, (1, 1),activation=image_softmax)(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        return model
    
      


# %%


class MaskDataGenerator(Sequence):
  'Generates data for Keras'
  def __init__(self, imgList, maskList, maskSamplingIndex, numberOfClasses, patchSize,batch_size=16, 
                bgClass=0,shuffle=True,preproc_fn= lambda x:np.float32(x)/255,
                augmentations=None):
      'Initialization'
      self.imgList=imgList
      self.numberOfPatches=len(maskSamplingIndex)
      self.maskList = maskList
      self.numberOfClasses=numberOfClasses
      self.batch_size = batch_size
      self.shuffle = shuffle
      self.preproc=preproc_fn
      self.patchSize=patchSize
      self.bgClass=bgClass
      self.samplingIndex=maskSamplingIndex
      self.augmentations=augmentations
      self.on_epoch_end()

  def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(self.numberOfPatches / self.batch_size))

  def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      indexes = self.samplingIndex[self.indexes[index*self.batch_size:(index+1)*self.batch_size]]
      #indexes = self.indexes[index]
      
      X=np.zeros((len(indexes),self.patchSize,self.patchSize,3),dtype=np.float32)
      Y=np.zeros((len(indexes),self.patchSize,self.patchSize,self.numberOfClasses),dtype=np.float32)
      for i,idx in enumerate(indexes):
         img=self.imgList[idx]
         mask=self.maskList[idx]
         #if((mask.shape[0]*mask.shape[1])/(self.patchSize*self.patchSize)<9):
         if img.shape[0]-(self.patchSize+1)>0:
             r=np.random.randint(img.shape[0]-(self.patchSize+1))
         else:
             r=0
         if img.shape[1]-(self.patchSize+1)>0:
            c=np.random.randint(img.shape[1]-(self.patchSize+1))
         else:
            c=0

         imgCrop=(img[r:(r+self.patchSize),c:(c+self.patchSize),:])
         maskCrop=mask[r:(r+self.patchSize),c:(c+self.patchSize)]
         if self.augmentations is not None:
             segmap=SegmentationMapsOnImage(np.int32(maskCrop),shape=(self.patchSize,self.patchSize),
                                            nb_classes=self.numberOfClasses)
             imgCrop,maskCrop=self.augmentations(images=np.expand_dims(imgCrop,0),
                                     segmentation_maps=segmap)
             imgCrop=np.squeeze(imgCrop)
             maskCrop=maskCrop.get_arr_int()

         X[i,:,:,:]=self.preproc(imgCrop)
         Y[i,:,:,:]=to_categorical(maskCrop,num_classes=self.numberOfClasses)
         
  
          
      return X, Y

  def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(self.numberOfPatches)
      if self.shuffle == True:
          np.random.shuffle(self.indexes)          
          

def accuracy_with_extra_layer(y_true,y_pred):
    trueClasses=K.argmax(y_true[:,:,:,:-1])     
    predClasses=K.argmax(y_pred)
    return K.mean(K.cast(trueClasses==predClasses,'float32'))
          
          
      
def CalculateSampling(maskList,patchSize,samplingFactor):
    maskAreas=np.array([m.shape[0]*m.shape[1] for m in maskList])      
    patchesPerMask=np.int32(np.round(samplingFactor*maskAreas/(patchSize*patchSize)))
    maskSampling=np.uint32(np.concatenate([mNum*np.ones(patchesPerMask[mNum]) for mNum in range(len(maskList))]))
    return maskSampling          

# %%

def LoadPatchesMasksAndBoxLocations(fileList,classDict):
    images, masks, boxLocations = [], [], []
    # boxLocations list contains tuples containing (svsFilepath, topLeft boxPosition)
    
    for pklFile in fileList:
        with open(pklFile, 'rb') as pfile:
          imgList, maskList, nameToNum, boxPositions, svsFile = pickle.load(pfile)
        images=images+imgList
        
        #numToName={}
        classMapper=np.zeros(len(nameToNum))
        for name in nameToNum:
          num=nameToNum[name]
          if(name in classDict):
            classMapper[num]=classDict[name]
          else:
            classMapper[num]=classDict['BG']
        maskList1=[]
        for m in maskList:
          maskList1.append(classMapper[m])
        masks=masks+maskList1

        for boxPosition in boxPositions:
            boxLocations.append((svsFile, tuple(boxPosition.tolist()) ))

    if len(images) != len(masks):
        raise Exception("numImages does not equal numMasks")

    if len(images) != len(boxLocations):
        raise Exception("numImages does not equal numBoxLocations")

    if len(masks) != len(boxLocations):
        raise Exception("numMasks does not equal numBoxLocations")
    
    return images, masks, boxLocations


def LoadPatchesAndMasks(fileList,classDict):
    images=[]
    masks=[]
    
    for pklFile in fileList:
        with open(pklFile, 'rb') as pfile:
          imgList,maskList,nameToNum,boxPositions=pickle.load(pfile)
        images=images+imgList  
        #numToName={}
        classMapper=np.zeros(len(nameToNum))
        for name in nameToNum:
          num=nameToNum[name]
          if(name in classDict):
            classMapper[num]=classDict[name]
          else:
            classMapper[num]=classDict['BG']
        maskList1=[]
        for m in maskList:
          maskList1.append(classMapper[m])
        masks=masks+maskList1  
    return images,masks

def GetMasks(annoFile,slide,outerBoxClassName='Box',minBoxSize=[256,256],
             magLevel=0,minAnnoBoxArea=25,markEdges=False):
  regionPos,regionNames,regionInfo,_=iu.GetQPathTextAnno(annoFile)
  outerBoxIdx=np.where(np.array(regionNames)=='+'+outerBoxClassName)[0]
  validNameIdx=np.where([(bool(re.compile(r"^[+-]\s*").match(i)) and not(i == '+'+outerBoxClassName) and not(i == '-'+outerBoxClassName)) for
                       i in regionNames])[0]  
  validNames=np.array(regionNames)[validNameIdx]  
  validIsPos=[s[0]=="+" for s in validNames] 
  validSuffixes=[s[1:] for s in validNames]
  isNeg=[s[0] =='-' for s in regionNames]
  numberOfAnnos=len(validNames)
  nameToNum={}
  for num,name in enumerate(np.unique(validSuffixes)):
    nameToNum[name]=num+1
  nameToNum['BG']=0
  if markEdges:
      nameToNum['Edge']=num+2
      edgeClass=nameToNum['Edge']
  else:
      edgeClass=nameToNum['BG']
  # Only preserve boxes that are big enough
  boxSizes=np.array([[regionInfo[b]['BoundingBox'][2]-regionInfo[b]['BoundingBox'][0],
    regionInfo[b]['BoundingBox'][3]-regionInfo[b]['BoundingBox'][1]] for b in outerBoxIdx])
  nPointsInBoundary=np.array([len(r) for r in regionPos])

  outerBoxIdx=outerBoxIdx[np.logical_and(boxSizes[:,0]>minBoxSize[0],boxSizes[:,1]>minBoxSize[1])]
  outerBoxIdx=outerBoxIdx[np.logical_or(nPointsInBoundary[outerBoxIdx]==4,nPointsInBoundary[outerBoxIdx]==5 )]
  
  imgList=[]
  maskList=[]
  boxPositions=np.zeros((len(outerBoxIdx),2))
  for boxCounter,b in enumerate(outerBoxIdx):
    bbox=regionInfo[b]['BoundingBox']
    boxSize=[int(np.round(bbox[2]-bbox[0]))+1,int(np.round(bbox[3]-bbox[1]))+1]
    boxPositions[boxCounter,:]=[np.round(bbox[0]),np.round(bbox[1])]
    img=np.array(slide.read_region((int(bbox[0]),int(bbox[1])),magLevel,boxSize))[:,:,range(3)]
    imgList.append(img)
  
      
    annosInBox=[]
    for idx in validNameIdx:
      annoBox=  regionInfo[idx]['BoundingBox']
      annoBoxArea=(int(np.round(annoBox[2]-annoBox[0]))+1)*(int(np.round(annoBox[3]-annoBox[1]))+1)
      if dlutil.isIntersecting(annoBox,bbox) and annoBoxArea>minAnnoBoxArea:
          annosInBox.append(idx)
    negVals= np.array([isNeg[x] for x in annosInBox])
    annoOrder=np.argsort(negVals)
    #annoIsNeg=negVals[annoOrder]
    annosInBox=np.array(annosInBox)[annoOrder]
    
    if(len(annosInBox)<254):
      mask=Image.new("P",boxSize,0)
      maskNeg=Image.new("P",boxSize,0)
    else:
      mask=Image.new("I",boxSize,0)
      maskNeg=Image.new("I",boxSize,0)
    
    for annoCounter,idx in enumerate(annosInBox):  
      annoName=regionNames[idx][1:]
      annoNumber=nameToNum[annoName]
      

      pos=regionPos[idx]
      pos[:,0]=pos[:,0]-bbox[0]
      pos[:,1]=pos[:,1]-bbox[1]
      poly=np.array(pos)
      poly=np.concatenate((poly,np.expand_dims(poly[0,:],axis=0))) 
      if(isNeg[idx]):
        # May want edge here to go to edge class?
        #ImageDraw.Draw(mask).polygon(poly.ravel().tolist(),outline=nameToNum['BG'],fill=nameToNum['BG'])
        ImageDraw.Draw(maskNeg).polygon(poly.ravel().tolist(),outline=nameToNum['BG'],fill=annoNumber)

      else:
        ImageDraw.Draw(mask).polygon(poly.ravel().tolist(),outline=edgeClass,fill=annoNumber)
    mask=np.array(mask)
    maskNeg=np.array(maskNeg)
    maskClean=mask.copy()
    maskClean[:,:]=nameToNum['BG']
    for className in nameToNum:
        if not className=='BG':
            maskClean[np.logical_and(mask==nameToNum[className],
                                     np.logical_not(maskNeg==nameToNum[className]))]=nameToNum[className]
    maskList.append(maskClean)
  return imgList,maskList,nameToNum,boxPositions

# %%

def hsvTransform(inImg,deltaHSV):
        (deltaH,deltaS,deltaV)=deltaHSV
        hsvImage = cv2.cvtColor(inImg, cv2.COLOR_RGB2HSV)
        # Hue, Saturation, Value
        h, s, v = cv2.split(hsvImage)
        h = h + h*deltaH
        s = s + s*deltaS
        v = v + v*deltaV

        h, s, v = np.clip(h, 0, 255), np.clip(s, 0, 255), np.clip(v, 0, 255)
        newHsvImage = np.dstack((h,s,v)).astype(np.uint8)
        return cv2.cvtColor(newHsvImage, cv2.COLOR_HSV2RGB)



# %%

class SlideAreaGenerator(Sequence):
  def __init__(self,slide,boxHeight=1000,boxWidth=1000, batch_size=4,borderSize=10, 
                shuffle=False):
      """ 
      Initialize the generator to create boxes of the slide. 
      
      ### Returns
      - None
      
      ### Parameters:
      - `slide: slide`  The loaded slide object.
      - `boxHeight: int`  Height of the box that will slide across the slide.
      - `boxWidth: int`  Width of the box that will slide across the slide.
      - `batchSize: int`  Number of boxes to fit in the GPU at time as we profile the slide.
      - `borderSize: int`  Border around the box to profile along with the box itself.
      - `shuffle: bool`  Shuffle indices of boxes in the epoch.
      - `downSampleFactor: int`  Supply reduced size SVS file to profile faster.
      - `preproc_fn: function`  supply preprocessing function. Else, defaults to input/255.
      
      """
      self.slide=slide
      self.boxHeight=boxHeight
      self.boxWidth=boxWidth
      self.batch_size = batch_size
      self.shuffle = shuffle
      (self.slideWidth,self.slideHeight)=slide.dimensions
      self.rVals,self.cVals=np.meshgrid(np.arange(0,self.slideHeight,boxHeight),np.arange(0,self.slideWidth,boxWidth))          
      self.numberOfBoxes=self.rVals.size
      self.rVals.resize(self.rVals.size)
      self.cVals.resize(self.cVals.size)
      self.borderSize=borderSize
      self.on_epoch_end()

  def __len__(self):
      """
      Denotes the number of batches per epoch
      
      ### Returns
      - `int`  Number of batches for the keras generator.
      
      ### Parameters
      - None
      
      """
      return int(np.ceil(self.numberOfBoxes / self.batch_size))

  def __getitem__(self, index):
      """
      Generate one batch of data
      
      ### Returns
      - `X: np.array()` of shape (numBoxesInBatch, numRows, numCols, numChannels)
      - `Y: List(np.array(), np.array())` where first numpy array is row values in the batch and second numpy array is col value in the batch.
      
      ### Parameters
      - `index: int`  batchIndex to be called.
      """
      # Generate indexes of the batch
      #indexes = self.samplingIndex[self.indexes[index*self.batch_size:(index+1)*self.batch_size]]
      #indexes = self.indexes[index]
      indexes=np.arange(index*self.batch_size,np.minimum((index+1)*self.batch_size,self.numberOfBoxes))
    
      X=np.zeros((len(indexes),self.boxHeight+2*self.borderSize,self.boxWidth+2*self.borderSize,3),dtype=np.float32)
      Y=[self.rVals[indexes],self.cVals[indexes]]
      for i,idx in enumerate(indexes):
         img=np.zeros((self.boxHeight+2*self.borderSize,self.boxWidth+2*self.borderSize,3))
         r=self.rVals[idx]
         c=self.cVals[idx]
         imgHeight=int(np.minimum(self.boxHeight+self.borderSize,self.slideHeight-(r))+self.borderSize)
         imgWidth=int(np.minimum(self.boxWidth+self.borderSize,self.slideWidth-(c))+self.borderSize)
         
         img[0:imgHeight,0:imgWidth]=np.array(self.slide.read_region((c-self.borderSize,r-self.borderSize),0,(imgWidth,imgHeight)))[:,:,range(3)]/255

         X[i,:,:,:]=img

         
  
          
      return X, Y

  def on_epoch_end(self):
      """ Updates indexes after each epoch """
      #self.indexes = np.arange(self.numberOfPatches)



# %%
def Profile_Slide_Fast(model,slide,boxHeight=1000,boxWidth=1000, batchSize=4,borderSize=10,
                       useMultiprocessing=True,nWorkers=64,verbose=1,responseThreshold=None,bgClass=0):
    """
    Runs the model across the slide and returns the prediction classes and activations of the whole slide.
    
    ### Returns
    - `slidePredictions:  np.array()`  numpy array of predictions after running the model across the slide.
    
    ### Parameters
    - `slidePredictionsList: np.array()`.  Predicted numpy array of whole slide. Dimensions of output is going to be the same as the slide.
    - `slide: slide`  The loaded slide object.
    - `boxHeight: int`  Height of the box that will slide across the slide.
    - `boxWidth: int`  Width of the box that will slide across the slide.
    - `batchSize: int`  Number of boxes to fit in the GPU at time as we profile the slide.
    - `useMultiprocessing: bool`  Use the multiprocessing to profile the slide faster.
    - `nWorkers:  int` number of parallel processes to run if useMultiprocessing = True.
    - `verbose: int`  print details as we profile the slides.
    - `responseThreshold: float or None`  Set the threshold under which predictions must be called background.
    - `bgClass: int`  The index for bgClass in the model.
    
    """
    
    slideGen=SlideAreaGenerator(slide,boxHeight=boxHeight,boxWidth=boxWidth, batch_size=batchSize,borderSize=borderSize)    
    if verbose>0:
        start_time = time.time()
    
    res=model.predict_generator(slideGen,workers=nWorkers,use_multiprocessing=useMultiprocessing,verbose=verbose)

    classes=np.argmax(res,axis=-1)
    if responseThreshold is not None:
        maxRes=np.max(res,axis=-1)
        classes[maxRes<responseThreshold]=bgClass


    (slideWidth,slideHeight)=slide.dimensions
    rVals,cVals=np.meshgrid(np.arange(0,slideHeight,boxHeight),np.arange(0,slideWidth,boxWidth))          
    numberOfBoxes=rVals.size
    rVals.resize(rVals.size)
    cVals.resize(cVals.size)
    slideClasses=np.zeros((slideHeight,slideWidth),dtype=np.uint8)
    for i in range(numberOfBoxes):
        r=rVals[i]
        c=cVals[i]
        imgHeight=int(np.minimum(boxHeight,slideHeight-(r)))
        imgWidth=int(np.minimum(boxWidth,slideWidth-(c)))
        slideClasses[r:(r+imgHeight),c:(c+imgWidth)]=classes[i][borderSize:(borderSize+imgHeight),borderSize:(borderSize+imgWidth)]
    if verbose>0:        
        print("--- %s seconds ---" % (time.time() - start_time))        
    return slideClasses
# %% meanIoU ignoring background(0) and edge(2) pixels

def MeanIOU(y_true, y_pred):
    # DO NOT USE WITHOUT GENERALIZING 
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    void_labels_BG = K.equal(K.sum(y_true, axis=-1), 0)
    void_labels_Edge = K.equal(K.sum(y_true, axis=-1), 2)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        # '& ~void_labels_Edge'  was added to ignore Edge pixels along with BG pixels
        true_labels = K.equal(true_pixels, i) & ~void_labels_BG & ~void_labels_Edge
        pred_labels = K.equal(pred_pixels, i) & ~void_labels_BG & ~void_labels_Edge
        # inter = tf.to_int32(true_labels & pred_labels)
        # union = tf.to_int32(true_labels | pred_labels)
        inter = tf.cast(true_labels & pred_labels, tf.int32)
        union = tf.cast(true_labels | pred_labels, tf.int32)
        # legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
        legal_batches = K.sum(tf.cast(true_labels, dtype=tf.int32), axis=1) > 0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(ious[legal_batches]))
    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou)
    iou = iou[legal_labels]
    return K.mean(iou)
MeanIOU.__name__='MeanIOU'    

# %% meanIoU ignoring background(0)

def vanillaMeanIOU(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    void_labels_BG = K.equal(K.sum(y_true, axis=-1), 0)
    for i in range(0, nb_classes): # exclude first label (background)
        true_labels = K.equal(true_pixels, i) & ~void_labels_BG
        pred_labels = K.equal(pred_pixels, i) & ~void_labels_BG
        inter = tf.cast(true_labels & pred_labels, tf.int32)
        union = tf.cast(true_labels | pred_labels, tf.int32)
        legal_batches = K.sum(tf.cast(true_labels, dtype=tf.int32), axis=1) > 0
        ious = K.sum(inter, axis=1) / K.sum(union, axis=1)
        iou.append( K.mean(ious[legal_batches]) )
    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou)
    iou = iou[legal_labels]
    return K.mean(iou)

vanillaMeanIOU.__name__ = 'vanillaMeanIOU'

# %%



def LoadUNet(modelFile, custom_objects={}):    
    w=np.random.rand(2)
    w=np.float32(np.max(w)/w)
    myLoss = partial(weighted_image_categorical_crossentropy,weights=w)  
    myLoss.__name__='weighted_loss'
    all_custom_objects={'image_softmax':image_softmax, 'weighted_loss':myLoss,'tf':tf,'MeanIOU':MeanIOU, 'vanillaMeanIOU': vanillaMeanIOU}
    # add new 
    for key in custom_objects:
        all_custom_objects[key] = custom_objects[key]
    uNetModel=load_model(modelFile,custom_objects=all_custom_objects)
    return uNetModel
