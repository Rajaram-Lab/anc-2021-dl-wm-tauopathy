def RegionModel_Training(scriptInputs):
    '''
    RegionModel_Training performs model training and inference for region classification
    Parameters and relevant files are provided in the accompanying yaml file (region_info.yml)
    
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
    import glob
    import numpy as np
    from scipy import ndimage as ndi
    from skimage.morphology import remove_small_holes, remove_small_objects    
    from imgaug import augmenters as iaa      
    import pickle
    from collections import Counter
    import PatchGen as pg
    
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Conv2D, MaxPooling2D,Input,UpSampling2D,Conv2DTranspose,Lambda, Cropping2D,AveragePooling2D
    from tensorflow.python.keras.layers import Activation,BatchNormalization,Dropout,Dense,Flatten,concatenate
    from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.python.keras.callbacks import TensorBoard
    from tensorflow.python.keras.callbacks import Callback
    from tensorflow.python.keras.utils import Sequence
    from tensorflow.python.keras.utils import to_categorical
    from tensorflow.python.keras.models import load_model
    from tensorflow.python.keras.models import Model
    from tensorflow.python.keras.applications import InceptionV3, VGG16
    from tensorflow.python.keras.optimizers import RMSprop, Adam, SGD
    
    from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.models import load_model
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Model
    from functools import partial
    import itertools
    import tensorflow as tf
    from tensorflow.keras.optimizers import RMSprop, Adam, SGD
    from imgaug import augmenters as iaa
    import cv2 as cv2
    import pandas as pd
    import seaborn as sns
    import progressbar
    tf.compat.v1.disable_eager_execution()
    import scipy
    from scipy import ndimage as ndi
    from skimage.morphology import binary_erosion,disk
    import yaml

    # %% Custom Data Generator
      
    class CustomDataGenerator(Sequence):
      'Generates data for Keras'
      def __init__(self, data, labels, numberOfClasses, batch_size=64, 
                    shuffle=True,preproc_fn= lambda x:np.float32(x)/255, augmentations = None):
          'Initialization'
          self.data=data
          self.numberOfPatches=data.shape[0]
          self.labels = labels       
          self.numberOfClasses=numberOfClasses
          self.batch_size = batch_size
          self.shuffle = shuffle
          self.preproc=preproc_fn
          self.augmentations = augmentations
          
               
          self.on_epoch_end()
    
      def __len__(self):
          'Denotes the number of batches per epoch'
          return int(np.floor(self.numberOfPatches / self.batch_size))
    
      def __getitem__(self, index):
          'Generate one batch of data'
          # Generate indexes of the batch
          indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
          
          X=np.zeros((len(indexes),self.data.shape[1],self.data.shape[2],self.data.shape[3]),dtype=np.float32)
          
          for i,idx in enumerate(indexes):
             # squeeze image to eliminate extra dimension
              currPatch = np.squeeze(self.data[idx, :, :, :])
              if self.augmentations is not None:
                  currPatch = self.augmentations.augment_image(currPatch)
    
              X[i,:,:,:]=self.preproc(currPatch)
    
          return X, to_categorical(self.labels[indexes], num_classes=self.numberOfClasses)
    
      def on_epoch_end(self):
          'Updates indexes after each epoch'
          self.indexes = np.arange(self.numberOfPatches)
          if self.shuffle == True:
              np.random.shuffle(self.indexes)
    # %% Section to read yaml info
    
    # SVS IMAGE SOURCES
    svsDir, imagesSubDirs = scriptInputs['imagesRootDir'], scriptInputs['imagesSubDirs']
    # %% Create (if needed) and Load Patch Data
    # Note: current code does trivial test train split, with 1-file per disease assigned to testing
    
    
    regeneratePatchData=scriptInputs['regeneratePatchData']# Only set this to be true if you want to recreate the patch data files
    annoFolder=scriptInputs['annotationsDir']
    patchDir=scriptInputs['patchDir'] 
    classDict=scriptInputs['classDict']
    numberOfClasses=len(classDict)
    
    
    distinguishAnnosInClass=scriptInputs['distinguishAnnosInClass'] # IF False, all annotations of the same class are treated as one, and sampling is performed from this
    downSampleLevels=scriptInputs['downSampleLevels']# Downsampling factor relative to max (typically 20X). So 4 will give the 5X image. Adding multiple values gives patches at different scales
    patchSizeList=scriptInputs['patchSizeList'] # Patch size (we assume patches are square) in pixels. Specify patch size separately for each scale in downSampleLevels
    maskDownSampleFactor=scriptInputs['maskDownSampleFactor'] # How much smaller is the mask. Leads to big speed ups, but loss of resolution, which is acceptable at tumor level
    showProgress=scriptInputs['showProgress']
    maxPatchesPerAnno=scriptInputs['maxPatchesPerAnno'] # Maximum number of patches sampled from an annotation
    maxAvgPatchOverlap=scriptInputs['maxAvgPatchOverlap'] # How tightly patches are allowed to overlap. 0 implies no overlap, 1 implies number of patches is selected so that combined area of patches= area of annotation
    minFracPatchInAnno=scriptInputs['minFracPatchInAnno'] # What percentage of the patch must belong to same class as the center pixel, for the patch to be considered
    
    if regeneratePatchData:
        print('Generating Patch Data')
        def TxtToSvs(txtFile,slidesFolder=svsDir):
          textString=os.path.splitext(os.path.split(txtFile)[-1])[0]
          disease,stain,caseId,sampleNumber,*rest=textString.split('_')
          diseaseToFolder={'ad':'PureAD','psp':'PurePSP','cbd':'PureCBD'}
          svsFile=os.path.join(slidesFolder,diseaseToFolder[disease.lower()],textString+'.svs')
          return svsFile
        annoFileList=glob.glob(os.path.join(annoFolder,'*.txt'))
        svsFileList=[TxtToSvs(f) for f in annoFileList]
    
    
        
        for fileNumber in range(len(annoFileList)):
          annoFile=annoFileList[fileNumber]
          svsFile=svsFileList[fileNumber]
          hdf5File=os.path.join(patchDir,os.path.split(svsFile)[-1].replace('svs','pkl'))
          slide=oSlide.open_slide(svsFile)
          
          mask,maskToClassDict=pg.MaskFromXML(annoFile,'NA',slide.dimensions,
                                              downSampleFactor=maskDownSampleFactor,
                                              distinguishAnnosInClass=distinguishAnnosInClass,
                                              outerBoxLabel='Box',outerBoxClassName='BG')   
        
          patchData,patchClasses,patchCenters=pg.PatchesFromMask(slide,mask,
                                                              downSampleLevels,patchSizeList,
                                                              maskToClassDict,
                                                              maxPatchesPerAnno=maxPatchesPerAnno,
                                                              showProgress=showProgress,
                                                              maxAvgPatchOverlap=maxAvgPatchOverlap,
                                                              minFracPatchInAnno=minFracPatchInAnno)
        
          pg.SaveHdf5Data(hdf5File,patchData,patchClasses,patchCenters,downSampleLevels,patchSizeList,svsFile)
          print(svsFile+ ' done!')   
    
          
    # % Patch Generation from annoFiles
    print('Loading Patch Data')
    for foldNumber in range(1,4):
        print(foldNumber)
        foldsDir=scriptInputs['foldsDir']
        trainHdf5File=os.path.join(foldsDir,'Training'+str(foldNumber)+'.txt')
        testHdf5File=os.path.join(foldsDir,'Testing'+str(foldNumber)+'.txt')
        
        trainHdf5List=[line.rstrip('\n') for line in open(trainHdf5File)]
        testHdf5List=[line.rstrip('\n') for line in open(testHdf5File)]
        testPatches,testClasses,classDict=pg.LoadPatchData(testHdf5List)
        trainPatches,trainClasses,classDict=pg.LoadPatchData(trainHdf5List)
        
        trainPatches=trainPatches[0]
        testPatches=testPatches[0]
         
         # %Create model
        # strideSize=16
        # patchSize=256
        epochsN =20
                 
        #  Stride 16-256
        model=Sequential()
        model.add(Conv2D(64,(4,4),strides=(2,2),input_shape=(None,None,3),data_format="channels_last",activation='relu'))
        model.add(MaxPooling2D(pool_size=(4,4),strides=(1,1)))
        model.add(Conv2D(32,(4,4),strides=(2,2),activation='relu'))
        model.add(MaxPooling2D(pool_size=(4,4),strides=(1,1)))
        model.add(Conv2D(32,(4,4),strides=(2,2),activation='relu'))
        model.add(MaxPooling2D(pool_size=(4,4),strides=(2,2)))
        model.add(Conv2D(32,(6,6),activation='relu'))
        model.add(Conv2D(numberOfClasses,(8,8)))
        model.add(Activation('softmax'))
        model.add(Flatten())
        model.summary()
        
        
        # % Define Generators and augmentation   
        aug = iaa.SomeOf((0,1), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.AddToHueAndSaturation((-50,50),per_channel=True)])#,
        
        trainGen=CustomDataGenerator(trainPatches,trainClasses,numberOfClasses,augmentations = aug)
        testGen=CustomDataGenerator(testPatches,testClasses,numberOfClasses)
        # % Train Model
        
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.00001),
                      metrics=['categorical_accuracy'])
        
        counter = Counter(trainGen.labels)                          
        max_val = float(max(counter.values()))       
        class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}  
        print('Training Region Model')
        model.fit_generator(
                trainGen,steps_per_epoch=np.floor(trainGen.numberOfPatches/trainGen.batch_size),
                epochs=epochsN, class_weight=class_weights,validation_data=testGen,
                validation_steps=10)
        # % Save model
        modelSaveFile = os.path.join(scriptInputs['modelDir'],'CTX_WM_Fold'+str(foldNumber)+'_E'+str(epochsN)+'.hdf5')
        model.save(modelSaveFile)
    
    
    print('Generating region masks')
    minArea = scriptInputs['minArea']
    
    # load classifiers
    regionClassifierFile = scriptInputs['modelFile_Fold1']
    regionClassifierF1=load_model(regionClassifierFile)
    
    
    regionClassifierFile = scriptInputs['modelFile_Fold2']
    regionClassifierF2=load_model(regionClassifierFile)
    
    
    regionClassifierFile = scriptInputs['modelFile_Fold3']
    regionClassifierF3=load_model(regionClassifierFile)
    
    
    # Set svs files to be analyszed 
    
    svsFileList = glob.glob(os.path.join(svsDir+imagesSubDirs[0],'AD_AT8*_2.svs'))
    pspFileList = glob.glob(os.path.join(svsDir+imagesSubDirs[1],'PSP_AT8*_2.svs'))
    svsFileList.extend(pspFileList)
    cbdFileList = glob.glob(os.path.join(svsDir+imagesSubDirs[2],'CBD_AT8*_2.svs'))
    svsFileList.extend(cbdFileList)  
    labelNames = ['BG', 'CTX', 'WM']
    
    resultsDir=scriptInputs['resultsDir']

    
    for svsFile in svsFileList:
        pklFile=os.path.splitext(os.path.split(svsFile)[-1])[0]
        pklFile=os.path.join(resultsDir,pklFile+'_RegionMasks.pkl')
    
        slide=oSlide.open_slide(svsFile)
        classes,responseF1=dlutil.Profile_Slide_Fast(regionClassifierF1,slide,16,256,3)
        classes,responseF2=dlutil.Profile_Slide_Fast(regionClassifierF2,slide,16,256,3)
        classes,responseF3=dlutil.Profile_Slide_Fast(regionClassifierF3,slide,16,256,3)
    
        response = (responseF1+responseF2+responseF3)/3
        regionResponseSmooth=np.zeros(response.shape)
        for i in range(response.shape[2]):
            regionResponseSmooth[:,:,i]=ndi.uniform_filter(response[:,:,i],100)
    
        aggFilter=  np.argmax(regionResponseSmooth,axis=-1)
        
        aggMask=np.zeros(response.shape)
        for i in range(response.shape[2]):
            aggMask[:,:,i] = remove_small_holes(aggFilter==i, minArea)
            aggMask[:,:,i] = remove_small_objects(np.bool_(aggMask[:,:,i]), minArea)
            
        regionClassesSmooth=np.argmax(aggMask,axis=-1)
        with open(pklFile, 'wb') as pfile:
            pickle.dump([regionClassesSmooth,labelNames], pfile, protocol=pickle.HIGHEST_PROTOCOL)
        print(svsFile+' done!')
    
                
    print('Region masks finished!')
    
    
