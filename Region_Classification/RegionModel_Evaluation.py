def RegionModel_Evaluation(scriptInputs):
    '''
    RegionModel_Evaluation performs model validation and figure generation for region classification
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
    from sklearn.metrics import confusion_matrix
    from scipy import ndimage as ndi
    from skimage.transform import resize
    from skimage.morphology import remove_small_holes, remove_small_objects
 
    from imgaug import augmenters as iaa  
    
    import matplotlib.pyplot as plt
    import PatchGen as pg
    from tensorflow.python.keras.utils import Sequence
    from tensorflow.python.keras.utils import to_categorical
    from tensorflow.python.keras.models import load_model
    import itertools
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    from skimage.morphology import binary_erosion,disk

    # %% Confusion matrix
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title,fontsize=32)
       # plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45,fontsize=32)
        plt.yticks(tick_marks, classes,fontsize=32)
    

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, np.around(cm[i, j],2),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",fontsize=32)
    
        plt.tight_layout()
        plt.ylabel('True label',fontsize=32)
        plt.xlabel('Predicted label',fontsize=32)
        plt.ylim([-0.5,2.5])
    # % Avg confusion matrix
    def plot_avg_confusion_matrix(cm_raw, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the average confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        cm = np.mean(cm_raw,axis=2)
        cm_s = np.std(cm_raw,axis=2)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_s = cm_s.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title,fontsize=32)
        #plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45,fontsize=32)
        plt.yticks(tick_marks, classes,fontsize=32)
        plt.ylim((-0.5,len(classes)-0.5))
    
        fmt = '.2f' if normalize else '0.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                      horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black",fontsize=32)
            plt.text(j, i-0.2, '+/- ' +format(cm_s[i, j], fmt),
                      horizontalalignment="center",verticalalignment="baseline",
                      color="white" if cm[i, j] > thresh else "black",fontsize=32)
    
        plt.tight_layout()
        plt.ylabel('True label',fontsize=32)
        plt.xlabel('Predicted label',fontsize=32)
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
    figureDir = scriptInputs['figureDir']
    
    # %% Plot confusion matrices by fold (Fig 2a)
    print('Generating fold-average confusion matrix (Fig 2a)')
    foldList = ['modelFile_Fold1','modelFile_Fold2','modelFile_Fold3']
    confMatTotal =  np.zeros([3,3,3])
    classDict={'BG':0,'WM':1, 'CTX':2}
    labels = ['BG', 'WM','CTX']
    numberOfClasses=len(classDict)
    
    for foldNumber in range(1,4):
        # foldNumber=1
        print(foldNumber)
        foldsDir=scriptInputs['foldsDir']
    
        #Load testing data
        testHdf5File=os.path.join(foldsDir,'Testing'+str(foldNumber)+'.txt')
        testHdf5List=[line.rstrip('\n') for line in open(testHdf5File)]
        testFileList=[os.path.split(h)[-1].split('_')[0] for h in testHdf5List]
        testPatches,testClasses,classDict=pg.LoadPatchData(testHdf5List)
        testPatches=testPatches[0]
        testGen=CustomDataGenerator(testPatches,testClasses,numberOfClasses)
        # Load model
        modelFile = scriptInputs[foldList[foldNumber-1]]
        model=load_model(modelFile)
        # Run CM
        trueLabels=[]
        predictedLabels=[]        
        for batchCounter in range(40):
            x,y=testGen[batchCounter]
            test=model.predict(x)
            trueLabels.append(np.argmax(y,-1).flatten())
            predictedLabels.append(np.argmax(test,-1).flatten())
        trueLabels=np.concatenate(trueLabels)    
        predictedLabels=np.concatenate(predictedLabels)
        confMat=confusion_matrix(trueLabels,predictedLabels)
        confMatNorm = confMat.astype('float') / confMat.sum(axis=1)[:, np.newaxis]
        confMatTotal[:,:,foldNumber-1] = confMatNorm
    
    
    fig = plt.figure(figsize=(10,10))
    plot_avg_confusion_matrix(confMatTotal,labels,normalize=False,title = 'Confusion Matrix Avg')
    fig.savefig(figureDir+'Figure2A.png')
    # %% Plot confusion matrices by fold and disease (Fig S2a)
    print('Generating fold-average confusion matrix for each disease (Fig S2a)')
    diseaseList=['AD','PSP','CBD']
    foldList = ['modelFile_Fold1','modelFile_Fold2','modelFile_Fold3']
    # Confusion matrix
    confMatTotalAD =  np.zeros([3,3,3])
    confMatTotalPSP =  np.zeros([3,3,3])
    confMatTotalCBD =  np.zeros([3,3,3])
    classDict={'BG':0,'WM':1, 'CTX':2}
    labels = ['BG', 'WM','CTX']
    numberOfClasses=len(classDict)
    for foldNumber in range(1,4):
        print(foldNumber)
        foldsDir=scriptInputs['foldsDir']
    
        #Load testing data
        testHdf5File=os.path.join(foldsDir,'Testing'+str(foldNumber)+'.txt')
        testHdf5List=[line.rstrip('\n') for line in open(testHdf5File)]
        testFileList=[os.path.split(h)[-1].split('_')[0] for h in testHdf5List] #SR - This is more a disease list, rename
        
        # Load model    
        modelFile = scriptInputs[foldList[foldNumber-1]]
        model=load_model(modelFile)
        for d in diseaseList:
            diseaseHdf5List = [testHdf5List[i] for i in range(len(testFileList)) if testFileList[i]==d]
            testPatchesD,testClassesD,classDict=pg.LoadPatchData(diseaseHdf5List)
            testPatchesD=testPatchesD[0]
            testGenD=CustomDataGenerator(testPatchesD,testClassesD,numberOfClasses)
            trueLabels=[]
            predictedLabels=[]        
            for batchCounter in range(20):
                x,y=testGenD[batchCounter]
                test=model.predict(x)
                trueLabels.append(np.argmax(y,-1).flatten())
                predictedLabels.append(np.argmax(test,-1).flatten())
            trueLabels=np.concatenate(trueLabels)    
            predictedLabels=np.concatenate(predictedLabels)
            confMat=confusion_matrix(trueLabels,predictedLabels)
            confMatNorm = confMat.astype('float') / confMat.sum(axis=1)[:, np.newaxis]
            if d=='AD':
                confMatTotalAD[:,:,foldNumber-1] = confMatNorm
            elif d=='PSP':
                confMatTotalPSP[:,:,foldNumber-1] = confMatNorm
            else:
                confMatTotalCBD[:,:,foldNumber-1] = confMatNorm
                    
    fig = plt.figure(figsize=(36,12))
    plt.subplot(1,3,1)       
    plot_avg_confusion_matrix(confMatTotalAD,labels,normalize=False,title = 'Confusion Matrix Avg AD')
    plt.subplot(1,3,2) 
    plot_avg_confusion_matrix(confMatTotalPSP,labels,normalize=False,title = 'Confusion Matrix Avg PSP')
    plt.subplot(1,3,3)
    plot_avg_confusion_matrix(confMatTotalCBD,labels,normalize=False,title = 'Confusion Matrix Avg CBD')
    fig.savefig(figureDir+'FigureS2A.png')
    
    # %% Run slide-level classification by consensus(Fig 2)
    # if regenerateData == 'True':
    print('Generating slide-level region classification (Fig 2)')
    minArea = 1000
    
    # load classifiers
    regionClassifierFile = scriptInputs['modelFile_Fold1']
    regionClassifierF1=load_model(regionClassifierFile)
    
    
    regionClassifierFile = scriptInputs['modelFile_Fold2']
    regionClassifierF2=load_model(regionClassifierFile)
    
    
    regionClassifierFile = scriptInputs['modelFile_Fold3']
    regionClassifierF3=load_model(regionClassifierFile)
    
    
    # Set svs files to be analyszed 
    
    svsFileList = glob.glob(os.path.join(svsDir+imagesSubDirs[0],'AD_AT8_53245_2.svs'))
    pspFileList = glob.glob(os.path.join(svsDir+imagesSubDirs[1],'PSP_AT8_49292_2.svs'))
    svsFileList.extend(pspFileList)
    cbdFileList = glob.glob(os.path.join(svsDir+imagesSubDirs[2],'CBD_AT8_32073_2.svs'))
    svsFileList.extend(cbdFileList)  

    
    figureDir = scriptInputs['figureDir']
    showVisual=True

    
    for svsFile in svsFileList:
        fileN = os.path.splitext(os.path.split(svsFile)[-1])[0]
        diseaseN=fileN.split('_')[0]
        pngFile=os.path.join(figureDir,'Fig2_RegionMask_'+diseaseN+'.png')

    
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

        if showVisual:
    
            fig = plt.figure(figsize=(7, 7))  
            slideImg=np.array(slide.read_region((0,0),2,slide.level_dimensions[2]))[:,:,range(3)]
            maskRescaled=resize(np.float_(regionClassesSmooth),slideImg.shape[:2],preserve_range=1)
            maskRescaled = np.round(maskRescaled)
            innerMaskCTX=binary_erosion(maskRescaled==1,selem=disk(25))
            innerMaskWM=binary_erosion(maskRescaled==2,selem=disk(25))
            innerMaskWM = innerMaskWM*2
            wmOutline = maskRescaled-innerMaskWM
            totalOutline = wmOutline-innerMaskCTX
            totalOutline[totalOutline==0]=np.NAN 
            plt.imshow(slideImg)
            plt.imshow(totalOutline,cmap='cool')
            fig.savefig(pngFile)
                
    print('Region model figures finished!')
    
        
    
        