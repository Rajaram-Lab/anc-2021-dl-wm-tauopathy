def DiseaseModel_Handcrafted_Visual(scriptInputs):
    '''
    DiseaseModel_Handcrafted_Visual extracts and clusters features from the disease classifier 
    and then generates figures displaying 1) UMAP cluster and 2)clusters with overlaid handcrafted 
    feature values
    Parameters and relevant files are provided in the accompanying yaml file (disease_info.yml)
    
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
    import MILCore as mil
    import PatchGen as pg
    import numpy as np
    import FeatureExtraction as feat
    import pickle
    import openslide as oSlide
    import umap
    from tensorflow.python.keras.utils import Sequence
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from matplotlib.colors import ListedColormap
    import scipy
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Model, load_model
    import seaborn as sns

    # %% Custom Data Generator
      
    class CustomDataGenerator(Sequence):
      'Generates data for Keras'
      def __init__(self, data, labels, numberOfClasses,coord, batch_size=1,
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
          self.coord = coord
          
               
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
    
          return X, to_categorical(self.labels[indexes], num_classes=self.numberOfClasses),indexes, self.coord[indexes]
    
      def on_epoch_end(self):
          'Updates indexes after each epoch'
          self.indexes = np.arange(self.numberOfPatches)
          if self.shuffle == True:
              np.random.shuffle(self.indexes)
              
    def Filter_Nuclei(featMat,clf):
    
        featNamesToUse=['Area','Eccentricity','Major_Axis_Length','Minor_Axis_Length',
                  'Solidity','Curvature','Extent','MaxDistToEdge','Extent_EdgeDist_Ratio',
                  'Hint_Haralick_2nd_moment','Hint_Haralick_sum_avg','Hint_Haralick_correlation']
        featIdxToUse=np.array([featNames.index(f) for f in featNamesToUse])
        featMatFilt=featMat[:,featIdxToUse]
        featLabel = clf.predict(featMatFilt)
        feats = featMat[featLabel==0]
    
        
        return feats,featLabel

    resultsDir=scriptInputs['resultsDir']
    patchDir = scriptInputs['patchDir']
    figureDir = scriptInputs['figureDir']


    # %% 
    if scriptInputs['regenerateData'] == True:
        #Load testing data into memory
        foldNumber=3
        region = 'WM'
        foldsDir=foldsDir=os.path.join(patchDir,region,'Folds')
        testHdf5File=os.path.join(foldsDir,'Testing'+str(foldNumber)+'.txt')
        testHdf5List=[line.rstrip('\n') for line in open(testHdf5File)]
        testSlideDiseaseList=[os.path.split(h)[-1].split('_')[0] for h in testHdf5List]
        
        classDict={'AD':0,'PSP':1,'CBD':2}
        
        testPatches,testAnnoLabels,temp,testSlideNumbers,testPatchPos=\
            pg.LoadPatchData(testHdf5List,returnSampleNumbers=True,returnPatchCenters=True)
        testPatches=testPatches[0]
        testDiseaseNames,testDiseaseNumbers=np.unique(testSlideDiseaseList,return_inverse=True)
        testClasses=testDiseaseNumbers[np.uint8(testSlideNumbers)]
        # % Create generator
        numberOfClasses=len(classDict)
        testGen=CustomDataGenerator(testPatches,testClasses,numberOfClasses,testPatchPos)
        # % Compile mask lists for region masks and aggregate masks
        aggMasksDir = scriptInputs['aggMasksDir']
        regionMasksDir=scriptInputs['regionMasksDir']
        svsDir=scriptInputs['imagesRootDir']
        
        #--
        aggMaskList=[]
        regionMaskList=[]
        svsList=[]
        diseaseList=[]
        
        for f in testHdf5List:
            prefix=os.path.split(f)[-1]
            disease=prefix.split('_')[0]
            aggMaskFile=os.path.join(aggMasksDir,'Pure'+disease,prefix.replace('.hdf5','.npz'))
            regionMaskFile=os.path.join(regionMasksDir,prefix.replace('.hdf5','_RegionMasks.pkl'))
            svsFile=os.path.join(svsDir,'Pure'+disease,prefix.replace('.hdf5','.svs'))
            if os.path.exists(regionMaskFile) and os.path.exists(svsFile) and os.stat(regionMaskFile).st_size>0:
                aggMaskList.append(aggMaskFile)
                regionMaskList.append(regionMaskFile)
                svsList.append(svsFile)
                diseaseList.append(disease)
        # % Load random forest classifier to get rid of artifacts
        resultsSaveDir=scriptInputs['resultsHandCraftedDir']
        resultsSaveFile=os.path.join(resultsSaveDir,'randomForest_ArtifactDetection.pkl')
        [clf]=pickle.load(open(resultsSaveFile, "rb" ) )
        # %Load MIL Model
        modelDir =scriptInputs['modelDir'] 
        milModelFile = os.path.join(modelDir,'mil_DiseaseClassifier_'+region+'_E3_Fold' + str(foldNumber)+ '.h5')
        model=load_model(milModelFile,compile=False,custom_objects=\
                            {'Normalize_Layer':mil.Normalize_Layer,
                             'Last_Sigmoid':mil.Last_Sigmoid,
                             'Last_Softmax':mil.Last_Softmax})
        # % Get profile model
        profileLayer=model.get_layer('flatten') #Change This
        profileModel=Model(inputs=model.input,outputs=profileLayer.output)
        # % Run classification and pull complementary handcrafted features
        numberOfBatches=2000
        minArea=30
        slideIdx=np.zeros(numberOfBatches,dtype=np.uint8)
        patchMat=np.zeros((numberOfBatches,224,224,3))
        predictedBatchProfiles=np.zeros((numberOfBatches,1024))
        trueBatchResults=np.zeros(numberOfBatches)
        featAvg=np.zeros((numberOfBatches,4))
        for batchCounter in range(numberOfBatches):
            # This part predicts for each patch the class--------------------------
            imgs,classes,idx,coord=testGen[batchCounter]
            profiles=profileModel.predict(imgs)
            patchMat[batchCounter] = imgs[0]
            slideIdx[batchCounter] = np.uint(testSlideNumbers[idx[0]])
            predictedBatchProfiles[batchCounter,:]=profiles#np.mean(profiles*weights,axis=0)
            trueBatchResults[batchCounter]=np.argmax(classes,axis=-1)[0]
            # This next part should read in appropriate handcrafted features-----------------
            xCor=np.uint(coord[0][1])
            yCor=np.uint(coord[0][0])
            xLen = np.uint(112)
            yLen = np.uint(112)
            aggMaskFile=aggMaskList[slideIdx[batchCounter]]
            regionMaskFile=regionMaskList[slideIdx[batchCounter]]
            svsFile=svsList[slideIdx[batchCounter]]
            aggMask= np.uint8(scipy.sparse.load_npz(aggMaskFile).todense())
            
            slide=oSlide.open_slide(svsFile)
            # aggImg=feat.ReadSlide(slide)
            cornerPos = (yCor-yLen,xCor-xLen)
            boxSize =np.uint( (2*xLen,2*xLen))
            aggImg = np.array(slide.read_region(cornerPos,0,boxSize))[:,:,range(3)]
            featList=[feat.Location(),feat.Size(),feat.Shape(),feat.Convexity(),feat.Haralick_Texture(0, -1, 1, 'Hint',transform=feat.GetD),\
                  feat.Curvature()]
            aggMaskWhiteBF = aggMask[xCor-xLen:xCor+xLen,yCor-yLen:yCor+yLen]
        
            if np.sum(aggMaskWhiteBF)>0:
                imgList=[aggImg]
                featMat,featNames=feat.ExtractFeatures(aggMaskWhiteBF,featList,minArea=minArea,imgList=imgList)
                # Filter list of outliers
                [featMatF,fLabel] = Filter_Nuclei(featMat,clf)
                #now create a new mask colored by different features
        
                featNamesToUse=['Area','MaxDistToEdge','Minor_Axis_Length','Eccentricity']
                featIdxToUse=np.array([featNames.index(f) for f in featNamesToUse])
                featMatFilt=featMatF[:,featIdxToUse]
                #Get average values 
                featAvg[batchCounter] = np.mean(featMatFilt,axis=0)
            else:
                featAvg[batchCounter] = [0,0,0,0]
            print(batchCounter)
        np.save(os.path.join(resultsDir,'featAvgMIL_2K.npy'),featAvg)
        np.save(os.path.join(resultsDir,'trueBatchResults_2K.npy'),trueBatchResults)
        np.save(os.path.join(resultsDir,'predictedBatchProfiles_2K.npy'),predictedBatchProfiles)
        np.save(os.path.join(resultsDir,'Analysis/slideIdx_2K.npy'),slideIdx)
        
    
    else:
        # %
        #Load data if necessary
        classDict={'AD':0,'PSP':1,'CBD':2}
        numberOfClasses=len(classDict)
        featAvg = np.load(os.path.join(resultsDir,'featAvgMIL_2K.npy'))
        trueBatchResults = np.load(os.path.join(resultsDir,'trueBatchResults_2K.npy'))
        predictedBatchProfiles = np.load(os.path.join(resultsDir,'predictedBatchProfiles_2K.npy'))
        slideIdx= np.load(os.path.join(resultsDir,'slideIdx_2K.npy'))
    
    # %% Run normal UMAP (generates new UMAP)
    print('Generating UMAP clustering of disease classifier data')
    strainColors=['r','y','c']
    reducer = umap.UMAP(n_neighbors=5,min_dist=1)
    test= np.sum(featAvg,axis=1) 
    # Ignore data that had insufficient amount of aggregates
    test2 = test>0
    fIdx = test2
    umapPos = reducer.fit_transform(predictedBatchProfiles[fIdx,:])
    # %%
    strainColors=['r','g','b']
    fig = plt.figure(figsize=(10,10))
    plt.scatter(umapPos[:,0],umapPos[:,1],40,trueBatchResults[fIdx],
                vmin=0,vmax=numberOfClasses-1,cmap=colors.ListedColormap(strainColors))
    fig.savefig(figureDir+'Figure6A.png')
    # %%Overlay UMAP with handcrafted features (uses new UMAP)

    featOrder = [0,3,2]
    nFeat=4
    featMinMax=np.zeros((nFeat,2))
    featMinMax[0] = np.array([40,90])#np.array([0,10]) 
    featMinMax[1] = np.array([3,4])#np.array([0,5])
    featMinMax[2] = np.array([4,8])#np.array([0,0.4])
    featMinMax[3] = np.array([0.8,1]) 
    my_cmap = ListedColormap(sns.color_palette("rocket")) 
    featNamesToUse=['Area','MaxDistToEdge','Minor_Axis_Length','Eccentricity']

   

    fig = plt.figure(figsize=(5,15))
    for i,featNum in enumerate(featOrder):
        name=featNamesToUse[featNum]
        plt.subplot(3,1,i+1)

        plt.scatter(umapPos[:,0],umapPos[:,1],10,featAvg[fIdx,featNum],cmap=my_cmap)
        plt.xticks([],[])
        plt.yticks([],[])


        plt.clim(featMinMax[featNum,0],featMinMax[featNum,1])
        plt.title(name)
        plt.colorbar()
    plt.tight_layout() 
    fig.savefig(figureDir+'Figure6B.png') 
    
