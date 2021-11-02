def AggregateModel_Training(scriptInputs):
    '''
    AggregateModel_Training performs model training and inference for aggregate classification
    Parameters and relevant files are provided in the accompanying yaml file (aggregate_info.yml)
    
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
    import UNet as UNet
    import glob
    import numpy as np
    from scipy import ndimage as ndi
    import pickle
    from tensorflow.keras import backend as K
    import tensorflow as tf
    from tensorflow.keras.optimizers import SGD
    from imgaug import augmenters as iaa
    tf.compat.v1.disable_eager_execution()
    import scipy

    # %% Section to read yaml info
    
    # SVS IMAGE SOURCES
    svsDir, imagesSubDirs = scriptInputs['imagesRootDir'], scriptInputs['imagesSubDirs']
    
    
    
    # %% Create (if needed) and Load Patch Data
    regeneratePatchData=scriptInputs['regeneratePatchData']# Only set this to be true if you want to recreate the patch data files
    annoFolder=scriptInputs['annotationsDir']
    patchDir=scriptInputs['patchDir'] 
    outerBoxClassName=scriptInputs['outerBoxClassName']
    minBoxSize=scriptInputs['minBoxSize'] 
    minAnnoBoxArea=scriptInputs['minAnnoBoxArea']#1Annotations with bounding boxes with area smaller than this are dropped
    classDict=scriptInputs['classDict']
    patchSize=scriptInputs['patchSize'] # Note, this value is only used during patch generation.
    addEdgeClass=scriptInputs['addEdgeClass'] # Note, this value is only used during patch generation.
    
    if regeneratePatchData:
    
        def TxtToSvs(txtFile,slidesFolder=svsDir):
            """
            This function converts txt file names to svs file names from default directory
    
            """
            textString=os.path.splitext(os.path.split(txtFile)[-1])[0]
            disease,stain,caseId,sampleNumber,*rest=textString.split('_')
            diseaseToFolder={'ad':'PureAD','psp':'PurePSP','cbd':'PureCBD'}
            svsFile=os.path.join(slidesFolder,diseaseToFolder[disease.lower()],textString+'.svs')
            return svsFile
        annoFileList=glob.glob(os.path.join(annoFolder,'*.txt'))
        svsFileList=[TxtToSvs(f) for f in annoFileList]
    
    
        magLevel=0 # Currently only 0 supported
        
        for fileNumber in range(len(annoFileList)):
          annoFile=annoFileList[fileNumber]
          svsFile=svsFileList[fileNumber]
          pklFile=os.path.join(patchDir,os.path.split(svsFile)[-1].replace('svs','pkl'))
          slide=oSlide.open_slide(svsFile)
          
          # Currently no support for negative annotations
          imgList,maskList,nameToNum,boxPositions=UNet.GetMasks(annoFile,slide,
                                              outerBoxClassName=outerBoxClassName,
                                              minBoxSize=minBoxSize,magLevel=magLevel,
                                              minAnnoBoxArea=minAnnoBoxArea,
                                              markEdges=addEdgeClass)
          with open(pklFile, 'wb') as pfile:
              pickle.dump([imgList,maskList,nameToNum,boxPositions], pfile, protocol=pickle.HIGHEST_PROTOCOL)
          print(svsFile+ ' done!')
    
    patchDir=scriptInputs['patchDir'] 
    classDict=scriptInputs['classDict']
    foldsDir=scriptInputs['foldsDir']
    for foldNumber in range(1,3):
        trainHdf5File=os.path.join(foldsDir,'Training'+str(foldNumber)+'.txt')
        testHdf5File=os.path.join(foldsDir,'Testing'+str(foldNumber)+'.txt')
        
        trainFileList=[line.rstrip('\n') for line in open(trainHdf5File)]
        testFileList=[line.rstrip('\n') for line in open(testHdf5File)]
        
        testImages,testMasks =UNet.LoadPatchesAndMasks(testFileList,classDict)
        trainImages,trainMasks = UNet.LoadPatchesAndMasks(trainFileList, classDict) 
               
        
        # % Define Sampling for Generators
        
        augmentations=iaa.SomeOf((0,2),[iaa.Fliplr(0.5),iaa.Flipud(0.5),
                                 iaa.ContrastNormalization((0.75,1.33)),
                                 ])
        numberOfClasses=len(classDict)
        samplingFactor=4  
        trainMaskSampling=UNet.CalculateSampling(trainMasks,patchSize,samplingFactor)
        testMaskSampling=UNet.CalculateSampling(testMasks,patchSize,samplingFactor)
        trainGen=UNet.MaskDataGenerator(trainImages,trainMasks,trainMaskSampling,numberOfClasses,patchSize,batch_size=16,augmentations=augmentations)
        testGen=UNet.MaskDataGenerator(testImages,testMasks,testMaskSampling,numberOfClasses,patchSize,batch_size=16)
        w=np.zeros(numberOfClasses,dtype=np.float32)
        for e in range(20):
            x,y=trainGen[e]
        
            counts=np.argmax(y,-1)
            for i in range(numberOfClasses):
              w[i]+=np.sum(counts==i)
        w=np.max(w)/w
        # %
        __EPS = 1e-5
        
        def EdgeLoss(classWeights,alpha,edgeClass=2):
            """
            This loss function encourages the prediction to produce the same oveall amount
            of edges as the ground truth to prevent over- or under-splitting of aggregates
            """
            def loss(y_true,y_pred):
                imgLoss=UNet.weighted_image_categorical_crossentropy(y_true,y_pred,classWeights)
            
                fracEdgeTrue=K.mean(y_true[:,:,:,edgeClass],axis=[1,2])
                fracEdgePred=K.mean(y_pred[:,:,:,edgeClass],axis=[1,2])
                fracEdgePred = K.clip(fracEdgePred, __EPS, 1 - __EPS)
                edgeLoss=-K.mean(fracEdgeTrue * K.log(fracEdgePred) + 
                                 (1 - fracEdgeTrue) * K.log(1 - fracEdgePred))
           
                
                
                totalLoss=imgLoss+alpha*edgeLoss
                
                
                return totalLoss
            return loss
        
        
        myLoss=EdgeLoss(w,0.5)
        myLoss.__name__='weighted_loss'  
              
        
        # % Run Training
        uN=UNet.UNet()
        unetModel=uN.create_model((None,None,3),numberOfClasses)
        unetModel.compile(optimizer = SGD(lr = 1e-3,momentum=0.5), loss = myLoss, metrics = ['accuracy'])
        unetModel.fit_generator(trainGen,epochs=50)
        unetModel.save(os.path.join(scriptInputs['modelDir'],'unet_AggClassifierEdgeLoss_Fold'+str(foldNumber)+'E50.h5'))
          
    
    # %% Default uses second fold
    modelSaveFile = scriptInputs['modelFile_Fold2']
    unetModel=UNet.LoadUNet(modelSaveFile)     
    #Directory where classified slides will be saved
    classDir= scriptInputs['resultsDir']
    
    
    # %% Create list with slide info
        
    dataDir=os.path.join(svsDir,imagesSubDirs[0])
    svsFileList = glob.glob(os.path.join(dataDir,'AD_AT8*_2.svs'))
    
    dataDir=os.path.join(svsDir,imagesSubDirs[1])
    pspFileList = glob.glob(os.path.join(dataDir,'PSP_AT8*_2.svs'))
    svsFileList.extend(pspFileList)
    
    dataDir=os.path.join(svsDir,imagesSubDirs[2])
    cbdFileList = glob.glob(os.path.join(dataDir,'CBD_AT8*_2.svs'))
    svsFileList.extend(cbdFileList)
    svsFile = [x.split('/')[-1] for x in svsFileList]
    disType = [x.split('/')[-2] for x in svsFileList]
    
    
    # %% Run Classification
    

    for numClass in range(len(svsFile)):
        #Load slide
        print(numClass)
        slide=oSlide.open_slide(os.path.join(svsDir,disType[numClass],svsFile[numClass]))
        
        # Run classification   
        imgClasses=UNet.Profile_Slide_Fast(unetModel,slide,boxHeight=400,
                      boxWidth=400,batchSize=2,borderSize=8,useMultiprocessing=False,
                      nWorkers=64,verbose=1,responseThreshold=0.5,bgClass=0)
        # Save as sparse to save space
        imgClassesSparse = scipy.sparse.csc_matrix(imgClasses)
        scipy.sparse.save_npz(os.path.join(classDir,disType[numClass],svsFile[numClass][:-4]), imgClassesSparse)
    
    
    
    
