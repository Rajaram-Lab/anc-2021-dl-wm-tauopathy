def AggregateModel_Evaluation(scriptInputs):
    '''
    AggregateModel_Evaluation performs model validation and figure generation for aggregate classification
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
    from sklearn.metrics import confusion_matrix    
    import pickle
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors    
    import itertools    
    from imgaug import augmenters as iaa

    # %% Confusion matrix code
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
        plt.xticks(tick_marks, classes, rotation=45,fontsize=40)
        plt.yticks(tick_marks, classes,fontsize=40)
    

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, np.around(cm[i, j],2),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",fontsize=32)
    
        plt.tight_layout()
        plt.ylabel('True label',fontsize=32)
        plt.xlabel('Predicted label',fontsize=32)
        plt.ylim([-0.5,len(labels)-0.5])
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
        plt.xticks(tick_marks, classes, rotation=45,fontsize=40)
        plt.yticks(tick_marks, classes,fontsize=40)
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
    
    # %% Section to read yaml info
    # SVS IMAGE SOURCES
    svsDir, imagesSubDirs = scriptInputs['imagesRootDir'], scriptInputs['imagesSubDirs']
    
    figureDir = scriptInputs['figureDir']
    # %% Create (if needed) and Load Patch Data
    # Note: current code does trivial test train split, with 1-file per disease assigned to testing
    foldNumber=2
    regeneratePatchData=scriptInputs['regeneratePatchData']# Only set this to be true if you want to recreate the patch data files
    annoFolder=scriptInputs['annotationsDir']
    patchDir=scriptInputs['patchDir'] 
    outerBoxClassName=scriptInputs['outerBoxClassName']
    minBoxSize=scriptInputs['minBoxSize'] # Should we also have a max box size?
    minAnnoBoxArea=scriptInputs['minAnnoBoxArea']#1,Annotations with bounding boxes with area smaller than this are dropped
    classDict=scriptInputs['classDict']
    patchSize=scriptInputs['patchSize'] # FYI, this is only used in the next bock 
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
    
    
    foldsDir=scriptInputs['foldsDir']
    trainHdf5File=os.path.join(foldsDir,'Training'+str(foldNumber)+'.txt')
    testHdf5File=os.path.join(foldsDir,'Testing'+str(foldNumber)+'.txt')
    
    trainFileList=[line.rstrip('\n') for line in open(trainHdf5File)]
    testFileList=[line.rstrip('\n') for line in open(testHdf5File)]
    
    testImages,testMasks =UNet.LoadPatchesAndMasks(testFileList,classDict)
    trainImages,trainMasks = UNet.LoadPatchesAndMasks(trainFileList, classDict) 
           
    
    # %% Define Sampling for Generators
    
    augmentations=iaa.SomeOf((0,2),[iaa.Fliplr(0.5),iaa.Flipud(0.5),
                             iaa.ContrastNormalization((0.75,1.33)),
                             ])
    numberOfClasses=len(classDict)
    samplingFactor=scriptInputs['samplingFactor']  
    trainMaskSampling=UNet.CalculateSampling(trainMasks,patchSize,samplingFactor)
    testMaskSampling=UNet.CalculateSampling(testMasks,patchSize,samplingFactor)
    trainGen=UNet.MaskDataGenerator(trainImages,trainMasks,trainMaskSampling,numberOfClasses,patchSize,batch_size=16,augmentations=augmentations)
    testGen=UNet.MaskDataGenerator(testImages,testMasks,testMaskSampling,numberOfClasses,patchSize,batch_size=16)
          
    # %% Load Model
    modelFile = scriptInputs['modelFile_Fold2']
    unetModel=UNet.LoadUNet(modelFile) 
    
    # %% Visualize ground truth and performance (Sample Images for 2b)
    print('Generating aggregate classification examples (Fig 2b)')
    classDictInv={}
    for cNum,cName in enumerate(classDict):
        classDictInv[cNum]=cName
    colorsList=['w', 'c', 'r']
    cmap = colors.ListedColormap(colorsList)
    x,y=testGen[7]
    test=unetModel.predict(x)
    plt.figure(figsize=(10,20))
    plt.subplot(1,2,1)
    plt.imshow(x[0])
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.title('IHC',fontsize=16)
    plt.subplot(1,2,2)
    plt.imshow(x[0])
    plt.imshow(np.argmax(test[0],axis=-1),vmin=0,vmax=3,cmap=cmap,alpha=0.5)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.title('Predicted Mask',fontsize=16)
        
    
    
    # %% Plot confusion matrices by fold (Fig. 2b)
    print('Generating fold-average aggregate classification accuracy (Fig 2b)')
    # Confusion matrix
    classDict={'BG':0,'Tau':1, 'Edge':2}
    patchSize=400 
    numberOfClasses=len(classDict)
    samplingFactor=4  
    foldList = ['modelFile_Fold1','modelFile_Fold2']
    nFolds=2
    confMatTotal =  np.zeros([3,3,nFolds])
    labels = ['BG', 'Aggregate','Edge']
    numberOfClasses=len(classDict)
    
    for foldNumber in range(1,nFolds+1):
        print(foldNumber)
    
        #Load testing data
        testHdf5File=os.path.join(foldsDir,'Testing'+str(foldNumber)+'.txt')
        testFileList=[line.rstrip('\n') for line in open(testHdf5File)]
        testImages,testMasks=UNet.LoadPatchesAndMasks(testFileList,classDict)
        testMaskSampling=UNet.CalculateSampling(testMasks,patchSize,samplingFactor)
        testGen=UNet.MaskDataGenerator(testImages,testMasks,testMaskSampling,numberOfClasses,patchSize,batch_size=16)
        modelFile = scriptInputs[foldList[foldNumber-1]]
        unetModel=UNet.LoadUNet(modelFile)
        
    
    
        # Run CM
        trueLabels=[]
        predictedLabels=[]        
        for batchCounter in range(20):
            x,y=testGen[batchCounter]
            test=unetModel.predict(x)
            trueLabels.append(np.argmax(y,-1).flatten())
            predictedLabels.append(np.argmax(test,-1).flatten())
        trueLabels=np.concatenate(trueLabels)    
        predictedLabels=np.concatenate(predictedLabels)
        confMat=confusion_matrix(trueLabels,predictedLabels)
        confMatNorm = confMat.astype('float') / confMat.sum(axis=1)[:, np.newaxis]
        confMatTotal[:,:,foldNumber-1] = confMatNorm
    
    
    fig = plt.figure(figsize=(10,10))
    plot_avg_confusion_matrix(confMatTotal,labels,normalize=False,title = 'Confusion Matrix Avg')
    fig.savefig(figureDir+'Figure2B.png')
    # %% Plot confusion matrices by fold and disease
    # Confusion matrix
    print('Generating fold-average aggregate classification accuracy for each disease (Fig S3)')
    diseaseList=['AD','PSP','CBD']
    foldList = ['modelFile_Fold1','modelFile_Fold2']
    patchSize=400 
    samplingFactor=4  
    confMatTotalAD =  np.zeros([3,3,nFolds])
    confMatTotalPSP =  np.zeros([3,3,nFolds])
    confMatTotalCBD =  np.zeros([3,3,nFolds])
    nFolds=2
    confMatTotal =  np.zeros([3,3,nFolds])
    classDict={'BG':0,'Tau':1, 'Edge':2}
    labels = ['BG', 'Aggregate','Edge']
    numberOfClasses=len(classDict)
    fig = plt.figure(figsize=(36,12))
    for foldNumber in range(1,nFolds+1):
        # foldNumber=1
        print(foldNumber)
    
        #Load testing data
        testHdf5File=os.path.join(foldsDir,'Testing'+str(foldNumber)+'.txt')
        testHdf5List=[line.rstrip('\n') for line in open(testHdf5File)]
        testFileList=[os.path.split(h)[-1].split('_')[0] for h in testHdf5List]
    
        modelFile = scriptInputs[foldList[foldNumber-1]]
        unetModel=UNet.LoadUNet(modelFile)
        for d in diseaseList:
            diseaseHdf5List = [testHdf5List[i] for i in range(len(testFileList)) if testFileList[i]==d]
            testImages,testMasks=UNet.LoadPatchesAndMasks(diseaseHdf5List,classDict)
            testMaskSampling=UNet.CalculateSampling(testMasks,patchSize,samplingFactor)
            testGen=UNet.MaskDataGenerator(testImages,testMasks,testMaskSampling,numberOfClasses,patchSize,batch_size=16)
            
            # Run CM
            trueLabels=[]
            predictedLabels=[]        
            for batchCounter in range(10):
                x,y=testGen[batchCounter]
                test=unetModel.predict(x)
                trueLabels.append(np.argmax(y,-1).flatten())
                predictedLabels.append(np.argmax(test,-1).flatten())
            trueLabels=np.concatenate(trueLabels)    
            predictedLabels=np.concatenate(predictedLabels)
            confMat=confusion_matrix(trueLabels,predictedLabels)
            confMatNorm = confMat.astype('float') / confMat.sum(axis=1)[:, np.newaxis]
            confMatTotal[:,:,foldNumber-1] = confMatNorm
            if d=='AD':
                confMatTotalAD[:,:,foldNumber-1] = confMatNorm
            elif d=='PSP':
                confMatTotalPSP[:,:,foldNumber-1] = confMatNorm
            else:
                confMatTotalCBD[:,:,foldNumber-1] = confMatNorm
    
          
    plt.subplot(1,3,1)        
    plot_avg_confusion_matrix(confMatTotalAD,labels,normalize=False,title = 'Confusion Matrix Avg AD')
    plt.subplot(1,3,2)
    plot_avg_confusion_matrix(confMatTotalPSP,labels,normalize=False,title = 'Confusion Matrix Avg PSP')
    plt.subplot(1,3,3)
    plot_avg_confusion_matrix(confMatTotalCBD,labels,normalize=False,title = 'Confusion Matrix Avg CBD')
    fig.savefig(figureDir+'FigureS3A.png')
    # %% Plot CM-adjusted for cortex (Supp 3b)
    #Adjust to only worry about two classes: BG and Tau
    print('Generating cortex aggregate classification accuracy S3b')
    modelFile = scriptInputs['modelFile_Fold2']
    unetModel=UNet.LoadUNet(modelFile)
    testHdf5File=scriptInputs['cortexFile']
    classDict={'BG':0,'Tau':1, 'Edge':2}
    numberOfClasses=len(classDict)
    samplingFactor=4
    
    testFileList=[line.rstrip('\n') for line in open(testHdf5File)]
    
    testImages,testMasks =UNet.LoadPatchesAndMasks(testFileList,classDict)   
    testMaskSampling=UNet.CalculateSampling(testMasks,patchSize,samplingFactor)
    testGen=UNet.MaskDataGenerator(testImages,testMasks,testMaskSampling,numberOfClasses,patchSize,batch_size=16)    
    trueLabels=[]
    predictedLabels=[]        
    for batchCounter in range(20):
        x,y=testGen[batchCounter]
        test=unetModel.predict(x)
        trueLabels.append(np.argmax(y,-1).flatten())
        predictedLabels.append(np.argmax(test,-1).flatten())
    trueLabels=np.concatenate(trueLabels)    
    predictedLabels=np.concatenate(predictedLabels)
    # Adjust here
    predictedLabels[np.where(predictedLabels==2)]=1
    trueLabels[np.where(trueLabels==2)]=1
    confMat=confusion_matrix(trueLabels,predictedLabels)
    confMatNorm = confMat.astype('float') / confMat.sum(axis=1)[:, np.newaxis]    
    
    labels = ['BG', 'Aggregate']
    fig = plt.figure(figsize=(10,10))
    plot_confusion_matrix(confMat,labels,normalize=True,title = 'Confusion Matrix')
    fig.savefig(figureDir+'FigureS3B.png')
    print('Aggregate model figures finished!')
    
         
    
    
    
    
    
    
    
