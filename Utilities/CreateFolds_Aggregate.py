def CreateFolds_Aggregate(scriptInputs):
    '''
    CreateFolds_Aggregate creates folds to stratify aggregate patch data for model training

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
    
    import glob
    import UNet as UNet
    import numpy as np
    from tensorflow.keras.utils import to_categorical
    import pickle
    from skimage.measure import label
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import yaml
    # %% Get directories
    classDict=scriptInputs['classDict']
    patchDir=scriptInputs['patchDir']
    foldNumber=1
    foldsDir=scriptInputs['foldsDir']
    # %%Get class proportions across all data
    trainHdf5File=os.path.join(foldsDir,'Training'+str(foldNumber)+'.txt')
    testHdf5File=os.path.join(foldsDir,'Testing'+str(foldNumber)+'.txt')
    
    allPatchFiles=[line.rstrip('\n') for line in open(trainHdf5File)]
    testFileList=[line.rstrip('\n') for line in open(testHdf5File)] 
    allPatchFiles.extend(testFileList)
    
    aggMaskName = [x.split('/')[-1].split('_')[0] for x in allPatchFiles]
    allImages,allMasks=UNet.LoadPatchesAndMasks(allPatchFiles,classDict)
    #create dictionary object
    propSlide = np.zeros([len(allPatchFiles),3])
    aggMaskNameT=[]
    allPatchFilesT=[]
    aggCount = 0
    regionCount = 0
    for ix, mask in enumerate(allPatchFiles):
        
        allImages,allMasks=UNet.LoadPatchesAndMasks(allPatchFiles[ix:ix+1],classDict)
        
        propRegion=np.zeros([len(allMasks),3])
        for iy, maskInd in enumerate(allMasks):
            j,counts = np.unique(maskInd,return_counts=True)
            propRegion[iy] = counts
            aggMask = maskInd==1
            aggLabel = label(aggMask)
            aggCount = aggCount+np.max(aggLabel)
            regionCount = regionCount+1
            print(aggCount)
        propSlide[ix] = np.sum(propRegion,axis=0)/np.sum(propRegion)
        
    aggDict = {"fileName":allPatchFiles,"disease":aggMaskName,"composition":propSlide}
    f = open(os.path.join(patchDir,'SlideComposition_Test.pkl'),"wb")
    pickle.dump(aggDict,f)
    f.close()
    # %%
    slideInfoFile=os.path.join(patchDir,'SlideComposition_Test.pkl')
    slideInfo=pickle.load(open(slideInfoFile,'rb'))
    
    
    fileNames=slideInfo['fileName']
    composition=slideInfo['composition']
    disease=slideInfo['disease']
    nFiles=len(fileNames)
    uDisease,diseaseNumber=np.unique(disease,return_inverse=True)
    diseaseHot=to_categorical(diseaseNumber,3)
    stratificationInput=np.hstack((diseaseNumber[None].transpose(),composition)) 
    stratificationFracs=np.hstack((diseaseHot,composition)) 
       
        
    # %% Stratify data based on slide composition
    nFold=2
    nRand=10000
    cvScore=np.zeros(nRand)
    for randSeed in tqdm(range(nRand)):
        
        foldCounter=0
        fracVals=np.zeros((nFold*2,6))
        skf = StratifiedKFold(n_splits=nFold,shuffle=True,random_state=randSeed); #np.random.seed(randSeed)
    
        for trainIx, testIx in skf.split(np.arange(nFiles), diseaseNumber):
    
            fracVals[foldCounter*2,:]=100*np.mean(stratificationFracs[trainIx],axis=0)
            fracVals[foldCounter*2+1,:]=100*np.mean(stratificationFracs[testIx],axis=0)
            
            foldCounter+=1        
        cvScore[randSeed]=np.mean(np.divide(np.std(fracVals,axis=0),np.mean(fracVals,axis=0)))
        
    randSeed=np.argmin(cvScore)
    
    skf = StratifiedKFold(n_splits=nFold,shuffle=True,random_state=randSeed); #np.random.seed(randSeed)
    
    foldCounter=1
    
    print('###Disease Stratification####')
    for trainIx, testIx in skf.split(np.arange(nFiles), diseaseNumber):
        trainFiles = np.array(fileNames)[trainIx]
        testFiles = np.array(fileNames)[testIx]
        trainFiles.tolist()
        testFiles.tolist()
        trainOutFile=os.path.join(foldsDir,'Training'+str(foldCounter)+'.txt')
        with open(trainOutFile, 'w') as f:
            for item in trainFiles:
                f.write("%s\n" % item)
                
        testOutFile=os.path.join(foldsDir,'Testing'+str(foldCounter)+'.txt')
        with open(testOutFile, 'w') as f:
            for item in testFiles:
                f.write("%s\n" % item)
        print('Fold '+str(foldCounter)+' Train', end=':')
        print(100*np.mean(stratificationFracs[trainIx],axis=0))
        print('Fold '+str(foldCounter)+' Test', end=':')
        print(100*np.mean(stratificationFracs[testIx],axis=0))
        foldCounter+=1  

