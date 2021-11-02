def HandCraftedFeatures_Extraction(scriptInputs):
    '''
    HandCraftedFeatures_Extraction performs a number of aggregate feature extraction analyses
    1. Slide level average feature extraction
    2. Aggregate level feature extraction
    3. Training on random forest classifier to remove aggregate artifacts
    4. Comparison of feature values from annotated and classified aggregate data
    Parameters and relevant files are provided in the accompanying yaml file (handcrafted_info.yml)
    
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
    import FeatureExtraction as feat
    import glob as glob
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import pickle
    import time
    import cv2
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics
    from sklearn.model_selection import train_test_split 
    from matplotlib.patches import Rectangle
    import umap
    import numpy.matlib
    from tqdm import tqdm
    import scipy
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import clear_border
    import UNet as UNet

    # %% Section to read yaml info
    
    # SVS IMAGE SOURCES
    svsDir, imagesSubDirs = scriptInputs['imagesRootDir'], scriptInputs['imagesSubDirs']
    
    #Mask directories
    aggMasksDir = scriptInputs['aggMasksDir']
    regionMasksDir= scriptInputs['regionMasksDir']
    
    # %% Compile slide,mask lists for future use
    
    #-- Get aggregate masks
    allAggMasks = glob.glob(os.path.join(aggMasksDir+imagesSubDirs[0],'AD_AT8*_2.npz'))
    pspFileList = glob.glob(os.path.join(aggMasksDir+imagesSubDirs[1],'PSP_AT8*_2.npz'))
    allAggMasks.extend(pspFileList)
    cbdFileList = glob.glob(os.path.join(aggMasksDir+imagesSubDirs[2],'CBD_AT8*_2.npz'))
    allAggMasks.extend(cbdFileList)
    #--populate rest of masks with this information
    aggMaskList=[]
    regionMaskList=[]
    svsList=[]
    diseaseList=[]
    
    for f in allAggMasks:
        prefix=os.path.split(f)[-1]
        disease=prefix.split('_')[0]
        regionMaskFile=os.path.join(regionMasksDir,prefix.replace('.npz','_RegionMasks.pkl'))
        svsFile=os.path.join(svsDir,'Pure'+disease,prefix.replace('.npz','.svs'))
        if os.path.exists(regionMaskFile) and os.path.exists(svsFile) and os.stat(regionMaskFile).st_size>0:
            aggMaskList.append(f)
            regionMaskList.append(regionMaskFile)
            svsList.append(svsFile)
            diseaseList.append(disease)
    
    
    # %% Slide level measurements
    minArea=scriptInputs['minArea']
    maxArea=scriptInputs['maxArea']
    featMatList=[]
    cl = np.zeros ([len(aggMaskList),3])      
    
    for fileCounter in range(len(aggMaskList)):
    
        aggMaskFile=aggMaskList[fileCounter]
        regionMaskFile=regionMaskList[fileCounter]
        svsFile=svsList[fileCounter]
        aggMask= np.uint8(scipy.sparse.load_npz(aggMaskFile).todense())
        if np.max(aggMask) ==2:
            aggInd = aggMask==1#Only if edge available
            aggMask = np.uint8(aggInd)
        regionMask,regionLabels=pickle.load(open(regionMaskFile,'rb'))
    
        wmMaskResized=cv2.resize(np.uint8(regionMask==regionLabels.index('WM')),(aggMask.shape[1],aggMask.shape[0]))
        aggMaskWhite=aggMask.copy()
        aggMaskWhite[np.logical_not(wmMaskResized)]=0
    
        aggMaskWhite = remove_small_objects(np.array(aggMaskWhite)>0, minArea)
        slide=oSlide.open_slide(svsFile)
        aggImg=feat.ReadSlide(slide)
        t0 = time.time()
    
    
        featList=[feat.Location(),feat.Size(),feat.Shape(),feat.Convexity(),feat.Haralick_Texture(0, -1, 1, 'Hint',transform=feat.GetD),\
              feat.Curvature()]
        imgList=[aggImg]
        featMat,featNames=feat.ExtractFeatures(aggMaskWhite,featList,
                                               minArea=minArea,maxArea=maxArea,
                                               imgList=imgList)#,imgList=imgList
        featMatList.append(featMat)
        np.save(scriptInputs['saveFeatMatSlideList'],featMatList)
        t1 = time.time()    
        print(t1-t0)
    for fileCounter in range(len(aggMaskList)):
        if  diseaseList[fileCounter] =='AD':
            cl[fileCounter,:] = [1,0,0]
        elif diseaseList[fileCounter] =='PSP':
            cl[fileCounter,:] = [0,0,1]
        elif diseaseList[fileCounter] =='CBD':
            cl[fileCounter,:] = [0,1,0]
        
    
    np.save(scriptInputs['saveFeatMatSlideList'],featMatList)
    
    # %%Aggregate level features via sub-sampling (used for outlier tsne in SuppFig 5)
    classDict={'BG':0,'NFT':1,'TA':2,'PLQ':3}
    featMatList=[]   
    colorList = []   
    count = 0
    sampleN = 300
    # patchList=[]
    filesToProfile=np.arange(0,len(aggMaskList))
    nFiles=len(filesToProfile)
    patchSize=128
    minArea=scriptInputs['minArea']
    
    patchMat=np.zeros((nFiles*sampleN,patchSize,patchSize,3),dtype=np.uint8)
    patchMatMask=np.zeros((nFiles*sampleN,patchSize,patchSize),dtype=np.uint8)
    
    # patchMatBad=np.zeros((nFiles*sampleN,patchSize,patchSize,3),dtype=np.uint8)
    # patchMatMaskBad=np.zeros((nFiles*sampleN,patchSize,patchSize),dtype=np.uint8)
    # boxSizeMatBad=np.zeros((nFiles*sampleN,2),dtype=np.uint8)
    
    boxSizeMat=np.zeros((nFiles*sampleN,2),dtype=np.uint8)
    # colorListAgg = np.zeros((nFiles*sampleN,3),dtype=np.uint8)
    patchCounter=0
    # patchCounterBad=0
    for fileCounter in filesToProfile:
    
        aggMaskFile=aggMaskList[fileCounter]
        regionMaskFile=regionMaskList[fileCounter]
        svsFile=svsList[fileCounter]
    
        aggMask= np.uint8(scipy.sparse.load_npz(aggMaskFile).todense())
        if np.max(aggMask) ==2:
            aggInd = aggMask==1#Only if edge available
            img_noBorder = clear_border(aggInd,buffer_size=256)
            aggMask = np.uint8(img_noBorder)
        else:
            img_noBorder = clear_border(aggMask,buffer_size=256)    
            aggMask = np.uint8(img_noBorder)
        #--------------------------
        regionMask,regionLabels=pickle.load(open(regionMaskFile,'rb'))
        wmMaskResized=cv2.resize(np.uint8(regionMask==regionLabels.index('WM')),(aggMask.shape[1],aggMask.shape[0]))#NOTE THIS should be adjusted
        aggMaskWhite=aggMask.copy()
        aggMaskWhite[np.logical_not(wmMaskResized)]=0
        
        slide=oSlide.open_slide(svsFile)
        aggImg=feat.ReadSlide(slide)
        
        t0 = time.time()
    
        
        featList=[feat.Location(),feat.Size(),feat.Shape(),feat.Convexity(),feat.Haralick_Texture(0, -1, 1, 'Hint',transform=feat.GetD),\
                  feat.Curvature()]
    
        imgList=[aggImg]
        featMat,featNames=feat.ExtractFeatures(aggMaskWhite,featList,minArea=minArea,imgList=imgList)
    
    
        goodIdx=np.where(featMat[:,featNames.index('Area')]>minArea)[0]
        subFeatInd = np.random.choice(goodIdx,np.min([len(goodIdx),sampleN]),replace= False)
    
        for idx in tqdm(subFeatInd):
            # xPos = int(featMat[idx,featNames.index('Centroid_X')])
            # yPos = int(featMat[idx,featNames.index('Centroid_Y')])
            # colorListAgg[patchCounter,aggMaskClass[yPos,xPos]-1] = 1
            hPS=int(patchSize/2)
            cornerPos=(int(featMat[idx,featNames.index('Centroid_X')]-hPS),\
                       int(featMat[idx,featNames.index('Centroid_Y')]-hPS))
            
           
            boxSize=(int(featMat[idx,featNames.index('X_End')]-featMat[idx,featNames.index('X_Start')])+1,\
                      int(featMat[idx,featNames.index('Y_End')]-featMat[idx,featNames.index('Y_Start')])+1)
            patchImg=np.array(slide.read_region(cornerPos,0,(patchSize,patchSize)))[:,:,range(3)]
            patchMat[patchCounter]=patchImg
            patchMatMask[patchCounter]=aggMaskWhite[cornerPos[1]:cornerPos[1]+patchSize,cornerPos[0]:cornerPos[0]+patchSize]
            boxSizeMat[patchCounter,:]=np.array(boxSize)
            patchCounter+=1
    
        subFeatMat = np.array(featMat[subFeatInd,:])#No longer excluding label here
    
        t1 = time.time()    
        print(t1-t0)
        sIdx = np.matlib.repmat(fileCounter,np.min([patchCounter,sampleN]),1)
        if  diseaseList[fileCounter] =='AD':
            cl = np.matlib.repmat([0,0,1],np.min([patchCounter,sampleN]),1)
        elif diseaseList[fileCounter] =='PSP':
            cl = np.matlib.repmat([0,1,0],np.min([patchCounter,sampleN]),1)
        elif diseaseList[fileCounter] =='CBD':
            cl = np.matlib.repmat([1,0,0],np.min([patchCounter,sampleN]),1)
        
        if count == 0:
            featMatList = subFeatMat
            colorList = cl
            slideIdx = sIdx
            count = count +1
        else:
            featMatList = np.concatenate([featMatList,subFeatMat])
            colorList = np.concatenate([colorList,cl])
            slideIdx = np.concatenate([slideIdx,sIdx])
        
    
    np.save(scriptInputs['saveFeatMatAggList'],featMatList)
    
    # %% Save aggregate level and plot UMAP
    _,classNum=np.unique(np.sum(np.multiply(colorList,np.array([4,2,1])),axis=1),return_inverse=True)
    
    numberOfClasses =3
    
    diseaseColors=['r','b','g']
    reducer = umap.UMAP(n_neighbors=50,min_dist=1)
    
    featNamesToUse=['Area','Eccentricity','Major_Axis_Length','Minor_Axis_Length',
                      'Solidity','Curvature','Extent','MaxDistToEdge','Extent_EdgeDist_Ratio',
                      'Hint_Haralick_2nd_moment','Hint_Haralick_sum_avg','Hint_Haralick_correlation']
    
    featIdxToUse=np.array([featNames.index(f) for f in featNamesToUse])
    normFeatMatList=scipy.stats.zscore(featMatList[:,featIdxToUse],axis=0)
    umapPos = reducer.fit_transform(normFeatMatList)
    
    plt.figure(figsize=(20,20))
    plt.scatter(umapPos[:,0],umapPos[:,1],40,classNum,
                vmin=0,vmax=numberOfClasses-1,cmap=colors.ListedColormap(diseaseColors))
    resultsSaveDir=scriptInputs['analysisDir']
    resultsSaveFile=os.path.join(resultsSaveDir,'aggUMAP.pkl')
    
    
    pickle.dump([patchMat,boxSizeMat,classNum,umapPos,patchMatMask],
                    open(resultsSaveFile, "wb" ) ) 
    
    # %% Load sub-sampled feat mat if not already loaded
    featMatList = np.load(scriptInputs['featMatAggList'],allow_pickle=True)
    featNames = np.load(scriptInputs['featNames'])
    featNames = featNames.tolist()
    
    resultsSaveFile=scriptInputs['aggUMAP']
    [patchData,boxSizes,tsneClasses,umapPos,patchMask]=pickle.load(open(resultsSaveFile, "rb" ) )
    # %% Train random forest classifier to identify nuclei artifacts
    #From our current data set, we will label either 0: Normal or 1:Nuclei
    #[patchData,boxSizes,tsneClasses,umapPos,patchMask,slideIdx,aggMaskName]
    #1. Only use certain features
    featNamesToUse=['Area','Eccentricity','Major_Axis_Length','Minor_Axis_Length',
                  'Solidity','Curvature','Extent','MaxDistToEdge','Extent_EdgeDist_Ratio',
                  'Hint_Haralick_2nd_moment','Hint_Haralick_sum_avg','Hint_Haralick_correlation']
    featIdxToUse=np.array([featNames.index(f) for f in featNamesToUse])
    # colNames=[featNames[i]+'_Avg' for i in featIdxToUse]  
    filtFeatMatList=featMatList[:,featIdxToUse]
    
    #Choose boundaries in data of where to extract examples of aggregates and artifacts
    confidenceBound1 = 20
    confidenceBound2=-35
    #Note: Boundaries from new UMAP may not come from first axis (umapPos[:,0]) 
    idxG = np.where(umapPos[:,0]>confidenceBound1)
    idxB= np.where(umapPos[:,0]<confidenceBound2)
    #Make new lists with only these
    filtFeatMatListG = filtFeatMatList[idxG]
    filtFeatMatListB = filtFeatMatList[idxB]
    #Do the same with patch data (verify all original data is filtered)
    patchDataG = patchData[idxG]
    patchDataB = patchData[idxB]
    #Stack to combine into one
    patchDataN = np.vstack([patchDataG,patchDataB])
    filtFeatMatList = np.vstack([filtFeatMatListG,filtFeatMatListB])
    #make labels for these
    idx = np.uint(np.vstack([np.zeros([len(idxG[0]),1]),np.ones([len(idxB[0]),1])]))
    numberOfRandomizations=10
    
    labels = np.array([0,1])
    confMat=0
    for r in range(numberOfRandomizations):
        trainIdx,testIdx=train_test_split(np.arange(filtFeatMatList.shape[0]),test_size=0.3)
        trainData=filtFeatMatList[trainIdx,:]
        testData=filtFeatMatList[testIdx,:]
        trainLabels=np.array(idx[trainIdx])
        testLabels=np.array(idx[testIdx])
        testPatches = patchDataN[testIdx]
    
        clf=RandomForestClassifier(n_estimators=250,class_weight='balanced')    
        clf.fit(trainData,np.ravel(trainLabels))
        testLabelsPredicted=np.round(clf.predict(testData))
        # trainLabelsPredicted=np.round(clf.predict(trainData))
        idxNuc = np.where(testLabelsPredicted==1)
        sampInd = np.random.choice(idxNuc[0],9,replace= False)
        plt.figure(figsize=(12,12))
        for i in range(9):

            ax = plt.subplot(3,3,i+1)
            ax.imshow(testPatches[sampInd[i]])
            rect = Rectangle((54,54),20,20,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            plt.title('True: '+str(testLabels[sampInd[i]][0]) +' Predicted: '+str(testLabelsPredicted[sampInd[i]]))
        idxGood = np.where(testLabelsPredicted==0)
        sampInd = np.random.choice(idxGood[0],9,replace= False)
        plt.figure(figsize=(12,12))
        for i in range(9):
            ax = plt.subplot(3,3,i+1)
            ax.imshow(testPatches[sampInd[i]])
            rect = Rectangle((54,54),20,20,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            plt.title('True: '+str(testLabels[sampInd[i]][0]) +' Predicted: '+str(testLabelsPredicted[sampInd[i]]))
    
        print(metrics.accuracy_score(testLabels,testLabelsPredicted))
        print(metrics.confusion_matrix(testLabels,testLabelsPredicted))
        confMat=confMat+metrics.confusion_matrix(testLabels,testLabelsPredicted,labels)
        
       
        
    
    
    # %% Save random forest
    resultsSaveDir=scriptInputs['analysisDir']
    resultsSaveFile=os.path.join(resultsSaveDir,'randomForest_ArtifactDetection.pkl')  
    pickle.dump([clf],
                    open(resultsSaveFile, "wb" ) )   
    
     
    # %% Compare the feature values of predicted and ground truth (Calculations for Supp Fig. 4)
    classDict={'BG':0,'Tau':1, 'Edge':2}
    foldNumber=2
    minArea=scriptInputs['minArea'] 
    maxArea=scriptInputs['maxArea']
    segDir = scriptInputs['aggMasksDir']
    foldsDir=scriptInputs['foldsDirBox']
    testHdf5File=os.path.join(foldsDir,'Testing'+str(foldNumber)+'.txt')
    testPatchFiles=[line.rstrip('\n') for line in open(testHdf5File)]
    # LoadPatchesAndMasks does mapping of classes as defined in classDict 
    testImages, testMasksGT, testBoxLocations = UNet.LoadPatchesMasksAndBoxLocations(testPatchFiles, classDict)
    featMatListPred = np.zeros ([len(testBoxLocations),32]) # 32 is number of features
    featMatListGT = np.zeros ([len(testBoxLocations),32])
    for boxLocation in enumerate((testBoxLocations)):
        #Load the corresponding mask
        fileName = boxLocation[1][0].split('/')[-1].replace('svs','npz')
        boxCorner = boxLocation[1][1]
    
        boxDim  = testMasksGT[boxLocation[0]].shape
    
        saveFile = os.path.join(segDir,fileName)
        predictedMaskFull=scipy.sparse.load_npz(saveFile).todense()
        #Crop based on box location
    
        predictedMask = np.array(predictedMaskFull[int(boxCorner[1]):int(boxCorner[1])+boxDim[0],
                                                   int(boxCorner[0]):int(boxCorner[0])+boxDim[1]])
        # Run code to filter size at class specific level
        predictedMask = predictedMask==1
        gtMask = testMasksGT[boxLocation[0]]==1
        predictedMaskFilter =remove_small_objects(np.array(predictedMask)>0, minArea)
    
        
        #Filter ground truth as well
        gtMaskFilter  =remove_small_objects(np.array(gtMask)>0, minArea)
        featList=[feat.Location(),feat.Size(),feat.Shape(),feat.Convexity(),feat.Haralick_Texture(0, -1, 1, 'Hint',transform=feat.GetD),\
          feat.Curvature()]
    
    
        imgList=[testImages[boxLocation[0]]]
        featMatGT,featNames=feat.ExtractFeatures(gtMaskFilter,featList,minArea=minArea,maxArea=maxArea,imgList=imgList)
        featMatPred,featNames=feat.ExtractFeatures(predictedMaskFilter,featList,minArea=minArea,maxArea=maxArea,imgList=imgList)
    
        featMatListPred[boxLocation[0]] = np.percentile(featMatPred,50,axis=0)
        featMatListGT[boxLocation[0]] = np.percentile(featMatGT,50,axis=0)
        if boxLocation[0] == 0:
            featMatListPredFull = featMatPred
            featMatListGTFull = featMatGT
        else:
            featMatListPredFull= np.vstack((featMatListPredFull,featMatPred))
            featMatListGTFull= np.vstack((featMatListGTFull,featMatGT))
            
    np.save(os.path.join(resultsSaveDir,'featMatListGTMed_Fold2.npy')  ,featMatListGT)
    np.save(os.path.join(resultsSaveDir,'featMatListPredMed_Fold2.npy')  ,featMatListPred)
    
    
    
    
    
    
    
    
    
    
    
    
    
       
    
    
    
    
    
