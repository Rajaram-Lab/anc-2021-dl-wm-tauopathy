def HandCraftedFeatures_Evaluation(scriptInputs):
    '''
    HandCraftedFeatures_Evaluation generates a number of aggregate feature extraction figures

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
    import cv2
    from matplotlib.cm import get_cmap
    import seaborn as sns
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.patches as patches
    import scipy.stats as stats
    import numpy.matlib
    import scipy
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import clear_border
    from statannot import add_stat_annotation

    # %
    def Filter_Nuclei(featMat,clf):
        """
        This function filters out nuclei detection artifacts using a trained random forect classifier
    
        """
        featNamesToUse=['Area','Eccentricity','Major_Axis_Length','Minor_Axis_Length',
                  'Solidity','Curvature','Extent','MaxDistToEdge','Extent_EdgeDist_Ratio',
                  'Hint_Haralick_2nd_moment','Hint_Haralick_sum_avg','Hint_Haralick_correlation']
        featIdxToUse=np.array([featNames.index(f) for f in featNamesToUse])
        featMatFilt=featMat[:,featIdxToUse]
        featLabel = clf.predict(featMatFilt)
        feats = featMat[featLabel==0]
    
        
        return feats, featLabel
    # %% Section to read yaml info
    svsDir, imagesSubDirs = scriptInputs['imagesRootDir'], scriptInputs['imagesSubDirs']
    
    #Mask directories
    aggMasksDir = scriptInputs['aggMasksDir']
    regionMasksDir= scriptInputs['regionMasksDir']
    #Figure Directory
    figureDir = scriptInputs['figureDir']
    regenerateData = scriptInputs['regenerateData']
    resultsDir = scriptInputs['resultsDir']
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
    
    # %% Load slide level features and feature names
    featMatList = np.load(scriptInputs['featMatSlideList'],allow_pickle=True)
    featNames = np.load(scriptInputs['featNames'])
    featNames = featNames.tolist()
    
    # %% Load random forest classifier
    [clf]=pickle.load(open(scriptInputs['artifactRF'], "rb" ) )
    # %% Display feature overlay (Fig 4a)
    print('Generating feature overlay (Fig 4a)')
    featsToShow=['Area','Eccentricity','Minor_Axis_Length']
    featNamesAdj = ['Area','Eccentricity','Minor Axis Length']
    alpha1=1
    alpha2=0.5
    nFeat=3
    featMinMax=np.zeros((nFeat,2))
    featMinMax[0] = np.array([0,1000])
    featMinMax[1] = np.array([0.5,1])
    featMinMax[2] = np.array([5,35])
    if regenerateData == True:
        minArea=30
        xLen = 250
        yLen=250

        #Select slide
        diseases=np.array([os.path.splitext(os.path.split(f)[-1])[0].split('.')[0] for f in aggMaskList])
        diseasesF = np.array('PSP_AT8_42120_2')
        xCor = np.array(21000)
        yCor = np.array(26000)
        
        fIdx = np.where(diseases == diseasesF)
        aggMaskFile=aggMaskList[fIdx[0][0]]
        regionMaskFile=regionMaskList[fIdx[0][0]]
        svsFile=svsList[fIdx[0][0]]
    
        
        aggMask= np.uint8(scipy.sparse.load_npz(aggMaskFile).todense())
        
        if np.max(aggMask) ==2:
            aggInd = aggMask==1#Only if edge available
            aggMask = np.uint8(aggInd)
        
        regionMask,regionLabels=pickle.load(open(regionMaskFile,'rb'))
        wmMaskResized=cv2.resize(np.uint8(regionMask==regionLabels.index('WM')),(aggMask.shape[1],aggMask.shape[0]))
        aggMaskWhiteP=aggMask.copy()
        aggMaskWhiteP[np.logical_not(wmMaskResized)]=0
        aggNoBorder = clear_border(aggMaskWhiteP==1,buffer_size=500)
        aggMaskWhite = remove_small_objects(np.array(aggNoBorder), minArea)
    
        
        slide=oSlide.open_slide(svsFile)
        aggImg=feat.ReadSlide(slide)
    
        featList=[feat.Location(),feat.Size(),feat.Shape(),feat.Convexity(),feat.Haralick_Texture(0, -1, 1, 'Hint',transform=feat.GetD),\
              feat.Curvature()]
    
        aggMaskWhiteB = aggMaskWhite[xCor-xLen:xCor+xLen,yCor-yLen:yCor+yLen]
        
        aggImgB = aggImg[xCor-xLen:xCor+xLen,yCor-yLen:yCor+yLen]
        imgList=[aggImgB]
        featMat,featNames=feat.ExtractFeatures(aggMaskWhiteB,featList,minArea=minArea,imgList=imgList)
        # Filter list of outliers
        [featMatF,fLabel] = Filter_Nuclei(featMat,clf)
    
        featNamesToUse=['Area','Eccentricity','Minor_Axis_Length']
        featIdxToUse=np.array([featNames.index(f) for f in featNamesToUse])
        featMatFilt=featMat[:,featIdxToUse]
        #Get average values for reference later
        featAvgP = np.mean(featMatFilt,axis=0)
    
        # Create label mask with same thresholding and nuclei filtering...
        nObj, labelTest, statsT, centroids=cv2.connectedComponentsWithStats(np.uint8(aggMaskWhiteB),8,cv2.CV_32S)
        labelMask = remove_small_objects(labelTest, minArea)
        labelList = np.unique(labelMask)
        labelList= labelList[1:]
    
        #Loop through features
        featureMaskTotal = np.zeros([500,500,3])
    
        for featS in range(featMatFilt.shape[1]):
            #We will populate featMask with label values equal to feature
            featMask = np.zeros(aggMaskWhiteB.shape)
            featNum=featS
    
            for aggObj in range(featMatFilt.shape[0]):
                #Get relevant idx
                idx = labelMask==labelList[aggObj]
                if fLabel[aggObj]==0:
                    featMask[idx] = featMatFilt[aggObj,featS]
                else:
                    featMask[idx] = 0
    
            featureMaskTotal[:,:,featS] = featMask
        pickle.dump([aggImgB,labelMask,featureMaskTotal,featNamesToUse,xCor,yCor,featAvgP],
                        open(resultsDir+'FeatureVisual.pkl', "wb" ) ) 
        # print(diseasesF+': Done!') 
    
    else:
        pklFile = os.path.join(resultsDir,'FeatureVisual.pkl')
        [aggImgB, labelMask,featureMaskTotal,featNamesToUse,xCor,yCor,featAvgP]=pickle.load(open(pklFile,'rb'))
        
    fig = plt.figure(figsize=(25,5))
    plt.subplot(1,nFeat+1,1)
    plt.imshow(aggImgB)
    plt.title('Raw Image',fontsize='xx-large')
    for featNum in range(nFeat):
    
        ax=plt.subplot(1,nFeat+1,featNum+2)
        plt.imshow(aggImgB)
        matToShow=featureMaskTotal[:,:,featNum].copy()
        matToShow[labelMask==0]=np.NAN
    
        plt.imshow(matToShow,alpha=alpha1,vmin=featMinMax[featNum,0],vmax=featMinMax[featNum,1],cmap='cool')
        plt.title(featNamesToUse[featNum],fontsize='xx-large')
        plt.xticks([],[])
        plt.yticks([],[])
        current_cmap = get_cmap()
        current_cmap.set_bad(color=[1,1,1,alpha2])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)
    fig.savefig(figureDir+'Figure4A.png')
    # %%
    print('Generating cluster maps (Fig 4b and S5d)')
    figureRef = ['4B','S5D']

    
    #Features for filtering
    featNamesToUseFilt=['Area','Eccentricity','Major_Axis_Length','Minor_Axis_Length',
                     'Solidity','Curvature','Extent','MaxDistToEdge','Extent_EdgeDist_Ratio',
                      'Hint_Haralick_2nd_moment','Hint_Haralick_sum_avg','Hint_Haralick_correlation']
    #Features for clustering
    featNamesToUse=['Area','Eccentricity','Major_Axis_Length','Minor_Axis_Length',
                     'Solidity','Curvature','Extent','MaxDistToEdge','Extent_EdgeDist_Ratio']#,
    # More inuitive names
    featNamesAdj=['Area','Eccentricity','Major Axis Length','Minor Axis Length',
                     'Solidity','Curvature','Extent','Width','Extent-to-Width Ratio']
    
    for figR in figureRef: 
        featIdxToUseFilt=np.array([featNames.index(f) for f in featNamesToUseFilt])
        featIdxToUse=np.array([featNames.index(f) for f in featNamesToUse])
        avgFeatMat=np.zeros((len(featMatList),len(featIdxToUse)))
        diseaseColors={'AD':'r','PSP':'b','CBD':'g'}
        plt.figure(figsize=(20,30))
        for sampleCounter in range(len(featMatList)):
            featMatListInd = featMatList[sampleCounter]
            #filter here and get labels
            featMatFilt=featMatListInd[:,featIdxToUseFilt]
            if figR == '4B':
                featMatLabels=np.round(clf.predict(featMatFilt))#filter
                featMat=featMatListInd[:,featIdxToUse]
                featMatToUse = featMat[featMatLabels==0]#filter
        
            else:
                #Then use full
                featMat=featMatListInd[:,featIdxToUse]
                featMatToUse = featMat #no filter
        
        
            avgFeatMat[sampleCounter,:]=np.percentile(featMatToUse,50,axis=0)
        
        
        
        combinedFeatMat=avgFeatMat
         
        
        plt.figure(figsize=(40,40))
        sns.clustermap(combinedFeatMat,yticklabels=[],xticklabels=featNamesAdj,
                       standard_scale=None,z_score=1,method='average',metric='correlation',
                       row_colors=[diseaseColors[d] for d in diseaseList],vmin=-3,vmax=4)
        plt.savefig(figureDir+'Figure'+figR+ '.png')
    
    # %% Compare median feature differences across samples, full and partial
    print('Generating boxplot comparisons (Fig 4c, S7)')
    figureRef = ['4C','S7']
    for figR in figureRef:
        if figR  == 'S7':
        
            featsToShow=['Area','Major_Axis_Length','Minor_Axis_Length','Eccentricity',
                              'Solidity','Extent_EdgeDist_Ratio','Extent','MaxDistToEdge','Curvature']
            featNamesAdj = ['Area','Major Axis Length','Minor Axis Length','Eccentricity',
                              'Solidity','Extent-to-Width Ratio','Extent','Width','Curvature']
        else:
            featsToShow=['Area','Eccentricity','Minor_Axis_Length']
            featNamesAdj = ['Area','Eccentricity','Minor Axis Length']
        diseaseColors=['red','blue','green'] 
        
        nSlides=len(featMatList)
        nFeat=len(featsToShow)
        nC=int(np.ceil(np.sqrt(nFeat)))
        nR=int(np.ceil(np.ceil(nFeat/nC)))
        filterNuc ='on'
        fig = plt.figure(figsize=(4*nC,4*nR))
        for featCounter,featName in enumerate(featsToShow):
            featNum=featNames.index(featName)
            
            featVec=np.zeros(nSlides)
            diseaseVec=[]
        
            for slideCounter in range(len(featMatList)):
                if filterNuc == 'on':
                    featsFull=featMatList[slideCounter]
                    [feats,featLabel] = Filter_Nuclei(featsFull,clf)
                else:
                    feats=featMatList[slideCounter]
                
                diseaseVec.append(diseaseList[slideCounter])
                featVec[slideCounter]=np.percentile(feats[:,featNum],50,axis=0)

                
            plt.subplot(nR,nC,featCounter+1)
            ax=sns.boxplot(x=diseaseVec,y=featVec,
                        palette=diseaseColors,dodge=False)
            sns.swarmplot(x=diseaseVec,y=featVec,color='gray',ax=ax)
            add_stat_annotation(ax,x=diseaseVec,y=featVec,
                                         box_pairs=[('AD','PSP'),('PSP','CBD'),('AD','CBD')],   
                                         test='Mann-Whitney',loc='inside', verbose=2,
                                         text_format='full') #full
        

            plt.title(featNamesAdj[featCounter])
        plt.tight_layout()
        fig.savefig(figureDir+'Figure'+figR+'.png')  
    # %% Compare annotated aggregate features to features from predicted aggregate masks of same aggregates
    featMatListGT = np.load(scriptInputs['featMatListGT'])
    featMatListPred = np.load(scriptInputs['featMatListPred'])
    
    # %% Plot scatter plots of features to look at (Plot for Supp Fig. 4)
    print('Generating scatter plots (Fig S4)')
    featNamesToUse=['Area','Eccentricity','Major_Axis_Length','Minor_Axis_Length',
                    'Solidity','Curvature','Extent','MaxDistToEdge','Extent_EdgeDist_Ratio']
    featNamesAdj=['Area','Eccentricity','Major Axis Length','Minor Axis Length',
                     'Solidity','Curvature','Extent','Width','Extent-to-Width Ratio']
    featIdxToUse=np.array([featNames.index(f) for f in featNamesToUse])
    nFeat=len(featIdxToUse)
    nC=int(np.ceil(np.sqrt(nFeat)))
    nR=int(np.ceil(np.ceil(nFeat/nC)))
    fig = plt.figure(figsize=(4*nC,4*nR))
    featCounter = 0
    for m in featIdxToUse:
        plt.subplot(nR,nC,featCounter+1)
        plt.scatter(featMatListGT[:,m],featMatListPred[:,m])
        plt.plot([0,np.max([featMatListGT[:,m],featMatListPred[:,m]])*1.5],[0,np.max([featMatListGT[:,m],featMatListPred[:,m]])*1.5],'r')
        corrS = stats.spearmanr(featMatListGT[:,m],featMatListPred[:,m])
        plt.title(featNamesAdj[featCounter]+': '+str("%.2f" % round(corrS[0],2)))
        plt.xlim(np.min([featMatListGT[:,m],featMatListPred[:,m]])*0.5,np.max([featMatListGT[:,m],featMatListPred[:,m]])*1.5)
        plt.ylim(np.min([featMatListGT[:,m],featMatListPred[:,m]])*0.5,np.max([featMatListGT[:,m],featMatListPred[:,m]])*1.5)
        featCounter = featCounter+1
        
    plt.tight_layout()
    fig.savefig(figureDir+'FigureS4.png')
    # %% Look at artifacts from aggregate features
    print('Generating UMAP for artifacts (Fig S5a)')
    numberOfClasses=3
    
    resultsSaveFile=scriptInputs['aggUMAP']
    [patchData,boxSizes,tsneClasses,umapPos,patchMask]=pickle.load(open(resultsSaveFile, "rb" ) )
    fig = plt.figure(figsize=(20,20))
    plt.scatter(umapPos[:,0],umapPos[:,1],40,tsneClasses,
                vmin=0,vmax=numberOfClasses-1,cmap=colors.ListedColormap(diseaseColors))
    
    plt.xlim([-90,90])
    plt.ylim([-90,90])
    fig.savefig(figureDir+'FigureS5A.png')
    # %% Test random forest classifier
    print('Generating Random forect classifier results (Fig S5b)')
    featMatList = np.load(scriptInputs['featMatAggList'],allow_pickle=True)
    [clf]=pickle.load(open(scriptInputs['artifactRF'], "rb" ) )
    featNamesToUse=['Area','Eccentricity','Major_Axis_Length','Minor_Axis_Length',
                  'Solidity','Curvature','Extent','MaxDistToEdge','Extent_EdgeDist_Ratio',
                  'Hint_Haralick_2nd_moment','Hint_Haralick_sum_avg','Hint_Haralick_correlation']
    featIdxToUse=np.array([featNames.index(f) for f in featNamesToUse])
 
    filtFeatMatList=featMatList[:,featIdxToUse]
    #New idea, let's try only using examples we are confident about
    #Get confident good and bad data
    idxG = np.where(umapPos[:,0]>20)
    idxB= np.where(umapPos[:,0]<-35)
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
    numberOfRandomizations=1
    
    labels = np.array([0,1])
    confMat=0
    for r in range(numberOfRandomizations):
        trainIdx,testIdx=train_test_split(np.arange(filtFeatMatList.shape[0]),test_size=0.3)
        trainData=filtFeatMatList[trainIdx,:]
        testData=filtFeatMatList[testIdx,:]
        trainLabels=np.array(idx[trainIdx])
        testLabels=np.array(idx[testIdx])
        testPatches = patchDataN[testIdx]
    
        testLabelsPredicted=np.round(clf.predict(testData))
        trainLabelsPredicted=np.round(clf.predict(trainData))
        idxNuc = np.where(testLabelsPredicted==1)
        sampInd = np.random.choice(idxNuc[0],9,replace= False)
        fig = plt.figure(figsize=(12,12))
        for i in range(9):

            ax = plt.subplot(3,3,i+1)
            ax.imshow(testPatches[sampInd[i]])
            rect = patches.Rectangle((54,54),20,20,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            plt.title('True: '+str(testLabels[sampInd[i]][0]) +' Predicted: '+str(testLabelsPredicted[sampInd[i]]))
        fig.savefig(figureDir+'FigureS5B.png')
        idxGood = np.where(testLabelsPredicted==0)
        sampInd = np.random.choice(idxGood[0],9,replace= False)
        fig = plt.figure(figsize=(12,12))
        for i in range(9):

            ax = plt.subplot(3,3,i+1)
            ax.imshow(testPatches[sampInd[i]])
            rect = patches.Rectangle((54,54),20,20,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            plt.title('True: '+str(testLabels[sampInd[i]][0]) +' Predicted: '+str(testLabelsPredicted[sampInd[i]]))
        fig.savefig(figureDir+'FigureS5C.png')
        print(metrics.accuracy_score(testLabels,testLabelsPredicted))
        print(metrics.confusion_matrix(testLabels,testLabelsPredicted))
        confMat=confMat+metrics.confusion_matrix(testLabels,testLabelsPredicted,labels)
        
    print('Hand crafted feature figures finished!')
    
    
    
    
    
    
    
      
    
    
     
    
    
    
    
    
    
    
