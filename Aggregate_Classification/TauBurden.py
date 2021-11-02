def TauBurden(scriptInputs):
    '''
    TauBurden calculates tau burden (fration of area covered by classified aggregates) 
    for cortex and white matter areas, and generates figures for this analysis
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
    import glob as glob
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    import time
    import cv2
    import numpy.matlib
    import statsmodels.api as sm
    import scipy
    from skimage.morphology import remove_small_objects

    # %% Section to read yaml info
    
    # SVS IMAGE SOURCES
    svsDir, imagesSubDirs = scriptInputs['imagesRootDir'], scriptInputs['imagesSubDirs']
    #Mask directories
    aggMasksDir = scriptInputs['aggMasksDir']
    regionMasksDir= scriptInputs['regionMasksDir']
    resultsDir=scriptInputs['resultsDir']
    figureDir = scriptInputs['figureDir']
    regenerateData = scriptInputs['regenerateData']
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
    diseaseList=[]
    
    for f in allAggMasks:
        prefix=os.path.split(f)[-1]
        disease=prefix.split('_')[0]
        regionMaskFile=os.path.join(regionMasksDir,prefix.replace('.npz','_RegionMasks.pkl'))
        if os.path.exists(regionMaskFile) and os.stat(regionMaskFile).st_size>0:
            aggMaskList.append(f)
            regionMaskList.append(regionMaskFile)
            diseaseList.append(disease)
    
    
     

    
    # %% Slide level tau burden
    print('Generating tau burden figure')
    if regenerateData == 'True':
        minArea=scriptInputs['minArea']
        
        
        tBurden = np.zeros([len(aggMaskList),2])
        cl = np.zeros ([len(aggMaskList),3])      
        
        for fileCounter in range(len(aggMaskList)):
            aggMaskFile=aggMaskList[fileCounter]
            regionMaskFile=regionMaskList[fileCounter]
            
            aggMask= np.uint8(scipy.sparse.load_npz(aggMaskFile).todense())
            if np.max(aggMask) ==2:
                aggInd = aggMask==1#Only if edge available
                aggMask = np.uint8(aggInd)
        
            regionMask,regionLabels=pickle.load(open(regionMaskFile,'rb'))
            
        
            wmMaskResized=cv2.resize(np.uint8(regionMask==regionLabels.index('WM')),
                                     (aggMask.shape[1],aggMask.shape[0]))
            aggMaskWhite=aggMask.copy()
            aggMaskWhite[np.logical_not(wmMaskResized)]=0
            #Burden begin-------------------
            ctxMaskResized=cv2.resize(np.uint8(regionMask==regionLabels.index('CTX')),
                                      (aggMask.shape[1],aggMask.shape[0]))
            aggMaskCTX=aggMask.copy()
            aggMaskCTX[np.logical_not(ctxMaskResized)]=0
            # #Now let's quickly apply a size threshold
            aggMaskWhite = remove_small_objects(np.array(aggMaskWhite)>0, minArea)
            aggMaskCTX = remove_small_objects(np.array(aggMaskCTX)>0, minArea)
        
            t0 = time.time()
            wmBurden = np.sum(aggMaskWhite)/np.sum(wmMaskResized)
            ctxBurden = np.sum(aggMaskCTX)/np.sum(ctxMaskResized)
            tBurden[fileCounter,0] = wmBurden
            tBurden[fileCounter,1] = ctxBurden
            #Burden-End-----------------
            np.save(os.path.join(resultsDir,'tauBurden_Fold2.npy')  ,tBurden)
            
            t1 = time.time()    
            print(t1-t0)
        
            
        
        np.save(os.path.join(resultsDir,'tauBurden_Fold2.npy')  ,tBurden)
    else:
        tBurden=np.load(scriptInputs['tauBurden'])
    # % Plotting tau burden
    
    cl = np.zeros ([len(aggMaskList),3])
    for fileCounter in range(len(aggMaskList)):
        if  diseaseList[fileCounter] =='AD':
            cl[fileCounter,:] = [1,0,0]
        elif diseaseList[fileCounter] =='PSP':
            cl[fileCounter,:] = [0,0,1]
        elif diseaseList[fileCounter] =='CBD':
            cl[fileCounter,:] = [0,1,0]
    
    aPos=4    
    fig = plt.figure(figsize=(5,5))
    plt.scatter(tBurden[:,1],tBurden[:,0],50,cl,edgecolors=[0,0,0])
    
    plt.ylabel('White matter tau burden',fontsize='x-large')
    plt.xlabel('Cortex tau burden',fontsize='x-large')  
    
    plt.ylim([-0.01,0.5])
    plt.xlim([-0.01,1.0])
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)
    
    #Show total fit
    m,b = np.polyfit(tBurden[diseaseList.index('AD'):diseaseList.index('PSP')-1,1],
                     tBurden[diseaseList.index('AD'):diseaseList.index('PSP')-1,0],1)
    X=tBurden[diseaseList.index('AD'):diseaseList.index('PSP')-1,1]
    y=tBurden[diseaseList.index('AD'):diseaseList.index('PSP')-1,0]
    X =sm.add_constant(X)
    mod =sm.OLS(y,X)
    res = mod.fit()
    print(res.conf_int(0.01))
    xD = np.arange(0,1.4,0.1)
    plt.plot(xD,m*xD+b,'r')
    
    plt.annotate(str(np.around(m,2)),(xD[aPos],m*xD[aPos]+b))
    m,b = np.polyfit(tBurden[diseaseList.index('PSP'):diseaseList.index('CBD')-1,1],
                     tBurden[diseaseList.index('PSP'):diseaseList.index('CBD')-1,0],1)
    
    plt.plot(xD,m*xD+b,'b')
    
    plt.annotate(str(np.around(m,2)),(xD[aPos],m*xD[aPos]+b))
    m,b = np.polyfit(tBurden[diseaseList.index('CBD'):,1],tBurden[diseaseList.index('CBD'):,0],1)
    X=tBurden[diseaseList.index('CBD'):,1]
    y=tBurden[diseaseList.index('CBD'):,0]
    X =sm.add_constant(X)
    mod =sm.OLS(y,X)
    res = mod.fit()
    print(res.conf_int(0.01))
    
    plt.plot(xD,m*xD+b,'g')
    plt.annotate(str(np.around(m,2)),(xD[aPos],m*xD[aPos]+b))
    fig.savefig(figureDir+'Figure3.png')
    # %% Plotting tau burden-Log
    print('Generating tau burden-log figure')
    cl = np.zeros ([len(aggMaskList),3])
    for fileCounter in range(len(aggMaskList)):
        if  diseaseList[fileCounter] =='AD':
            cl[fileCounter,:] = [1,0,0]
        elif diseaseList[fileCounter] =='PSP':
            cl[fileCounter,:] = [0,0,1]
        elif diseaseList[fileCounter] =='CBD':
            cl[fileCounter,:] = [0,1,0]
    
       
    fig = plt.figure(figsize=(5,5))
    plt.scatter(np.log(tBurden[:,1]),np.log(tBurden[:,0]),50,cl,edgecolors=[0,0,0])
    plt.ylabel('Log( White matter tau burden)',fontsize='x-large')
    plt.xlabel('Log( Cortex tau burden)',fontsize='x-large')
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)
    fig.savefig(figureDir+'FigureS6A.png')
    
