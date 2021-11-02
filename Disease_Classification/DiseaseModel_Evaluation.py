def DiseaseModel_Evaluation(scriptInputs):
    '''
    DiseaseModel_Evaluation performs model inference and figure generation for disease classification
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
    import pickle
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    from tensorflow.keras.models import load_model
    import itertools
    import seaborn as sns

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
    
        plt.tight_layout()
        plt.ylabel('True label',fontsize=32)
        plt.xlabel('Predicted label',fontsize=32)
        
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
    
    def save_dict(new_dict,name,pFolder):
        #First check that secondary_dict is in fact and dictionary
        if type(new_dict) is dict:
            print('Dictionary found')
            with open(pFolder+ name + '.pkl', 'wb') as f:
                        pickle.dump(new_dict, f, pickle.HIGHEST_PROTOCOL)
            
        else:
            print('No dictionary found! Saving Failed')
            
    
    
    def load_dict(name ,pFolder):
        with open(pFolder + name + '.pkl', 'rb') as f:
            return pickle.load(f)
    # %% Section to read yaml info
    #Relevant directories
    patchDir = scriptInputs['patchDir']
    modelDir= scriptInputs['modelDir']
    resultsDir= scriptInputs['resultsDir']
    regenerateData = scriptInputs['regenerateData']
    figureDir = scriptInputs['figureDir']
    
    
    # %% Run analysis to apply both CTX and WM disease classifiers on all test folds (Fig 5b)
    if regenerateData == True:
        regionTypes=['CTX','WM']
        results={}
        for region in regionTypes:
            #  Create variables
            results[region]={}
            #What fraction of patches were correctly classified?
            results[region]['slideAcc'] = np.zeros([49,2])
            # What is the class composition of each slide
            results[region]['slideComp'] = np.zeros([49,3])
            # Confusion matrix
            results[region]['confMatTotal'] =  np.zeros([3,3,3])
            #Counter to keep track
            compCount=0    
            #milSlideIdx
            results[region]['milSlideList'] = []
            
                
            for foldNumber in range(1,4):
                # foldNumber=1
                print(foldNumber)
                foldsDir=os.path.join(patchDir,region,'Folds')
        
                testHdf5File=os.path.join(foldsDir,'Testing'+str(foldNumber)+'.txt')
        
                testHdf5List=[line.rstrip('\n') for line in open(testHdf5File)]
                testSlideDiseaseList=[os.path.split(h)[-1].split('_')[0] for h in testHdf5List]
        
                milSlideName = [x.split('/',)[-1] for x in testHdf5List]
                results[region]['milSlideList'] =results[region]['milSlideList']+milSlideName
                classDict={'AD':0,'CBD':1,'PSP':2}
        
                testPatches,testAnnoLabels,temp,testSlideNumbers,testPatchPos=\
                    pg.LoadPatchData(testHdf5List,returnSampleNumbers=True,returnPatchCenters=True)
                testPatches=testPatches[0]
                testDiseaseNames,testDiseaseNumbers=np.unique(testSlideDiseaseList,return_inverse=True)
                testClasses=testDiseaseNumbers[np.uint8(testSlideNumbers)]
                
                # % Load Model ---------------------------------
                milModelFile = os.path.join(modelDir,'mil_DiseaseClassifier_'+region+'_E3_Fold' 
                                            + str(foldNumber)+ '.h5') 
                model=load_model(milModelFile,compile=False,custom_objects=\
                                    {'Normalize_Layer':mil.Normalize_Layer,
                                     'Last_Sigmoid':mil.Last_Sigmoid,
                                     'Last_Softmax':mil.Last_Softmax})
                
                # % Apply model to testing data------------------------------
                numberOfClasses=len(classDict)#train
                
                nPatchesPerSlide=1000
                numberOfTestSlides=len(np.unique(testSlideNumbers))
                testIdxToProfile=[]
                for slideNumber in range(numberOfTestSlides):
                    testSlideIdx = np.random.choice(np.where(testSlideNumbers==slideNumber)[0],nPatchesPerSlide)#N
                    testIdxToProfile.append(testSlideIdx)#N

                testIdxToProfile=np.array(testIdxToProfile).flatten()
           
                classResponses=model.predict(testPatches[testIdxToProfile]/255,verbose=1)
                predClasses=np.argmax(classResponses,axis=-1)
            
                confMat=confusion_matrix(testClasses[testIdxToProfile], predClasses)
                results[region]['confMatTotal'][:,:,foldNumber-1] = confMat.astype('float') / confMat.sum(axis=1)[:, np.newaxis]  
                gtClass = testClasses[testIdxToProfile]
        
                for slideNumber in range(numberOfTestSlides):
                    isInSlide=testSlideNumbers[testIdxToProfile]==slideNumber
                    slideCompFull = np.argmax(classResponses[isInSlide,:],axis=-1)
                    gtComp = gtClass[isInSlide]
                    correctComp = np.uint((gtComp-slideCompFull)==0)
                    results[region]['slideAcc'][compCount,0] = np.sum(correctComp)/nPatchesPerSlide
                    results[region]['slideAcc'][compCount,1] = gtComp[0]
                    n = plt.hist(slideCompFull,[0,1,2,3])
                    results[region]['slideComp'][compCount] = n[0]/1000
                    compCount = compCount+1
        
               
        # %Save Results here
        resultsSaveFile = 'milResults'
        save_dict(results,resultsSaveFile,resultsDir)
    else:
    
        # % Or load previous results
        resultsLoadFile = 'milResults'
        results = load_dict(resultsLoadFile,resultsDir)
        
        
    # % Plot all confusion matrices (Fig 5b)---------------------------------------
    print('Generating disease classification accuracy (Fig 5b, S8)') 
    labels = ['AD', 'CBD','PSP']
    fig= plt.figure(figsize=(20,10))
    for regionCounter,region in enumerate(results):
        plt.subplot(1,2,regionCounter+1)
        plot_avg_confusion_matrix(results[region]['confMatTotal'],
                                  labels,normalize=True,
                                  title = region) 
    fig.savefig(figureDir+'Figure5B.png')
    
    
    # % Compare CTX and WM accuracy (Fig S8a)-------------------------------------
    nBins=4
    hist2d,xBins,yBins=np.histogram2d(results['CTX']['slideAcc'][:,0],
                                      results['WM']['slideAcc'][:,0],
                                      bins=nBins,
                                      range=[[0,1],[0,1]])
    fig= plt.figure(figsize=(7,7))
    sns.heatmap(hist2d,annot=True,cmap='Wistia')
    plt.ylim(0,nBins+0.5)    
    
    plt.plot([nBins/2,nBins/2],[0,nBins],'--k')
    plt.plot([0,nBins],[nBins/2,nBins/2],'--k')
    makePct = lambda pctList: [str(int(p))+'%' for p in pctList]
    plt.xticks(xBins*nBins,makePct(100*xBins))
    plt.yticks(yBins*nBins,makePct(100*yBins),rotation=0)
    plt.xlabel('WM Accuracy')
    plt.ylabel('CTX Accuracy')
    plt.axis('square')
    fig.savefig(figureDir+'FigureS8A.png')
    
    
    # %Compare accuracy with consensus(Fig S8b)------------------------------------
    
    
    isHitCtx=np.argmax(results['CTX']['slideComp'],axis=1)!=results['CTX']['slideAcc'][:,1]
    isHitWm=np.argmax(results['WM']['slideComp'],axis=1)!=results['WM']['slideAcc'][:,1]
    isHitCons=np.argmax(results['CTX']['slideComp']+results['WM']['slideComp'],
                        axis=1)!=results['WM']['slideAcc'][:,1]
    
    
    isHit=np.hstack([isHitWm,isHitCtx,isHitCons])
    disease=np.hstack([results['CTX']['slideAcc'][:,1],results['CTX']['slideAcc'][:,1],results['CTX']['slideAcc'][:,1]])
    modelType=np.hstack([0*np.ones(isHitWm.shape),1*np.ones(isHitWm.shape),2*np.ones(isHitWm.shape)])
    
    adCounts= np.array([np.sum(np.logical_and(disease[isHit]==0,modelType[isHit]==m)) for m in range(3)])
    pspCounts= np.array([np.sum(np.logical_and(disease[isHit]==2,modelType[isHit]==m)) for m in range(3)])
    cbdCounts= np.array([np.sum(np.logical_and(disease[isHit]==1,modelType[isHit]==m)) for m in range(3)])
    
    fig= plt.figure(figsize=(7,7))
    plt.bar(np.arange(3),adCounts,color='R')
    plt.bar(np.arange(3),pspCounts,bottom=adCounts,color='b')
    plt.bar(np.arange(3),cbdCounts,bottom=adCounts+pspCounts,color='g')
    plt.xticks(np.arange(3),['WM','CTX','Consensus'],rotation=0)
    plt.yticks(np.arange(5))
    plt.ylabel('Misclassified Samples')
    fig.savefig(figureDir+'FigureS8B.png')
    print('Disease classification figures finished!')
