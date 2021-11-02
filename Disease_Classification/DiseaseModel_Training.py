def DiseaseModel_Training(scriptInputs):
    '''
    DiseaseModel_Training performs model training for disease classification
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
    from collections import Counter 

   
    
    # %% Section to read yaml info
    #Relevant directories
    patchDir = scriptInputs['patchDir']
    modelDir= scriptInputs['modelDir']
    
    # %% 
    #Select training and testing data from given region and fold 
    classDict={'AD':0,'PSP':1,'CBD':2}
    
    regions = ['CTX','WM']
    
    for region in regions:
        for foldNumber in range(1,4):
            foldsDir=os.path.join(patchDir,region,'Folds')
            trainHdf5File=os.path.join(foldsDir,'Training'+str(foldNumber)+'.txt')
            testHdf5File=os.path.join(foldsDir,'Testing'+str(foldNumber)+'.txt')
            
            
            trainHdf5List=[line.rstrip('\n') for line in open(trainHdf5File)]
            testHdf5List=[line.rstrip('\n') for line in open(testHdf5File)]
            testSlideDiseaseList=[os.path.split(h)[-1].split('_')[0] for h in testHdf5List]
            trainSlideDiseaseList=[os.path.split(h)[-1].split('_')[0] for h in trainHdf5List]    
            # %Load the training and testing data into memory
            trainPatches,trainAnnoLabels,_,trainSlideNumbers=\
                pg.LoadPatchData(trainHdf5List,returnSampleNumbers=True,classDict=classDict)
            trainPatches=trainPatches[0]
            
            testPatches,testAnnoLabels,temp,testSlideNumbers,testPatchPos=\
                pg.LoadPatchData(testHdf5List,returnSampleNumbers=True,returnPatchCenters=True)
            testPatches=testPatches[0]
            trainDiseaseNames,trainDiseaseNumbers=np.unique(trainSlideDiseaseList,return_inverse=True)
            trainClasses=trainDiseaseNumbers[np.uint8(trainSlideNumbers)]
            
            testDiseaseNames,testDiseaseNumbers=np.unique(testSlideDiseaseList,return_inverse=True)
            testClasses=testDiseaseNumbers[np.uint8(testSlideNumbers)]
            
            
             
            # % Create generators
            
            numberOfClasses=len(classDict)#train
            trainGen=mil.MILGenerator(trainPatches, trainClasses, \
                                      np.uint16(trainSlideNumbers), numberOfClasses,batch_size=16)   
            testGen=mil.MILGenerator(testPatches, testClasses, \
                                     np.uint16(testSlideNumbers), numberOfClasses,batch_size=16)    
        # % Create Model if need be
        
            counter = Counter(trainClasses)
            max_val = float(max(counter.values()))
            class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
            class_weights=np.array([class_weights[c] for c in class_weights],dtype=np.float32)
            
            inputDim=(224,224,3)
            trainingParams={'momentum':0.5,'lr':0.0001} #init_lr is not used0.9,1E-5
            model=mil.MilNetwork(inputDim,trainingParams,class_weights,\
                                 numberOfClasses=numberOfClasses,activationType='softmax') 
            
            # %
            # Save Model after training
            epochN = 3
            model.fit_generator(generator=trainGen, 
                                          steps_per_epoch=np.floor(trainGen.numberOfPatches/trainGen.batch_size),\
                                          epochs=epochN, validation_data=testGen,\
                                          use_multiprocessing=True)
            
            # % Save model
            modelSaveFile = os.path.join(modelDir,'mil_DiseaseClassifier_'+region+'_E'+str(epochN)+'_Fold'+str(foldNumber)+'.h5') #Put the directory (including file name here ex. /project/Models/model1.hdf5) -TONY
            model.save(modelSaveFile)
    
    
    
    
     
    
        
        
    
        
        
        
    
    
    
    
    
    
    
