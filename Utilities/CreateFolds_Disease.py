def CreatFolds(scriptInputs):    
    '''
    CreateFolds_Aggregate creates folds to stratify aggregate patch data for model training

    Parameters and relevant files are provided in the accompanying yaml file (region_info.yml or disease_info.yaml)
    
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
    import sys
    import glob
    import numpy as np
    import yaml

    # %% Build list of annotated svs files
    
    numberOfFolds=3
    region = scriptInputs['region']
    patchDir=os.path.join(scriptInputs['patchDir'],region)
    foldsDir=foldsDir=os.path.join(patchDir,'Folds')
    hdf5List=np.array(glob.glob(os.path.join(patchDir,'*AT8*.hdf5')))
    
    
    caseIds=np.array([os.path.splitext(os.path.split(f)[-1])[0].split('_')[2] for f in hdf5List])
    diseases=np.array([os.path.splitext(os.path.split(f)[-1])[0].split('_')[0] for f in hdf5List])
    
    uniqueDiseases=np.unique(diseases)
    
    diseaseFolds={}
    for disease in uniqueDiseases:
        diseaseFolds[disease]=[]
        casesInDisease=np.unique(caseIds[diseases==disease])
        # np.random.shuffle(casesInDisease)#not sure why this is here, consider removing for consistency
        
        casesPerFold=int(np.floor(len(casesInDisease)/numberOfFolds))
        for fold in range(numberOfFolds):
            if(fold<(numberOfFolds-1)):
                diseaseFolds[disease].append(casesInDisease[np.arange(fold*casesPerFold,(fold+1)*casesPerFold)])
            else:
                diseaseFolds[disease].append(casesInDisease[np.arange(fold*casesPerFold,len(casesInDisease))])
    
    trainFolds=[]
    testFolds=[]            
    for fold in range(numberOfFolds):
        testFiles=[]
        trainFiles=[]
        for disease in uniqueDiseases:
            
            for fold1 in range(numberOfFolds):
                foldCases=diseaseFolds[disease][fold1]
                if fold==fold1:
                    for case in foldCases:
                        testFiles=testFiles+(hdf5List[caseIds==case].tolist())
                else:
                    for case in foldCases:
                        trainFiles=trainFiles+(hdf5List[caseIds==case].tolist())
            
        trainFolds.append(trainFiles)
        testFolds.append(testFiles)
        
    for fold in range(numberOfFolds):
        trainOutFile=os.path.join(foldsDir,'Training'+str(fold+1)+'.txt')
        with open(trainOutFile, 'w') as f:
            for item in trainFolds[fold]:
                f.write("%s\n" % item)
                
        testOutFile=os.path.join(foldsDir,'Testing'+str(fold+1)+'.txt')
        with open(testOutFile, 'w') as f:
            for item in testFolds[fold]:
                f.write("%s\n" % item)