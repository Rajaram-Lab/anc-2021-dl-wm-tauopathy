'''
FigureGeneration is the main code for generating figures 
for the paper 'Deep learning reveals diseas-specific signatures
of white matter pathology in tauopathies '
Code is divided into blocks for generating figures from different analyses.

All analysis can be re-run if desired, by changing regenerateData variable to True (default: False) in each yaml file 

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
import yaml  
# %% 1. Directories
#Directory where code is located
codeDir = '/home2/avega/Documents/Deep_Learning/Code/Neuro/neuro-wm-paper/'

#Directory where all data is located
mainDir = '/project/bioinformatics/Rajaram_lab/shared/Neuro/White_Matter_Project/'

sys.path.insert(0,os.path.join(codeDir,'Utilities'))
sys.path.insert(0,os.path.join(codeDir,'Extern/statannot'))
sys.path.insert(0,os.path.join(codeDir,'Region_Classification'))
sys.path.insert(0,os.path.join(codeDir,'Aggregate_Classification'))
sys.path.insert(0,os.path.join(codeDir,'Disease_Classification'))
sys.path.insert(0,os.path.join(codeDir,'Hand_Crafted'))

from RegionModel_Training import RegionModel_Training
from RegionModel_Evaluation import RegionModel_Evaluation

from AggregateModel_Training import AggregateModel_Training
from AggregateModel_Evaluation import AggregateModel_Evaluation

from TauBurden import TauBurden

from HandCraftedFeatures_Extraction import HandCraftedFeatures_Extraction
from HandCraftedFeatures_Evaluation import HandCraftedFeatures_Evaluation 

from Patch_Generation import Patch_Generation

from DiseaseModel_Training import DiseaseModel_Training
from DiseaseModel_Evaluation import DiseaseModel_Evaluation

from DiseaseModel_Handcrafted_Visual import DiseaseModel_Handcrafted_Visual 


os.chdir(mainDir)

# %% 2. Region Classification: Code for generating figures for Region Classifier accuracy (Fig. 2a, Fig. S2a)
scriptInputsFile = os.path.join(codeDir, 'region_info.yml')
scriptInputs = yaml.safe_load(open(scriptInputsFile, 'r'))

#Re-train models and regenerate region mask data
if scriptInputs['regenerateData'] == True:
    RegionModel_Training(scriptInputs)
    
# Figure Generation
RegionModel_Evaluation(scriptInputs)


# %% 3. Aggregate Classification: Code for generating figures for Aggregate Classifier accuracy (Fig. 2b, Fig. S3)
scriptInputsFile = os.path.join(codeDir, 'aggregate_info.yml')
scriptInputs = yaml.safe_load(open(scriptInputsFile, 'r'))

#Re-train models and regenerate aggregate mask data
if scriptInputs['regenerateData'] == True:
    AggregateModel_Training(scriptInputs)
    
# Figure Generation
AggregateModel_Evaluation(scriptInputs)

# %% 4. Tau Burden: Code for generating figures for Tau Burden Results (Fig. 3, Fig. S6)
scriptInputsFile = os.path.join(codeDir, 'handcrafted_info.yml')
scriptInputs = yaml.safe_load(open(scriptInputsFile, 'r'))

# Figure Generation
TauBurden(scriptInputs)  


# %% 5. Handcrafted Features: Code for generating figures for Handcrafted Feature Results (Fig. 4,Fig. S4, Fig. S5, Fig. S7)
scriptInputsFile = os.path.join(codeDir, 'handcrafted_info.yml')
scriptInputs = yaml.safe_load(open(scriptInputsFile, 'r'))

#Regenerate handcrafted feature data
if scriptInputs['regenerateData'] == True:
    HandCraftedFeatures_Extraction(scriptInputs)
    
# Figure Generation
HandCraftedFeatures_Evaluation(scriptInputs)


# %% 6. Disease Classification: Code for generating figures for Disease Classifier Results (Fig. 5, Fig. S8)
scriptInputsFile = os.path.join(codeDir, 'disease_info.yml')
scriptInputs = yaml.safe_load(open(scriptInputsFile, 'r'))

#Re-train models and regenerate aggregate mask data
if scriptInputs['regenerateData']== True:
    if scriptInputs['regeneratePatchData']==True:
        Patch_Generation(scriptInputs)
    DiseaseModel_Training(scriptInputs)
    
# Figure Generation 
DiseaseModel_Evaluation(scriptInputs)

# %%7. Clustering of disease features: Code for generating figures for clustering of disease  (Fig. 6)
scriptInputsFile = os.path.join(codeDir, 'disease_info.yml')
scriptInputs = yaml.safe_load(open(scriptInputsFile, 'r'))

# Figure Generation     
DiseaseModel_Handcrafted_Visual(scriptInputs)


