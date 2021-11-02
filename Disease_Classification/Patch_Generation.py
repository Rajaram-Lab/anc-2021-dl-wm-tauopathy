def Patch_Generation(scriptInputs):
    '''
    PatchGeneration generates image patches for training the disease classifier model
    Parameters and relevant files are provided in the accompanying yaml file (region_info.yml)
    
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
    import PatchGen as pg
    import openslide as oSlide
    import glob
    import pickle
    import numpy as np

    
    
    # %% Build list of annotated svs files
    
    # THIS CODE ASSUMES THE ANNO FILES USE THE SAME NAMING CONVENTION AS THE SVS

    slidesDir=scriptInputs['imagesRootDir']
    
    
    masksDir=scriptInputs['regionMasksDir']
    diseaseDirs=['PureAD','PurePSP','PureCBD']
    svsList=[]
    maskFileList=[]
    for dirName in diseaseDirs:
        at8Files=glob.glob(os.path.join(slidesDir,dirName,'*AT8*.svs'))
        for svs in at8Files:
                regionMaskFile=os.path.splitext(os.path.split(svs)[-1])[0]
                regionMaskFile=os.path.join(masksDir,regionMaskFile+'_RegionMasks.pkl')    

                slide=oSlide.open_slide(svs)
                if slide.properties['aperio.AppMag']=='20':
                  svsList.append(svs)  
                  maskFileList.append(regionMaskFile)
                else:
                    print(svs+' excluded!')

           
    # %% Patch Generation from annoFiles
    regions=['CTX','WM']
    for regionToProfile in regions:
        assert regionToProfile in ['WM','CTX']
        basePatchDir=scriptInputs['patchDir']
        outputPatchDir=os.path.join(basePatchDir,regionToProfile) # Change name as needed
        downSampleLevels=scriptInputs['downSampleLevels'] # Downsampling factor relative to max (typically 20X). So 4 will give the 5X image. Adding multiple values gives patches at different scales
        patchSizeList=scriptInputs['patchSizeList'] # Patch size (we assume patches are square) in pixels. Specify patch size separately for each scale in downSampleLevels
        showProgress=scriptInputs['showProgress']
        maxPatchesPerAnno=scriptInputs['maxPatchesPerAnno'] # Maximum number of patches sampled from an annotation10000
        maxAvgPatchOverlap=scriptInputs['maxAvgPatchOverlap'] # How tightly patches are allowed to overlap. 0 implies no overlap, 1 implies number of patches is selected so that combined area of patches= area of annotation
        minFracPatchInAnno=scriptInputs['minFracPatchInAnno'] # What percentage of the patch must belong to same class as the center pixel, for the patch to be considered
        for maskFile,svsFile in zip(maskFileList,svsList):
            hdf5File=os.path.join(outputPatchDir,os.path.split(svsFile)[-1].replace('.svs','.hdf5'))  

            slide=oSlide.open_slide(svsFile)
             
            mask,maskLabels=pickle.load(open(maskFile,'rb'))
            mask=np.uint8(mask==maskLabels.index(regionToProfile))
            disease=os.path.split(svsFile)[-1].split('_')[0].upper()
            maskToClassDict={1:disease}
          
            #MAKE maskToClassDict
            patchData,patchClasses,patchCenters=pg.PatchesFromMask(slide,mask,
                                                                downSampleLevels,patchSizeList,
                                                                maskToClassDict,
                                                                maxPatchesPerAnno=maxPatchesPerAnno,
                                                                showProgress=showProgress,
                                                                maxAvgPatchOverlap=maxAvgPatchOverlap,
                                                                minFracPatchInAnno=minFracPatchInAnno)
            
            pg.SaveHdf5Data(hdf5File,patchData,patchClasses,patchCenters,downSampleLevels,patchSizeList,svsFile)   
