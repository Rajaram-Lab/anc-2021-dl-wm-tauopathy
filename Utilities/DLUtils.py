'''
DLUtils contains code for training and validation of deep learning models

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
import ImageUtils as iu
import os
import sys
import openslide as oSlide
import numpy as np
import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as mplCB
from matplotlib.path import Path
from matplotlib.collections import PatchCollection

import scipy.misc
import re
from PIL import Image, ImageDraw
from skimage import morphology as morph
import glob
import cv2 as cv2
import tensorflow as tf
if tf.__version__[0]=='1':
    from tensorflow.python.keras.models import load_model
    from tensorflow.python.keras.models import Model
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.python.keras.layers import Activation,BatchNormalization,Dropout,Dense,Flatten
    from tensorflow.python.ops.nn_ops import softmax
    from tensorflow.python.keras.utils import Sequence
else: 
    from tensorflow.keras.models import load_model
    from tensorflow.keras.models import Model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.keras.layers import Activation,BatchNormalization,Dropout,Dense,Flatten
    from tensorflow.keras.utils import Sequence
    if tf.__version__=='2.0.0':
        from tensorflow.python.ops.nn_ops import softmax
    else:
        from tensorflow.nn import softmax


import pickle
from scipy import ndimage as ndi
from joblib import Parallel, delayed

from skimage import measure
import progressbar
import fnmatch
import time
# %% Importing Functions
def isIntersecting1d(a_min,a_max,b_min,b_max):
    '''Neither range is completely greater than the other
    '''
    return (a_min <= b_max) and (b_min <= a_max)

def isIntersecting(bb1,bb2):
    return isIntersecting1d(bb1[0],bb1[2],bb2[0],bb2[2])  and isIntersecting1d(bb1[1],bb1[3],bb2[1],bb2[3])


def Generate_Patch_Data(xmlFileList,svsFileList,outputDir,patchSize,annoLayerName,
                        magLevel=0,maxPatchesPerRegion=300,
                        openRadius=5,whiteThresh=200,backgroundClassLabel='b',
                        nameMappingFunction = lambda x:x,patchOverlapFactor=20):
    
    regionPatchCounts={}
    patch2File={}
    for imgNum,xmlFile in enumerate(xmlFileList):
    
        # Get region annotation for XML file
        annoNames,subAnno=iu.GetXmlAnnoNames(xmlFile)
        
        layersToUse=[i for i,x in enumerate(annoNames) if x==annoLayerName]
        
        for annoLayer in layersToUse:
        
            annos=iu.ReadRegionsFromXML(xmlFile,layerNum=annoLayer)
            regionPos=annos[0]
            regionNames=annos[1]
            regionInfo=annos[2]
            #isNegative=annos[3]
            
            # Initialize Slide Image File
            slide=oSlide.open_slide(svsFileList[imgNum])
            slideDims=np.array(slide.dimensions)
            
            # only use names in "+text" format
            validNameIdx=np.where([bool(re.compile(r"^[+-]\s*").match(i)) for 
                                   i in regionNames])[0]
            if(len(validNameIdx)>0):
                validNames=np.array(regionNames)[validNameIdx]
                validIsPos=[s[0]=="+" for s in validNames]
                validSuffixes=[s[1:] for s in validNames]
                validSuffixes=[nameMappingFunction(n) for n in validSuffixes]
                
                uniqueSuffixes, uniqueInverse=np.unique(np.array(validSuffixes),return_inverse=True)  
            
                # For each region type identify +/- idx
                regionsInImg={}
                for idx,r in enumerate(uniqueSuffixes):
                    regionsInImg[r] = [validNameIdx[np.logical_and(uniqueInverse==idx,validIsPos)], 
                                validNameIdx[np.logical_and(uniqueInverse==idx,np.logical_not(validIsPos))]]
            
                regionPairs={} # for each label this contains a list of +/- pair idx that overlap   
                for rName, indices in regionsInImg.items():
                    posIdx=indices[0]
                    negIdx=indices[1]
                    pairList=[]
                    for pCounter,pIdx in enumerate(posIdx):
            
                        intersectingNegIdx=[]
                        for nIdx in negIdx:
                            if(isIntersecting(regionInfo[pIdx]['BoundingBox'],
                                             regionInfo[nIdx]['BoundingBox'])):
                                intersectingNegIdx.append(nIdx)
            
                        pairList.append({'positive':pIdx,'negative':intersectingNegIdx})
            
                    regionPairs[rName]=pairList
            
            
                for className in regionPairs:
                    for regionNum in range(len(regionPairs[className])):
            
                        regionTypeDir=os.path.join(outputDir,className)
                        if(not os.path.isdir(regionTypeDir)):
                            os.makedirs(regionTypeDir)
                        # Generate mask for a given region in a given image
                        posIdx=regionPairs[className][regionNum]['positive']
                        negIdx=regionPairs[className][regionNum]['negative']
            
                        # Start with area as large as positive mask
                        bBox=regionInfo[posIdx]['BoundingBox']
                        xLeft=bBox[0]
                        yTop=bBox[1]
                        xRight=bBox[2]
                        yBottom=bBox[3]
            
                        # Extend area as needed to include negative masks
                        for nIdx in negIdx:
                            bBox=regionInfo[nIdx]['BoundingBox'] 
                            if (bBox[0] < xLeft):
                                xLeft=bBox[0] 
                            if (bBox[1] < yTop):
                                yTop=bBox[1] 
                            if (bBox[2] > xRight):
                                xRight=bBox[2] 
                            if (bBox[3] > yBottom):
                                yBottom=bBox[3] 
            
                        # Extend by patch size to allow filters to be calculated 
                        patchSizeScaled=int(patchSize*slide.level_downsamples[magLevel])
                        xLeft=int(max(0,xLeft-patchSizeScaled/2)) 
                        yTop=int(max(0,yTop-patchSizeScaled/2))            
                        xRight=int(min(slideDims[0],xRight+patchSizeScaled/2))  
                        yBottom=int(min(slideDims[1],yBottom+patchSizeScaled/2))
                        xWidth=int(((xRight-xLeft)+1)/slide.level_downsamples[magLevel])
                        yWidth=int(((yBottom-yTop)+1)/slide.level_downsamples[magLevel])
            
                        # Load Image corresponding to area
                        img=np.asarray(slide.read_region((xLeft,yTop),magLevel,
                                                         (xWidth,yWidth)))
                        #plt.imshow(img)
                        #plt.title(svsFileList[imgNum] + " Class:" + className   )
                        #plt.show()
                        #print(img.shape)
                        #print((xWidth,yWidth))
                        print(svsFileList[imgNum] + " Class:" + className   )
            
                        poly=np.array(regionPos[posIdx])
                        poly[:,0]=(poly[:,0]-xLeft)/slide.level_downsamples[magLevel]
                        poly[:,1]=(poly[:,1]-yTop)/slide.level_downsamples[magLevel]
                        poly=np.concatenate((poly,np.expand_dims(poly[0,:],axis=0)))   
                        mask = Image.new("L",(xWidth,yWidth), 0)
                        ImageDraw.Draw(mask).polygon(poly.ravel().tolist(), outline=1, fill=1)
                        mask = np.array(mask)
                        #plt.imshow(mask)
                        #plt.show()
            
                        for nIdx in negIdx:
                            poly=np.array(regionPos[nIdx])
                            poly[:,0]=(poly[:,0]-xLeft)/slide.level_downsamples[magLevel]
                            poly[:,1]=(poly[:,1]-yTop)/slide.level_downsamples[magLevel]
                            poly=np.concatenate((poly,np.expand_dims(poly[0,:],axis=0)))   
                            maskNeg = Image.new("L",(xWidth,yWidth), 0)
                            ImageDraw.Draw(maskNeg).polygon(poly.ravel().tolist(), outline=1, fill=1)
                            maskNeg = np.array(maskNeg)
                            mask=np.logical_and(mask,np.logical_not(maskNeg==1))
                            #plt.imshow(maskNeg)
                            #plt.show()
            
                        #plt.imshow(mask)
                        #plt.show()    
            
            
                        sElem=morph.disk(3)
                        if(className != backgroundClassLabel):
                            isWhite=np.all(img[:,:,0:2]>whiteThresh,axis=2)
                        else:
                            isWhite=np.all(img[:,:,0:2]>256,axis=2)
                        #plt.imshow(isWhite)
                        #plt.show()
                        isBG=morph.opening(isWhite,sElem)
                        #bgMask=np.logical_not(morph.closing(np.logical_not(isWhite),sElem))
            
                        #plt.imshow(isBG)
                        #plt.show()
            
                        candidateMask=np.logical_and(mask,np.logical_not(isBG))
                        candidateMask[range(int(patchSize/2)),:]=False
                        candidateMask[range(0,int(patchSize/2),-1),:]=False
                        candidateMask[:,range(int(patchSize/2))]=False
                        candidateMask[:,range(0,int(patchSize/2),-1)]=False
                        candidatePos=np.where(candidateMask)
            
                        #plt.imshow(candidateMask)
                        #plt.show()
            
            
                        #nP=min(100,int(candidatePos[0].size/5))
                        nP=min(maxPatchesPerRegion,int(patchOverlapFactor*candidatePos[0].size/(patchSize*patchSize)))
                        if candidatePos[0].size>0:
                            try:
                                chosenIdx=np.random.choice(range(candidatePos[0].size),nP,replace=True)
                            
                                
                                cX=candidatePos[0][chosenIdx]
                                cY=candidatePos[1][chosenIdx]
                    
                                #plt.figure(figsize=(15,15))
                                for n in range(nP): #loop over selected patches and save to disk
                                    #print('Patch Number:' +str(n))  
                                    patchRange=np.asarray(range(-int(np.floor(patchSize/2)),int(np.ceil(patchSize/2))))
                                    patchImg=img[cX[n]+patchRange,:,:]
                                    patchImg=patchImg[:,cY[n]+patchRange,:]
                                    patchImg=patchImg[:,:,range(3)]/255
                    
                                    #plt.subplot(10,10,n+1)
                                    #plt.imshow(patchImg)
                                    #plt.axis('off')
                    
                                    
                                    if className in regionPatchCounts:
                                        regionPatchCounts[className]=regionPatchCounts[className]+1
                                    else:
                                        regionPatchCounts[className]=1
                                    
                                    scipy.misc.toimage(patchImg, cmin=0.0, cmax=1.0).save(os.path.join(regionTypeDir,
                                                         className + str(regionPatchCounts[className]) + '.png'))
                                    #print( className + str(regionPatchCounts[className]) + " " +svsFileList[imgNum]
                                    #      )
                                    patch2File[(className + str(regionPatchCounts[className]) + '.png')]=svsFileList[imgNum]
                            except:
                                print('Failed')
                                print(candidatePos[0].size)
                        #plt.show()    
    return patch2File





def Create_Testing_Dir(origDataDir,testingDir,testFraction=0.1):
    #  Moves a randomly selected fraction (sepecifid by testFraction) of files
    # from origDataDir to testingDir, while preserving the directory structure etc

    subdirs=[d for d in os.listdir(origDataDir) if os.path.isdir(os.path.join(origDataDir,d))]
    for dirName in subdirs:
        fullDir=os.path.join(origDataDir,dirName)
        fileList=np.array([f for f in os.listdir(fullDir) if os.path.isfile(os.path.join(fullDir,f))])
        randIdx=np.random.choice(len(fileList),replace=False,size=(np.int32(testFraction*len(fileList)),1))    
        filesToMove=fileList[randIdx]
        if not os.path.isdir(os.path.join(testingDir,dirName)):
            os.makedirs(os.path.join(testingDir,dirName))
        for f in np.ndarray.tolist(filesToMove):
            source=os.path.join(fullDir,f[0])
            destination=os.path.join(testingDir,dirName,f[0])
            os.rename(source,destination)
            


def Create_Directory_With_Subset_Of_Classes(sourceDirRoot,destDirRoot,dirNamesToCopy):
    # This is meant to address the need of building a classifier on a subset of classes
    # without needed to rerun the data generation script
    # simply specify the sourceDirRoot which contains the patch data, and the 
    # classes to copy as dirNamesToCopy and the function will create symbolic links
    # in destRootDir with the appropriate directory structure
    
    for dirName in dirNamesToCopy:
            sourceDir=os.path.join(sourceDirRoot,dirName)
            destDir=os.path.join(destDirRoot,dirName)
            
            if(not os.path.isdir(destDir)):
                os.makedirs(destDir)
    
           
            print([sourceDir,destDir])
            for f in glob.glob(os.path.join(sourceDir,'*.png')):
                filename=os.path.split(f)[1]
                os.symlink(os.path.join(sourceDir,filename),os.path.join(destDir,filename))    

def Get_Eff_Kernel_Params(model):
    nLayers=len(model.layers)
    k=-1
    s=-1
    for layerNum in range(nLayers):
        layerConfig=model.layers[layerNum].get_config()
        if('kernel_size' in layerConfig):
            k1=layerConfig['kernel_size'][0]
        else:
            k1=1
        if('strides' in layerConfig):
            s1=layerConfig['strides'][0]
        else:
            s1=1
            
        #print(layerNum,k1,s1)
        if(k>0 and s>0):    
            k=k+(k1-1)*s
            s=s1*s
        else:
            k=k1
            s=s1
        
    return k,s


class Classifier():
    
    def __init__(self):
        self.model=[]
        self.patchSize=[]
        self.magLevel=[]
        self.labelNames=[]
        self.intModel=[]
        self.numberOfClasses=[]
        self.effectiveKernel=[]
        self.effectiveStride=[]
        

    def Init(self,model, patchSize, magLevel,labelNames,savefile='classifier.pkl'):

        self.model=model
        self.patchSize=patchSize
        self.magLevel=magLevel
        self.labelNames=labelNames
    
        self.intModel=Model(inputs=model.input,outputs=model.get_layer(index=-2).output)
        self.numberOfClasses=self.intModel.layers[-1].output_shape[3]
        self.effectiveKernel,self.effectiveStride=Get_Eff_Kernel_Params(self.intModel)
        
    def Load(self,filename):
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close() 
        
        try:
          self.model=load_model(tmp_dict['hd5'])         
        except:
          self.model=load_model(filename+'.hd5')   
        self.patchSize=tmp_dict['patchSize']
        self.magLevel=tmp_dict['magLevel']
        self.labelNames=tmp_dict['labelNames']

    
        self.intModel=Model(inputs=self.model.input,outputs=self.model.get_layer(index=-2).output)
        self.numberOfClasses=self.intModel.layers[-1].output_shape[3]
        self.effectiveKernel,self.effectiveStride=Get_Eff_Kernel_Params(self.intModel)
        


    def Save(self,fileName,hd5File=None):
        
        if hd5File is None:
            hd5File=fileName+'.hd5'
            
        extraData={'patchSize':self.patchSize,'magLevel':self.magLevel,
                          'labelNames':self.labelNames,'hd5':hd5File}        
        
        pickle.dump(extraData, open(fileName, "wb" ) )
        
        self.model.save(hd5File,save_format='h5')
        


def Load_Data(dataDir,patchSize):
    fileList = [y for x in os.walk(dataDir) for y in glob(os.path.join(x[0], '*.png'))]
    dirList=[os.path.split(os.path.split(x)[0])[1] for x in fileList]
    labelNames,Y=np.unique(dirList,return_inverse=True)
    X=np.zeros((len(fileList),patchSize,patchSize,3))
    for counter,fileName in enumerate(fileList):
        X[counter,:,:,:]=np.array(Image.open(fileName))
    return X,Y,labelNames,fileList

def load_file(im_file):
    return np.array(Image.open(im_file))

def Load_And_Transform(dataDir,patchSize,trainingFraction, augmentFactor):
    print('from git', dataDir)
    #fileList = np.array([y for x in os.walk(dataDir) for y in glob(os.path.join(x[0], '*.png'))])
    fileList = []
    for root, dirnames, filenames in os.walk(dataDir):
        for filename in fnmatch.filter(filenames, '*.png'):
            fileList.append(os.path.join(root, filename))

    dirList=[os.path.split(os.path.split(x)[0])[1] for x in fileList]
    labelNames,Y=np.unique(dirList,return_inverse=True)
    Y=np.array(Y)
    numberFiles=len(fileList)
    numberTraining=np.int32(np.round(trainingFraction*numberFiles))
    numberTesting=numberFiles-numberTraining
    
    trainingIdx=np.random.choice(np.arange(numberFiles),size=numberTraining,replace=False)
    testingIdx=np.delete(np.arange(numberFiles),trainingIdx)
    
    
    augmentIdx= np.random.choice(trainingIdx,size=np.int32(np.round(augmentFactor*numberTraining)),replace=True)
    augmentIdx=np.concatenate((trainingIdx,augmentIdx))
    np.random.shuffle(augmentIdx)
    
    numberTrainingFull=augmentIdx.size
    
    X_train=np.zeros((numberTrainingFull,patchSize,patchSize,3))
    
    
    print(augmentIdx)
    Y_train=Y[augmentIdx]
    Y_test=Y[testingIdx]
    
    #X_test=np.zeros((numberTesting,patchSize,patchSize,3))
    #bar=progressbar.ProgressBar(max_value=numberTesting)
    #for counter,idx in enumerate(testingIdx):
    #    X_test[counter,:,:,:]=np.array(Image.open(fileList[idx]))
    #    bar.update(counter)
    #bar.finish()     
    
  
    
    X_test=[]
    X_test.extend(Parallel(n_jobs=10)(delayed(load_file)(fileList[im_idx]) for im_idx in testingIdx))
    X_test=np.array(X_test)/255
        
    bar=progressbar.ProgressBar(max_value=numberTrainingFull)
    for counter,idx in enumerate(augmentIdx):
        X_train[counter,:,:,:]=Img_Transform(np.array(Image.open(fileList[idx]))/255)
        bar.update(counter)
    bar.finish()      
   
        
    return X_train,Y_train,X_test,Y_test,labelNames,fileList,testingIdx
  

def Region_Label_Plot(labelImg,numberOfClasses,cmap,alpha=0.2,classLabels=[]):
    def ring_coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = ob.shape[0]
        codes = np.ones(n, dtype=Path.code_type) * Path.LINETO
        codes[0] = Path.MOVETO
        return codes
    
    ax=plt.gca()
    if(len(classLabels)>0):
        fig=plt.gcf()
        axCB=fig.add_axes([0.95,0.15,0.05,0.7])
  
   
    
    for c in range(numberOfClasses):
        if(labelImg.ndim==3):
            mask,bdry,hierarchy=cv2.findContours(np.uint8(np.pad(np.array(np.argmax(labelImg,axis=2)==c),1,'constant')),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        else:
            mask,bdry,hierarchy=cv2.findContours(np.uint8(np.pad(np.array(labelImg==c),1,'constant')),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)    
        hierarchy=np.squeeze(hierarchy)
        patchList=[]
        #for regionNum in range(len(bdry)):
        if(bdry):
            contourTree={}
            try:
                isChild=hierarchy[0:,3]>=0
                for l in range(hierarchy.shape[0]):
                    if(not isChild[l]):
                        contourTree[l]=np.where(hierarchy[0:,3]==l)[0]
            except:
                isChild=np.array(hierarchy[3]>=0)
                contourTree[0]=[]
            
            
            for v in contourTree:
                vertices=np.squeeze(np.concatenate([bdry[v]]+ [bdry[i] 
                                                for i in contourTree[v]]))
                codes=np.concatenate([ring_coding(bdry[v])]+ [ring_coding(bdry[i]) 
                                                for i in contourTree[v]])
                if(vertices.size>2):
                    path = Path(vertices,codes)
                    #patch = patches.PathPatch(path, alpha=0.1,
                    #                          facecolor=cmap[c,:], edgecolor='k')
                    patch = patches.PathPatch(path)
                    #patch.set_alpha(0.005)
                    #x.add_patch(patch) 
                    patchList.append(patch)
                            
            p=PatchCollection(patchList,alpha=alpha)
            p.set_edgecolor('k')
            p.set_facecolor(cmap[c,:])
            #print(cmap[c,:])
            ax.add_collection(p)
    if(len(classLabels)>0):
        cmap1=np.concatenate((cmap,alpha*np.ones((cmap.shape[0],1))),axis=1)
        cbar=mplCB.ColorbarBase(axCB,colors.ListedColormap(cmap1),orientation='vertical',
                                ticks=range(0,numberOfClasses),boundaries=np.linspace(0,numberOfClasses,numberOfClasses+1)-0.5)
        cbar.ax.set_yticklabels(classLabels)
    
    



def Img_Transform(img,maxHue=15,satLow=0.75,satHigh=1.25):
    I=Image.fromarray(np.uint8(img*255))
    iHSV=np.array(I.convert('HSV'))
    randHue=np.random.randint(2*maxHue)-maxHue
    randSat=np.random.uniform(low=satLow,high=satHigh)
    iHSV[:,:,0]=np.remainder(np.array(iHSV[:,:,0])+randHue,255)
    satVals=iHSV[:,:,1]*randSat
    satVals[satVals<0]=0
    satVals[satVals>255]=255
    iHSV[:,:,1]=satVals
    iTrans=Image.fromarray(iHSV,mode='HSV').convert('RGB')
    if np.random.randint(2)==1:
        iTrans=iTrans.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.randint(2)==1:
        iTrans=iTrans.transpose(Image.FLIP_TOP_BOTTOM)
    imgTrans=np.array(iTrans)/255
    return imgTrans



def Augment_Data(data,labels,randsPerImg):
    #sess = tf.Session()
    (nImg,xRes,yRes,colorDepth)=data.shape
    augData=np.zeros((nImg*randsPerImg,xRes,yRes,colorDepth))
    augLabels=np.zeros((labels.shape[0]*randsPerImg,labels.shape[1]))
    bar=progressbar.ProgressBar(max_value=nImg)
    for imgNum in range(nImg):
        img=data[imgNum,:,:,:]
        #r=np.random.randint(2,size=randsPerImg)
        for randNum in range(randsPerImg):
            counter=imgNum*randsPerImg+randNum
            augLabels[counter,:]=labels[imgNum,:]
            augData[counter,:,:,:]=Img_Transform(img)
            #if(r[randNum]==0):
            #    
            #    augData[counter,:,:,:]=tfImg.random_flip_up_down(tfImg.random_flip_left_right(
            #            tfImg.random_saturation(img,lower=0.5, upper=1.5))).eval(session=sess)    
            #else:
             #   
             #   augData[counter,:,:,:]=tfImg.random_flip_up_down(tfImg.random_flip_left_right(
             #           tfImg.random_hue(img,0.1))).eval(session=sess)    
        bar.update(imgNum)           
    randomOrder=np.random.shuffle(np.arange(augData.shape[0]))
    augData=np.squeeze(augData[randomOrder,:,:,:])    
    augLabels=np.squeeze(augLabels[randomOrder,:])       
    bar.finish()
    return augData,augLabels


def Drop_Tiny_Regions(mask,minPixelSize=50,maxEccentricity=0.9,minAreaToPerimeterRatio=0):
    nuclearLabels=measure.label(mask)
    props=measure.regionprops(nuclearLabels)
    areas=np.array([r['area'] for r in props])
    ecc=np.array([r['eccentricity'] for r in props])
    areaToPerimeterRatio=np.array([r['area']/r['perimeter'] for r in props])
    regionsToDropIdx=np.where(np.logical_or(np.logical_or(
            areas<minPixelSize,
            ecc>maxEccentricity),
            areaToPerimeterRatio<minAreaToPerimeterRatio))[0]+1
    cleanMask=np.copy(mask)
    for a in regionsToDropIdx:
        cleanMask[nuclearLabels==a]=False
    return cleanMask
        



def Profile_Region(classifier,slide,cornerPos,regionSize):
    patchSize=classifier.patchSize
    magLevel=classifier.magLevel
    #stride=classifier.effectiveStride 
    (tumorWidth,tumorHeight)=slide.level_dimensions[magLevel]
    
    x=int((cornerPos[0]-int((patchSize)/(2/slide.level_downsamples[magLevel])))-1)
    y=int((cornerPos[1]-int((patchSize)/(2/slide.level_downsamples[magLevel])))-1)
    
    imgSize=(int(regionSize[0]/slide.level_downsamples[magLevel]+patchSize-1),
             int(regionSize[1]/slide.level_downsamples[magLevel]+patchSize-1))
             
    img=np.asarray(slide.read_region((x,y),magLevel,imgSize))[:,:,range(3)]
    img2Show=img[range(int(patchSize/2),int(img.shape[0]-patchSize/2)),:,:]
    img2Show=img2Show[:,range(int(patchSize/2),int(img.shape[1]-patchSize/2)),:]
    #plt.imshow(img[range(int(patchSize/2),int(img.shape[0]-patchSize/2)),range(int(patchSize/2),int(img.shape[1]-patchSize/2)),:])
    #plt.imshow(img2Show)
    #print(img2Show.shape)
    #plt.show()
    response=np.squeeze(classifier.intModel.predict(img.reshape((1,img.shape[0],img.shape[1],3))/255))
    return response,img2Show


def ProfileRegionMultiscale(classifier,slide,cornerPos,regionSize,smoothingSize=5,responseThresh=0):

    intModel=Model(inputs=classifier.model.input, 
                         outputs=classifier.model.get_layer(index=-2).output)
    
  
    cornerOffset=int((4*classifier.patchSize[1]-classifier.patchSize[0])/2)
    sizeCorrection=int(cornerOffset/2)
    
    cornerPos1=[cornerPos[0]-cornerOffset,cornerPos[1]-cornerOffset]
    regionSize1=[int(regionSize[0]/4)+sizeCorrection,int(regionSize[1]/4)+sizeCorrection]
    
    img0=np.asarray(slide.read_region(cornerPos,0,regionSize))[:,:,range(3)]
    img1=np.asarray(slide.read_region(cornerPos1,1,regionSize1))[:,:,range(3)]
    imgList=[]
    imgList.append(img0.reshape((1,img0.shape[0],img0.shape[1],3))/255)
    imgList.append(img1.reshape((1,img1.shape[0],img1.shape[1],3))/255)
    response=np.squeeze(intModel.predict(imgList))
    
    if(smoothingSize==0):
      smoothResponse=response
      
    else:
      sElem=morph.disk(smoothingSize)
      smoothResponse=np.zeros(response.shape)
      for c in range(classifier.numberOfClasses):
          smoothResponse[:,:,c]=ndi.convolve(response[:,:,c],sElem)/np.sum(sElem)
    
    imgClasses=np.uint8(np.argmax(smoothResponse,axis=2))
    imgClasses[np.max(smoothResponse,axis=2)<responseThresh]=0
    
   
    imgOffset=int(classifier.patchSize[0]/2)
    img2Show=img0[(imgOffset-1):(-imgOffset),(imgOffset-1):(-imgOffset),:]

    return imgClasses,smoothResponse,img2Show

def Label2Idx(labels):
    label2Idx={}
    for idx,l in enumerate(labels):
        label2Idx[l]=idx
    return label2Idx


# %%
#NOTE: This class is also implemented in patchgen. Remove one of these.
class PatchReader():
    # Convenience Class to Read Image Regions With Arbitrary Magnification from a Slide
    def __init__(self,slide,downSampleFactor,downSampleTolerance=0.025):
        # Inputs:
        # slide - openslide slide that image regions need to be sampled from
        # downSampleFactor - number specifying the magnfication level relative 
        #                    to the highest resolution available. e.g. downSampleFactor
        #                    of 3 would yield patches with one third the max resolution
        # downSampleTolerance - a number specifying numerical accuracy of the downSampleFactor.
        #                    It is used to select a pre-existing maglevel. For example, 
        #                    if it is 0.025 and a downsampleFactor or 4 was used, 
        #                    and a layer with 4.024 was available, it would be used
        self.slide=slide
        assert downSampleFactor>=1 # cannot get higher resolution than original image
        self.downSampleFactor=downSampleFactor
        
        if np.min(np.abs(np.array(slide.level_downsamples)-downSampleFactor))<downSampleTolerance:
          self.magLevel=int(np.argmin(np.abs(np.array(slide.level_downsamples)-downSampleFactor)))
          self.isResizeNeeded=False
          self.downSampleExtracted=self.downSampleFactor

        else:
          self.magLevel=int(np.where(np.array(slide.level_downsamples)<downSampleFactor)[0][-1])
          self.isResizeNeeded=True
          self.downSampleExtracted=slide.level_downsamples[self.magLevel]
  
    
    def LoadPatches(self,patchCenters,patchSize,showProgress=False):
      # Generate patch data given patch centers and patch sizes
      # Inputs:
      # patchCenters - a numpy array, with 2 columns (specifying xy positions 
      #                in ABSOLUTE coordinates, i.e. at the max mag level). 
      #                Rows correspoind to different patches
      # patchSize - a 1x2 numpy array/list/tuple specifying OUTPUT patch size 
      #             (i.e. at the output downSample space)
      
      assert ((type(patchCenters) is np.ndarray) and patchCenters.shape[1]==2),sys.error('Invalid Patch Centers')
      numberOfPatches  =patchCenters.shape[0]
      
      if self.isResizeNeeded:
        patchSizeLoaded=np.int32(np.round(np.array(patchSize)*self.downSampleFactor/self.downSampleExtracted))
      else:
        patchSizeLoaded=np.array(patchSize)
      
      patchData=np.zeros((numberOfPatches,patchSize[0],patchSize[1],3),np.uint8)
      
      cornerPositions=np.int32(np.floor(np.array(patchCenters)-self.downSampleExtracted*(patchSizeLoaded+1)/2.0))
      if showProgress:
        bar=progressbar.ProgressBar(max_value=numberOfPatches)
      for patchCounter in range(numberOfPatches):
        img=self.slide.read_region(cornerPositions[patchCounter],self.magLevel,patchSizeLoaded)
        if self.isResizeNeeded:
          img=img.resize(patchSize,Image.BILINEAR)
        patchData[patchCounter]=np.array(img,np.uint8)[:,:,np.arange(3)]
        if showProgress:
          bar.update(patchCounter)
      if showProgress:  
        bar.finish()      
      
      return patchData


class SlideAreaGenerator(Sequence):
  """ Uses Keras' generator to create boxes out of the slide. """
  def __init__(self,slide,boxHeight=1000,boxWidth=1000, batch_size=4,borderSize=10, 
                shuffle=False, downSampleFactor=1, preproc_fn = lambda x:np.float32(x)/255):
      """ 
      Initialize the generator to create boxes of the slide. 
      
      ### Returns
      - None
      
      ### Parameters:
      - `slide: slide`  The loaded slide object.
      - `boxHeight: int`  Height of the box that will slide across the slide.
      - `boxWidth: int`  Width of the box that will slide across the slide.
      - `batchSize: int`  Number of boxes to fit in the GPU at time as we profile the slide.
      - `borderSize: int`  Border around the box to profile along with the box itself.
      - `shuffle: bool`  Shuffle indices of boxes in the epoch.
      - `downSampleFactor: int`  Supply reduced size SVS file to profile faster.
      - `preproc_fn: function`  supply preprocessing function. Else, defaults to input/255.
      
      """
      self.slide=slide
      self.boxHeight=boxHeight
      self.boxWidth=boxWidth
      self.batch_size = batch_size
      self.shuffle = shuffle
      
      self.dsf=downSampleFactor
      self.on_epoch_end()
      if self.dsf!=1:
          self.imgReader=PatchReader(slide, downSampleFactor)
          
      (self.slideWidth,self.slideHeight)=slide.dimensions
      self.rVals,self.cVals=np.meshgrid(np.arange(0,self.slideHeight,self.dsf*boxHeight),
                                        np.arange(0,self.slideWidth,self.dsf*boxWidth))          
      self.numberOfBoxes=self.rVals.size
      self.rVals.resize(self.rVals.size)
      self.cVals.resize(self.cVals.size)
      self.borderSize=borderSize
      self.preproc=preproc_fn


  def __len__(self):
      """
      Denotes the number of batches per epoch
      
      ### Returns
      - `int`  Number of batches for the keras generator.
      
      ### Parameters
      - None
      
      """
      return int(np.ceil(self.numberOfBoxes / self.batch_size))

  def __getitem__(self, index):
      """
      Generate one batch of data
      
      ### Returns
      - `X: np.array()` of shape (numBoxesInBatch, numRows, numCols, numChannels)
      - `Y: List(np.array(), np.array())` where first numpy array is row values in the batch and second numpy array is col value in the batch.
      
      ### Parameters
      - `index: int`  batchIndex to be called.
      """
      # Generate indexes of the batch
      #indexes = self.samplingIndex[self.indexes[index*self.batch_size:(index+1)*self.batch_size]]
      #indexes = self.indexes[index]
      indexes=np.arange(index*self.batch_size,np.minimum((index+1)*self.batch_size,self.numberOfBoxes))
    
      
      Y=[self.rVals[indexes],self.cVals[indexes]]
      
      if self.dsf==1:           
          X=np.zeros((len(indexes),self.boxHeight+2*self.borderSize,self.boxWidth+2*self.borderSize,3),dtype=np.float32)   
          for i,idx in enumerate(indexes):
             img=np.zeros((self.boxHeight+2*self.borderSize,self.boxWidth+2*self.borderSize,3))
             r=self.rVals[idx]
             c=self.cVals[idx]
            
             imgHeight=int(np.minimum(self.boxHeight+self.borderSize,self.slideHeight-(r))+self.borderSize)
             imgWidth=int(np.minimum(self.boxWidth+self.borderSize,self.slideWidth-(c))+self.borderSize)
            
             img[0:imgHeight,0:imgWidth]=np.array(self.slide.read_region((c-self.borderSize,r-self.borderSize),0,(imgWidth,imgHeight)))[:,:,range(3)]

             X[i,:,:,:]=self.preproc(img)
      else:
          patchCenters=np.zeros((len(indexes),2))
          #patchCenters[:,1]=self.rVals[indexes]+self.dsf*int(self.boxHeight/2+self.borderSize)
          #patchCenters[:,0]=self.cVals[indexes]+self.dsf*int(self.boxWidth/2+self.borderSize)
          patchCenters[:,1]=self.rVals[indexes]+self.dsf*int(self.boxHeight/2)
          patchCenters[:,0]=self.cVals[indexes]+self.dsf*int(self.boxWidth/2)
          
          patchSize=(self.boxHeight+2*self.borderSize,self.boxWidth+2*self.borderSize)
          X=self.preproc(self.imgReader.LoadPatches(patchCenters,patchSize))

         
  
          
      return X, Y

  def on_epoch_end(self):
      'Updates indexes after each epoch'
      #self.indexes = np.arange(self.numberOfPatches)

def Profile_Slide_Fast(model, slide, stride, patchSize, numberOfClasses,
                       isMultiOutput=False, downSampleFactor=1, isModelFlattened=True,
                       boxHeight=2000, boxWidth=2000, batchSize=4,
                       useMultiprocessing=True, nWorkers=64, verbose=1,
                       returnActivations=True, preproc_fn = lambda x:np.float32(x)/255):
    """
    Runs the model across the slide and returns the prediction classes and activations of the whole slide.
    
    ### Returns
    - `slidePredictionsList: [np.array(), ...] or np.array()`. List is returned when `isMultiOutput=True`
    - `slideActivationsList: [np.array(), ...] or np.array()`. List is returned when `isMultiOutput=True`
    
    ### Parameters
    - `model: model`  The loaded keras model. Can be intermediate CNN model or FCN.
    - `slide: slide`  The loaded slide object.
    - `stride: int`  The model's stride.
    - `patchSize: int`  The model's patchsize.
    - `numberOfClasses: int`  Total number of classes (including background) the model was trained on.
    - `isMultiOutput: int`  If the model returns one or more outputs for one input.
    - `downSampleFactor: int`  Supply reduced size SVS file to profile faster.
    - `isModelFlattened: int:`  If FCN, mark as True. If intermediate model, then mark False.
    - `boxHeight: int`  Height of the box that will slide across the slide.
    - `boxWidth: int`  Width of the box that will slide across the slide.
    - `batchSize: int`  Number of boxes to fit in the GPU at time as we profile the slide.
    - `useMultiprocessing: bool`  Use the multiprocessing to profile the slide faster.
    - `nWorkers:  int` number of parallel processes to run if useMultiprocessing = True.
    - `verbose: int`  print details as we profile the slides.
    - `returnActivations: bool`  return NN activation outputs along with predicted classes.
    - `preproc_fn: function`  supply preprocessing function. Else, defaults to input/255.
    
    """
    
    if verbose>0:
        start_time = time.time()
    borderSize=int(patchSize/2)
    
    slideGen=SlideAreaGenerator(slide,downSampleFactor=downSampleFactor,
                                boxHeight=boxHeight,boxWidth=boxWidth, 
                                batch_size=batchSize,borderSize=borderSize,
                                preproc_fn=preproc_fn)    

    
    res=model.predict_generator(slideGen,workers=nWorkers,
                                use_multiprocessing=useMultiprocessing,
                                verbose=verbose)
    
    (slideWidth,slideHeight)=slide.dimensions
    outHeight=int(np.ceil((slideHeight-(downSampleFactor*patchSize))/(downSampleFactor*stride)))+1
    outWidth=int(np.ceil((slideWidth-(downSampleFactor*patchSize))/(downSampleFactor*stride)))+1
    
    if not isModelFlattened:
        outBoxHeight=res.shape[1]
        outBoxWidth=res.shape[2]
    else:
        outBoxHeight=int(np.floor(boxHeight/stride))+1
        outBoxWidth=int(np.floor(boxWidth/stride))+1
        if res.size!=res.shape[0]*numberOfClasses*outHeight*outWidth:
            s=int(np.sqrt(res.size/(res.shape[0]*numberOfClasses)))
            outBoxHeight=s
            outBoxWidth=s
            print('Warning, automatic sizing failed, selecting boxsize='+str(outBoxHeight))
            assert res.size==res.shape[0]*numberOfClasses*outBoxHeight*outBoxWidth,'Failed!'
    
    
    
    rVals1,cVals1=np.meshgrid(np.arange(0,slideHeight,boxHeight*downSampleFactor),
                                            np.arange(0,slideWidth,boxWidth*downSampleFactor))
    rVals1.resize(rVals1.size)
    cVals1.resize(cVals1.size)
    
    rVals=np.uint32(np.ceil(rVals1/(downSampleFactor*stride)))
    cVals=np.uint32(np.ceil(cVals1/(downSampleFactor*stride)))
    numberOfBoxes=rVals.size
    
    if isMultiOutput:
        numberOfOutputs=len(model.output)
    else:
        numberOfOutputs=1
   
       
    slideClassesList=[]
    slideActivationsList=[]
    for outNum in range(numberOfOutputs):
        outHeightR=int(outBoxHeight*np.ceil(outHeight/outBoxHeight))
        outWidthR=int(outBoxWidth*np.ceil(outWidth/outBoxWidth))
        slideClasses=np.zeros((outHeightR,outWidthR),dtype=np.uint8)
        if returnActivations:
            slideActivations=np.zeros((outHeightR,outWidthR,numberOfClasses))
        else:
            slideActivations=None
        
        if isMultiOutput:
            response=res[outNum]
        else:
            response=res
            
        
        if isModelFlattened:
            classes=np.argmax(response.reshape(response.shape[0],outBoxHeight,outBoxWidth,numberOfClasses),axis=-1)
            activations=response.reshape(response.shape[0],outBoxHeight,outBoxWidth,numberOfClasses)
        else:
            classes=np.argmax(response,axis=-1)
            activations=response
            
        for i in range(numberOfBoxes):
            r=rVals[i]
            c=cVals[i]
            imgHeight=int(np.minimum(outBoxHeight,outHeightR-(r)))
            imgWidth=int(np.minimum(outBoxWidth,outWidthR-(c)))
            bS=0
            #try:
            slideClasses[r:(r+imgHeight),c:(c+imgWidth)]=classes[i][bS:(bS+imgHeight),bS:(bS+imgWidth)]
            # except:
            #     print(res.shape)
            #     print(classes.shape)
            #     print(classes[i].shape)
            #     print(slideClasses[r:(r+imgHeight),c:(c+imgWidth)].shape)
            #     print(classes[i][bS:(bS+imgHeight),bS:(bS+imgWidth)].shape)
            #     import sys
            #     sys.exit('Error')
            if returnActivations:
                slideActivations[r:(r+imgHeight),c:(c+imgWidth)]=activations[i][bS:(bS+imgHeight),bS:(bS+imgWidth)]
                
        
        if isMultiOutput:
            slideClassesList.append(slideClasses[0:outHeight,0:outWidth])
            if returnActivations:
                slideActivationsList.append(slideActivations[0:outHeight,0:outWidth,:])
        else:
            slideClassesList=slideClasses[0:outHeight,0:outWidth]
            if returnActivations:
                slideActivationsList=slideActivations[0:outHeight,0:outWidth,:]            
  
        

    
    if verbose>0:        
        print("--- %s seconds ---" % (time.time() - start_time))        
    return slideClassesList,slideActivationsList

