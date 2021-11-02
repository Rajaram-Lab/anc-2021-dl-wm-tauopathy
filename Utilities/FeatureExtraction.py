'''
FeatureExtraction contains functions to extract shape, size. and texture features
from aggregate masks and/or accompanying images

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

import mahotas
import numpy as np
from scipy.stats import median_absolute_deviation,mode,kurtosis,skew
from scipy import ndimage as ndi
from skimage import morphology as morph
from skimage.morphology import remove_small_objects,skeletonize
from skimage.transform import resize
from skimage.morphology import medial_axis
from scipy.spatial.distance import pdist,squareform
from skimage import measure
import cv2 as cv2
from abc import ABC,abstractmethod
from multiprocessing import Pool
from math import sqrt,pi as PI
from skimage.color import rgb2hed,rgb2lab
import colour
import warnings
import networkx as nx
from math import sqrt, atan2, pi as PI
# try:
#     import skan
# except:
#     print('Could not find skan. Skeletonization features will not work')


# %%

def ReadImg(blockCoords):
    (cornerPos,magLevel,blockSizeInt,isResizeNeeded,blockSizeOut,cornerPosOut)=blockCoords
    blockImg=np.array(mySlide.read_region(cornerPos,magLevel,blockSizeInt),order='f')[:,:,range(3)]
    if isResizeNeeded:
        blockOut=cv2.resize(blockImg,blockSizeOut)
    else:
        blockOut=blockImg
    return ((cornerPosOut,blockSizeOut),blockOut)


def ReadSlide(slide,blockWidth=2000,nWorkers=64,downSampleFactor=1,
              normalizer=None,downSampleTolerance=0.025):

    global mySlide
    mySlide=slide

    # Determine what level in the image pyramid we will pul images from
    # Not this is possibly an intermediate between highest magnification in pyrmaid and outpur mag desired
    if np.min(np.abs(np.array(slide.level_downsamples)-downSampleFactor))<\
        downSampleTolerance: #Is one of the existing levels close enough to desired downsampling
      
      # Use an existing downsampled imaged, no manual downsampling needed
      magLevel=int(np.argmin(np.abs(np.array(slide.level_downsamples)-downSampleFactor)))
      isResizeNeeded=False
      downSampleFactor=slide.level_downsamples[magLevel]

    else:
      # we will need to manually dowsnample
      magLevel=int(np.where(np.array(slide.level_downsamples)<downSampleFactor)[0][-1])
      isResizeNeeded=True
    
    downSampleExtracted=slide.level_downsamples[magLevel]

    dim=slide.dimensions
    # Output image size
    nR=int(np.ceil(dim[1]/downSampleFactor))
    nC=int(np.ceil(dim[0]/downSampleFactor))
     
    # Number of blocks needed
    nBR=int(np.ceil(nR/blockWidth))
    nBC=int(np.ceil(nC/blockWidth))

    # Detemine various image extraction parameters
    dSInt=downSampleFactor/downSampleExtracted # downsampling between pyramid level at which we extract image and output image
    
    # if self.isResizeNeeded:
    #   patchSizeLoaded=np.int32(np.round(np.array(patchSize)*self.downSampleFactor/self.downSampleExtracted))
    # else:
    #   patchSizeLoaded=blockWidth

    blockList=[]
    for r in range(nBR):
        for c in range(nBC):
            
            # CornerPosition for read_region in openSlide: expects in highest magLevel coords
            cornerPosOut=np.array([int(c*blockWidth),int(r*blockWidth)])
            cornerPos=np.uint32(cornerPosOut*downSampleFactor)
            
            # Image size for read_region: expects in coors at magLevel being read (i.e. intermediate)
            rSizeOut=int(min(blockWidth,(nR-cornerPosOut[1])))
            cSizeOut=int(min(blockWidth,(nC-cornerPosOut[0])))
            blockSizeOut=(cSizeOut,rSizeOut)
            rSizeInt=int(dSInt*rSizeOut)
            cSizeInt=int(dSInt*cSizeOut)
            blockSizeInt=(cSizeInt,rSizeInt)
            
            blockList.append((cornerPos,magLevel,blockSizeInt,isResizeNeeded,\
                              blockSizeOut,cornerPosOut))
            
    hImg=np.zeros((nR,nC,3),dtype=np.uint8)
    # bar=progressbar.ProgressBar(max_value=len(blockList))
    # for blockCounter,blockCoords in enumerate(blockList):
    #     blockImg=np.array(slide.read_region(blockCoords[0],0,blockCoords[1]))[:,:,range(3)]
    #     bar.update(blockCounter)
    # bar.finish()
    

    
    pool = Pool(processes=nWorkers,maxtasksperchild=100)    
    result=pool.map(ReadImg,blockList)
    for imgStuff in result:
        blockCoords,blockSize=imgStuff[0]
        img=imgStuff[1]
        boxSlice=np.s_[blockCoords[1]:((blockCoords[1]+blockSize[1])),
                       blockCoords[0]:((blockCoords[0]+blockSize[0])),
                       0:3]
        if normalizer is None:
            hImg[boxSlice]=img
        else:
            hImg[boxSlice]=normalizer(img)
    pool.close()
    pool.join()
    return hImg    

def SampleSlide(slide,numberOfPatches=250,patchSize=50,rgbThresh=220):                                                                                                                                                               
                                                                                                                                                                                                           
       lowResImg=np.array(slide.read_region((0,0),
                                            slide.level_count-1,
                                            slide.level_dimensions[slide.level_count-1]))[:,:,range(3)]
       
       lowResPos=np.where(np.any(lowResImg<rgbThresh,axis=-1))
       chosenIdx=np.random.choice(range(lowResPos[0].size),numberOfPatches,replace=True)             
       cX=np.int32(lowResPos[0][chosenIdx]*slide.level_downsamples[slide.level_count-1])                                                                                                                                              
       cY=np.int32(lowResPos[1][chosenIdx]*slide.level_downsamples[slide.level_count-1])
       pixelData=np.zeros((numberOfPatches*patchSize*patchSize,3))                                                         
       for n in range(numberOfPatches): #loop over selected patches and save to disk
           patchImg=np.asarray(slide.read_region((cY[n],cX[n]),0,(patchSize,patchSize)))[:,:,range(3)]
           pixelData[range(n*patchSize*patchSize,(n+1)*patchSize*patchSize),:]=np.resize(patchImg,(patchSize*patchSize,3))
       pixelData=np.uint8(pixelData[np.any(pixelData<rgbThresh,axis=-1),:])   
       isNotGreen=np.logical_and(pixelData[:,1]<1.0*pixelData[:,0],pixelData[:,1]<1.0*pixelData[:,2])
       pixelData=pixelData[isNotGreen,:]
       isNotDark=np.any(pixelData>50,axis=-1)
       pixelData=pixelData[isNotDark,:]
       return pixelData.reshape(pixelData.shape[0],1,3)    
   
# %%


def GetH(x):
    return rgb2hed(x)[:,:,0]    

def GetD(x):
    return rgb2hed(x)[:,:,2]    
def GetLabL(x):
    return rgb2lab(x)[:,:,0]    
def GetHslL(x):
    return colour.RGB_to_HSL(x/255)[:,:,2]    
def Identity(x):
    return x

class Feature(ABC):
    @abstractmethod
    def __len__(self):
        pass
    @abstractmethod
    def names(self):
        pass
    @abstractmethod
    def profile(self,mask,imgList,objSlice):
        pass

class Location(Feature):
    def __len__(self):
        return 6
    def names(self):
        return ['Y_Start','Y_End','X_Start','X_End','Centroid_Y','Centroid_X']
    def profile(self,mask,img,objSlice):
        yStart=objSlice[0].start
        yEnd=objSlice[0].stop
        xStart=objSlice[1].start
        xEnd=objSlice[1].stop
        centroid=np.mean(np.nonzero(mask),axis=1)
        return np.array([yStart,yEnd,xStart,xEnd,centroid[0]+yStart,centroid[1]+xStart])
    

class Size(Feature):
    def __len__(self):
        return 3
    def names(self):
        return ['Area','BBox_Area','Equivalent_Diameter']
    def profile(self,mask,img,objSlice):
        area= np.sum(mask)
        eqDiameter=sqrt(4 * area / PI)
        return np.array([area,mask.size,eqDiameter])
    
class Shape(Feature):
    def __len__(self):
        return 3
    def names(self):
        return ['Eccentricity','Major_Axis_Length','Minor_Axis_Length']
    def profile(self,mask,imgList,objSlice):
        l1, l2 = measure._moments.inertia_tensor_eigvals(mask)
        if l1 == 0:
            eccentricity=0
        else :
            eccentricity=sqrt(1-l2/l1)
        majAxLength=4*sqrt(l1)
        minAxLength= 4 * sqrt(l2)
        return np.array([eccentricity,majAxLength,minAxLength])
    
class Orientation(Feature):
    def __len__(self):
        return 4
    def names(self):
        return ['yProj','xProj','atan','orientation']
    def profile(self,mask,imgList,objSlice):
        M = measure._moments.moments(np.uint(mask), 3)
        ndim=2
        local_centroid=tuple(M[tuple(np.eye(ndim, dtype=int))] /
                     M[(0,) * ndim])
        mu = measure._moments.moments_central(np.uint8(mask),
                                      local_centroid, order=3)
        inertia_tensor=measure._moments.inertia_tensor(mask, mu)
        a, b, b, c = inertia_tensor.flat
        yProj=-2*b
        xProj=c-a
        warnings.filterwarnings('ignore', '.*divide by zero.*', )
        if xProj ==0 and yProj==0:
            atan=0
        else:
            tan=yProj/xProj
            atan=np.arctan(tan)

        r=sqrt((yProj*yProj)+(xProj*xProj))
        if r>0:
            yProj=yProj/r
            xProj=xProj/r


        if a - c == 0:
            if b < 0:
                orient= -PI / 4.
            else:
                orient= PI / 4.
        else:
            orient= 0.5 * atan2(-2 * b, c - a)     
        return np.array([yProj,xProj,atan,orient])    

class Convexity(Feature):
    def __len__(self):
        return 2
    def names(self):
        return ['Convex_Area','Solidity']
    def profile(self,mask,imgList,objSlice):
        convexHull=morph.convex_hull_image(mask)
        convexArea=np.sum(convexHull)
        solidity=mask.size/convexArea
        return np.array([convexArea,solidity])    

def area(tPos):
   #    return (tPos[1,0]-tPos[0,0])*(tPos[2,1]-tPos[0,1])-(tPos[1,1]-tPos[0,1])*(tPos[2,0]-tPos[0,0])
   return np.abs(tPos[0,0]*(tPos[1,1]-tPos[2,1])+tPos[1,0]*(tPos[2,1]-tPos[0,1])+\
       tPos[2,0]*(tPos[0,1]-tPos[1,1]))/2

def curvature(tPos):
    return 4*area(tPos)/np.prod(pdist(tPos))

def GetCurvatureExtent(mask,minDist=8,maxDist=10,maxRand=50,returnCoords=False):
    cImg=np.NAN*np.ones(mask.shape)
    diam=0
    
    r,c=np.where(mask)
    coords=np.transpose(np.stack([r,c]))
    areNbrs=squareform(pdist(coords))<2
    adj=nx.Graph(areNbrs)
    diam=nx.diameter(adj)
    if np.sum(mask)>=minDist:
        
        path = dict(nx.all_pairs_shortest_path(adj,cutoff=maxDist))
        distMat=np.zeros(areNbrs.shape)
        for i in range(areNbrs.shape[0]):
            for j in range(i):
                if j in path[i]:
                    distMat[i,j]=distMat[j,i]=len(path[i][j])
                else:
                    distMat[i,j]=distMat[j,i]=maxDist+1
                    
                    
        p1,p2=np.where(np.logical_and(distMat>=minDist,distMat<=maxDist))         
        
        if returnCoords:
            cDict={}  
        cList=[]
        if len(p1>0):
            nRand=min(maxRand,len(p1))
            idxList=np.random.choice(len(p1),size=nRand,replace=False)
            for i in range(nRand):
                idx=idxList[i]
                startP=p1[idx]
                endP=p2[idx]
                p=path[startP][endP]
                midP=p[int(len(p) / 2)]
                cv=curvature(coords[[startP,midP,endP],:])
                if returnCoords:
                    if midP in cDict:
                        cDict[midP].append(cv)
                    else:
                        cDict[midP]=[cv]
                cList.append(cv)
                 
            if returnCoords:  
                cImg=np.NAN*np.ones(mask.shape)  
                
                for idx in cDict:
                    cImg[r[idx],c[idx]]=np.mean(cDict[idx])
            
            # 1) select valid pairs at random
            # 2) select point near middle of pair
            # 3) calculate curvature
            outCv=np.mean(cList)
        else:
            outCv=0
    else:
        outCv=0
        
    if returnCoords:
        return outCv,diam,cImg
    else:
        return outCv,diam


class Curvature(Feature):
    def __init__(self,minDist=8,maxDist=10,maxRand=50):
        self.minDist=minDist
        self.maxDist=maxDist
        self.maxRand=maxRand  
        
    def __len__(self):
        return 4
    def names(self):
        return ['Curvature','Extent','MaxDistToEdge','Extent_EdgeDist_Ratio']
    def profile(self,mask,imgList,objSlice):
        skelMask,medDist=medial_axis(mask,return_distance=True)
        maxDistToEdge=np.max(medDist[mask])
        cv,extent=GetCurvatureExtent(skelMask,minDist=self.minDist,\
                                    maxDist=self.maxDist,maxRand=self.maxRand,\
                             returnCoords=False)
        return np.array([cv,extent,maxDistToEdge,extent/maxDistToEdge])    
    
class Skeleton(Feature):
    def __len__(self):
        return 1
    def names(self):
    #    return ['Skeleton_Length','NumberOf_Skel_Branches','Mean_Skel_Branch_Length']
        return ['Skeleton_Length']
    def profile(self,mask,imgList,objSlice):
        skeleton=skeletonize(mask)
        skelLength= np.sum(skeleton)  
        
        # if skelLength<=2:
            
        #     nBranch=1
        #     meanBranchDist=0
        # else:
            
        #     #try:
        #     # pixel_graph, coordinates, degrees = skan.skeleton_to_csgraph(skeleton)
        #     # nbgraph = skan.csr.csr_to_nbgraph(pixel_graph, None)
        #     # paths = skan.csr._build_skeleton_path_graph(nbgraph,
        #     #                     _buffer_size_offset=None)
        #     # nBranch = paths.shape[0]
        #     # meanBranchDist=skelLength/nBranch
        #     nBranch=0
        #     meanBranchDist=0
        #     #branchData = skan.summarize(skan.Skeleton(skeleton))
        #     #nBranch=branchData.shape[0]
        #     #meanBranchDist=np.mean(branchData['branch-distance'])
        #     # except:
        #     #     nBranch=-1
        #     #     meanBranchDist=-1
        #return np.array([skelLength,nBranch,meanBranchDist])

        return np.array([skelLength])

class IntensityStats(Feature):
    def __init__(self,imgNum,featPrefix,transform=Identity):
        self.imgNum=imgNum
        self.featPrefix=featPrefix
        self.transform=transform  
        
    def __len__(self):
        return 8
    def names(self):
        suffixes=['_Mean','_Median','_Std','_MAD','_Min','_Max','_Kurtosis','_Skewness']
        return [self.featPrefix+s for s in suffixes]
    def profile(self,mask,imgList,objSlice):
        imgVals=np.float32(self.transform(imgList[self.imgNum])[mask])
        
        return np.array([np.mean(imgVals),np.median(imgVals),
                         np.std(imgVals),median_absolute_deviation(imgVals),
                         np.min(imgVals),np.max(imgVals),kurtosis(imgVals),
                         skew(imgVals)])        

class Zernike_Shape(Feature):

        
    def __len__(self):
        return 25
    def names(self):
        n=[]
        for featNum in np.arange(1,26):
              n.append('Zernike_'+str(featNum))
        return n
    def profile(self,mask,imgList,objSlice):
            centroid=np.mean(np.nonzero(mask),axis=1)    
            zMoments=mahotas.features.zernike_moments(mask,np.max(mask.shape),cm=centroid)
            return zMoments

class Haralick_Texture(Feature):
    def __init__(self,imgNum,minVal,maxVal,featPrefix,transform=Identity):
        self.imgNum=imgNum
        self.featPrefix=featPrefix
        self.transform=transform  
        self.minVal=minVal
        self.maxVal=maxVal

    def __len__(self):
        return 13
    def names(self):
        n=[]
        featNames1=['2nd_moment','contrast','correlation','variance',
                            'inv_diff_moment','sum_avg','sum_variance',
                            'sum_entropy','entropy','diff_var','diff_entropy',
                            'inf_corr1','inf_corr2']    

        
        for featNum in np.arange(13):
            n.append(self.featPrefix+'_Haralick_'+featNames1[featNum])
                
                
                
        return n
    def profile(self,mask,imgList,objSlice):
        imgVals=(self.transform(imgList[self.imgNum])-self.minVal)/(self.maxVal-self.minVal)
        
        imgVals[imgVals<0]=0
        imgVals[imgVals>1]=1
        imgValsInt=np.uint8(255*imgVals)
        imgValsInt[~mask]=0
        try:
            haralickAll=np.concatenate(mahotas.features.haralick(imgValsInt,ignore_zeros=True))
            haralick=np.mean(haralickAll.reshape((4,13)),axis=0)
        except:
            haralick=np.zeros(13)
        return haralick     


class Label(Feature):
    def __init__(self,imgNum,featPrefix):
        self.imgNum=imgNum
        self.featPrefix=featPrefix
        
    def __len__(self):
        return 1
    def names(self):
        suffixes=['_label']
        return [self.featPrefix+s for s in suffixes]
    def profile(self,mask,imgList,objSlice):
        
        
        return mode(imgList[self.imgNum][mask].flatten())[0][0]

class FeatureExtractor():
    def __init__(self,featureList):
        nFeat=0
        if not all([isinstance(f,Feature) for f in featureList]):
            raise SystemError('Passed classes must be of class Feature')
        self.featureNames=['label']
        for f in featureList:
            if len(f) != len(f.names()) :
               raise SystemError('Wrong Size in '+ f.__name__())
            nFeat=nFeat+len(f)
            self.featureNames=self.featureNames+f.names()
        self.numberOfFeatures=nFeat            
        self.fList=featureList
    
    def Run(self,inputs)   :
        label,mask,img,objSlice=inputs
        outMat=np.zeros(self.numberOfFeatures+1)
        outMat[0]=label
        counter=1
        for i,f in enumerate(self.fList):
            outMat[counter:counter+len(f)]=f.profile(mask,img,objSlice)
            counter+=len(f)
        return outMat    
    
    def Names(self):
        return [f.__name__ for f in self.fList]
     


def ExtractFeatures(mask,featureList,imgList=[],minArea=None, maxArea=None): 
    #mask is the matrix which defines the objects
    # featureList is a list of objects of type Feature 
    # imgList is a list of images corresponding to mask (e.g. H&E image)
    nObj, labelMat, stats, centroids=cv2.connectedComponentsWithStats(np.uint8(mask),8,cv2.CV_32S)
    if minArea is not None or maxArea is not None:
        mapMat=np.zeros(nObj,dtype=np.int32)
        objCounter=1
        for i in range(1,nObj):
            isGood=True
            if minArea is not None and stats[i,-1]<minArea:
                isGood=False
            if maxArea is not None and stats[i,-1]>maxArea:
                isGood=False
            
            if isGood:
                mapMat[i]=objCounter
                objCounter+=1
            else:
                mapMat[i]=0
        labelMat=mapMat[labelMat]
    objects=ndi.measurements.find_objects(labelMat)
    dataList=[]
    for objNum,objSlice in enumerate(objects):
        temp=labelMat[objSlice]
        label=objNum+1
        objMask=temp.copy()==label
        iList=[]
        for i in range(len(imgList)):
            iList.append(imgList[i][objSlice])
        dataList.append((label,objMask,iList,objSlice)) # this contains all the inputs for a single object
        #proc=Process(target=Area,args=(objMask,))
        #procs.append(proc)
        #proc.start()
    pool = Pool(processes=64,maxtasksperchild=100)    
    
    featCalc=FeatureExtractor(featureList)  
    result=pool.map(featCalc.Run,dataList)
    pool.close()
    pool.join()
        
    featureMat=np.zeros((len(result),result[0].size))
    for f in result:
        featureMat[int(f[0]-1),:]=f
    featureNames=featCalc.featureNames        
    return featureMat,featureNames

