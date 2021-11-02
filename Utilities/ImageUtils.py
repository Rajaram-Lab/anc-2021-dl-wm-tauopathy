'''
ImageUtils contains functions to extract information from image data (i.e., AT8 images)

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

import numpy as np
from sklearn.decomposition import NMF
import xml.etree.ElementTree as ET

def HnEDeconv(hneImg):

    odThresh = 0.1

    odImg = -np.log10(np.double(hneImg)/256)
    odImg[np.isinf(odImg)] = -np.log10(1.0/256)
    odMean = np.mean(odImg, axis=2)

    isFG = odMean>odThresh
    fgOdVals = np.zeros((np.count_nonzero(isFG), 3))
    for channel in range(3):
        channelImg=odImg[:,:,channel]
        fgOdVals[:,channel]=channelImg[isFG]

    model=NMF(n_components=2)
    w=model.fit_transform(fgOdVals)
    basisVecs=model.components_

    mostBlue=np.argmin(basisVecs[:,2])
    mostRed = np.argmin(basisVecs[:, 0])
    eComp=mostRed
    hComp=mostBlue
    # need to add check of mostBlue and mostRed are the same
    if(mostRed==mostBlue):
        if hComp==0:
            eComp=1
        else:
            eComp=0
    s=odImg.shape
    odVals=odImg.reshape((s[0]*s[1],3))
    scores=np.linalg.lstsq(odVals.transpose(),basisVecs.transpose())[0]
    scores[scores<0]=0
    scoresImg=scores.reshape(s[0],s[1],2)

    hImg=scoresImg[:,:,hComp]
    eImg=scoresImg[:,:,eComp]

    return hImg,eImg,model

def GetXmlAnnoNames(xmlFile):
    tree = ET.parse(xmlFile)
    # get root element
    root = tree.getroot()
    annoNames=[]
    subAnnoNames=[]  
    
    for annoNum,anno in enumerate(root.iter('Annotation')):
        
        annoNames.append(anno.get('Name'))
        regionList = anno.find('Regions')
        regionNames=[]
                 
 

        for item in regionList:
            regionName=item.get('Text')
            if(regionName != None):
                regionNames.append(item.get('Text'))   
            
        subAnnoNames.append(regionNames)        

    return annoNames,subAnnoNames

def ReadRegionsFromXML(xmlFile,layerNum=0):

    # create element tree object
    tree = ET.parse(xmlFile)

    # get root element
    root = tree.getroot()

    #regionList = root.iter('Region')
    #numberOfRegions = sum(1 for i in regionList)
    regionPos = []
    regionNames = []
    isNegative=[]
    regionInfo=[]
    
    
    for annoNum,anno in enumerate(root.iter('Annotation')):

        if(annoNum==layerNum):
            regionList = anno.find('Regions')
        
                 
            for item in regionList:
                # print item.tag,item.attrib
                regionName=item.get('Text')
                if(regionName != None):
                    regionNames.append(item.get('Text'))
                    isNegative.append(item.get('NegativeROA')=='1')
                    idNum=item.get('Id')
                    inputRegionId=item.get('InputRegionId')
                    vertexList = item.find('Vertices')
                    
                    numberOfVertices = sum(1 for i in vertexList)
                    pos = np.zeros((numberOfVertices, 2))
                    counter = 0
                    for v in vertexList:
                        pos[counter, 0] = int(float(v.get('X')))
                        pos[counter, 1] = int(float(v.get('Y')))
                        counter = counter + 1
                    regionPos.append(pos)
                    info={'Length':float(item.get('Length')),'Area':float(item.get('Area')), 
                          'BoundingBox':np.array([np.min(pos[:,0]),np.min(pos[:,1]),np.max(pos[:,0]),np.max(pos[:,1])]),
                          'id':idNum,'inputRegionId':inputRegionId,'Type':float(item.get('Type'))}
                    regionInfo.append(info)
    regionPos = np.array(regionPos)
    isNegative=np.array(isNegative)
    return regionPos,regionNames,regionInfo,isNegative

def GetLoops(pos):
  rowCounter=0
  continueSearch=True
  loops=[]
  while continueSearch:
    idx=np.where(np.logical_and(pos[(rowCounter+1):,0]==pos[rowCounter,0], 
                                    pos[(rowCounter+1):,1]==pos[rowCounter,1]))[0]
    if(len(idx)>0):
      newPos=idx[0]+rowCounter+1
      
      loops.append(pos[rowCounter:min(newPos+1,len(pos)),:])
      rowCounter=newPos+1
      if(rowCounter>=len(pos)):
        continueSearch=False
    else:
      continueSearch=False  
      loops.append(pos[rowCounter:,:])
  return loops      

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def GetQPathTextAnno(annoFile, checkForDisjointAnnos=True):

  with open(annoFile) as f:
      content = f.readlines()
  
  regionPos = []
  regionNames = []
  isNegative=[]
  regionInfo=[]
     
      

  for p in range(len(content)):
    line=content[p]  
    className,data=line.replace(']','').split('[')
    if(isinstance(className,str) and len(className)>0 and (className[0]=='+' or className[0]=='-')):
      className=className  
    else:
      className='+'+className
    regionNames.append(className)
    
    data=np.array([float(x) for x in  data.replace('Point: ','').split(',')])
    coords=np.zeros((int(len(data)/2),2))
    coords[:,0]=data[0::2]
    coords[:,1]=data[1::2]
    #coords=np.concatenate((coords,np.expand_dims(coords[0,:],axis=0)))   
    if(checkForDisjointAnnos):
      loops=GetLoops(coords)
      loopAreas=[PolyArea(l[:,0],l[:,1]) for l in loops]
      coords=loops[np.argmax(loopAreas)]
    
    regionPos.append(coords)
  
    bbox=np.array([np.min(coords[:,0]),np.min(coords[:,1]),np.max(coords[:,0]),np.max(coords[:,1])])
    l=((bbox[2]-bbox[0])+1)
    w=((bbox[3]-bbox[1])+1)
    area=l*w
    info={'Length':max(l,w),'Area':area,'BoundingBox':bbox,'id':p,'inputRegionId':0,'Type':0}
    regionInfo.append(info)
    
  regionPos=np.array(regionPos)
  isNegative=np.zeros(len(content))==1
  return regionPos,regionNames,regionInfo,isNegative


def DiskFilter(filterWidth,discRadius):

    x, y = np.meshgrid(np.array(range(filterWidth)), np.array(range(filterWidth)))
    x = x.astype(float) - (filterWidth - 1.0) / 2
    y = y.astype(float) - (filterWidth - 1.0) / 2
    dist = np.sqrt(x * x + y * y)
    discFilter = (dist <= discRadius).astype(float)
    return discFilter

def prettify_xml(root):
  from xml.dom import minidom
  
  xml_string = ET.tostring(root, encoding='utf-8')
  reparsed = minidom.parseString(xml_string).documentElement
  return reparsed.toprettyxml(indent="  ")

def pos_to_xml(xmlfilename, data, mpp):
  from xml.etree import cElementTree as ET
  
  # data[layer_name] = [pos, region]
  root = ET.Element('root')   # root that will contain the xml structure Aperio Imagescope needs

  # Layers = Annotations in Aperio XML
  Layers_obj = ET.SubElement(root, "Annotations", MicronsPerPixel=str(mpp))

  for layer_ix, layer_name in enumerate(data):
      curr_layer_data = data[layer_name]
      pos_list, name_list = curr_layer_data[0], curr_layer_data[1]

      Layer_obj = ET.SubElement(Layers_obj, "Annotation", Name=layer_name, Id=str(layer_ix), ReadOnly="0", NameReadOnly="0", \
              Incremental="0", Type="4", LineColorReadOnly="0", Visible="1", Selected="1", MarkupImagePath="", MacroName="")
      
      # Boundaries = Regions  in Aperio XML
      Boundaries_obj = ET.SubElement(Layer_obj, "Regions")

      for pos_ix in range(len(pos_list)):
          # Boundary = Region  in Aperio XML
          Boundary_object = ET.SubElement(Boundaries_obj, "Region", Id=str(pos_ix), Type="0", Selected="0", ImageLocation="", ImageFocus="-1", \
                  Text=name_list[pos_ix], NegativeROA="0", InputRegionId="0", Analyze="1", DisplayId=str(pos_ix), \
                  Length="", Area="", LengthMicrons = "", AreaMicrons="", Zoom="")

          # Points = Vertices in Aperio XML
          Points_obj = ET.SubElement(Boundary_object, "Vertices")

          for point_ix in range(pos_list[pos_ix].shape[0]):
              # Point = Vertex in Aperio XML
              Point_obj = ET.SubElement(Points_obj, "Vertex", X=str(pos_list[pos_ix][point_ix][0]), Y=str(pos_list[pos_ix][point_ix][1]), Z="0")

  xml = prettify_xml(Layers_obj) # pretty xml

  print("Writing XML to", xmlfilename)

  with open(xmlfilename, 'w') as fd:
      fd.write(xml)