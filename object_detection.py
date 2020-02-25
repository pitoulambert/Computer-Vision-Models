#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt 
import numpy as np 
import cv2 
import glob
import os
from PIL import Image
import json
from json import loads
import xmltodict
from yattag import indent


# In[2]:


def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles    


# In[3]:


def object_detection(image_name,f_name,bx,by,tx,ty):
    
    # plate_path = '/home/secquraise/Desktop/vehicle/Indian_Number_Plate'
    # dirName = '/home/secquraise/Desktop/vehicle'
    cnt = np.random.randint(1, 10000)
    
    try:
        # creating a folder named data 
        if not os.path.exists(f_name):
            os.makedirs(f_name) 
    # if not created then raise error 
    except OSError:
        print ('Error: Creating directory of data')

    print("Object Detection:image_name", image_name)
    img = cv2.imread(image_name)
    h = int(img.shape[0])
    w = int(img.shape[1])
    img = cv2.resize(img, dsize=(w,h))
    tx = int(tx)
    ty = int(ty)
    bx = int(bx)
    by = int(by)
    plate = img[by:ty, bx:tx]
    img = cv2.rectangle(img, (bx,by),(tx,ty),(0,255,255),3)
    name = f_name + str(cnt) + '.jpg'
    print('Creating....' + name)
    cv2.imwrite(os.path.join(f_name, name), plate)

def xml_to_json():
    
    dirName = 'name of directory/input'
    
    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)
    #os.mkdir("Indian Number Plates")
    # # Print the files
    # for elem in listOfFiles:
    #     print(elem)
 
    print ("****************")
    
    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames if file.endswith('.' +'xml')]        
        
    # Print the files    
    for k in listOfFiles:
#         print(k)
        
        with open(k, 'r') as f:
            xmlString = f.read()            
            # print(xmlString)
            
            jsonString = json.dumps(xmltodict.parse(xmlString), indent=4)
            jsonString = json.loads(jsonString)
    
            
            if ('object' not in jsonString['annotation']):
                print('No Object Detected')
                pass
                # os.remove(k)
                # os.remove(xmlString)
                # os.remove(jsonString)
            else:
                #print(jsonString)
                path_name = jsonString['annotation']['path']
                folder_name = jsonString['annotation']['folder']
                objects = jsonString['annotation']['object']
                width = jsonString['annotation']['size']['width']
                height = jsonString['annotation']['size']['height']
            
                if (type(objects) == list):
                    for i in range(len(objects)):
                        objects[i] = dict(objects[i])
                        xmin = objects[i]['bndbox']['xmin']
                        ymin = objects[i]['bndbox']['ymin']
                        xmax = objects[i]['bndbox']['xmax']
                        ymax = objects[i]['bndbox']['ymax']
                        object_detection(path_name,folder_name,xmin,ymin,xmax,ymax)
#                         print(xmin)
#                         print(ymin)
#                         print(xmax)
#                         print(ymax)



                elif (type(objects) == dict):
                    xmin = objects['bndbox']['xmin']
                    ymin = objects['bndbox']['ymin']
                    xmax = objects['bndbox']['xmax']
                    ymax = objects['bndbox']['ymax']
                    object_detection(path_name,folder_name,xmin,ymin,xmax,ymax)
#                     print(xmin)

                column = ['img_name', 'xmin', 'ymin', 'xmax', 'ymax' ]

                jFileName = k.replace(".xml",".json")
                with open(jFileName, 'w') as f:
                    json.dump(jsonString,f, indent=4)
                
            
                
                
xml_to_json()