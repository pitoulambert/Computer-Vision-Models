#!/usr/bin/env python
# coding: utf-8

# In[1]:

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
    
    cnt = np.random.randint(1, 10000)
    
    try:
        # creating a folder named data 
        if not os.path.exists(f_name):
            os.makedirs(f_name) 
    # if not created then raise error 
    except OSError:
        print ('Error: Creating directory of data')
        
        '''
            . Find the height an width of each images
            . Resize with those two values
            . Fetch out the bottom and top values and save into a rectangle size in RGB format
            . Save the images in a directory
        '''
    
def xml_to_json():
    
    dirName = 'name of directory/input'
    
    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)
 
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
            '''
                Convert the xm into json
            '''
                
                '''
                   . Check if the name 'objects' exist 
                            - if no, delete those xml files
                            - if yes, check again if it is a list or a dictionary
                   . Find the coordinates of the bounding boxes [ xmin, ymin, xmax, ymax ]
                   
                '''
                object_detection(path_name,folder_name,xmin,ymin,xmax,ymax)

                '''
                    You can create a csv document with the following details
                    column = ['img_name', 'xmin', 'ymin', 'xmax', 'ymax' ]
                '''
                
                jFileName = k.replace(".xml",".json")
                with open(jFileName, 'w') as f:
                    json.dump(jsonString,f, indent=4)
                
                
xml_to_json()
