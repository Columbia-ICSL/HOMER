#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:38:53 2019

@author: hugomeyer
"""

import cv2
import numpy as np
from imutils import face_utils
import imutils
import dlib

FD_PARAMS= {
                'minDetectRatio': 0.3,
                'scaleFactor': 1.2,
                'minNeighbors': 3,
                'model': 'dlib' #['haarcascade', 'dlib', 'both']

           }


#Detect the user's face on an image and crop it if detected
def face_detection(img, mode, model='haarcascade'):
    #Downscale image and convert to grayscale
    m = img.copy()
    m = imutils.resize(m, width=500)
    gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) 
    minFaceHeight = int(FD_PARAMS['minDetectRatio']*m.shape[0])

    #Different models for detecting the face: 'dlib' or 'haarcascade'
    #'both' mode uses both by first using 'haarcascade' and if not detected, using 'dlib' (more robust but longer computation)
    if model == 'haarcascade' or model == 'both':
        face_cascade = cv2.CascadeClassifier('../Models/Face/haarcascade_frontalface_default.xml') 
        faces = face_cascade.detectMultiScale(gray, FD_PARAMS['scaleFactor'], FD_PARAMS['minNeighbors'], 0, (minFaceHeight, minFaceHeight)) 
    elif model == 'dlib':
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray, 1)   
        faces = np.asarray([face_utils.rect_to_bb(face) for face in faces]) 
    else: 
        raise ValueError("The chosen model is not valid")
    
    #Assumption that the user's face will be the biggest on the image
    face = find_biggest_face(faces, minFaceHeight)
    

    if face is not None:
        (x, y, w, h) = face[0]
        #Crop the face on the image
        if mode == 'crop':
            if y < 0:
                y = 0
            if y > gray.shape[0]-1:
                y = gray.shape[0]-1
            if x < 0:
                x = 0
            if x > gray.shape[1]-1:
                x = gray.shape[1]-1
            return m[y:y+h, x:x+w], True
        #Only draw a rectangle around the face
        elif mode == 'rect':
            cv2.rectangle(m,(x,y),(x+w,y+h),(0,0,255),2)
            return m, True
        else:
            raise ValueError("The chosen mode is invalid")
    return m, False
            
            

def find_biggest_face(faces, minFaceHeight):
    if type(faces) is np.ndarray and faces.size != 0:
        if faces.shape[0]>1:
            biggest_face = faces[np.argmax(faces[:, 3]*faces[:, 2])]
            biggest_face = np.expand_dims(biggest_face, axis=0)
            return biggest_face
        if faces[0, 3] < minFaceHeight:
            return None
        else:
            return faces
    else:
        return None
    
    
#Bin the value of the pixels in the HSV domain for frame similarity computation
def from_img_to_bins(img, compression_ratio, bin_size):
    w, h = int(img.shape[1]*compression_ratio), int(img.shape[0]*compression_ratio)
    img = cv2.resize(img,(w,h))

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_red = 16*img_hsv[:, :, 1] + img_hsv[:, :, 0] + 1

    img_binned = (img_red/bin_size).astype(int)
    img_binned[img_binned==int(256/bin_size)]=int(256/bin_size)-1
    return img_binned

#Compute the triagonal matrix resulting from two consecutive binned frames
def from_binned_img_to_tridiag(mat1,mat2, bin_size):
    (h, w)=mat1.shape[:2]
    
    #Init 3D matrices to count the number of pixels that remained in the same bin 
    #in both consecutive frames
    D_i = np.zeros((int(256/bin_size), h, w)).astype('int16')
    D_j = np.zeros((int(256/bin_size), h, w)).astype('int16')

    for k in range(h):
        for l in range(w):
            D_i[mat1[k, l], k, l]+=1
            D_j[mat2[k, l], k, l]+=1

    #Computing only the three main diagonals of the matrix out of the previous counts
    main_diag=np.sum(D_i&D_j, axis=(1, 2)).tolist()
    upper_diag=np.sum(D_i[:-1]&D_j[1:], axis=(1, 2)).tolist()
    lower_diag=np.sum(D_i[1:]&D_j[:-1], axis=(1, 2)).tolist()

    
    return main_diag, upper_diag, lower_diag


def diags_sum(diags):
    sum1 = sum([sum(diag) for diag in diags[0]])
    sum2 = sum([sum(diag) for diag in diags[1]])
    return (sum1 + sum2)/2
    
#Compute the sum of minimal values and maximal values between to pair of consecutive frames (3 consectutive frames)
#To later compute min/max ratio
def min_max_tridiags_pair(tridiags1, tridiags2):
    min_ = max_ = 0
    for diag1, diag2 in zip(tridiags1, tridiags2):
        min_ += np.minimum(diag1, diag2).sum()
        max_ += np.maximum(diag1, diag2).sum()
        
    return min_, max_







    
 

