#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 22:20:24 2019

@author: hugomeyer
"""


import pandas as pd
import numpy as np
from peak_processing import peak_detection
import matplotlib.pyplot as plt    
import os



#Class handling the pre-processing and feature extraction of emotions
class Emotion(object):
    def __init__(self, preds, emo_labels, fps, interlude=False):
        self.labels_3 = ['p3_neg', 'p3_neu', 'p3_pos']
        self.labels_7 = ['p7_ang', 'p7_disg', 'p7_fear', 'p7_hap', 'p7_sad', 'p7_surp', 'p7_neu']
        self.tresholds = {'p7_ang': None, 
                          'p7_disg': None, 
                          'p7_fear': None, 
                          'p7_hap': 0.8,
                          'p7_sad': None, 
                          'p7_surp': 0.3, 
                          'p7_neu': None}
        self.best_cut = {'p7_ang': None, 
                          'p7_disg': None, 
                          'p7_fear': None, 
                          'p7_hap': None,
                          'p7_sad': None, 
                          'p7_surp': None, 
                          'p7_neu': None}
        self.emo_dict = {'p7_ang': 'Anger', 
                        'p7_disg': 'Disgust', 
                        'p7_fear': 'Fear', 
                        'p7_hap': 'Happiness', 
                        'p7_sad': 'Sad', 
                        'p7_surp': 'Surprise', 
                        'p7_neu': 'Neutral',
                        'p3_neg': 'Negative', 
                        'p3_neu': 'Neutral', 
                        'p3_pos': 'Positive'}
        self.max_peak_time={'p7_ang': None,   #In seconds
                          'p7_disg': None, 
                          'p7_fear': None, 
                          'p7_hap': 4,
                          'p7_sad': None, 
                          'p7_surp': 1, 
                          'p7_neu': None}
        self.preds = self.init_preds(preds)
        self.emo_labels = emo_labels
        self.best_emo_label = None
        self.no_face = self.no_face_interpolation(0.5)
        self.fps=fps
        self.peaks = []
        self.preds = preds
        self.T = self.preds.shape[0]
        
        
        
    #Initialize the dataframe containing emotion predictions
    def init_preds(self, preds):
        labels=  self.labels_7 + ['7_best']
        preds.columns=labels
        for label in preds.columns:
            if label[0] == 'p':
                preds[label] = preds[label].astype(float)
              
        preds.index = range(1, preds.shape[0]+1)
        
        return preds
        
    
    
    #In case the case where a face appeared on the front video, the feature extraction was done 
    #by detecting emotion peaks in the probability signal.
    def extract_features(self):

        if not self.no_face:
            self.signals = dict()
            for emo_label in self.emo_labels:
                max_peak_size=self.max_peak_time[emo_label]*self.fps
                peaks = peak_detection(self.preds, self.T, emo_label, self.emo_dict, 
                                                  max_peak_size, self.tresholds[emo_label])
               
                self.peaks = self.peaks + peaks
        


                
    #Interpolate the emoiton probabily signal when some value misses because of frames without face on it
    #Do it as long as the ratio of no face frames is below a threshold
    def no_face_interpolation(self, discard_clip_treshold):
        #Count number of value missing
        no_face_ratio = self.preds['7_best'][self.preds['7_best']=='No face'].count()/self.preds.shape[0]
        df = self.preds.copy()

        if no_face_ratio < discard_clip_treshold:
            if no_face_ratio == 0:
                return False
            
            indices_not_missing=df.index[df['7_best']!='No face'].copy().values
            indices_to_interp=df.index[df['7_best']=='No face'].copy().values
            values_not_missing=df[df['7_best']!='No face'].copy()

            #Linearly interpolate the missing values of the signal
            for col in df.columns:
                if col != '7_best':
                    self.preds.loc[indices_to_interp, col]=np.interp(indices_to_interp, indices_not_missing, values_not_missing[col].values)
            preds = self.preds.iloc[:, :-1].copy()
            max_preds = preds.loc[indices_to_interp].idxmax(axis=1)
            self.preds.loc[indices_to_interp, '7_best'] = [self.emo_dict[max_pred]  for max_pred in max_preds]
            ratio = indices_to_interp.shape[0]/self.preds.shape[0]
            print("{:.3f}% of emotion predictions were interpolated" .format(ratio))
            
            return False
        else:
            return True
        
        
