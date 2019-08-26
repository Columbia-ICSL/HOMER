#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:21:00 2019

@author: hugomeyer
"""

import numpy as np

#This class is the parent class of start/end cuts. It contains the common variables of both start/end cuts
class Cut(object):
    def __init__(self, features, time, fps, sound_dict, hl_max_size, hl_min_size):
        self.features = features
        self.time = time
        self.fps = fps
        self.sound_dict = sound_dict
        
        self.score_fct = np.zeros(self.time.shape[0])
        self.time_score_fct = np.zeros(self.time.shape[0])
        self.sound_score_fct = np.zeros(self.time.shape[0])
        self.emo_score_fct = np.zeros(self.time.shape[0])
        self.simil_score_fct = np.zeros(self.time.shape[0])
        self.test = np.zeros(self.time.shape[0])
        
        self.hl_max_size = hl_max_size
        self.hl_min_size = hl_min_size
        
        self.automatic = self.hl_max_size == -1 or self.hl_min_size == -1
        
        self.L = self.time.shape[0]
        
        #All hyper-parameters for each signal that can be tuned to personalize the algorithm
        self.params = {
                            'general': {'auto_min_hl_time': 3},
                            'sound': {
                                        'shape':  {
                                                    'constant': {'laugh': 0.5, 'speech': 1, 'misc': 0.5},
                                                    'linear': {'laugh': 1, 'speech': 2, 'misc': 1}
                                                },
                                        'during_speech_penalty': 1.5, 
                                        'laugh_front_cue_max': 1,
                                        'misc_scale_down': 0.7,
                                        'misc_cue_ratio': 0.33,
                                        'max_misc_score_end_cut': 0.5,
                                        'during_speech_laugh_start_cut_penalty': 2
                                      },
            
                            'emotion': {
                                        'surp_hap_diff': 1, 
                                        'max_cue_width': 4,
                                        'max_peak_duration': 4,
                                        'neg_score_slope': 0.3,
                                        'upscale_factor': 2,
                                        'front_neg_cue': 2
                                       },
            
                            'similarity': {
                                            'max_time_range': 1, 
                                            'downscale_factor_end': 1,
                                            'factor_start': 0.7,
                                            'upscale_start': 1.2,
                                            'labels_rescale': 0.5,
                                            'upscale_valley': 3,
                                            'upscale_plateau': 1.5,
                                            'upscale_valley_pair': 4,
                                            'upscale_plateau_pair': 2,
                                            'downscale_hill': 0.5,
                                            'upscale_hill': 1.5
                                          },
            
                            'time': {
                                     'order': 'quadratic', #['quadratic', 'linear']
                                     'penalty': 0.2,
                                     'end_penalty_score': -1,
                                    } 
                       }
        
        if self.hl_min_size == -1:
            self.hl_min_size = self.params['general']['auto_min_hl_time']*self.fps
        if self.hl_max_size == -1:
            self.hl_max_size = self.L
        
    #Find the the constant interval (plateau) of a function with the highest value between 2 boundaries 
    def find_plateau(self, function, start=None, end=None):
        start = 0 if start is None else start
        end = function.shape[0]-1 if end is None else end
        spot = start + np.argmax(function[start:end+1])
      
        left, right = spot, spot
        while left > 0 and function[left-1] == function[spot]:
            left -= 1
        while right < len(function)-1 and function[right+1] == function[spot]:
            right += 1
            
        return left, right
        
        