#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:24:55 2019

@author: hugomeyer
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.scimath import logn
from math import e

from cut import Cut




#Class handling the computation of the Highlight end extraction
class End_cut(Cut):
    def __init__(self, features, time, fps, sound_dict, hl_max_size=-1, hl_min_size=-1):
        Cut.__init__(self, features, time, fps, sound_dict, hl_max_size, hl_min_size)
        
        self.best_intervals = []
        self.simil_labels = ['plateau_end', 'valley_end', 'hill_top', 'plateau_start']
        self.simil_ranges = []
        self.intervals_on_similarity_zone = []
    
    
    #Find all the interval
    def find_n_best_end_cut_intervals(self):
        self.emotions_scoring()
        self.sound_scoring()
        self.similarity_scoring()
        #self.plot_score_fct()
        self.find_best_intervals()
        
        
    
    
    #Find all the intervals of the end cut score function as long as the values are above the overall median
    def find_best_intervals(self):
        function = self.score_fct.copy()
        #self.plot_score_fct()
        interval_middle = np.argmax(function)
        median = np.median(function)
        
        #Find the first best interval
        function[:self.hl_min_size] = median-1
        left, right = self.find_plateau(function)
        interval_middle = int((left + right)/2)
        
        #Find all the best intervals (values above the median)
        while function[int(interval_middle)] >= median:
            #Prevent to get interval with score == 0
            interval_on_similarity_zone = False

            for (start, end) in self.simil_ranges:
                if interval_middle >= start and interval_middle <= end:
                    interval_on_similarity_zone = True

            self.intervals_on_similarity_zone.append(interval_on_similarity_zone)

            interval = [{'time': self.time[el], 'index': el, 'score': function[el]} for el in range(left, right+1)]
            self.best_intervals.append(interval)

            function[left:right+1] = median-1
            
            #Find the i-th best interval
            left, right = self.find_plateau(function)
            interval_middle = int((left + right)/2)
     
            
            
    #Compute the score function to score frame similarity events
    def similarity_scoring(self):
        factor = self.params['similarity']['downscale_factor_end']

        #Only consider frame similarity events relevant for the end cut of the highlight (on the contrary to start cut)
        relevant_simil = [feat for feat in self.features['similarity'] 
                          if feat['label'] in self.simil_labels and feat['time'] > self.hl_min_size]
        
        #In the case where we have 2 events on the same timestamps -> merge them into one
        relevant_simil = self.merge_double_points(relevant_simil)
        
        for feat in relevant_simil:
            #Adapt the score of the event given its class: 'valley', 'plateau' or 'hill'
            score = feat['score'] if feat['time'] > 1*self.fps else feat['score']/2
            ratio1 = self.params['similarity']['upscale_valley']
            ratio2 = self.params['similarity']['upscale_plateau']
            if not self.features['sound'] and len([simil['label'] for simil in relevant_simil if simil['label']=='hill_top'])==len(relevant_simil):
                ratio3 = self.params['similarity']['upscale_hill']
            else:
                ratio3 = self.params['similarity']['downscale_hill']
            valley_end_or_plateau_start = feat['label'] == 'valley_end' or feat['label'] == 'plateau_start'
            score = score**(1/ratio1) if valley_end_or_plateau_start else score**(1/ratio2) if feat['label'] == 'plateau_end' else score*ratio3

            #Compute the width of the event in the score function
            half_t_range = max(int(round(self.params['similarity']['max_time_range']*self.fps*feat['score']/2)), 1)
            start = feat['time']-half_t_range
            end = feat['time']+half_t_range+1

            #Add the event to the function
            self.simil_score_fct[start:end] += score*factor#gaussian_vals*feat['score']*0.5/gaussian_vals.max()
            self.simil_ranges.append([start, end])
        self.score_fct += self.simil_score_fct
    
    
    #Fuzzy logic to reajust the score of an event given its class
    def reajust_scores_given_labels(self, label, score):
        ratio = self.params['similarity']['labels_rescale']
        if label == 'valley_end' or label == 'hill_end':
            return (score+ratio)/(1+ratio)
        elif label == 'plateau_end' or label == 'plateau_start':
            return (score+ratio/2)*(2-ratio)/(2+ratio)
        else:
            return score*(1-ratio)
    
    
    #In the case where we have 2 events on the same timestamps -> merge them into one
    def merge_double_points(self, simil_pts):
        to_rm = list()
        for i in range(len(simil_pts)-1):
            if simil_pts[i]['time'] == simil_pts[i+1]['time']:
                indices = [i, i+1]
                best_index = np.argmax([simil_pts[i]['score'], simil_pts[i+1]['score']])
                simil_pts[indices[best_index]]['score'] = max(simil_pts[i]['score'], simil_pts[i+1]['score'])
                to_rm.append(indices[1-best_index])
                
        return np.delete(simil_pts, to_rm)
    
        
    #Compute the score function in order to score audio events
    def sound_scoring(self):
        events_labels = [self.sound_dict[feat['label']] for feat in self.features['sound']]
        rescale_laugh_and_speech = 'laugh' in events_labels and 'speech' in events_labels
        #downscale_misc = True #'laugh' in events_labels or 'speech' in events_labels
        upscale_similarity = not 'laugh' in events_labels and not 'speech' in events_labels
        
        if upscale_similarity:
            self.params['similarity']['downscale_factor_end'] = 1
            
        
        for feat in self.features['sound']:
            sound_label = self.sound_dict[feat['label']]
            if feat['end'] > self.hl_min_size:
                #sound_scale_up = (sound_label == 'laugh')^(sound_label == 'speech')
                if sound_label == 'laugh':
                    score = feat['score']#*(feat['end']-feat['start'])/(3*self.fps)
                    #Upscale laugh score as long as speech is present in the clip
                    score = score**(1/2) if rescale_laugh_and_speech else score
                    score *= 1.5
                    event_vals = np.linspace(-score, score, feat['end']-feat['start'])
                    self.sound_score_fct[feat['start']:feat['end']] += event_vals

                    pattern_end = self.score_pattern('constant', 'laugh', feat['end'], score)
                    self.score_pattern('linear', 'laugh', pattern_end, score)
                    
                elif sound_label == 'speech':
                    score = feat['score']#*(feat['end']-feat['start'])/(3*self.fps)
                    #Downscale speech score as long as laugh is present in the clip
                    score = logn(e, 1+score) if rescale_laugh_and_speech else score
                    score *= 1.5
                    #Penalize more for cutting during speech and scale up the low values with square function
                    neg_score = score*self.params['sound']['during_speech_penalty']

                    self.sound_score_fct[feat['start']:feat['end']] -= neg_score
                    self.test[feat['start']:feat['end']] -= neg_score
                    pattern_end = self.score_pattern('constant', self.sound_dict[feat['label']], feat['end'], score)
                    self.score_pattern('linear', self.sound_dict[feat['label']], pattern_end, score)
                    
                else:
                    #Miscellaneous class -> adjust score 
                    score = feat['score']*(feat['end']-feat['start'])/(3*self.fps)
                    score = min(score, self.params['sound']['max_misc_score_end_cut'])
                    self.sound_score_fct[feat['start']:feat['end']] -= score/2
                    duration = (feat['end']-feat['start'])/self.fps
                    pattern_end = self.score_pattern('constant', self.sound_dict[feat['label']], feat['end'], score, duration)
                    self.score_pattern('linear', self.sound_dict[feat['label']], pattern_end, score, duration)
          
        self.score_fct += self.sound_score_fct
        
        
        
        
    #Compute the score pattern -> same for 'speech' and 'miscellaneous' classes
    #Can be other a constant function (shape == 'constant') or a linear function (shape == 'linear')
    def score_pattern(self, shape, event, last_pattern_end, score, duration=None):
        #Constant values differents depending on the event class ('speech or 'miscellaneous)
        if event == 'misc':
            ratio = self.params['sound']['misc_cue_ratio'] 
            param = duration*ratio if shape == 'constant' else duration*(1-ratio)
        else:
            param = self.params['sound']['shape'][shape]['speech']
            
        width = int(round(min(param*self.fps, self.L - last_pattern_end)))
        #Order of the function, depending on the portion of the event pattern we are computing
        if shape == 'constant':
            vals = score
        else:
            end_val = score*(1-width/(param*self.fps))
            vals = np.linspace(score, end_val, width)
        end = last_pattern_end+width 
        self.sound_score_fct[last_pattern_end:end] += vals
        
        return end
        
    
    
    
        
   
    def emotions_scoring(self):
        #Score the emotion peaks with their respective score as squares 
        #with a cue at the end proportional to the peak width        
        for emotion in ['happiness', 'surprise']:
            #Init by starting at the end of the video
            last_emo_peak_start = self.L
            for feat in self.features[emotion][::-1]:
                #Don't consider peak at the very beginning of the video (happening before the minimal size of the HL)
                if feat['fall_end'] > self.hl_min_size:
                    #Build the score pattern for an emotion event and add it to the score function
                    score = feat['score']*self.params['emotion']['upscale_factor']
                    rise_vals = np.linspace(0, score, feat['start']-feat['rise_start']+1)[:-1]
                    ratio_cue = max(1-(feat['end'] - feat['start'])/(self.params['emotion']['max_peak_duration']*self.fps), 0)
                    const_width = int(round(ratio_cue*self.params['emotion']['max_cue_width']*self.fps))
                    const_end = min(const_width+feat['fall_end'], last_emo_peak_start)
                    self.emo_score_fct[feat['rise_start']:feat['start']] = rise_vals
                    self.emo_score_fct[feat['start']:const_end] += score
                    last_emo_peak_start = feat['rise_start']
        self.score_fct += self.emo_score_fct
    
    
    
    
    