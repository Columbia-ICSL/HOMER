#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 19:11:16 2019

@author: hugomeyer
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import load_workbook
import openpyxl
from string import ascii_uppercase
from start_cut import Start_cut
from end_cut import End_cut
from copy import deepcopy
from moviepy.video.io.VideoFileClip import VideoFileClip
import os


#Class handling the computation of the event timeline -> synchronisation of the feature signals for each of
#the three signals: sound, FS and emotion
class Events_timeline(object):
    def __init__(self, signals, features, duration, fps=10, ground_truth=None):
        self.signals = signals
        self.features = features
        
        self.duration = duration
        self.fps = fps
        self.time = np.linspace(0, self.duration, round(self.duration*self.fps))
        self.ground_truth = ground_truth
        self.sound_dict = {0: 'laugh', 1: 'speech', 2: 'misc'}
        
        self.time_features = deepcopy(self.features)
        self.best_hls = None
        self.many_bests_hls = None

        self.build_event_timeline()
        
        
        
    #Readjust shifted values and distortions because of approximation issues when adapting signals to the same fps
    def build_event_timeline(self):

        self.signals = dict((key, self.resize_to_fixed_fps(signal, key)) for key, signal in self.signals.items()) 
   
        #Compute signals from the event features
        self.reconstruct_emotion_simplified_signal('surprise')
        self.reconstruct_emotion_simplified_signal('happiness')
        
        #Relocate shifted values (events start/end)
        self.reajust_similarity_features()
        self.reajust_sound_features()
        
        #Go from timestamps to indices
        self.from_time_to_index()
        #Simplify the emotion signals 
        self.emotion_preprocessing()
        
        
    #Function that finds the highlight, and, in the case of many HLs allowed, find the n most optimal ones
    # with n < 3 and some stop conditions
    def compute_highlight(self, hl_min_size=-1, hl_max_size=-1, many_hls=False, hl_overlap_ratio=0.25, 
                         hl_duration_ratio=0.33, max_hl_nb=3, rm_tresh_score=0.15):
        
        #Convert min and max hl durations to indice lengths
        hl_min_size = self.t_to_i(hl_min_size) if hl_min_size!=-1 else hl_min_size
        hl_max_size = self.t_to_i(hl_max_size) if hl_max_size!=-1 else hl_max_size
           
        #Find the best highlight
        best_hl = self.find_best_hl(hl_max_size, hl_min_size)[0]
        self.best_hls = [best_hl]

        if many_hls:
            best_hls = []
            score = 10
            next_hl = best_hl
            cumul_duration = best_hl['end']-best_hl['start']
            
            
           #In case of many highlight, pull a supplementary Hl as long as it fulfills the criteria 
            while not best_hls or not (poor_score or score_stuck or cumul_duration_too_long or too_many_hls):
                best_hls.append(next_hl)
                last_score = next_hl['score']
       
                #Once a HL has been found and was triggered by an emotion peak, delete the emotion peak from 
                #the features to compute the next best HL.
                for emotion in ['happiness', 'surprise']:
                    to_rm = []
                    for i, feat in enumerate(next_hl['start_cut'].features[emotion]):
                        if feat['rise_start'] < next_hl['start_cut'].end_cut:
                            emo_start = self.time[feat['emo_mask_start']]
                            emo_end = self.time[feat['emo_mask_end']-1]
                            if emo_start < next_hl['end'] and emo_end > next_hl['end']:
                                to_rm.append(i)
                    self.features[emotion] = np.delete(self.features[emotion], to_rm).tolist()
                
                #Compute nest best HL
                hls = self.find_best_hl(hl_max_size, hl_min_size)
                overlap=True
                i = 0

                #Consider the current computed HL as long as it doesn't overlap with the already
                #found HLs
                while overlap and i < len(hls):
                    overlap = False
                    next_hl = hls[i]
                    for prev_hl in best_hls:
                        both_hls = [prev_hl, next_hl]
                        starts = [prev_hl['start'], next_hl['start']]
                        ends = [prev_hl['end'], next_hl['end']]
                        durations = [prev_hl['end']-prev_hl['start'], next_hl['end']-next_hl['start']]
                        diff = both_hls[np.argmin(starts)]['end'] - both_hls[1-np.argmin(starts)]['start']

                        if diff > 0.25 * min(durations):
                            overlap = True
                    i += 1      

                #Compute the variables used for the stop condiitons of the HLs search          
                cumul_duration += next_hl['end']-next_hl['start']
                poor_score = next_hl['score'] < best_hls[0]['score']/3
                score_stuck = last_score == next_hl['score'] 
                cumul_duration_too_long = cumul_duration > self.duration*hl_duration_ratio
                too_many_hls = len(best_hls)+1 > max_hl_nb
        else:
            best_hls = [best_hl]

            
        self.best_hls = best_hls if best_hls[0]['score'] >= rm_tresh_score else []

        if self.best_hls:
            for hl in self.best_hls:
                print('BEST HL: start: {:.2f} | end: {:.2f} | score: {:.2f}'
                      .format(hl['start'], hl['end'], hl['score']))
        else:
            print('NO RELEVANT HIGHLIGHT IN THE VIDEO')
                
        

    #Find the best highglight in the video
    def find_best_hl(self, hl_max_size, hl_min_size):
        best_hls = []
        #Determine the end cut score function
        end_cut = End_cut(self.features, self.time, self.fps, self.sound_dict,
                              hl_max_size, hl_min_size)
        #Determine intervals with highest values
        end_cut.find_n_best_end_cut_intervals()
        #For each end interval candidate, find the associated best start cut
        for i, interval in enumerate(end_cut.best_intervals):
            hls = []
            #Find the optimal start cut for each end cut of the interval and take the best one
            for cut in interval:
                start_cut = Start_cut(self.features, self.time, self.fps, self.sound_dict, 
                                   hl_max_size, hl_min_size)
                start_cut.find_best_start_cut(cut)
                hl = dict()
                hl['start'] = start_cut.best['time']
                hl['end'] = cut['time']
                hl['start_i'] = start_cut.best['index']
                hl['end_i'] = cut['index']
                hl['score'] = (cut['score']+start_cut.best['score'])/2
                hl['score_end'] = cut['score']
                hl['score_start'] = start_cut.best['score']
                hl['end_score_fct'] = end_cut.score_fct
                hl['start_score_fct'] = start_cut.score_fct
                hl['start_cut'] = start_cut
                hl['end_cut'] = end_cut
                hls.append(hl)


            if hls:

                interval_min_score = min([hl['score_start'] for hl in hls])
                interval_max_score = max([hl['score_start'] for hl in hls])
                
                #If the interval is on a plateau (constant portion of the function) due to similarity event, cut in the middle
                if end_cut.intervals_on_similarity_zone[i]:
                    best_hl_in_interval = hls[int(len(hls)/2)]
                #If the interval is a plateau but not on similarity, take the furthest cut
                elif interval_max_score-interval_min_score < 0.1:
                    best_hl_in_interval = hls[0]
                #Otherwise, take the cut of the interval with the highest score 
                else:
                    best_hl_in_interval = hls[np.argmax([hl['score_start'] for hl in hls])]
                best_hls.append(best_hl_in_interval)
        #Return the bests start/end cuts pair for each end cut interval and sort them
        return np.asarray(best_hls)[np.argsort([hl['score'] for hl in best_hls])[::-1]]#[np.argmax([hl['score'] for hl in best_hls])]
            
    

    #Converting the all the time features into indices 
    def from_time_to_index(self):
        for emotion in ['happiness', 'surprise']:
            for i in range(len(self.features[emotion])):
                for key in self.features[emotion][i].keys():
                    #In the case where rise start or end fall times are on the video boundaries, the value is None -> needs to be considered
                    if key == 'rise_start' and self.features[emotion][i][key] is None:
                        self.features[emotion][i][key] = self.features[emotion][i]['start']
                    if key == 'fall_end' and self.features[emotion][i][key] is None:
                        self.features[emotion][i][key] = self.features[emotion][i]['end']
        for emotion in ['happiness', 'surprise']:
            for i in range(len(self.features[emotion])):
                for key in self.features[emotion][i].keys():
                    if key != 'score':
                        self.features[emotion][i][key] = self.t_to_i(self.features[emotion][i][key])
                  
                    
        for i in range(len(self.features['sound'])):
            self.features['sound'][i]['start'] = self.t_to_i(self.features['sound'][i]['start'])
            self.features['sound'][i]['end'] = self.t_to_i(self.features['sound'][i]['end'])
            
        for i in range(len(self.features['similarity'])):
            self.features['similarity'][i]['time'] = self.t_to_i(self.features['similarity'][i]['time'])
            
            
    def t_to_i(self, ts):
        return np.abs(self.time-ts).argmin()
         


    
    def emotion_preprocessing(self, two_consec_peak_diff=0.25, surp_hap_diff=1):
        #Give more importance to the surprise emotion than happiness
        #Remove happiness peak if right after a surprise peak
        if self.features['surprise']:
            self.features['surprise'] = [dict((k, v if k!='score' else 1) for k, v in feat.items())
                                        for feat in self.features['surprise']]
            self.features['happiness'] = [dict((k, v if k!='score' else 0 
                                                if (feat2['fall_end'] >= feat1['rise_start']
                                                and feat2['fall_end'] <= feat1['fall_end'])
                                                or abs(feat2['fall_end'] - feat1['rise_start'])<surp_hap_diff
                                                else v/2) 
                                                for k, v in feat1.items()
                                                for feat2 in self.features['surprise'])
                                         for feat1 in self.features['happiness']]
            
        #Merge consecutive peaks, with max of 0.25 seconds time distance
        for emotion in ['happiness', 'surprise']:
            peaks = self.features[emotion]
            to_rm = []

            i = 0
            while i < len(peaks)-1:
                peak1_end = peaks[i]['fall_end']
                peak2_start = peaks[i+1]['rise_start']
                if abs(peak1_end - peak2_start) < two_consec_peak_diff*self.fps:
                    peaks[i]['end'] = peaks[i+1]['end']
                    peaks[i]['fall_end'] = peaks[i+1]['fall_end']
                    w1 = peaks[i]['fall_end'] - peaks[i]['rise_start']
                    w2 = peaks[i+1]['fall_end'] - peaks[i+1]['rise_start']
                    score1 = peaks[i]['score']
                    score2 = peaks[i+1]['score']
                    peaks[i]['score'] = (w1*score1 + w2*score2)/(w1 + w2)
                    del peaks[i+1]
                    i-=1
                i+=1
            
                
            self.features[emotion] = peaks

        
    #When similarity events belong to the same pattern (valley or plateau), homogenize their score to the max of them
    def pair_similarity_scores(self):
        labels = [feat['label'] for feat in self.features['similarity']]
        
        for i in range(len(labels)):
            if labels[i] == 'plateau_start':
                j = i+1
                while j<len(labels) and labels[j] != 'plateau_end':
                    j+=1
                if j != len(labels):
                    score1 = self.features['similarity'][i]['score']
                    score2 = self.features['similarity'][j]['score']
                    self.features['similarity'][i]['score'] = max(score1, score2)#(score1+score2)/2
                    self.features['similarity'][j]['score'] = max(score1, score2)#(score1+score2)/2
                    
            if labels[i] == 'valley_start':
                j = i+1
                while j<len(labels) and labels[j] != 'valley_end':
                    j+=1
                if j != len(labels):
                    score1 = self.features['similarity'][i]['score']
                    score2 = self.features['similarity'][j]['score']
                    self.features['similarity'][i]['score'] = max(score1, score2)
                    self.features['similarity'][j]['score'] = max(score1, score2)
                    
                    
    
    #In the case where two close consecutive speech events occur,
    #they are merged as long as the botained total duration is below a threshold
    def merge_speech(self):
        if self.features['sound']:
            i = 0
            to_rm = []
            nb_sound_events = len(self.features['sound'])
            while i < nb_sound_events-1:
                #conditions for merging two events
                curr_label_is_speech = self.features['sound'][i]['label'] == 1
                next_label_is_speech = self.features['sound'][i+1]['label'] == 1
                both_events_are_close = (self.features['sound'][i+1]['start']-self.features['sound'][i]['end'])<0.5
                merge_is_not_too_long = (self.features['sound'][i+1]['end']-self.features['sound'][i]['start'])<8
                if curr_label_is_speech and next_label_is_speech and both_events_are_close and merge_is_not_too_long:
                    #Merge of both events
                    t1 = self.features['sound'][i]['end']-self.features['sound'][i]['start']
                    t2 = self.features['sound'][i+1]['end']-self.features['sound'][i+1]['start']
                    score1 = self.features['sound'][i]['score']
                    score2 = self.features['sound'][i+1]['score']
                    self.features['sound'][i]['end'] = self.features['sound'][i+1]['end']
                    self.features['sound'][i]['score'] = (t1*score1+t2*score2)/(t1+t2)
                    del self.features['sound'][i+1]
                    i -= 1
                    nb_sound_events -= 1
                i += 1
                    
    
            
        


            
            
    #Build a feature signal (a value for each timestamp at a 10fps rate) from the feature events
    def reconstruct_emotion_simplified_signal(self, label):
        #Init signal
        signal = np.zeros(round(self.duration*self.fps))
        #Iterate over emotion events
        for feat in self.features[label]:
            #Convert from times to indices
            start = self.t_to_i(feat['start'])
            end = self.t_to_i(feat['end'])
            #Add up event values to the signal
            if feat['rise_start'] is not None:
                rise_start = self.t_to_i(feat['rise_start'])
                rise_vals = np.linspace(0, feat['score'], start-rise_start+1)[:-1]
                signal[rise_start:start] = rise_vals
            if feat['fall_end'] is not None:
                fall_end = self.t_to_i(feat['fall_end'])
                fall_vals = np.linspace(feat['score'], 0, fall_end-end+1)[1:]
                signal[end+1:fall_end+1] = fall_vals
            #print(rise_start, fall_end, start, end)
            signal[start:end+1] = feat['score']
            
        self.signals[label] = signal
        
        
    #Fine relocating of each frame similarity event timestamp, 
    #that shifted because of sampling frequency changes and approximations
    def reajust_similarity_features(self):
        signal = self.signals['similarity']
        #Iterate over events
        for i in range(len(self.features['similarity'])):
            feat = self.features['similarity'][i]
            #2 points initialized to the current event timestamps that explore on the right/left of the event to find the optimal location
            moving_pts = [self.t_to_i(feat['time']), self.t_to_i(feat['time'])]
            distances = [0, 0]
            #Check if event is assigned to the right timestamp (local maxima)
            if moving_pts[0]<1 or moving_pts[0]>=len(signal)-1 or (signal[moving_pts[0]-1]-signal[moving_pts[0]])*(signal[moving_pts[0]+1]-signal[moving_pts[0]])<0:
                #Explore the left of the event until finding local maximum and save the travelled distance from the initial event timestamp
                if moving_pts[0] < 1:
                    distances[0] = float('inf')
                else:                    
                    init_diff = signal[moving_pts[0]-1]-signal[moving_pts[0]]
                    while moving_pts[0] >= 1 and init_diff*(signal[moving_pts[0]-1]-signal[moving_pts[0]])>0:
                        moving_pts[0] -= 1
                        distances[0] += 1
                    if moving_pts[0] == 0:
                        distances[0] = float('inf')
                        
                #Explore the right of the event until finding local maximum and save the travelled distance from the initial event timestamp
                if moving_pts[1] >= len(signal)-1:
                    distances[1] = float('inf')
                else:                    
                    init_diff = signal[moving_pts[1]+1]-signal[moving_pts[1]]
                    while moving_pts[1]+1 < len(signal) and init_diff*(signal[moving_pts[1]+1]-signal[moving_pts[1]])>0:
                        moving_pts[1] += 1
                        distances[1] += 1
                    if moving_pts[1] == len(signal)-1:
                        distances[1] = float('inf')
                    
                #Assign the nearest local maxima (between right and left)
                self.features['similarity'][i]['time'] = self.time[moving_pts[np.argmin(distances)]]
                    
        self.pair_similarity_scores()
                
                
    #Fine relocating of each sound event start/end timestamps, 
    #that shifted because of sampling frequency changes and approximations
    def reajust_sound_features(self):
        signal = self.signals['sound']
        #Iterate over events
        for i in range(len(self.features['sound'])):
            feature = self.features['sound'][i]
            feat_start = self.t_to_i(feature['start'])
            feat_end = self.t_to_i(feature['end'])
            
            #Init the event start search at the middle of the sound event
            prev = int((feat_start+feat_end)/2)
            next_ = prev-1
            #Decrement the searching point until reaching the true event start (when the signal nullifies)
            while prev > 0 and signal[next_] == signal[prev]:
                prev = next_
                next_ -= 1
            self.features['sound'][i]['start'] = self.time[prev]
            
            #Init the event start search at the middle of the sound event
            prev = int((feat_start+feat_end)/2)
            next_ = prev+1
            #Increment the searching point until reaching the true event end (when the signal nullifies)
            while next_ < len(signal) and signal[next_] == signal[prev]:
                prev = next_
                next_ += 1
            #Update event boundaries
            self.features['sound'][i]['end'] = self.time[prev]
            
        self.merge_speech()
            
            
    #Reajust the fps of a signal by downsampling/interpolating
    def resize_to_fixed_fps(self, signal, label):
        nb_pts = abs(len(signal)-round(self.duration*self.fps))
        if nb_pts:
            step = int(len(signal)/(nb_pts+1))
            #Indices subject to down-/upsampling
            indices = np.linspace(step, len(signal)-step-1, nb_pts).astype(int)
            diff = indices[:-1]-indices[1:]
            if diff[diff == 0].shape[0] != 0:
                pass
            #Downsampling condition
            if len(signal)/self.duration > self.fps:
                signal = np.delete(signal, indices).tolist()
            #Upsampling condition
            else:
                if label == 'sound':
                    values = [signal[i-1] if i else signal[0] for i in indices]
                else:
                    values = [(signal[i] + signal[i-1])/2 if i else signal[0] for i in indices]
                signal = np.insert(signal, indices, values).tolist()
        return signal
    
    
                        
    
   
    #Using highlight start/end timestamps to trim the original video into a highlight video
    def export_hl(self, input_path, output_path):      
        for i, hl in enumerate(self.best_hls):
            start_time = max(hl['start']-0.1, 0)
            end_time = hl['end']
            hl_path = '.'.join(output_path.split('.')[:-1]) + '_{}'.format(i+1) + '.mp4' if len(self.best_hls) > 1 else output_path
            
            print(hl_path)

            #Extract the trimmed video sequence and write it on the memory
            with VideoFileClip(input_path) as video:
                new = video.subclip(start_time, min(end_time, video.duration))
                new.write_videofile(hl_path, audio_codec='aac')


            path = '/'.join(hl_path.split('/')[:-1])
            start_end_times = pd.DataFrame(np.array([[start_time, end_time]]), columns = ['start_time', 'end_time'])
            start_end_times.to_csv(os.path.join(path, 'start_end_cuts.csv'))
            
            
            