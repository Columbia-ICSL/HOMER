#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:44:10 2019

@author: hugomeyer
"""


from emotion import Emotion
from similarity import Similarity
from img_processing import from_img_to_bins, from_binned_img_to_tridiag, min_max_tridiags_pair
from media import Video
from sound import Sound
from events_timeline import Events_timeline

import pandas as pd
import os
import math
import numpy as np
from scipy.io.wavfile import read
import subprocess


class Trim(object):
    def __init__(self, start, end, fps):
        self.start=start
        self.end=end
        self.fps = fps
        self.time=(self.end-self.start)/fps
        
    def info(self):
        return {
            'start':self.start,
            'end': self.end,
            'time': self.time,
            'fps': self.fps
        }
    


#Highest level class, containing and initiating the computation of the three input signals under the form of classes: 
#emotion, frame similarity and sound.
#The event timeline is also contained as a class which handles the highlight extraciton
class Clip(object):
    def __init__(self, ID=None, back_video_path='', front_video_path='', 
                 trimmed_video_export='', target_cut=False):
        self.ID = ID
        self.duration = None
         
        self.emotion=None
        self.similarity=None
        self.sound=None
        
        self.events_timeline = None
        
        
        self.front_video=front_video_path
        self.back_video=back_video_path
        self.trim_export=trimmed_video_export
        self.trim = None
        
        
    def compute_events_timeline(self, fps):
        signals = dict()

        signals['sound'] = self.sound.y
        signals['similarity'] = self.similarity.values
        features = dict()
        
        if self.emotion is not None:
            adjusted_fps_ratio = self.duration/self.emotion.preds.shape[0]
        
            features['happiness'] = [
                                        {
                                                'start': peak.start*adjusted_fps_ratio,
                                                'end': peak.end*adjusted_fps_ratio,
                                                'rise_start': peak.rise_start,
                                                'fall_end': peak.fall_end,
                                                'score': peak.avg
                                        } 
                                        for peak in self.emotion.peaks if peak.emotion == 'p7_hap'
                                    ]
            
            features['surprise'] = [
                                        {
                                                'start': peak.start*adjusted_fps_ratio,
                                                'end': peak.end*adjusted_fps_ratio,
                                                'rise_start': peak.rise_start,
                                                'fall_end': peak.fall_end,
                                                'score': peak.avg
                                        } 
                                        for peak in self.emotion.peaks if peak.emotion == 'p7_surp'
                                    ]
        else:
            features['happiness'] = []
            features['surprise'] = []
        
        features['sound'] = [
                                    {
                                            'start': event.start_t,
                                            'end': event.end_t,
                                            'score': event.score,
                                            'label': event.label
                                    } 
                                    for event in self.sound.events
                            ]
        
        features['similarity'] = [
                                    {
                                            'time': feat.time,
                                            'score': feat.score,
                                            'label': feat.label,
                                    } 
                                    for feat in self.similarity.features
                            ]
        for emo_label in ['happiness', 'surprise']:
            for i in range(len(features[emo_label])):
                if features[emo_label][i]['rise_start'] is not None:
                    features[emo_label][i]['rise_start'] *= adjusted_fps_ratio
                if features[emo_label][i]['fall_end'] is not None:
                    features[emo_label][i]['fall_end'] *= adjusted_fps_ratio
        
     
    
        
        self.events_timeline = Events_timeline(signals, features, self.duration, fps)
        
        
        
        
    def compute_audio_signal(self, input_path):
        #Create an audio file from the video file and import it as a 1D array along with its sampling frequency
        
        try: 
            fs, track = read(input_path)
        except:      
            command = "ffmpeg -i "+self.back_video+" -ab 160k -ac 2 -ar 44100 -vn "+input_path
            subprocess.call(command, shell=True)
            try:
                fs, track = read(input_path)
            except:
                return 0
            
        
        self.sound = Sound(track[:, 0], fs, self.ID, input_path)
        self.duration = self.sound.L
        return 1
        

    def compute_emotions(self, emo_labels, fps=4):
        #import video
        video = Video([], self.duration, path=self.front_video, name=self.ID, fps=fps)
        
        if video.successful_loading:
            #Detect and crop the face of the user
            video.face_detect(mode='crop', model='both')
            #Predict emotions with the CNN with the cropped face as input
            video.emotions_prediction()
            #7 predicitons outputed from the NN for each timestamp (4 fps) along the video 
            #contenated with the winning emotion for each timestamp
            preds7 = np.asarray(video.emotions.preds7)
            best_guess7 = np.asarray(np.expand_dims(video.emotions.best_guess7, axis=1))
            concat = np.concatenate((preds7, best_guess7), axis=1)
            df = pd.concat([pd.Series(x) for x in concat], axis=1).T
            #Create an emotion class for further processing 
            self.emotion = Emotion(df, emo_labels, fps=fps)
            return 1
        else:
            print('Front video of clip {} not found.' .format(self.ID))
            return 0
        
        
        
        
    def compute_img_similarity(self, simil_fps, importing_time_limit=30, ratio=0.1, bin_size=2):  
        
        tot_simil_vect=[]
        #Import the video in sequences of 30sec to not overload the memory
        for start in range(0, int(self.duration), importing_time_limit):
            #Import video sequence of 30sec
            video=Video(frames=[], path=None, name='clip_{}'.format(self.ID), fps=simil_fps, duration=self.duration) 
            if start:
                start -=1
                importing_time_limit += 1
            end = min(start+importing_time_limit, self.duration)
            successful_loading = video.load_frames(path=self.back_video, fps=simil_fps, record_start_time=start, time_limit=end)
            if not successful_loading:
                raise ValueError('The video file was not found. Check the path again')

            video.remove_black_frame()
            imgs = [frame.pix_vals for frame in video.frames]             
            simil_vect=[]
            
            #Compute the frame similarity for each consecutive 3 frames of the video
            for i in range(1, len(imgs)-1):
                triplet_imgs = [imgs[j] for j in [i-1, i, i+1]]
                binned_imgs = [from_img_to_bins(img, ratio, bin_size) for img in triplet_imgs]
                diags = [from_binned_img_to_tridiag(binned_imgs[i], binned_imgs[i+1], bin_size) for i in range(2)]
                min_, max_ = min_max_tridiags_pair(diags[0], diags[1])
                simil_vect.append(min_/max_)
                
            simil_vect = [simil_vect[0]] + simil_vect

            
            if start == 0:
                tot_simil_vect = simil_vect
            else:
                tot_simil_vect += simil_vect[9:]


        #Create an object for the frame similarity for further processing
        self.similarity = Similarity(1, len(tot_simil_vect), tot_simil_vect, simil_fps)
            
            
        