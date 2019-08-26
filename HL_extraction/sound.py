#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:05:53 2019

@author: hugomeyer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import pywt
from sklearn.cluster import KMeans 
from scipy.integrate import simps

from sound_recognition import predict_wrapper


#Class countaining the feature of a sound event
class Sound_event(object):
    def __init__(self, start_i, end_i, start_t, end_t, score):
        self.start = start_i
        self.end = end_i
        self.start_t = start_t
        self.end_t = end_t
        self.score = score
        self.label = None
        


class Sound(object):
    def __init__(self, signal, fs, index, path):
        self.index = index
        self.y = np.asarray(signal)
        self.fs = fs
        self.L = signal.shape[0]/fs
        self.t = np.linspace(0, self.L, len(signal))
        self.T = 1/fs
        self.N = self.y.shape[-1]
        self.only_noise = False
        self.treshold = None
        self.events = []
        self.origY = self.y
        self.origT = self.t
        self.path = path
        self.event_labels = None
        
    #Subsample the sound signal
    def subsample(self, f):
        self.y = self.y[::f]
        self.t = self.t[::f]
        self.fs /= f
        self.T = 1/self.fs
        self.N = self.y.shape[-1]
        
        
    #Compute the Fast Fourier Transform (FFT) of the signal
    def fft(self, db=False):
        signal = self.y.copy()
        yf = fft(signal)
        self.xf = np.linspace(0.0, 1.0/(2.0*self.T), self.t.shape[0]//2)
        self.yf = 2.0/self.N * np.abs(yf[0:self.N//2])
        if db == True:
            yf = 20 * np.log10(yf / np.max(yf))
            
            
    #Compute the three-class sound event classification over the whole signal (1sec intervals)
    def events_recognition(self):
        self.event_labels = predict_wrapper(self.path, '../Models/Sound/laughter', '../Models/Sound/speech')
            
            
    #Pre-process of the sound signal for event segmentation
    def segmentation_preprocess(self, subsamp_freq=1000, nb_clusters=3, wavelet_lvl=7):        
        self.subsample(int(self.fs/subsamp_freq))
        self.origY = self.y
        self.origT = self.t
        self.y = (np.abs(self.y)/np.max(np.abs(self.y)))**2#Square the signal for positive values and widen gap between high and low values
        self.y = self.wavelet_filtering(self.y, nb_coeffs=1, lvl=wavelet_lvl)
        #Reajust signal properties after wavelet transform 
        self.N = self.y.shape[-1]
        self.t = np.linspace(0, self.L, self.N)
        self.fs = self.N/self.L
        self.T = 1/self.fs
        
        #Discard signal if too noisy background has too high amplitude
        integral = simps((self.y/self.y.max())**2, self.t)
        if integral/self.L >= 0.3:
            self.only_noise = True
            
        #Adjust signal with its median to remove the constant background noise
        self.y = (self.y/self.y.max())-np.percentile(self.y, 50)
        self.y = np.where(self.y<0, 0, self.y)
     


    #Compute the wavelet transform and its inverse to reconstruct a simplified, filtered signal
    def wavelet_filtering(self, signal, nb_coeffs, lvl):
        coeffs = pywt.wavedec(signal, 'db1', level=lvl)
        return pywt.waverec(coeffs[:nb_coeffs], 'db1')
    
    #Perform event segmentation over the whole sound signal
    def events_segmentation(self):
        #Compute adaptative threshold to detect only relevant sound sub-events in the pre-processed signal
        self.treshold = self.adaptive_treshold(2)
        #If threshold is null, it means that the signal is null and therefore does not contain any event
        if self.treshold < 0.01:
            self.only_noise = True
            return

        #Find events 
        events_boundaries = self.find_events_boundaries()
        #Check if the cumulative duration of detected events is less than a % of the total duration
        coverage = sum([bound[1]-bound[0] for bound in events_boundaries])/self.N
        if coverage > 0.7:
            self.events = []
            self.treshold = self.adaptive_treshold(2, nb_pts=4, elbow_constraint_free=True)
            events_boundaries = self.find_events_boundaries()

        #Compute sound event score given its energy density
        events_score = self.compute_event_score(events_boundaries)
        self.events = [Sound_event(start, end, (start+1)*self.T, (end+1)*self.T, score) 
                       for score, (start, end) in zip(events_score, events_boundaries)]
        
        #Fuse both event segmentation and recognition by applying each of the three class to 
        #the detected events (majority vote)
        if self.event_labels is not None:
            self.merge_segmentation_and_recognition()
            
        #Turn sound events into a simplified sound feature signal
        self.signal_simplification(events_score, events_boundaries)

        
    #Turn sound events into a simplified sound feature signal  
    def signal_simplification(self, events_score, events_boundaries):
        feat_signal = np.zeros(len(self.y))
        for score, (start, end) in zip(events_score, events_boundaries):
            feat_signal[start:end+1] = score
        self.y = feat_signal
        
        
        
    #Fuse both event segmentation and recognition by applying each of the three class to 
    #the detected events (majority vote)
    def merge_segmentation_and_recognition(self):
        #Iterate over all sound event segmented in the signal
        for i in range(len(self.events)):
            start = self.events[i].start_t
            end = self.events[i].end_t
            
            durations = [0, 0, 0]
            event_indices = np.arange(int(start), min(int(end)+1, len(self.event_labels)), 1) 
            #Compute the respective durations of each of the three sound classes with the sound event
            for j in event_indices:
                if start > j:
                    durations[self.event_labels[j]] += (1-start+j)
                elif abs(j-end) < 1:
                    durations[self.event_labels[j]] += end-j
                else:
                    durations[self.event_labels[j]] += 1
    
            #Vote for speech/laugh more easily than miscellaneous 
            if durations[0] + durations[1] > 0.25*(end-start):
                #Majority vote between speech and laugh
                self.events[i].label = np.argmax(durations[:2])
            else:
                self.events[i].label = len(durations)-1
            
        
        
    #Find events
    def find_events_boundaries(self):
        #Find sound sub-events by considering the signal values over the adaptive threshold
        peaks_pts_ind = np.squeeze(np.argwhere(self.y > self.treshold))
        #Create intervals out of it
        if peaks_pts_ind.shape:
            indices_shifted = np.append(peaks_pts_ind[1:], peaks_pts_ind[-1]+1)
            bounds = [0] + list(np.nonzero(indices_shifted-peaks_pts_ind-1)[0]+1) + [len(peaks_pts_ind)]
            peaks_intervals = [(peaks_pts_ind[bounds[i]], peaks_pts_ind[bounds[i+1]-1]) for i in range(len(bounds)-1)]
        else:
            peaks_intervals = [(peaks_pts_ind, peaks_pts_ind)]
        #Each sub-event detected is extended on its right and left with gradient descent to find its "true" boundaries 
        raw_boundaries = self.find_subevents_boundaries(peaks_intervals, momentum=0.5)
        #Cluster sub-events into events
        return self.merge_subevents_into_event(raw_boundaries)

        
    #Score sound events based on their energy density
    def compute_event_score(self, boundaries):
        scores=[]
        for start, end in boundaries:                
            event_integral = simps(self.y[start:end+1], self.t[start:end+1])
            event_duration = self.t[end]-self.t[start]
            score = 2*event_integral/event_duration
            scores.append(score)
            
        return scores
        
    
            
    #Each sub-event detected is extended on its right and left with gradient descent to find its "true" boundaries 
    def find_subevents_boundaries(self, peaks_intervals, momentum=0.5, eps=0.02): #momentum in seconds
        boundaries = []
        #Iterates over the sub-events
        for interval in peaks_intervals:
            #Intilize
            (start, end) = interval
            momentum_ind = int(momentum/self.T)+1
            stop_condition = False
            i = start
            #Gradient descent on the left side to find the true start of the sub-event
            while not stop_condition and i >= 0:
                curr_val = self.y[i]
                proj_val = self.y[max(i-momentum_ind, 0)]
                #Condition for finding a relevant window where the sub-event should start -> start of peak side
                if curr_val - proj_val < eps:
                    #Continue to go down within the window until reaching a less sloppy point
                    while i > 0 and self.y[i]-self.y[i-1] > 0.01:
                        i -= 1
                    stop_condition = True
                i -= 1
            i += 1
            j = end
            stop_condition = False
            #Gradient descent on the right side to find the true end of the sub-event
            while not stop_condition and j<len(self.y):
                curr_val = self.y[j]
                proj_val = self.y[min(j+momentum_ind, len(self.y)-1)]
                #Condition for finding a relevant window where the sub-event should end -> end of peak side
                if curr_val - proj_val < eps:
                    #Continue to go down within the window until reaching a less sloppy point
                    while j < len(self.y)-1 and self.y[j]-self.y[j+1] > 0.01:
                        j += 1
                    stop_condition = True
                j += 1
            j -= 1
            if i!=j:
                boundaries.append([i, j])
            else:
                boundaries.append([i-1, j+1])
        return boundaries
    
    #Find an adapted threshold used later for selecting sub-events in the signal
    def adaptive_treshold(self, nb_crosses_limit=2, nb_pts=2, elbow_constraint_free=False):
        nb_crosses=[]
        tresh_vals = np.arange(0, 0.5, 0.01)
        #Iterate over a grid of candidate threshold values and compute their associated number of crosses with the sound signal
        for i in tresh_vals:
            nb_crosses.append(self.nb_of_treshold_crosses(self.y, i)/self.L)

        #Find the optimal threshold value that will constitute a good compromise between 
        #allowing too much or not enough sub-events. 
        #By partially minimizing the number of crosses between the threshold and the signal
        if nb_crosses:
            index_max = np.argmax(nb_crosses)
            
            if index_max != 0:
                #An elbow on the number of crosses means we got rid of a lot of sub-events very quickly
                #what often means getting rid of uninteresting sound noise
                elbow=self.check_if_elbow(nb_crosses, index_max, 0.2, nb_pts, elbow_constraint_free)
                if elbow is None:
                    if nb_crosses[index_max] > nb_crosses_limit:
                        i = index_max
                        while nb_crosses[i]>=2 and i<20:
                            i += 1

                        return tresh_vals[i]
                    else:
                        return 0.01
                else:
                    return tresh_vals[min(elbow, 20)]
        return 0

    #Number of times the signal crosses the threshold 
    def nb_of_treshold_crosses(self, data, treshold=None):
        if treshold is None:
            treshold = data.mean()
        return len([i for i in range(data.shape[0]-1) if (data[i+1] - treshold)*(data[i] - treshold)<0])


    #An elbow on the number of crosses means we got rid of a lot of sub-events very quickly
    #what often means getting rid of uninteresting sound noise
    def check_if_elbow(self, data, start, end, nb_pts, elbow_constraint_free):
        i = start
        slope=0
        while i < len(data)-1 and data[i+1] == data[i] and i<20:
            i += 1
        while i < len(data)-2 and data[i]-data[i+nb_pts] > 0.1 and i<20:
            i+=1
        if i!=start:
            slope = 100*(data[start]-data[i])/(i-start)
        #The criteria for having an elbow is based on the sharpness of the slope
        if elbow_constraint_free or (slope > 15 and data[i] < 0.5*data[start]):
            return i
        return None
    
    
    #Compute the mean between the best third of local maxima in the data
    def comp_mean_maxs(self, data):
        if len(data)>2:
            tops = [data[i] for i in range(1, len(data)-1) if data[i]>data[i-1] and data[i]>data[i+1]]
        if len(data)<=2 or not tops:
            return data.max()
        rank = max(round(len(tops)/3), 1)
        relevant_tops = np.sort(tops)[-rank:]
        return np.asarray(relevant_tops).mean()
    
    

    def merge_subevents_into_event(self, boundaries, eps=0.6, delta_t=1):
        eps = round(eps/self.T)
        #Initialize the first cluster
        merged_boundaries = [boundaries[0]]
        max_vals = []
        cumul_max = self.comp_mean_maxs(self.y[merged_boundaries[0][0]: merged_boundaries[0][1]+1])
        cumul_width = merged_boundaries[0][1] - merged_boundaries[0][0] 
        counter=1
        #Iterate over all subevents 
        for i in range(1, len(boundaries)):
            start1, end1 = merged_boundaries[-1]
            start2, end2 = boundaries[i]
            max_diff = abs(cumul_max - self.comp_mean_maxs(self.y[start2: end2+1]))
            treshold = 0.2+0.2*max(cumul_max, (self.y[start2: end2+1]).max())

            #Merge the new sub-event with the current growing cluster if:
            # -They are close enough to eachother in time
            # -They are close enough to eachother in amplitude
            merging = start2-end1 <= eps and (start2-end1 < 0 or max_diff <= treshold)
            if merging:
                merged_boundaries[-1][1] = end2
                treshold = 0.2+0.2*max(cumul_max, (self.y[start2: end2+1]).max())
                if max_diff < treshold:
                    coeff = (end2-start2)/(cumul_width+end2-start2)
                    cumul_max = cumul_max*(1-coeff)+self.comp_mean_maxs(self.y[start2: end2+1])*coeff#Weighted average of the events score
                else:
                    cumul_max = max(cumul_max, self.comp_mean_maxs(self.y[start2: end2+1]))
                cumul_width += end2-start2
                counter+=1
            else:
                merged_boundaries.append([start2, end2])
                max_vals.append(cumul_max)
                cumul_max = self.comp_mean_maxs(self.y[start2: end2+1])
                cumul_width = end2-start2
                counter=1
                
        max_vals.append(cumul_max)
                            
        return self.merge_unsimilar_profiles(merged_boundaries, max_vals, eps, delta_t)
    
    #Once events are obtained they can be merged again between them in a second round
    def merge_unsimilar_profiles(self, boundaries, max_vals, eps, delta_t):
        delta_ind = int(delta_t/self.T)+1
        if len(boundaries)>1:
            #Compute distances between events in sorting order
            dists = np.asarray([boundaries[i+1][0]-boundaries[i][1] for i in range(len(boundaries)-1)])
            dist_ind = np.argsort(dists)
            dists = dists[dist_ind]

            i=0
            #Iterate as long as the distance remains smaller than the allowed threshold for merging
            while i<len(dists) and dists[i] < eps:
                bounds = [boundaries[dist_ind[i]], boundaries[dist_ind[i]+1]]
                maxis = [max_vals[dist_ind[i]], max_vals[dist_ind[i]+1]]
                
                low_peak_ind = np.argmin(maxis)
                max_diff = abs(maxis[0] - maxis[1])
                treshold = 0.2+0.2*max(maxis)
                #Merge events if their amplitude are similar or if the more quiet event is very small and can be annexed by the loud one
                if max_diff < treshold or bounds[low_peak_ind][1] - bounds[low_peak_ind][0] <= delta_ind:
                    width0 = bounds[0][1]-bounds[0][0]
                    width1 = bounds[1][1]-bounds[1][0]
                    coeff = width0/(width0+width1)
                    max_vals[dist_ind[i]+1] = max_vals[dist_ind[i]]*coeff+max_vals[dist_ind[i]+1]*(1-coeff)#Weighted average of the events score
                    boundaries[dist_ind[i]+1][0] = boundaries[dist_ind[i]][0]
                    
                    del boundaries[dist_ind[i]]
                    del max_vals[dist_ind[i]]
                    dists = np.delete(dists, i)
                    index=dist_ind[i]
                    dist_ind = np.where(dist_ind>index, dist_ind-1, dist_ind)
                    dist_ind = np.delete(dist_ind, i)
                    i-=1
                i+=1
        return boundaries
        
    
    