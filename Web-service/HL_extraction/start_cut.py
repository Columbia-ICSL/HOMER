#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:24:24 2019

@author: hugomeyer
"""


import numpy as np
import matplotlib.pyplot as plt

from cut import Cut











class Start_cut(Cut):
    def __init__(self, features, time, fps, sound_dict, hl_max_size=-1, hl_min_size=-1):
        Cut.__init__(self, features, time, fps, sound_dict, hl_max_size, hl_min_size)
        
        self.W = None
        self.end = None
        self.start = None
        self.end_cut = None
        self.best = dict()
        self.simil_labels = ['plateau_start', 'plateau_end', 'valley_start', 'hill_top', 'valley_pit']
    
    def find_best_start_cut(self, end_cut=None):
        
        self.initialize_params(end_cut)
        self.time_scoring()
        self.sound_scoring()
        self.emotions_scoring()
        self.similarity_scoring()
        self.find_cut()
        
    def find_cut(self):

        left, right = self.find_plateau(self.score_fct, self.start, self.end)
        index = max(min(int((left + right)/2), self.end), self.start)
            
        self.best['time'] = self.time[index]
        self.best['index'] = index
        self.best['score'] = self.score_fct[self.start:self.end+1].max()
        '''
        print('index')
        print(index)
        print('left, right')
        print(left, right)
        print()
        '''
    
    def initialize_params(self, end_cut):
        self.end_cut = end_cut['index']

        self.end = end_cut['index'] - self.hl_min_size
        self.hl_max_size = min(self.hl_max_size, end_cut['index'])
        self.W = self.hl_max_size-self.hl_min_size
        self.start = self.end-self.W
        '''
        print('end_cut')
        print(end_cut['index'])
        print('hl_min_size, hl_max_size')
        print(self.hl_min_size, self.hl_max_size)
        print('start, end')
        print(self.start, self.end)
        print()
        '''
    def time_scoring(self):
        nb_pts = self.end_cut-self.start
        #penalty_per_sec = self.params['time']['penalty']/(1+logn(e, self.time[-1]/3)**2/10)
        #penalty = self.params['time']['penalty']
        duration = self.time[-1]
        penalty_per_sec = 0.25 if duration<10 else 0.25-(duration-10)*0.01 if duration < 30 else 0.05
        #penalty_per_sec = 0.2 if duration<10 else 0.2-(duration-10)*0.0075 if duration < 30 else 0.05
        #penalty_per_sec = 0.15 if duration<10 else 0.15-(duration-10)*0.005 if duration < 30 else 0.05
        #penalty_per_sec = penalty if duration < 10 else penalty/(1+logn(e, duration/10)/6)
        #print(duration, penalty_per_sec)
        last_lin_value = -penalty_per_sec*(nb_pts+1)/self.fps
        #print(last_lin_value)
        

        last_quad_value = last_lin_value*(nb_pts+1)*0.25/self.fps
        #last_quad_value = -penalty_per_sec*(nb_pts+1)/self.fps

        if not self.automatic:
            #print(self.params['time']['end_penalty_score'], last_quad_value)
            last_quad_value = max(self.params['time']['end_penalty_score'], last_quad_value)




        last_lin_value = last_quad_value*self.fps/((nb_pts+1)*0.25)
        linear_vals = np.linspace(0, last_lin_value, nb_pts+1)
        integrated = np.cumsum(linear_vals/self.fps, dtype=float)/2
        last_quad_value = integrated[-1]

        #if not self.automatic:
         #   last_quad_value = max(self.params['time']['end_penalty_score'], last_acc_value)
        self.time_score_fct[self.start:self.end_cut+1] += integrated[::-1]
        self.score_fct[self.start:self.end_cut+1] += integrated[::-1]
        
    
    
    def similarity_scoring(self):
        last_simil_range_end = 0
        last_time = -1
        #print(self.features['similarity'])
        for feat in self.features['similarity']:
            
            if feat['label'] in self.simil_labels:
                score = feat['score'] if feat['time'] > 1*self.fps else feat['score']/2
                half_t_range = max(int(round(self.params['similarity']['max_time_range']*self.fps*score/2)), 1)
                if feat['time']!=last_time:
                    start = max(feat['time']-half_t_range, last_simil_range_end)
                else:
                    start = feat['time']-half_t_range
                end = feat['time']+half_t_range+1
                last_simil_range_end = end
                
                if start < self.end and end > self.start:
                    last_time=feat['time']
                    #print(start, end, half_t_range)
                    #print(feat)

                    simil_pt = [feat['label'] for feat in self.features['similarity']
                                                   if feat['time']-0.5*self.fps < self.end_cut
                                                   and feat['time']+0.5*self.fps > self.end_cut
                                                   and feat['label'] != 'hill_top']
                    
                    #print(simil_pt, feat['label'], feat['time'], score, self.end_cut)
                    if simil_pt:
                        start_end_cuts_doing_valley = feat['label'] == 'valley_start' and 'valley_end' in simil_pt
                        start_end_cuts_doing_plateau = feat['label'] == 'plateau_start' and 'plateau_end' in simil_pt
                        start_end_cuts_doing_plateau2 = feat['label'] == 'plateau_end' and 'plateau_start' in simil_pt
                        #score = score*self.params['similarity']['upscale_valley_pair'] if start_end_cuts_doing_valley else score
                        #score = score*self.params['similarity']['upscale_plateau_pair'] if start_end_cuts_doing_plateau else score
                        #score = score*self.params['similarity']['upscale_valley_pair'] if start_end_cuts_doing_plateau2 else score
                        
                        duration = min((self.end_cut-feat['time'])/self.fps, 10)
                        score = score*1.5*duration**(1/2) if start_end_cuts_doing_valley else score
                        score = score*duration**(1/2) if start_end_cuts_doing_plateau else score
                        score = score*1.5*duration**(1/2) if start_end_cuts_doing_plateau2 else score
                    else:
                        ratio1 = self.params['similarity']['upscale_valley']
                        ratio2 = self.params['similarity']['upscale_plateau']
                        valley_start_or_plateau_end = feat['label'] == 'valley_start' or feat['label'] == 'plateau_end'
                        score = score**(1/ratio1) if valley_start_or_plateau_end else score**(1/ratio2) if feat['label'] == 'plateau_start' else score
                    #print(score)
                    upscale_hill_top = not self.features['sound'] and len([simil['label'] for simil in self.features['sound'] if simil['label']=='hill_top'])==len(self.features['sound'])
                    
                    ratio3 = self.params['similarity']['downscale_hill'] if not upscale_hill_top else self.params['similarity']['upscale_hill']
                    score = score*ratio3 if feat['label'] == 'hill_top' else score
     
                    start = max(start, self.start)
                    end = min(end, self.end)
                    
                    #print(score*self.params['similarity']['factor_start'])
                    #print()
                    self.simil_score_fct[start:end] += score*self.params['similarity']['factor_start']
  
                    self.score_fct[start:end] += score*self.params['similarity']['factor_start']
                    values = self.score_fct[start:end]
                    diff_means = sum([1 for i in range(len(values)-1) if abs(values[i]-values[i+1])>0.1])
                    if not diff_means:
                        self.score_fct[start:end] = self.score_fct[start:end].mean()
        #print()
    
        #self.plot_score_fct()
    
    def reajust_scores_given_labels(self, label, score):
        ratio = self.params['similarity']['labels_rescale']
        if label == 'valley_start' or label == 'hill_start' or label == 'plateau_end':
            return (score+ratio)/(1+ratio)
        elif label == 'plateau_start' or label == 'valley_pit':
            return (score+ratio/2)*(2-ratio)/(2+ratio)
        else:
            return score*(1-ratio)
    
    
        
    def sound_scoring(self):
        margin = int(0.3*self.fps)
        
        sound_labels = [self.sound_dict[feat['label']] for feat in self.features['sound']
                        #if feat['end'] <= self.end_cut+margin and feat['start'] >= self.start]
                        if feat['end'] > self.hl_min_size]
        
        laugh_indices = np.array([])
        speech_indices = np.array([])

        if sound_labels:
            sound_labels = np.asarray(sound_labels)
            laugh_indices = np.where(sound_labels=='laugh')[0]
            speech_indices = np.where(sound_labels=='speech')[0]
            
        #delete_laugh_if_preceded_by_speech = sum([1 for i in range(len(sound_labels)-1) 
         #                                     if sound_labels[i] == 'speech' 
         #                                     and sound_labels[i+1] == 'laugh'])
        
        #Scale speech up and scale laugh down in case both are presents
        rescale_laugh_and_speech = laugh_indices.any() and speech_indices.any() and speech_indices.min() < laugh_indices.max()
        
        #If no sound, 
        upscale_similarity = not speech_indices.shape[0] and not laugh_indices.shape[0]
        if upscale_similarity:
            self.params['similarity']['factor_start'] = self.params['similarity']['upscale_start']
            #if 'misc' not in sound_labels:
            self.score_fct /= 2
            self.time_score_fct /= 2
        labels=[]
        for feat in self.features['sound']:
            sound_label = self.sound_dict[feat['label']]
            #no_laugh_deleting = (delete_laugh_if_preceded_by_speech==0 or sound_label is not 'laugh')
            if feat['start'] < self.end_cut and feat['end'] >= self.start: #and no_laugh_deleting:
                
                end = min(self.end_cut, feat['end'])
                start = feat['start']
                score = feat['score']
                                
                if labels and self.sound_dict[labels[-1]] == 'laugh' and self.sound_dict[feat['label']] == 'laugh':
                    score /= 2
                
                labels.append(feat['label'])
                
                #Scale down miscellenaeous
                if sound_label == 'misc': #and (laugh_indices.any() or speech_indices.any()):
                    score *=  self.params['sound']['misc_scale_down']#(1/self.params['sound']['speech_and_laugh_scale_up'])
                    neg_score = score
                else:
                    neg_score = score**(1/self.params['sound']['during_speech_laugh_start_cut_penalty'])
                
                #print('score1: ', score*(end-start)/self.fps)
                integrated_score = min(score*(end-start)/self.fps, 1.5)
                #print('score2: ', integrated_score)
                '''
                if rescale_laugh_and_speech and sound_label == 'speech':
                    integrated_score *= (1+self.params['sound']['speech_laugh_rescale_fact'])
                if rescale_laugh_and_speech and sound_label == 'laugh':
                    integrated_score *= (1-self.params['sound']['speech_laugh_rescale_fact'])
                '''   
   
                sound_vals = np.linspace(integrated_score, 0, end-start+1)
                #self.score_fct[start: end+1] += sound_vals - score
                if sound_label != 'misc':
                    self.score_fct[start: end+1] -= neg_score
                    self.sound_score_fct[start: end+1] -= neg_score
                else:
                    self.score_fct[start: end+1] += sound_vals
                    self.sound_score_fct[start: end+1] += sound_vals
                    
                self.score_fct[:start] += integrated_score
                self.sound_score_fct[:start] += integrated_score
                
                if sound_label == 'laugh':
                    max_laughing_time = 3
                    d_score = score/self.fps
                    ratio = min((feat['end']-feat['start'])/(max_laughing_time*self.fps), 1)
                    param = self.params['sound']['laugh_front_cue_max']*ratio*self.fps
                    cue_width = int(round(min(param, start)))
                    if cue_width:
                        cue_start = start - cue_width
                        start_val = d_score*(1-cue_width/(param))
                        front_cue_vals = np.linspace(d_score, start_val, cue_width)

                        integrated = np.cumsum(front_cue_vals, dtype=float)[::-1]
                        self.test[cue_start: start] += integrated
                        self.test[:cue_start] += integrated[0]
                        self.score_fct[cue_start: start] += integrated
                        self.score_fct[:cue_start] += integrated[0]
                        self.sound_score_fct[cue_start: start] += integrated
                        self.sound_score_fct[:cue_start] += integrated[0]

                    
        
     
            
    
            
    
    def emotions_scoring(self):        
        #Score the emotion peaks with their respective score as squares 
        #with a cue at the end proportional to the peak width
        max_time_peak = 5
        for emotion in ['happiness', 'surprise']:
            last_emo_peak_start = self.L
            for i, feat in enumerate(self.features[emotion][::-1]):
                if feat['rise_start'] < self.end_cut:
                    ratio_cue = max(1-(feat['end'] - feat['start'])/(self.params['emotion']['max_peak_duration']*self.fps), 0)
                    #ratio_const = (feat['end'] - feat['start'])/(feat['fall_end']-feat['start'])
                    const_width = int(round(ratio_cue*self.params['emotion']['max_cue_width']*self.fps))
                    
                    emo_mask_end = min(const_width+feat['fall_end'], self.L)
                    if last_emo_peak_start < emo_mask_end:
                        emo_mask_end = feat['fall_end']

                    emo_mask_start = max(int(feat['rise_start']-self.params['emotion']['front_neg_cue']*self.fps), 0)
            
                    if i < len(self.features[emotion])-1:
                        emo_mask_start = max(self.features[emotion][-i-2]['fall_end'], emo_mask_start)
                        
                    self.features[emotion][-1-i]['emo_mask_start'] = emo_mask_start
                    self.features[emotion][-1-i]['emo_mask_end'] = emo_mask_end

                    #max_val = (emo_mask_end-emo_mask_start)*self.params['emotion']['neg_score_slope']/(2*self.fps)
                    max_val = self.params['emotion']['upscale_factor']*feat['score']
       
                    middle = int((emo_mask_end+emo_mask_start)/2)
                    #vals = np.linspace(max_val, -max_val, middle-emo_mask_start)
                    vals = np.linspace(max_val, -max_val, emo_mask_end-emo_mask_start)

                    self.emo_score_fct[emo_mask_start:emo_mask_end] = vals
                    #self.emo_score_fct[emo_mask_start:middle] = vals
                    #self.emo_score_fct[middle:emo_mask_end] = -max_val
                    self.emo_score_fct[:emo_mask_start] = max_val
                    

                last_emo_peak_start = feat['rise_start']

                    
        self.score_fct += self.emo_score_fct
        
    
    def plot_score_fct(self, save=False):
        title = 'Score function for the start cut of the highlight'
        x_label = 'Time'
        y_label='Score'
            
            
        x = self.time#+self.offset
        y = np.asarray(self.score_fct)
    
        fig, ax = plt.subplots(figsize=(10, 4), dpi=80)
        
        factor = 1/self.fps
        
        if x.shape[0] > 400:
            step = round(50*factor)
        elif x.shape[0] > 200:
            step = round(20*factor)
        elif x.shape[0] > 100:
            step = round(10*factor)
        elif x.shape[0] > 30:
            step = max(round(5*factor, 1), 0.1)
        else:
            step = max(round(1*factor, 1), 0.1)
        
        
        glob_min = min(self.simil_score_fct.min(), self.emo_score_fct.min(), 
                       self.sound_score_fct.min(), self.time_score_fct.min(), y.min())
        
        glob_max = max(self.simil_score_fct.max(), self.emo_score_fct.max(), 
                       self.sound_score_fct.max(), self.time_score_fct.max(), y.max())
        
        ystep = round((glob_max-glob_min)/10, 1)
        
        X_ticks_maj = np.arange(max(int(x[0])-factor, 0), int(x[-1])+2, step)
        y_ticks = np.arange(int(glob_min)-ystep-0.1, round(glob_max, 1)+ystep+0.1, max(ystep, 0.1))
        X_ticks_min = np.arange(x[0]-factor, x[-1]+1, step*0.2)
        X_ticks_min[0] = int(x[0])

        
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        
        ax.set_xticks(X_ticks_maj)
        ax.set_xticks(X_ticks_min, minor=True)
        ax.set_yticks(y_ticks)
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.4)
        ax.grid(which='major', alpha=1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.title(title)
        
        min_, max_ = y.min(), y.max()
        
        ax.axvline(x=self.time[self.start], linewidth=1, color='r')
        ax.axvline(x=self.time[self.end], linewidth=1, color='r')
        ax.fill_between([self.time[self.start], self.time[self.end]], [min_, min_], [max_, max_], facecolor='red', alpha=0.1, label='Window')
            
        

        plt.plot(x, self.simil_score_fct, label='similarity')
        plt.plot(x, self.emo_score_fct, label='emotion')
        plt.plot(x, self.sound_score_fct, label='sound')
        plt.plot(x, self.time_score_fct, label='time')
        
        plt.plot(x, y, label='score')
        
        #ax.axvline(x=self.best['time'], linewidth=1, color='g')
        #plt.plot(x, self.emo_score_fct)
        #plt.plot(x, self.test)
        plt.legend()
            
        if save == True:
            plt.savefig(os.path.join(path, 'Clip_{}_score_fct' .format(clip_ID)+string+'.png'))
        else:
            plt.show()
    
        