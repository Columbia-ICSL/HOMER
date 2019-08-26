#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 20:35:43 2019

@author: hugomeyer
"""

import os, json, glob
import tensorflow as tf
import numpy as np
import librosa as lr
from pydub import AudioSegment
import math

### Inference

## Audio helper functions

# Loads the audio file given by path using LibROSA
def audio_from_file(path, sr=None, ext='', offset=0.0):
    return lr.load('{}{}'.format(path, ext), sr=sr, mono=True, offset=offset, 
                 duration=None, dtype=np.float32, res_type='kaiser_best')

# Converts the given audio to frames
def audio_to_frames(x, n_frame, n_step=None):
    if (n_step is None):
        n_step = n_frame
    if (len(x.shape) == 1):
        x.shape = (-1,1)
    n_overlap = n_frame - n_step
    n_frames = (x.shape[0] - n_overlap) // n_step       
    n_keep = n_frames * n_step + n_overlap
    strides = list(x.strides)
    strides[0] = strides[1] * n_step
    return np.lib.stride_tricks.as_strided(x[0:n_keep,:], (n_frames,n_frame), strides)

# Exports the audio to path
def audio_to_file(path, x, sr):    
    lr.output.write_wav(path, x.reshape(-1), sr, norm=False)
    
# Gets the audio duration
def get_audio_duration(input_audio):
    import wave
    import contextlib
    with contextlib.closing(wave.open(input_audio,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

# Rounds the audio by repeating the tail
def round_audio(input_audio):
    duration = get_audio_duration(input_audio)
    
    audio = AudioSegment.from_wav(input_audio)
    tail_duration = (duration % 1) * 1000 # ms

    if duration > 1:
        tail = audio[-tail_duration:]
    else:
        tail = audio

    for i in range(math.ceil(1 / (duration % 1))):
        audio += tail
        
    
    audio.export(input_audio.split('.')[0] + '_rounded.wav', format='wav')
    
    return audio

## OS helper functions

# Functions the same as os.listdir but ignores hidden files
def listdir_nohidden(path):
    for f in os.listdir(path):
        if (not f.startswith('.')):
            yield f

## Core prediction function

#   model_path - the path to the directory containing the model information
#   input_audio - either a directory containing the .wav files to predict on or 
#                 a single .wav file to predict on 
#   n_batch - the number of points to predict on
#   offset - where to starting read the audio from

def predict(model_path, input_audio, n_batch=256, offset=0.0):

  
    if (os.path.isdir(model_path)):
        candidates = glob.glob(os.path.join(model_path, 'model.ckpt-*.meta'))
        if (candidates):
            candidates.sort()                
            checkpoint_path, _ = os.path.splitext(candidates[-1])
    else:
        checkpoint_path = model_path  
    if (not all([os.path.exists(checkpoint_path + x) for x in 
              ['.data-00000-of-00001', '.index', '.meta']])):
        print('ERROR: Could Not Load Model')
        raise FileNotFoundError
    
    vocabulary_path = checkpoint_path + '.json'
    if (not os.path.exists(vocabulary_path)):
        vocabulary_path = os.path.join(os.path.dirname(checkpoint_path), 'vocab.json')
    if (not os.path.exists(vocabulary_path)):
        print('ERROR: Could Not Load Vocabulary')
        raise FileNotFoundError
    with open(vocabulary_path, 'r') as fp:
        vocab = json.load(fp)
    
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
    
        x = graph.get_tensor_by_name(vocab['x'])
        y = graph.get_tensor_by_name(vocab['y'])            
        init = graph.get_operation_by_name(vocab['init'])
        logits = graph.get_tensor_by_name(vocab['logits'])            
        ph_n_shuffle = graph.get_tensor_by_name(vocab['n_shuffle'])
        ph_n_repeat = graph.get_tensor_by_name(vocab['n_repeat'])
        ph_n_batch = graph.get_tensor_by_name(vocab['n_batch'])
        sr = vocab['sample_rate']
    
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
            outputs = list()
            
            if os.path.isfile(input_audio):
                file = input_audio
            else:
                raise ValueError('input_audio Must Be a Directory or File ({})'.format(input_audio))

            
            if os.path.isdir(input_audio):
                file = os.path.join(input_audio, file)
            
            
            if (os.path.exists(file)):
                sound, _ = audio_from_file(file, sr=sr, offset=offset)
                input_audio = audio_to_frames(sound, x.shape[1])
                labels = np.zeros((input_audio.shape[0],), dtype=np.int32)
                sess.run(init, feed_dict = {x : input_audio, y : labels, 
                                      ph_n_shuffle : 1, ph_n_repeat : 1, 
                                      ph_n_batch : n_batch})
                count = 0
                while (True):
                    try:
                        output = sess.run(logits)
                        labels[count:count+output.shape[0]] = np.argmax(output, axis=1)
                        count += output.shape[0]
                        outputs.append(output)
                    except tf.errors.OutOfRangeError:
                        break
            else:
                print('Skip [File Not Found]')

            return outputs

## Combined laughter/speech classifier

# Laughter/non-laughter + speech/non-speech
def predict_one_window_thresholded(laughter_prob, speech_prob):
    if (laughter_prob > 0.8):
        return 0
    elif (laughter_prob <= 0.8 and laughter_prob > 0.5 and speech_prob > 0.5):
        return 0
    elif (laughter_prob <= 0.8 and laughter_prob > 0.5 and speech_prob <= 0.5):
        return 2
    elif (laughter_prob <= 0.5 and speech_prob > 0.5):
        return 1
    elif (laughter_prob <= 0.5 and speech_prob <= 0.5):
        return 2

# Outputs the prediction for each window
def predict_window_batch(laughter_probs, speech_probs):
    assert(len(laughter_probs) == len(speech_probs))
    predictions = [None] * len(laughter_probs)
    for i in range(len(laughter_probs)):
        window_predictions = np.zeros(len(laughter_probs[i]), dtype=int)
        for j in range(len(laughter_probs[i])):
            laughter_prob = laughter_probs[i][j][1]
            speech_prob = speech_probs[i][j][1]
            window_predictions[j] = predict_one_window_thresholded(laughter_prob, speech_prob)
        predictions[i] = window_predictions
    return list(list(predictions)[0])

## Smoothing algorithm

# Get the run length encoding
def get_rle(x):
    d = dict()
    d['values'] = list()
    d['lengths'] = list()
    if (len(x) == 0):
        return d
    else:
        cur_run_val = x[0]
        cur_run_length = 1
        for i in range(1, len(x)):
            if (x[i] == cur_run_val):
                cur_run_length += 1
            else:
                d['values'].append(cur_run_val)
                d['lengths'].append(cur_run_length)
                cur_run_val = x[i]
                cur_run_length = 1
        d['values'].append(cur_run_val)
        d['lengths'].append(cur_run_length)
        assert(len(d['values']) == len(d['lengths']))
        return d

# Get the inverse run length encoding
def get_inverse_rle(run_values, run_lengths):
    out = list()
    for i in range(len(run_values)):
        out += [run_values[i]] * run_lengths[i]
    return out

# Smooths values (the crux of the smoothing algorithm)
def get_switch_value(i, run_values, run_lengths, frame_duration=1):
    num_runs = len(run_values)

    prev_run_value = run_values[i-1]
    cur_run_value = run_values[i]
    next_run_value = run_values[i+1]

    prev_run_length = run_lengths[i-1]
    next_run_length = run_lengths[i+1]

  # Ignore edges
    if (i == 0 or i == num_runs-1):
        return cur_run_value

    # Smooth short speech within laughter
    if (cur_run_value == 1 and prev_run_value == 0 and next_run_value == 0 and
        prev_run_length + next_run_length > 3*frame_duration):
        return 0

    # Smooth short noise within laughter
    if (cur_run_value == 2 and prev_run_value == 0 and next_run_value == 0 and
        prev_run_length + next_run_length > 3*frame_duration):
        return 0

    # Smooth short noise within speech
    if (cur_run_value == 2 and prev_run_value == 1 and next_run_value == 1 and
        prev_run_length + next_run_length > 3*frame_duration):
        return 1
  
    return cur_run_value
  

def erase_audio_files(audio1, audio2):
    for audio in [audio1, audio2]:
        if os.path.exists(audio):
            command = 'rm ' + audio
            os.system(command)

## Wrapper

# Serves as a wrapper function for prediction
def predict_wrapper(input_audio, laughter_model_path, speech_model_path, offset=0.0):
    audio_signal = round_audio(input_audio)
    rounded_input_audio = input_audio.split('.')[0] + '_rounded.wav'
    laughter_outputs = predict(laughter_model_path, rounded_input_audio, offset=offset)
    speech_outputs = predict(speech_model_path, rounded_input_audio, offset=offset)
    predictions = predict_window_batch(laughter_outputs, speech_outputs)
    erase_audio_files(input_audio, rounded_input_audio)
    return predictions