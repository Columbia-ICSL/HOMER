
import random
import os
import sys
from moviepy.editor import VideoFileClip, concatenate_videoclips

#Create one montage file regrouping the shuffled highlights produced by the algorithm
def concatenate(input_folder, output_file):
    random.seed(251)
    file_list = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.lower().endswith('.mp4')]
    video_list = [VideoFileClip(file) for file in file_list]
    concatenated_video = concatenate_videoclips(video_list, method='compose')
    concatenated_video.write_videofile(output_file, audio_codec='aac')

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print('Usage: python3 montage.py [input_folder] [output_file]')
        sys.exit()
    else:
        input_folder = sys.argv[1]
        output_file = sys.argv[2]
        concatenate(input_folder, output_file)
