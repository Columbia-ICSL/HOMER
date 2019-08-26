import os
import cv2
import numpy as np
from emo_detect import EmoDetector
#import matplotlib.pyplot as plt
from img_processing import face_detection
import matplotlib.pyplot as plt
import subprocess
from openpyxl import load_workbook


#Class handling the import, export and processing of videos
class Video(object):
    def __init__(self, frames, duration, path=None, name='video', fps=5):
        self.frames=frames
        self.name=name
        self.duration = duration
        self.successful_loading = 1
        if path is not None:
            self.name = path.split('/')[-1].split('.')[0]
            self.successful_loading = self.load_frames(path, fps)
        elif frames:
            self.width=frames[0].pix_vals.shape[1]
            self.height=frames[0].pix_vals.shape[0]
        self.emotions = EmoDetector()


    

    #Export the video as a .mp4 file
    def export(self, path, width=None, height=None, codec='mp4v', exp_format='mp4', fps=20, start=-1, end=-1):
        
        if width is None or height is None:
            width=self.width
            height=self.height
        else:
            self.resize(width, height)
        
        #Set the encoding, file path and set the writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        file_path = os.path.join(path, self.name)+'.'+exp_format
        out = cv2.VideoWriter(file_path, fourcc, fps, (width,height))
        
       # 
        if start == -1 and end == -1:
            frames = self.frames
        else:
            frames = self.frames[start-1:end]
            
        #Write frames one by one
        for i, frame in enumerate(frames):
            out_img = frame.pix_vals
            out.write(out_img)
            
        out.release()
        cv2.destroyAllWindows()



    #Load frames one by one from the video files and save them as numpy matrices
    def load_frames(self, path, fps, record_start_time=0, time_limit=None):
        cap = cv2.VideoCapture(path)
        sec=0
        i=0
        if time_limit is None:
            time_limit = self.duration
        while(cap.isOpened()):
            sec = sec + 1/fps
            sec = round(sec, 2)
            cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            ret, frame = cap.read()
            #print(sec)
            #Record frame as long as we didn't reach the end of the video or a pre-specified time range
            if ret==True and sec <= time_limit:
                #Record frame as soon as we the video begins or the pre-specified time range starts
                if sec > record_start_time:
                    i += 1
                    #Capture video size
                    if i == 1:
                        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
                        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
                    frame_name = "frame_{:d}" .format(i)
                    self.frames.append(Image(frame, name=frame_name))
                    #print(frame)

            else:
                break
            
        if i == 0:
            return 0
        else:
            print("Video "+self.name+" successfully imported.")
        cap.release()
        cv2.destroyAllWindows()
        return 1
    
    #Detect and crop face for each frame of the video
    def face_detect(self, mode='crop', model='both'):
        for i in range(len(self.frames)):
            self.width, self.height = self.frames[i].face_detect(mode, model)


    #Compute emotion probabilities for each frame of the video using the CNN model
    def emotions_prediction(self):
        for frame in self.frames:
            if frame.face_detected == True:
                frame.resize(48, 48)
                frame.convert_to_gray()
                self.emotions.predict_emotion(frame.pix_vals[np.newaxis, :, :, np.newaxis])
            else:
                self.emotions.preds7.append(np.asarray([-0.1]*7))
                self.emotions.best_guess7.append("No face")
        

                
    #Resize frames of the video
    def resize(self, width, height):
        for i in range(len(self.frames)):
            self.frames[i].resize(width, height)
            
    #Rotate frames of the video
    def rotate(self, angle):
        if angle%90 != 0:
            raise ValueError("The rotation angle must be a multiplier of 90")
        for i in range(len(self.frames)):
            self.frames[i].rotate(angle) 
            
        


    #Remove the black frame around each frame of the video if there is one
    def remove_black_frame(self):
        middle_frame = self.frames[int(len(self.frames)/2)].pix_vals
        start = 0
        end=self.width-1
        black_pix = [0, 0, 0]
        middle_index=int(self.height/2)
        
        #take the frame that is in the middle of the video, initialize to the pixel at mid-height and on the left 
        #and go to the right as long as the pixel is black
        #Pointer will then be stopped at the begining of the intersting x-location of the frame
        while start < self.width and middle_frame[middle_index][start].tolist() == black_pix:
            start+=1
        #Same process than above starting from the right to determine the end of the interesting x-location of the frame
        while end >= 0 and middle_frame[middle_index][end].tolist() == black_pix:
            end-=1
        #Crop if black frame detected
        if not (end < 0 or start >= self.width or end > self.width-10 or start < 10):
            for i in range(len(self.frames)):
                self.frames[i].crop(0, self.height-1, start, end, inplace=True)
            self.width = self.frames[0].pix_vals.shape[1]
            self.height = self.frames[0].pix_vals.shape[0]
            
            
        
    
        
            

#Class handling regular image processes (for video frames)
class Image(object):
    def __init__(self, img=None, path=None, name=None):
        if img is not None:
            self.pix_vals = img
        elif path is not None:
            self.pix_vals = cv2.imread(path)
        else:
            raise ValueError("Provide either an image array (img) or a file (path)")
        if name is not None:
            self.name = name
        else:
            self.name = "IMG"


    #Detect the face on the image and crop if found
    #Use three different models: 'haarcascade', 'dlib' or 'both'
    #'both' being a tradeoff using the less robust but quicker one first (haarcascade) and if not found, apply the second
    def face_detect(self, mode, model):
        img, self.face_detected = face_detection(self.pix_vals.copy(), mode, model)
        if self.face_detected == False and model=='both':
            img, self.face_detected = face_detection(self.pix_vals.copy(), mode, 'dlib')
        self.pix_vals = img
            
        return img.shape[1], img.shape[0]



    
    def convert_to_gray(self): 
        self.pix_vals = cv2.cvtColor(self.pix_vals, cv2.COLOR_BGR2GRAY)



    def resize(self, w, h):
        self.pix_vals = cv2.resize(self.pix_vals,(w,h))

    def rotate(self, angle, keep_same_dim=False):
        if keep_same_dim == False:
            k = angle/90
            rotated = np.rot90(self.pix_vals, k)
            self.pix_vals = rotated

        else:
            (h, w) = self.pix_vals.shape[:2]
            scale=h/w
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            self.pix_vals = cv2.warpAffine(self.pix_vals, M, (w, h))
        
        
        
    def crop(self, yi, yf, xi, xf, inplace=True):
        img = self.pix_vals[yi:yf, xi:xf]
        
        if inplace == True:
            self.pix_vals = img
        else:
            return img
        return 1
    
    def emotions_prediction(self):
        
        self.emotions=EmoDetector()
        #self.resize(48, 48)
        self.resize(64, 64)
        #self.convert_to_gray()
        self.emotions.predict_emotion(self.pix_vals[np.newaxis, :, :, np.newaxis])
        
        
        
        
