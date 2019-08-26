## Imports ##

import socket
import sys
import os
import ast
import threading
import shutil
from multiprocessing import Process, Queue

## Globals ##

SERVER_HOST = '23.239.22.55'   # IP address of the ICSL server
# SERVER_HOST = '209.2.214.30'   # IP address of my laptop
BACK_PORT = 8274               # port number for the back camera video
FRONT_PORT = 8933              # port number for the front camera video
CONTROL_PORT = 8186            # port number for any control information
HIGHLIGHT_PORT = 8844          # port number for the highlight video
BACKLOG = 10                   # number of unacceptable connections before refusing to accept new connections
SERVER_BUFSIZE = 4096          # buffer size for receiving data from client

CLIENT_HOST = '160.39.200.228' # IP address of the HTC One M8
CLIENT_BUFSIZE = 1024          # buffer size for sending data to client

CONTROL_DICT = dict()          # dictionary holding settings for algorithm

## Arg Functions ##

file_index = 0

def getPrefixDir(index):
    return '../Montage_Video/%s' % str(index)

def getFrontCameraDir(index):
    return os.path.join(getPrefixDir(index), 'front')

def getFrontCameraFilename(index):
    return os.path.join(getFrontCameraDir(index), 'front_camera.mp4')
    
def getBackCameraDir(index):
    return os.path.join(getPrefixDir(index), 'back')

def getBackCameraFilename(index):
    return os.path.join(getBackCameraDir(index), 'back_camera.mp4')

def getHighlightFilename(index):
    return os.path.join(getHighlightDir(index), getBackCameraFilename(index).split('/')[-1])

def getHighlightDir(index):
    return os.path.join(getPrefixDir(index), 'highlight')

def getMinDuration():
    if ('min_duration' in CONTROL_DICT):
        return str(min(CONTROL_DICT['min_duration'], CONTROL_DICT['max_duration']))
    else:
        return '-1'

def getMaxDuration():
    if ('max_duration' in CONTROL_DICT):
        return str(max(CONTROL_DICT['max_duration'], CONTROL_DICT['min_duration']))
    else:
        return '-1'
        
def getMontageStageDir():
    return '../Montage_Video/montage_highlights'
    
def getMontageFilename():
    dir = '../Montage_Video/montage'
    if (not os.path.exists(dir)):
        os.mkdir(dir)
    return os.path.join(dir, 'montage.mp4')

## Server Functions ##

def initSocket(port):
    # Create socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('[receiveData] Socket created')
    # Bind socket
    try:
        s.bind((SERVER_HOST, port))
    except socket.error as err:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((SERVER_HOST, port))
        except socket.error as err:
            print('[receiveData] Bind failed with %s: %s; %s' % (SERVER_HOST, port, err))
            sys.exit()
    print('[receiveData] Socket bind success with %s: %s' % (SERVER_HOST, port))
    # Enable socket listening
    s.listen(BACKLOG)
    print('[receiveData] Socket is now listening')
    return s
    
back_socket = initSocket(BACK_PORT)
front_socket = initSocket(FRONT_PORT)
control_socket = initSocket(CONTROL_PORT)
    
def receiveData(s):
    buffer = bytearray()
    while (True):
        conn, addr = s.accept()
        if (addr[0] != CLIENT_HOST):
            print('[receiveData] Invalid connection address: %s: %s' % (addr[0], str(addr[1])))
            continue
        else:
            print('[receiveData] Connected with %s: %s' % (addr[0], str(addr[1])))
        while (True):
            data = conn.recv(SERVER_BUFSIZE)
            if (data):
                print('[receiveData] Received %d bytes' % len(data))
                buffer += bytearray(data)
            else:
                print('[receiveData] Finished receiving data')
                return buffer
    
def receiveBackVideoData():
    global back_socket
    return receiveData(back_socket)
    
def receiveFrontVideoData():
    global front_socket
    return receiveData(front_socket)

def receiveControlData():
    global control_socket
    return receiveData(control_socket)
    
def decodeBinaryToMP4(buffer, write_filename):
    try:
        print('Writing binary to .mp4')
        write_file = open(write_filename, 'wb+')
        write_file.write(buffer)
        write_file.close()
        print('Finished writing to %s' % write_filename)
    except:
        print('Error decoding binary to mp4')
    
def decodeBinaryToDict(buffer):
    try:
        s = bytes(buffer).decode('utf-8')
        d = ast.literal_eval(s)
        return d
    except:
        print('Error decoding binary to dictionary')
        
class BackVideoThread(threading.Thread):
    def run(self):
        index = self._kwargs['index']
        back_video_data = receiveBackVideoData()
        decodeBinaryToMP4(back_video_data, getBackCameraFilename(index))
    
class FrontVideoThread(threading.Thread):
    def run(self):
        index = self._kwargs['index']
        front_video_data = receiveFrontVideoData()
        decodeBinaryToMP4(front_video_data, getFrontCameraFilename(index))

class ControlThread(threading.Thread):
    d = dict()
    def run(self):
        self.d = decodeBinaryToDict(receiveControlData())
        
    def getDict(self):
        return self.d

def serverWrapper():
    global CONTROL_DICT
    global file_index
    # Make the new directory to hold the data (if necesssary)
    if (not os.path.exists(getPrefixDir(file_index))):
        os.mkdir(getPrefixDir(file_index))
    if (not os.path.exists(getFrontCameraDir(file_index))):
        os.mkdir(getFrontCameraDir(file_index))
    if (not os.path.exists(getBackCameraDir(file_index))):
        os.mkdir(getBackCameraDir(file_index))
    if (not os.path.exists(getHighlightDir(file_index))):
        os.mkdir(getHighlightDir(file_index))
    control_thread = ControlThread(name='Control-Thread')
    back_video_thread = BackVideoThread(name='Back-Video-Thread', kwargs={'index':file_index})
    front_video_thread = FrontVideoThread(name='Front-Video-Thread', kwargs={'index':file_index})
    threads = [control_thread, back_video_thread, front_video_thread]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    CONTROL_DICT = control_thread.getDict()
    if ('done' in CONTROL_DICT and bool(CONTROL_DICT['done']) == True):
        done = True
    else:
        done = False
    q.put((file_index, done))
    file_index += 1

## Client Functions ##

def sendData(highlight_filename):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('[sendData] Socket created')
    s.connect((CLIENT_HOST, HIGHLIGHT_PORT))
    print('[sendData] Connected with %s: %s' % (CLIENT_HOST, HIGHLIGHT_PORT))
    with open(highlight_filename, 'rb') as f:
        bytes = f.read(CLIENT_BUFSIZE)
        while (bytes):
            print('[sendData] Sending %d bytes' % len(bytes))
            s.send(bytes)
            bytes = f.read(CLIENT_BUFSIZE)
    print('[sendData] Finished sending data')

def clientWrapper(index):
    while (True):
        # get(block=True) blocks until an item is available
        (index, done) = q.get(block=True)
        genHighlights(index)
        if (not os.path.exists(getMontageStageDir())):
            os.mkdir(getMontageStageDir())
        # 0_0.mp4, 0_1.mp4, 1_0.mp4, etc.
        idx = 0
        for file in os.listdir(getHighlightDir(index)):
            if (file.lower().endswith('.mp4')):
                src = os.path.join(getHighlightDir(index), file)
                dst = os.path.join(getMontageStageDir(), str(index) + '_' + str(idx) + '.mp4')
                shutil.copy(src, dst)
                idx += 1
        # Check if all highlights are generated
        if (done):
            # Call montage generation API
            # Send montage back to client
            args = [getMontageStageDir(), getMontageFilename()]
            command = 'python3 montage.py ' + ' '.join(args)
            print('Executing %s' % command)
            os.system(command)
            sendData(getMontageFilename())

## Highlight Generation Functions ##

def genHighlights(index):
    # arg1 : filepath to back video
    # arg2 : filepath to front video
    # arg3 : filepath to directory to put highlight
    # arg4 : int representing min highlight duration
    # arg5 : int representing max highlight duration
    # arg6 : bool to allow multiple highlights or not 
    args = [getBackCameraFilename(index), getFrontCameraFilename(index), getHighlightDir(index), 
            getMinDuration(), getMaxDuration(), 'True']
    command = 'python3 main.py ' + ' '.join(args)
    print('Executing %s' % command)
    os.system(command)

## Main ##

if __name__ == '__main__':
    q = Queue()
    # Spawn process to generate highlights and send data
    p = Process(target=clientWrapper, args=(q,))
    p.start()
    # Listen for incoming data
    while (True):
        serverWrapper()
        print(CONTROL_DICT)
