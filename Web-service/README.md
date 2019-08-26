# HOMER---Web-service

This repository contains the code and necessary files to build and run the Highlight Origination using Multimodal Emotion Recognition (HOMER) web-service. 
Two different modes can be set, one corresponding to the first application of the dual camera single highlight generation, and the other corresponding to highlights montage generation.

## Set-up/build

Here are the steps to follow in order to successfully have the HOMER web-service running on any server:
1. Provide your server with either python 2 or 3.
2. Upload on the server all three folders `/src` , `/HL_extraction` and `/Models` along with the `env.yml`  file.
3. Set-up the python environment by either following one of the two approaches:
	* Install conda on your server and create an environment using the *env.yml* file with the following command: **conda env create -f env.yml**
	* Import each library listed in the `env.yml` file on the terminal with pip install.
4. Open each of the three files `/src/montage_server.py`,  `/src/par_server.py` and  `/src/seq_server.py` and edit *SERVER_HOST* and *CLIENT_HOST* parameters of lines 12 and 21 with your own case.
5. In order to prevent any conflict with matplotlib library, run the following command: **export DISPLAY=:0**
6. Run one of the following files:
	* `/src/montage_server.py`: for running Montage application &amp; generate a montage video file containing the highlights of all the videos presented in input
	* `/src/par_server.py`: for running DualCamera application in parallel -> generate a single highlight of the video presented in input (along with the user's face video). Up to two video highlight extraction can be performed in parallel, decreasing the computation time
	* `/src/seq_server.py`: for running DualCamera application sequentially.
7. The sever is now listening on its ports any incoming data from the client having the following respective parameters: 
	* *Scene Video*: video to be highlight extracted. Both video frames (10fps) and audio signals (1 kHz) are extracted as distinct raw signals inputs for the highlight extraction pipeline.
	* *Facial Video*: video of the user's face recorded in reaction to the recorded scene. Its duration must then be equal to the *Scene video* duration. Only frames are extracted from this video (4fps) to further derive user's emotions.
	* *T<sup>min</sup><sub>HL</sub>*: parameter indicating the desired highlight minimal duration. If set to -1, the algorithm will decide automatically the highlight length by itself.
	* *T<sup>max</sup><sub>HL</sub>*: parameter indicating the desired highlight maximal duration. If set to -1, the algorithm will decide automatically the highlight length by itself.
	* *&Gamma;<sub>mult</sub>*: boolean deciding whether multiple highlights can be generated or not.
