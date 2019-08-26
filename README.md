# HOMER

This repository contains the code and necessary files to build and run the Highlight Origination using Multimodal Emotion Recognition (HOMER).

HOMER is a system which, provided with both an original video and the corresponding video of the user's face, allows to derive the highlight of the original video. HOMER was designed as a web-service with a provided API along with two Android application examples.

The contents of this repository are summarized below.

* **Web-service**: Contains all the necessary files that need to be uploaded on a server to perform highlight extraction.
	* src: python files to run on the server to retrieve the data stream from any pre-specified client and generate either a single highlight or a montage of many highlights.
	* HL_extraction: all .py files called by the source files and perform the highlight extraction.
	* Models: Open-source models used by HOMER for the emotion recognition, speech recognition and face detection
* **Applications**: Contains all the the necessary files to mount two Android applications on a HTC One M8 phone.
	* DualCamera: source files for the first application that allows to automatically generate a highlight after having recorded a video.
	* Montage: source files for the second application that submits to the user his own videos in a certain time range and generates a montage of the extracted highlights.
	
	
## Set-up and build HOMER web-service

All the details for setting up the system and build it on your server is explained in the **HOMER/Web-service/** folder.

## Run HOMER applications

 Both Android applications are ran on the HTC One M8 phone by following the steps described in the **HOMER/Applications** folder.